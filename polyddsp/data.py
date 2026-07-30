"""Single RawAudioDataset — handles every experiment via constructor flags.

Each epoch yields one randomly-cropped 4 s clip per non-overlapping 4 s window
contained in the active split's source audio. Train crops are stochastic;
val crops are deterministic per `idx`.
"""
from __future__ import annotations

import functools
import random
from pathlib import Path
from typing import Literal

import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset


def _file_duration(path: Path) -> float:
    """Audio duration in seconds without loading samples."""
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def _load_resample(path: Path, target_sr: int) -> torch.Tensor:
    """Load mono float32 at target_sr."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    audio = torch.from_numpy(wav)
    if sr != target_sr:
        audio = AF.resample(audio, orig_freq=sr, new_freq=target_sr)
    return audio


@functools.lru_cache(maxsize=64)
def _load_cached(path_str: str, target_sr: int) -> torch.Tensor:
    """Per-process cache of resampled full-file audio. ~15 MB per 4-min file."""
    return _load_resample(Path(path_str), target_sr)


@functools.lru_cache(maxsize=64)
def _load_bp_cache_cached(path_str: str, suffix: str) -> dict[str, torch.Tensor]:
    """Per-process cache of full-file BP allocation: {pitch, velocity} of shape (V, T)."""
    audio_path = Path(path_str)
    cache = audio_path.with_suffix(audio_path.suffix + f".{suffix}.f0.pt")
    return torch.load(cache, map_location="cpu", weights_only=True)


def _random_window_with_start(
    audio: torch.Tensor, n_samples: int, rng: random.Random
) -> tuple[torch.Tensor, int]:
    """Random n_samples crop. Zero-pads files shorter than the window. Returns (clip, start)."""
    if audio.shape[-1] >= n_samples:
        start = rng.randint(0, audio.shape[-1] - n_samples)
        return audio[start:start + n_samples].clone(), start
    pad = n_samples - audio.shape[-1]
    return torch.cat([audio, audio.new_zeros(pad)], dim=0), 0


class RawAudioDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val"],
        sample_rate: int = 16_000,
        clip_seconds: float = 4.0,
        seed: int = 0,
        file_glob: str = "**/*.wav",
        pitch_cache_kind: Literal["basic_pitch"] | None = None,
        pitch_cache_suffix: str | None = None,
        f0_hop: int = 64,
        n_voices: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds
        self.clip_samples = int(clip_seconds * sample_rate)
        self.seed = seed
        self.split = split
        self.f0_hop = f0_hop
        self.n_voices = n_voices

        self.pitch_cache_kind = pitch_cache_kind
        self.pitch_cache_suffix = pitch_cache_suffix

        files = sorted(self.root.glob(file_glob))
        if not files:
            raise FileNotFoundError(f"No audio files in {self.root} matching {file_glob}")
        rng = random.Random(seed)
        shuffled = files.copy()
        rng.shuffle(shuffled)
        cut = int(0.8 * len(shuffled))
        self.active = shuffled[:cut] if split == "train" else shuffled[cut:]

        self.file_durations = [_file_duration(p) for p in self.active]
        self.total_duration = float(sum(self.file_durations))
        self._virtual_len = max(int(self.total_duration / clip_seconds), 1)

        if pitch_cache_kind is not None:
            if n_voices is None:
                raise ValueError("pitch_cache_kind='basic_pitch' requires n_voices")
            missing = [
                p for p in self.active
                if not p.with_suffix(p.suffix + f".{pitch_cache_suffix}.f0.pt").exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"pitch cache missing for {len(missing)}/{len(self.active)} files "
                    f"(suffix={pitch_cache_suffix}); run "
                    f"`python -m polyddsp.preprocess --root {self.root} --n-voices {n_voices}`. "
                    f"First missing: {missing[0]}"
                )

    def __len__(self) -> int:
        return self._virtual_len

    def _pick_file_idx(self, rng: random.Random) -> int:
        return rng.choices(range(len(self.active)), weights=self.file_durations, k=1)[0]

    def _rng_for(self, idx: int, salt: str = "") -> random.Random:
        if self.split == "train":
            return random.Random()
        return random.Random(f"{self.seed}-{idx}-{self.split}-{salt}")

    def __getitem__(self, idx: int):
        rng = self._rng_for(idx)
        file_idx = self._pick_file_idx(rng)
        audio_path = self.active[file_idx]
        wav = _load_cached(str(audio_path), self.sample_rate)
        clip, start = _random_window_with_start(wav, self.clip_samples, rng)
        if self.pitch_cache_kind is None:
            return clip
        target_frames = self.clip_samples // self.f0_hop
        f0_start = start // self.f0_hop
        blob = _load_bp_cache_cached(str(audio_path), self.pitch_cache_suffix)
        pitch = blob["pitch"][:, f0_start : f0_start + target_frames]
        velocity = blob["velocity"][:, f0_start : f0_start + target_frames]
        if pitch.shape[-1] < target_frames:
            pad = target_frames - pitch.shape[-1]
            pitch = torch.cat([pitch, pitch.new_zeros(pitch.shape[0], pad)], dim=-1)
            velocity = torch.cat([velocity, velocity.new_zeros(velocity.shape[0], pad)], dim=-1)
        return {"audio": clip, "pitch": pitch, "velocity": velocity}
