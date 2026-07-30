"""Dataset returns {audio, pitch, velocity} when bp cache present."""
from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf
import torch

from polyddsp.data import RawAudioDataset
from polyddsp.preprocess import bp_cache_suffix, cache_path_for


def _write_silence(path: Path, sr: int = 16000, duration_s: int = 6) -> None:
    sf.write(str(path), torch.zeros(sr * duration_s).numpy(), sr)


def test_dataset_returns_pitch_velocity_when_bp_cache_present(tmp_path: Path) -> None:
    sr, hop, n_voices = 16000, 64, 6
    audio = tmp_path / "fake.wav"
    _write_silence(audio, sr=sr, duration_s=6)

    suffix = bp_cache_suffix(n_voices, sr, hop)
    n_target_frames_full = (sr * 6) // hop
    cache = cache_path_for(audio, suffix)
    torch.save({
        "pitch": torch.full((n_voices, n_target_frames_full), 220.0),
        "velocity": torch.full((n_voices, n_target_frames_full), 0.7),
    }, cache)

    ds = RawAudioDataset(
        root=str(tmp_path), split="val", sample_rate=sr, clip_seconds=4.0,
        seed=0, file_glob="*.wav",
        pitch_cache_kind="basic_pitch", pitch_cache_suffix=suffix,
        f0_hop=hop, n_voices=n_voices,
    )
    item = ds[0]
    assert set(item) == {"audio", "pitch", "velocity"}
    assert item["audio"].shape == (4 * sr,)
    assert item["pitch"].shape == (n_voices, 4 * sr // hop)
    assert item["velocity"].shape == (n_voices, 4 * sr // hop)


def test_dataset_pads_when_file_shorter_than_window(tmp_path: Path) -> None:
    sr, hop, n_voices = 16000, 64, 6
    audio = tmp_path / "short.wav"
    _write_silence(audio, sr=sr, duration_s=2)  # shorter than 4 s clip

    suffix = bp_cache_suffix(n_voices, sr, hop)
    n_target_frames_full = (sr * 2) // hop
    torch.save({
        "pitch": torch.full((n_voices, n_target_frames_full), 220.0),
        "velocity": torch.full((n_voices, n_target_frames_full), 0.7),
    }, cache_path_for(audio, suffix))

    ds = RawAudioDataset(
        root=str(tmp_path), split="val", sample_rate=sr, clip_seconds=4.0,
        seed=0, file_glob="*.wav",
        pitch_cache_kind="basic_pitch", pitch_cache_suffix=suffix,
        f0_hop=hop, n_voices=n_voices,
    )
    item = ds[0]
    assert item["pitch"].shape == (n_voices, 4 * sr // hop)


def test_dataset_raises_clearly_when_bp_cache_missing(tmp_path: Path) -> None:
    sr, hop, n_voices = 16000, 64, 6
    _write_silence(tmp_path / "fake.wav", sr=sr)
    with pytest.raises(FileNotFoundError, match="pitch cache missing"):
        RawAudioDataset(
            root=str(tmp_path), split="val", sample_rate=sr, clip_seconds=4.0,
            seed=0, file_glob="*.wav",
            pitch_cache_kind="basic_pitch",
            pitch_cache_suffix=bp_cache_suffix(n_voices, sr, hop),
            f0_hop=hop, n_voices=n_voices,
        )
