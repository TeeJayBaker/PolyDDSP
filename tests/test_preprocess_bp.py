"""Smoke + idempotence tests for the Basic Pitch precompute cache."""
from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf
import torch

from polyddsp.preprocess import (
    MIDI_VELOCITY_64,
    bp_cache_suffix,
    cache_path_for,
    neutone_cache_suffix,
    precompute_one_bp,
    precompute_one_neutone_amt,
)


def _make_chirp_wav(path: Path, duration_s: float = 6.0, sr: int = 16000) -> None:
    t = torch.linspace(0, duration_s, int(duration_s * sr))
    # Slow chirp 220→440 Hz; should give some BP onsets.
    audio = torch.sin(2 * 3.14159 * (220 + (440 - 220) * (t / duration_s)) * t).numpy()
    sf.write(str(path), audio, sr)


def test_precompute_one_bp_writes_cache_with_correct_shape(tmp_path: Path) -> None:
    audio = tmp_path / "chirp.wav"
    _make_chirp_wav(audio, duration_s=6.0, sr=16000)
    cache, recomputed = precompute_one_bp(
        audio, sample_rate=16000, target_hop=64, n_voices=4, device="cpu",
    )
    assert recomputed
    assert cache.exists()
    blob = torch.load(cache, weights_only=True)
    assert set(blob) == {"pitch", "velocity"}
    expected_target_frames = (16000 * 6) // 64
    assert blob["pitch"].shape == (4, expected_target_frames)
    assert blob["velocity"].shape == (4, expected_target_frames)


def test_precompute_one_bp_is_idempotent(tmp_path: Path) -> None:
    audio = tmp_path / "chirp.wav"
    _make_chirp_wav(audio, duration_s=4.0, sr=16000)
    _, first = precompute_one_bp(audio, 16000, 64, n_voices=4, device="cpu")
    _, second = precompute_one_bp(audio, 16000, 64, n_voices=4, device="cpu")
    assert first is True
    assert second is False


def test_bp_cache_suffix_includes_voices_sr_hop() -> None:
    assert bp_cache_suffix(n_voices=6, sample_rate=16000, target_hop=64) == "bp_v6_sr16000_hop64"


def test_neutone_cache_suffix_includes_voices_sr_hop() -> None:
    assert neutone_cache_suffix(6, 16000, 64) == "neutone_v6_sr16000_hop64"


def test_cache_path_for_mirrors_nested_path_in_output_dir(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    audio = root / "player_00" / "take.wav"
    output_dir = tmp_path / "pitch-cache"

    cache = cache_path_for(
        audio,
        "bp_v6_sr16000_hop64",
        root=root,
        output_dir=output_dir,
    )

    assert cache == output_dir / "player_00" / "take.wav.bp_v6_sr16000_hop64.f0.pt"


def test_precompute_one_neutone_amt_uses_constant_midi_velocity(tmp_path: Path) -> None:
    class FakeSpec:
        hop_length = 512

    class FakeModel:
        spec = FakeSpec()
        delays: list[int] = []
        target_shift = 0

        def __call__(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
            shape = (1, 88, 90)
            onset = torch.full(shape, -20.0)
            frame = torch.full(shape, -20.0)
            offset = torch.full(shape, -20.0)
            pitch_idx = 60 - 21
            onset[0, pitch_idx, 10] = 20.0
            frame[0, pitch_idx, 10:30] = 20.0
            offset[0, pitch_idx, 30] = 20.0
            return {"onset": onset, "frame": frame, "offset": offset}

    audio = tmp_path / "note.wav"
    sf.write(str(audio), torch.zeros(44_100).numpy(), 44_100)
    cache, recomputed = precompute_one_neutone_amt(
        audio,
        sample_rate=16_000,
        target_hop=64,
        n_voices=4,
        device="cpu",
        model=FakeModel(),
    )

    assert recomputed
    blob = torch.load(cache, weights_only=True)
    active = blob["pitch"] != 0
    assert active.any()
    assert torch.all(blob["velocity"][active] == MIDI_VELOCITY_64)
    assert torch.all(blob["velocity"][~active] == 0)
    assert blob["pitch"].shape == (4, 250)
