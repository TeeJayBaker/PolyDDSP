"""Smoke + idempotence tests for the Basic Pitch precompute cache."""
from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf
import torch

from polyddsp.preprocess import bp_cache_suffix, cache_path_for, precompute_one_bp


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
