"""Cheap tests for the cached_basic_pitch PitchEncoder source."""
from __future__ import annotations

import pytest
import torch

from polyddsp.model.pitch import PitchEncoder


def test_cached_basic_pitch_returns_pitch_and_velocity() -> None:
    enc = PitchEncoder(sr=16000, n_voices=6, target_frame_hop=64, source="cached_basic_pitch")
    audio = torch.zeros(2, 64000)
    pitch_hint = torch.full((2, 6, 1000), 220.0)
    velocity_hint = torch.full((2, 6, 1000), 0.5)
    out = enc(audio, pitch_hint=pitch_hint, velocity_hint=velocity_hint)
    assert out["pitch"].shape == (2, 6, 1000)
    assert out["velocity"].shape == (2, 6, 1000)
    assert torch.equal(out["pitch"], pitch_hint)
    assert torch.equal(out["velocity"], velocity_hint)
    assert out["bp_post"] == {}


def test_cached_basic_pitch_truncates_to_target_frames() -> None:
    enc = PitchEncoder(sr=16000, n_voices=6, target_frame_hop=64, source="cached_basic_pitch")
    audio = torch.zeros(1, 64000)  # → 1000 target frames
    pitch_hint = torch.full((1, 6, 1500), 220.0)
    velocity_hint = torch.full((1, 6, 1500), 0.5)
    out = enc(audio, pitch_hint=pitch_hint, velocity_hint=velocity_hint)
    assert out["pitch"].shape == (1, 6, 1000)
    assert out["velocity"].shape == (1, 6, 1000)


def test_cached_basic_pitch_without_hints_raises() -> None:
    enc = PitchEncoder(sr=16000, n_voices=6, target_frame_hop=64, source="cached_basic_pitch")
    with pytest.raises(RuntimeError, match="pitch_hint and velocity_hint"):
        enc(torch.zeros(1, 64000))


def test_cached_basic_pitch_allows_polyphonic() -> None:
    enc = PitchEncoder(sr=16000, n_voices=10, target_frame_hop=64, source="cached_basic_pitch")
    assert enc.n_voices == 10


def test_unknown_pitch_source_raises() -> None:
    with pytest.raises(ValueError, match="unknown pitch source"):
        PitchEncoder(source="bogus")
