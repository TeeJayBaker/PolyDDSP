"""Tests for the decoder-conditioning audio preview."""
from __future__ import annotations

import pytest
import torch

from polyddsp.midi_synthesis import render_pitch_image, synthesize_midi


def test_render_pitch_image_draws_active_voices() -> None:
    pitch = torch.zeros(2, 8)
    velocity = torch.zeros_like(pitch)
    pitch[0, 1:5] = 440.0
    velocity[0, 1:5] = 1.0
    pitch[1, 4:7] = 880.0
    velocity[1, 4:7] = 0.5

    image = render_pitch_image(pitch, velocity)

    assert image.shape == (3, 256, 8)
    assert image.min() >= 0.0
    assert image.max() <= 1.0
    # A4 (MIDI 69) and A5 (81) use different voice colours and both appear.
    a4_row = round((127 - 69) * 255 / 127)
    a5_row = round((127 - 81) * 255 / 127)
    assert image[:, a4_row, 2].max() > 0.9
    assert image[:, a5_row, 5].max() > 0.4
    assert image[:, a4_row, 0].max() == 0.0


def test_synthesize_midi_renders_pitch_and_polyphony() -> None:
    sample_rate = 8_000
    frame_hop = 80
    frames = 100
    pitch = torch.zeros(2, 2, frames)
    velocity = torch.zeros_like(pitch)
    pitch[0, 0] = 440.0
    velocity[0, 0] = 0.75
    pitch[0, 1] = 660.0
    velocity[0, 1] = 0.5

    audio = synthesize_midi(
        pitch, velocity, sample_rate=sample_rate, frame_hop=frame_hop,
    )

    assert audio.shape == (2, sample_rate)
    assert torch.isfinite(audio).all()
    assert audio[0].abs().max() == pytest.approx(0.95)
    assert torch.count_nonzero(audio[0]) > 0
    assert torch.count_nonzero(audio[1]) == 0


def test_synthesize_midi_uses_exact_requested_length() -> None:
    pitch = torch.full((1, 1, 3), 440.0)
    velocity = torch.ones_like(pitch)
    audio = synthesize_midi(
        pitch, velocity, sample_rate=16_000, frame_hop=64, n_samples=160,
    )
    assert audio.shape == (1, 160)


def test_synthesize_midi_validates_controls() -> None:
    pitch = torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match="velocity must match"):
        synthesize_midi(
            pitch, torch.ones(1, 1, 3), sample_rate=16_000, frame_hop=64,
        )
