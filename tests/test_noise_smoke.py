"""Filtered-noise synth — shape and finiteness."""
from __future__ import annotations

import torch

from polyddsp.model.noise import FilteredNoise


def test_noise_shape_and_finite() -> None:
    synth = FilteredNoise(frame_hop=64, n_bands=65, window_size=257)
    B, T_frames, n_bands = 2, 1_000, 65
    noise_mags = torch.randn(B, T_frames, n_bands)
    audio = synth(noise_mags)
    assert audio.shape == (B, T_frames * 64)
    assert torch.isfinite(audio).all()


def test_noise_amplitude_bounded() -> None:
    synth = FilteredNoise(frame_hop=64, n_bands=65, window_size=257)
    B, T_frames, n_bands = 2, 1_000, 65
    noise_mags = torch.zeros(B, T_frames, n_bands)
    audio = synth(noise_mags)
    assert audio.abs().max().item() < 1.0


def test_filtered_noise_window_zero_uses_ir_size() -> None:
    fn = FilteredNoise(frame_hop=64, n_bands=65, window_size=0)
    noise_mags = torch.zeros(1, 4, 65)
    out = fn(noise_mags)
    assert out.shape == (1, 4 * 64)
    assert torch.isfinite(out).all()
