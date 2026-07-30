"""Decoder shapes and finiteness."""
from __future__ import annotations

import torch

from polyddsp.model.decoder import MonoDecoder


def test_mono_decoder_shapes(fake_features: dict) -> None:
    dec = MonoDecoder(z_dim=16, n_harmonics=100, n_bands=65, use_z=True)
    harm_dist, amp_v, noise_mags = dec(**fake_features)
    B, V, T = 2, 2, 1_000
    assert harm_dist.shape == (B, V, T, 100)
    assert amp_v.shape == (B, V, T)
    assert noise_mags.shape == (B, V, T, 65)
    # harm_dist + noise_mags are raw logits; finiteness only.
    assert torch.isfinite(harm_dist).all()
    assert torch.isfinite(noise_mags).all()
    assert (amp_v >= 0).all()


def test_mono_decoder_no_z(fake_features: dict) -> None:
    feats = {**fake_features, "z": None}
    dec = MonoDecoder(z_dim=16, n_harmonics=100, n_bands=65, use_z=False)
    harm_dist, amp_v, noise_mags = dec(**feats)
    assert harm_dist.shape == (2, 2, 1_000, 100)
    assert noise_mags.shape == (2, 2, 1_000, 65)


def test_mono_decoder_solo_no_velocity(fake_features: dict) -> None:
    """Solo case: use_velocity=False drops the velocity input MLP entirely."""
    feats = {**fake_features, "z": None}
    dec = MonoDecoder(
        z_dim=16, n_harmonics=60, n_bands=65, use_z=False, use_velocity=False
    )
    harm_dist, amp_v, noise_mags = dec(**feats)
    assert harm_dist.shape == (2, 2, 1_000, 60)
    assert noise_mags.shape == (2, 2, 1_000, 65)
    assert not hasattr(dec, "vel_mlp")
