"""Decoder structural checks (residual + raw-logit head + joint output)."""
from __future__ import annotations

import torch

from polyddsp.model.decoder import MonoDecoder


def test_mono_decoder_post_mlp_takes_residual_skip() -> None:
    """The post-MLP consumes `cat(gru_input, gru_output)`, not the GRU output
    alone — the residual skip DDSP's `RnnFcDecoder` uses. Verified on the input
    width of the post-MLP's first Linear."""
    dec = MonoDecoder(use_z=False, gru_hidden=512, mlp_hidden=512, use_velocity=True)
    # cat_dim = (pitch, velocity, loudness) × mlp_hidden = 3 × 512.
    cat_dim = 3 * 512
    assert dec.post_mlp.net[0].in_features == 512 + cat_dim


def test_mono_decoder_returns_raw_logits() -> None:
    """Heads emit raw logits — synthesisers apply exp_sigmoid + normalize."""
    dec = MonoDecoder(use_z=False, n_harmonics=4, n_bands=8)
    pitch = torch.full((2, 1, 8), 220.0)
    velocity = torch.ones_like(pitch)
    loudness = torch.zeros(2, 8)
    raw_h, amp, raw_n = dec(pitch, velocity, loudness, z=None)
    assert raw_h.shape == (2, 1, 8, 4)
    assert amp.shape == (2, 1, 8)
    assert raw_n.shape == (2, 1, 8, 8)
    # Raw logits — sums won't be 1, values can be outside [0, 1].
    assert not torch.allclose(raw_h.sum(dim=-1), torch.ones_like(raw_h.sum(dim=-1)), atol=1e-3)


def test_mono_decoder_joint_head_couples_amp_and_noise() -> None:
    """Amp logit and noise mags share the final Dense — gradient on either
    backprops through the same trunk weights. Sanity-check that the head
    weight matrix has rows for both."""
    dec = MonoDecoder(use_z=False, n_harmonics=60, n_bands=65)
    # head: Linear(mlp_hidden, 1 + 60 + 65) = (126, 512)
    assert dec.head.weight.shape == (1 + 60 + 65, 512)
