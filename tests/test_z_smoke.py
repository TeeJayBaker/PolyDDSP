"""Z encoder — shape, trailing frame alignment, and causal conditioning."""
from __future__ import annotations

import pytest
import torch

from polyddsp.model.z import ZEncoder


def test_z_shape_and_finite(four_s_sine: torch.Tensor) -> None:
    enc = ZEncoder(sr=16_000, frame_hop=64, z_dim=16)
    z = enc(four_s_sine)
    assert z.shape == (2, 1_000, 16)
    assert torch.isfinite(z).all()


def test_default_mfcc_window_and_filterbank() -> None:
    enc = ZEncoder(gru_hidden=16)
    spectrogram = enc.mfcc.MelSpectrogram.spectrogram
    assert spectrogram.win_length / enc.sr == 0.128
    assert spectrogram.hop_length / enc.sr == 0.004
    assert not spectrogram.center
    filters = enc.mfcc.MelSpectrogram.mel_scale.fb
    assert torch.all(filters.amax(dim=0) > 0)


def test_mfcc_uses_only_completed_control_blocks() -> None:
    """Changing audio from 32 ms onward cannot affect the first eight frames."""
    torch.manual_seed(11)
    enc = ZEncoder(gru_hidden=16)
    audio = torch.randn(1, 4_096) * 0.02
    changed = audio.clone()
    changed[:, 512:] = torch.randn(1, 3_584) * 0.5  # from 32 ms onward

    with torch.no_grad():
        original_mfcc = enc.mfcc_frames(audio)
        changed_mfcc = enc.mfcc_frames(changed)

    torch.testing.assert_close(original_mfcc[..., :8], changed_mfcc[..., :8], rtol=0, atol=0)
    assert (original_mfcc[..., 32:] - changed_mfcc[..., 32:]).abs().max().item() > 1e-3


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("prefix_samples", [64, 1_984, 4_096, 4_127])
def test_appending_audio_preserves_existing_z(training: bool, prefix_samples: int) -> None:
    """Analysis, normalization, frame timing, and the GRU must all be causal."""
    torch.manual_seed(7)
    enc = ZEncoder(gru_hidden=16).train(training)
    audio = torch.randn(1, 8_192) * 0.02

    with torch.no_grad():
        prefix_z = enc(audio[:, :prefix_samples])
        full_z = enc(audio)

    frames = prefix_samples // enc.frame_hop
    assert prefix_z.shape == (1, frames, enc.z_dim)
    torch.testing.assert_close(prefix_z, full_z[:, :frames], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_samples", [0, 1, 63, 64, 65, 2_047, 2_048, 2_049])
def test_startup_and_partial_hops(n_samples: int) -> None:
    enc = ZEncoder(gru_hidden=16)
    with torch.no_grad():
        z = enc(torch.zeros(1, n_samples))
    assert z.shape == (1, n_samples // enc.frame_hop, enc.z_dim)
    assert torch.isfinite(z).all()


def test_z_gradient_does_not_reach_future_audio() -> None:
    enc = ZEncoder(gru_hidden=16)
    audio = (torch.randn(1, 4_096) * 0.02).requires_grad_()
    z = enc(audio)
    grad, = torch.autograd.grad(z[:, 31].square().sum(), audio)
    assert torch.count_nonzero(grad[:, :2_048]) > 0
    assert torch.count_nonzero(grad[:, 2_048:]) == 0
