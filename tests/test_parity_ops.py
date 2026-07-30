"""Numeric checks for `polyddsp.model.parity_ops`."""
from __future__ import annotations

import torch
import pytest

from polyddsp.model.parity_ops import (
    F0_RANGE,
    DB_RANGE,
    hz_to_midi,
    midi_to_hz,
    scale_f0_hz,
    scale_db,
    normalize_harmonics,
    upsample_with_windows,
    safe_divide,
)


def test_hz_to_midi_a440():
    assert hz_to_midi(torch.tensor(440.0)).item() == pytest.approx(69.0, abs=1e-5)
    assert hz_to_midi(torch.tensor(880.0)).item() == pytest.approx(81.0, abs=1e-5)


def test_hz_to_midi_zero_clamped_to_zero():
    assert hz_to_midi(torch.tensor(0.0)).item() == pytest.approx(0.0, abs=1e-3)


def test_midi_to_hz_inverse():
    midi = torch.tensor([21.0, 60.0, 69.0, 108.0])
    assert torch.allclose(hz_to_midi(midi_to_hz(midi)), midi, atol=1e-4)


def test_scale_f0_hz_a440():
    assert scale_f0_hz(torch.tensor(440.0)).item() == pytest.approx(69.0 / F0_RANGE, abs=1e-5)


def test_scale_db_endpoints():
    assert scale_db(torch.tensor(0.0)).item() == pytest.approx(1.0)
    assert scale_db(torch.tensor(-DB_RANGE)).item() == pytest.approx(0.0)


def test_safe_divide_zero_denominator_matches_ddsp():
    """DDSP swaps eps in for den only where den==0; sub-eps non-zero den is untouched."""
    a = torch.tensor([1.0, 2.0, 1.0])
    b = torch.tensor([0.0, 4.0, 1e-9])
    out = safe_divide(a, b, eps=1e-7)
    assert out[0].item() == pytest.approx(1.0 / 1e-7, rel=1e-5)  # num/eps, not 0
    assert out[1].item() == pytest.approx(0.5)
    # 1e-9 < eps but is NOT zero → untouched, returns 1.0 / 1e-9.
    assert out[2].item() == pytest.approx(1.0 / 1e-9, rel=1e-5)


def test_normalize_harmonics_zero_distribution_still_returns_zeros():
    """All-zero hd must still produce zeros, not eps-divisions blowing up."""
    hd = torch.zeros(1, 1, 4)
    f0 = torch.tensor([[[440.0]]])
    out = normalize_harmonics(hd, f0, sample_rate=16000)
    # 0 / eps = 0, finite. (Was already covered by the existing test, but keep
    # an explicit check now that safe_divide semantics changed.)
    assert torch.isfinite(out).all().item()
    assert out.abs().max().item() == 0.0


def test_normalize_harmonics_masks_above_nyquist_and_sums_to_one():
    # f0 = 4000, sr = 16000 → nyquist = 8000.
    # Harmonics: 4000 (keep), 8000 (drop, >=nyquist), 12000 (drop), 16000 (drop).
    f0 = torch.tensor([[[4000.0]]])
    hd = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]])
    out = normalize_harmonics(hd, f0, sample_rate=16000)
    assert out.shape == (1, 1, 4)
    assert out[0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert out[0, 0, 1:].abs().max().item() == pytest.approx(0.0)


def test_normalize_harmonics_no_mask_when_sr_is_none():
    f0 = torch.tensor([[[1000.0]]])
    hd = torch.tensor([[[0.4, 0.4, 0.2]]])
    out = normalize_harmonics(hd, f0, sample_rate=None)
    assert torch.allclose(out, hd, atol=1e-6)


def test_normalize_harmonics_zero_distribution_returns_zeros():
    # All-zero hd → safe_divide should return zeros, not NaN.
    hd = torch.zeros(1, 1, 4)
    f0 = torch.tensor([[[440.0]]])
    out = normalize_harmonics(hd, f0, sample_rate=16000)
    assert torch.isfinite(out).all().item()
    assert out.abs().max().item() == 0.0


def test_upsample_with_windows_shape():
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # (B=1, T=4, C=1)
    y = upsample_with_windows(x, 64, add_endpoint=True)
    assert y.shape == (1, 64, 1)


def test_upsample_with_windows_constant_input_yields_constant_output():
    x = torch.full((1, 4, 1), 5.0)
    y = upsample_with_windows(x, 64, add_endpoint=True)
    assert torch.allclose(y, torch.full_like(y, 5.0), atol=1e-4)


def test_upsample_with_windows_multi_channel():
    x = torch.randn(2, 5, 3)
    y = upsample_with_windows(x, 100, add_endpoint=True)
    assert y.shape == (2, 100, 3)
