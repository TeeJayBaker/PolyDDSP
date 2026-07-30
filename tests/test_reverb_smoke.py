"""Reverb — shape, finiteness, dry-mask sanity."""
from __future__ import annotations

import torch

from polyddsp.model.reverb import Reverb


def test_reverb_shape_and_finite(four_s_sine: torch.Tensor) -> None:
    rev = Reverb(reverb_len=64_000)
    out = rev(four_s_sine)
    assert out.shape == four_s_sine.shape
    assert torch.isfinite(out).all()


def test_reverb_first_sample_masked() -> None:
    """Dry-mask: ir[0] = 0 by construction so wet doesn't double-count dry.

    Reverb output is `dry + wet`, so firing an impulse in makes sample 0 equal
    `1.0 (dry) + ir[0] (wet)`. With the mask in place that's exactly 1.0.
    """
    rev = Reverb(reverb_len=64_000)
    with torch.no_grad():
        rev.ir.data = torch.randn_like(rev.ir.data)
    impulse = torch.zeros(1, 64_000)
    impulse[0, 0] = 1.0
    out = rev(impulse)
    assert abs(out[0, 0].item() - 1.0) < 1e-6
