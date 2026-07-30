"""Loudness extractor — shape and finiteness."""
from __future__ import annotations

import torch

from polyddsp.model.loudness import LoudnessExtractor


def test_loudness_shape_and_finite(four_s_sine: torch.Tensor) -> None:
    extractor = LoudnessExtractor(sr=16_000, frame_hop=64)
    out = extractor(four_s_sine)
    assert out.shape == (2, 1_000)
    assert torch.isfinite(out).all()


def test_loudness_normalisation_applies_static_scale(four_s_sine: torch.Tensor) -> None:
    """normalise=True must apply DDSP's static (db/80)+1 mapping."""
    extractor = LoudnessExtractor(sr=16_000, frame_hop=64)
    raw = extractor(four_s_sine, normalise=False)
    norm = extractor(four_s_sine, normalise=True)
    assert torch.allclose(norm, (raw / 80.0) + 1.0, atol=1e-6)
