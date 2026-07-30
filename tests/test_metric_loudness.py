"""Loudness L1 metric."""
from __future__ import annotations

import numpy as np
import torch

from polyddsp.metrics.loudness import loudness_l1


def test_identical_audio_zero_l1(four_s_sine: torch.Tensor) -> None:
    out = loudness_l1(four_s_sine, four_s_sine)
    assert "loudness_l1" in out
    np.testing.assert_allclose(out["loudness_l1"], np.zeros(2), atol=1e-6)


def test_returns_per_example_array(four_s_sine: torch.Tensor) -> None:
    other = four_s_sine * 0.5
    out = loudness_l1(four_s_sine, other)
    assert out["loudness_l1"].shape == (2,)
