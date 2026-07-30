"""Multipitch (BP posteriorgram MSE)."""
from __future__ import annotations

import numpy as np
import torch

from polyddsp.metrics.multipitch import multipitch_mse


def test_identical_audio_zero_mse(four_s_sine: torch.Tensor) -> None:
    out = multipitch_mse(four_s_sine, four_s_sine)
    for k in ("multipitch_onset", "multipitch_contour", "multipitch_note", "multipitch_mean"):
        assert k in out
        np.testing.assert_allclose(out[k], np.zeros(2), atol=1e-6)


def test_per_example_shape(four_s_sine: torch.Tensor) -> None:
    out = multipitch_mse(four_s_sine, four_s_sine * 0.5)
    assert out["multipitch_mean"].shape == (2,)
