"""MSS L1 metric."""
from __future__ import annotations

import numpy as np
import torch

from polyddsp.metrics.mss import mss_l1


def test_identical_audio_zero(four_s_sine: torch.Tensor) -> None:
    out = mss_l1(four_s_sine, four_s_sine)
    np.testing.assert_allclose(out["mss"], np.zeros(2), atol=1e-6)


def test_per_example_shape(four_s_sine: torch.Tensor) -> None:
    out = mss_l1(four_s_sine, four_s_sine * 0.5)
    assert out["mss"].shape == (2,)
