"""MFCC L1 metric."""
from __future__ import annotations

import numpy as np
import torch

from polyddsp.metrics.mfcc import mfcc_l1


def test_identical_audio_zero(four_s_sine: torch.Tensor) -> None:
    out = mfcc_l1(four_s_sine, four_s_sine)
    np.testing.assert_allclose(out["mfcc"], np.zeros(2), atol=1e-6)


def test_per_example_shape(four_s_sine: torch.Tensor) -> None:
    out = mfcc_l1(four_s_sine, four_s_sine * 0.5)
    assert out["mfcc"].shape == (2,)
