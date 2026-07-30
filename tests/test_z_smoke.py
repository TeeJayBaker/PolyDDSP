"""Z encoder — shape and finiteness."""
from __future__ import annotations

import torch

from polyddsp.model.z import ZEncoder


def test_z_shape_and_finite(four_s_sine: torch.Tensor) -> None:
    enc = ZEncoder(sr=16_000, frame_hop=64, z_dim=16)
    z = enc(four_s_sine)
    assert z.shape == (2, 1_000, 16)
    assert torch.isfinite(z).all()
