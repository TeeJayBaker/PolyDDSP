"""Multi-scale spectral L1 — re-uses the training loss."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from polyddsp.losses import MultiScaleSpectral


def mss_l1(
    ref: torch.Tensor,
    gen: torch.Tensor,
    sr: int = 16_000,
    fft_sizes: Iterable[int] = (2048, 1024, 512, 256, 128, 64),
) -> dict[str, np.ndarray]:
    """Per-example multi-scale spectral L1 (no log term — matches DDSP's eval setup)."""
    loss = MultiScaleSpectral(fft_sizes=fft_sizes, alpha=0.0)
    out = []
    for r, g in zip(ref, gen):
        v = loss(r.unsqueeze(0), g.unsqueeze(0)).item()
        out.append(v)
    return {"mss": np.asarray(out, dtype=np.float64)}
