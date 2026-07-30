"""Loudness L1 metric — re-uses the model's A-weighted extractor."""
from __future__ import annotations

import numpy as np
import torch

from polyddsp.model.loudness import LoudnessExtractor


def loudness_l1(ref: torch.Tensor, gen: torch.Tensor, sr: int = 16_000) -> dict[str, np.ndarray]:
    """Per-frame |ref - gen| averaged over time → per-example value."""
    extractor = LoudnessExtractor(sr=sr).to(ref.device)
    a = extractor(ref, normalise=False)
    b = extractor(gen, normalise=False)
    diff = (a - b).abs().mean(dim=-1)
    return {"loudness_l1": diff.detach().cpu().numpy()}
