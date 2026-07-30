"""FAD via VGGish (audio-metrics)."""
from __future__ import annotations

import torch

from polyddsp.metrics.fad import fad

def _noise_batch(n: int, sr: int = 16_000, seconds: float = 4.0) -> torch.Tensor:
    rng = torch.Generator().manual_seed(0)
    return torch.randn(n, int(seconds * sr), generator=rng) * 0.1


def test_identical_distribution_low_fad() -> None:
    audio = _noise_batch(16)
    out = fad(audio, audio)
    assert out["fad"].shape == (1,)
    assert out["fad"][0] < 1.0
