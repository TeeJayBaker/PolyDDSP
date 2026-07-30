"""CLAP set-vs-set Fréchet distance."""
from __future__ import annotations

import torch

from polyddsp.metrics.clap import clap_cos, clap_fd

def _noise_batch(n: int, sr: int = 16_000, seconds: float = 4.0) -> torch.Tensor:
    rng = torch.Generator().manual_seed(0)
    return torch.randn(n, int(seconds * sr), generator=rng) * 0.1


def test_identical_distribution_near_zero() -> None:
    audio = _noise_batch(16)
    out = clap_fd(audio, audio)
    assert out["clap_fd"].shape == (1,)
    assert out["clap_fd"][0] < 1.0


def test_clap_cos_identical_audio_is_one() -> None:
    audio = _noise_batch(4)
    out = clap_cos(audio, audio)
    assert out["clap_cos"].shape == (4,)
    assert (out["clap_cos"] > 0.99).all()
