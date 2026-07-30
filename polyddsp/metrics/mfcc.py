"""MFCC L1 metric (n_mfcc=30, matches Z-encoder config)."""
from __future__ import annotations

import numpy as np
import torch
import torchaudio


def mfcc_l1(ref: torch.Tensor, gen: torch.Tensor, sr: int = 16_000) -> dict[str, np.ndarray]:
    transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=30,
        log_mels=True,
        melkwargs=dict(n_fft=2048, hop_length=64, n_mels=128, f_min=20.0, f_max=8_000.0),
    ).to(ref.device)
    a = transform(ref)
    b = transform(gen)
    diff = (a - b).abs().mean(dim=(-2, -1))
    return {"mfcc": diff.detach().cpu().numpy()}
