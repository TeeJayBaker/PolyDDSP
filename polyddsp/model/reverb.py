"""Single learnable IR reverb (frequency-domain convolution)."""
from __future__ import annotations

import torch
import torch.nn as nn


class Reverb(nn.Module):
    """One learnable IR shared across the dataset.

    Returns dry + wet. The first IR sample is masked to zero so the dry leak
    from the IR doesn't double-count the dry signal added back at the end.
    """

    def __init__(self, reverb_len: int = 48_000) -> None:
        super().__init__()
        self.reverb_len = reverb_len
        self.ir = nn.Parameter(torch.randn(reverb_len) * 1e-6)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        B, T = audio.shape
        ir = torch.cat([self.ir.new_zeros(1), self.ir[1:]], dim=0)
        ir = ir.unsqueeze(0).expand(B, -1)

        n_fft = 1
        target = T + ir.shape[-1] - 1
        while n_fft < target:
            n_fft <<= 1
        audio_fft = torch.fft.rfft(audio, n=n_fft, dim=-1)
        ir_fft = torch.fft.rfft(ir, n=n_fft, dim=-1)
        wet = torch.fft.irfft(audio_fft * ir_fft, n=n_fft, dim=-1)[:, :T]
        return wet + audio

