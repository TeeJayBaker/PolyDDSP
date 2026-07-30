"""Multi-scale spectral loss (DDSP eq. 4).

Both terms are added: `L = ||S - Ŝ||_1 + α * ||log S - log Ŝ||_1`, summed over
FFT sizes. α is positive, so the log-magnitude term always *increases* the loss.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSpectral(nn.Module):
    """Multi-scale spectral L1 loss with log-magnitude term.

    For each FFT size in `fft_sizes`:
        L_i = || S - Ŝ ||_1  +  α · || log S - log Ŝ ||_1

    Sum across sizes. α defaults to 1.0 per DDSP, and the log term is added
    (never subtracted) — regression-tested in tests/test_loss_sign.py.

    Matches DDSP `core.safe_log` (substitutive epsilon) and
    `tf.signal.stft(pad_end=True)` (right-zero-pad framing) instead of
    PyTorch's `torch.log(x + eps)` and `center=True, pad_mode='reflect'`.
    """

    def __init__(
        self,
        fft_sizes: Iterable[int] = (2048, 1024, 512, 256, 128, 64),
        overlap: float = 0.75,
        alpha: float = 1.0,
        log_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.overlap = overlap
        self.alpha = alpha
        self.log_eps = log_eps

    def _spectrogram(self, audio: torch.Tensor, n_fft: int) -> torch.Tensor:
        hop = int(n_fft * (1.0 - self.overlap))
        n_t = audio.shape[-1]
        # Match `tf.signal.stft(pad_end=True)`: end-only zero-pad to align
        # frame count, no leading symmetric pad. Number of frames is
        # `1 + ceil((n_t - n_fft) / hop)` when `n_t >= n_fft`, else 1.
        if n_t < n_fft:
            n_frames = 1
        else:
            n_frames = 1 + (n_t - n_fft + hop - 1) // hop
        padded_len = (n_frames - 1) * hop + n_fft
        pad_amount = max(padded_len - n_t, 0)
        if pad_amount:
            audio = F.pad(audio, (0, pad_amount))
        window = torch.hann_window(n_fft, device=audio.device, dtype=audio.dtype)
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=False,
            return_complex=True,
        )
        return spec.abs()

    def _safe_log(self, x: torch.Tensor) -> torch.Tensor:
        # Mirrors `ddsp.core.safe_log`: `log(where(x <= 0, eps, x))`.
        # Substitutive — only floors non-positive values; `log(eps)` ≈ -11.5
        # at eps=1e-5 vs additive `log(x+eps)` which biases every magnitude.
        eps = x.new_tensor(self.log_eps)
        return torch.log(torch.where(x <= 0, eps, x))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = x.new_zeros(())
        for n_fft in self.fft_sizes:
            sx = self._spectrogram(x, n_fft)
            sy = self._spectrogram(y, n_fft)
            mag_term = F.l1_loss(sx, sy)
            log_term = F.l1_loss(self._safe_log(sx), self._safe_log(sy))
            total = total + mag_term + self.alpha * log_term
        return total
