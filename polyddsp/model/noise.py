"""LTV-FIR filtered-noise synth (DDSP §3.4)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def exp_sigmoid(
    x: torch.Tensor,
    exponent: float = 10.0,
    max_value: float = 2.0,
    threshold: float = 1e-7,
) -> torch.Tensor:
    """Exponentiated sigmoid pointwise nonlinearity (DDSP utility).

    Bounds output to [threshold, max_value]; controls slope via `exponent`.
    """
    x = x.float()
    return max_value * torch.sigmoid(x).pow(math.log(exponent)) + threshold


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _apply_hann_window_to_ir(ir: torch.Tensor, window_size: int) -> torch.Tensor:
    """Window an IR (zero-phase) using a Hann window of `window_size`, then
    return the windowed IR in causal form trimmed to the original length.
    """
    ir_size = ir.shape[-1]
    window_size = min(window_size, ir_size)
    window = torch.hann_window(window_size, device=ir.device, dtype=ir.dtype)

    padding = ir_size - window_size
    if padding > 0:
        half = (window_size + 1) // 2
        window = torch.cat(
            [window[half:], torch.zeros(padding, device=ir.device, dtype=ir.dtype), window[:half]],
            dim=0,
        )
    else:
        window = torch.fft.fftshift(window, dim=-1)

    windowed = ir * window
    return torch.fft.fftshift(windowed, dim=-1)


# Added to the raw band logits before `exp_sigmoid`, matching DDSP's
# `ddsp.synths.FilteredNoise(initial_bias=-5.0)`. A decoder head initialised
# near zero would otherwise start at `exp_sigmoid(0) ≈ 0.14` per band — loud
# broadband hiss that the harmonic path has to fight. The −5 offset starts the
# noise floor near silence so the optimiser has to ask for noise explicitly.
_INITIAL_BIAS = -5.0


class FilteredNoise(nn.Module):
    """Apply a frame-wise LTV FIR filter (Hann-windowed) to uniform noise."""

    def __init__(
        self,
        frame_hop: int = 64,
        n_bands: int = 65,
        window_size: int = 0,
    ) -> None:
        super().__init__()
        self.frame_hop = frame_hop
        self.n_bands = n_bands
        self.window_size = window_size

    def forward(self, noise_mags: torch.Tensor) -> torch.Tensor:
        """noise_mags: (B, T_frames, n_bands) raw network outputs."""
        B, T_frames, n_bands = noise_mags.shape
        assert n_bands == self.n_bands
        T_samples = T_frames * self.frame_hop

        magnitudes = exp_sigmoid(noise_mags + _INITIAL_BIAS)

        ir = torch.fft.irfft(magnitudes, dim=-1)  # (B, T_frames, ir_size)
        ws = self.window_size if self.window_size > 0 else ir.shape[-1]
        ir = _apply_hann_window_to_ir(ir, ws)
        ir_size = ir.shape[-1]

        noise = torch.empty(B, T_samples, device=noise_mags.device).uniform_(-1.0, 1.0)
        noise_frames = noise.reshape(B, T_frames, self.frame_hop)

        fft_size = _next_pow2(self.frame_hop + ir_size - 1)
        noise_fft = torch.fft.rfft(noise_frames, n=fft_size, dim=-1)
        ir_fft = torch.fft.rfft(ir, n=fft_size, dim=-1)
        out_fft = noise_fft * ir_fft
        out_frames = torch.fft.irfft(out_fft, n=fft_size, dim=-1)  # (B, T_frames, fft_size)

        # Overlap-add via conv_transpose1d with an identity filter.
        identity = torch.eye(out_frames.shape[-1], device=noise_mags.device).unsqueeze(1)
        oa = F.conv_transpose1d(
            out_frames.transpose(1, 2), identity, stride=self.frame_hop, padding=0
        ).squeeze(1)
        # Group-delay compensation. The Hann-windowed zero-phase IR has its peak
        # at sample (ir_size-1)//2, so OLA output trails the harmonic signal by
        # ~ir_size/2 samples (~4 ms at ir_size=128). Mirrors
        # `ddsp.core.crop_and_compensate_delay(padding='same',
        # delay_compensation=-1)`, which drops `start = (ir_size-1)//2 - 1` from
        # the front to realign noise with the harmonic path.
        start = (ir_size - 1) // 2 - 1
        oa = oa[:, start : start + T_samples]
        return oa
