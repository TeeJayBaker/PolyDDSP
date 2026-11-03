"""LTV-FIR filtered-noise synth (DDSP §3.4)."""
from __future__ import annotations

import math
from dataclasses import dataclass

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


@dataclass(frozen=True)
class NoiseState:
    """Caller-owned overlap, startup compensation, and independent RNG state."""

    overlap: torch.Tensor
    skip: int
    samples_seen: int
    rng_state: torch.Tensor


class FilteredNoise(nn.Module):
    """Apply a frame-wise LTV FIR filter (Hann-windowed) to uniform noise."""

    def __init__(
        self,
        frame_hop: int = 64,
        n_bands: int = 65,
        window_size: int = 0,
    ) -> None:
        super().__init__()
        if frame_hop < 1 or n_bands < 2 or window_size < 0:
            raise ValueError("frame_hop must be positive, n_bands >= 2, and window_size >= 0")
        self.frame_hop = frame_hop
        self.n_bands = n_bands
        self.window_size = window_size

    @property
    def ir_size(self) -> int:
        return 2 * (self.n_bands - 1)

    @property
    def latency_samples(self) -> int:
        """Samples withheld to preserve the offline group-delay compensation."""
        return max(0, (self.ir_size - 1) // 2 - 1)

    def _filter_frames(self, noise_mags: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Filter excitation blocks; leave their convolution tails intact."""
        B, T_frames, n_bands = noise_mags.shape
        if n_bands != self.n_bands:
            raise ValueError(f"expected {self.n_bands} noise bands, got {n_bands}")
        if noise.shape != (B, T_frames * self.frame_hop):
            raise ValueError("noise must have shape (batch, frames * frame_hop)")
        magnitudes = exp_sigmoid(noise_mags + _INITIAL_BIAS)
        ir = torch.fft.irfft(magnitudes, dim=-1)  # (B, T_frames, ir_size)
        ws = self.window_size if self.window_size > 0 else ir.shape[-1]
        ir = _apply_hann_window_to_ir(ir, ws)
        ir_size = ir.shape[-1]
        noise_frames = noise.to(magnitudes).reshape(B, T_frames, self.frame_hop)
        fft_size = _next_pow2(self.frame_hop + ir_size - 1)
        noise_fft = torch.fft.rfft(noise_frames, n=fft_size, dim=-1)
        ir_fft = torch.fft.rfft(ir, n=fft_size, dim=-1)
        out_fft = noise_fft * ir_fft
        return torch.fft.irfft(out_fft, n=fft_size, dim=-1)

    def _overlap_add(self, frames: torch.Tensor) -> torch.Tensor:
        """Sum filtered blocks without allocating a quadratic identity kernel."""
        B, count, size = frames.shape
        length = (count - 1) * self.frame_hop + size
        return F.fold(
            frames.transpose(1, 2), output_size=(1, length),
            kernel_size=(1, size), stride=(1, self.frame_hop),
        ).reshape(B, length)

    def forward(
        self, noise_mags: torch.Tensor, *, noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Synthesize a clip; optional explicit excitation enables exact comparisons.

        noise_mags: (B, frames, n_bands) raw outputs. noise, when supplied, is
        (B, frames * frame_hop). Otherwise fresh uniform [-1, 1) noise is drawn.
        """
        B, count, _ = noise_mags.shape
        length = count * self.frame_hop
        if count == 0:
            return noise_mags.new_empty(B, 0, dtype=torch.float32)
        if noise is None:
            noise = torch.empty(B, length, device=noise_mags.device).uniform_(-1.0, 1.0)
        oa = self._overlap_add(self._filter_frames(noise_mags, noise))
        # Group-delay compensation. The Hann-windowed zero-phase IR has its peak
        # at sample (ir_size-1)//2, so OLA output trails the harmonic signal by
        # ~ir_size/2 samples (~4 ms at ir_size=128). Mirrors
        # `ddsp.core.crop_and_compensate_delay(padding='same',
        # delay_compensation=-1)`, which drops `start = (ir_size-1)//2 - 1` from
        # the front to realign noise with the harmonic path.
        start = self.latency_samples
        return oa[:, start:start + length]

    @torch.no_grad()
    def stream(
        self,
        noise_mags: torch.Tensor,
        state: NoiseState | None = None,
        *,
        noise: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, NoiseState]:
        """Filter new control frames, retaining overlap and delay compensation.

        Returns only finalized samples. The first latency_samples samples of raw
        convolution are dropped once, not once per call. At EOF, flush(state)
        completes the clip to the same length as forward. None resets a stream.

        A private CPU generator produces frame-major excitation, then copies it
        to the synthesis device. Its saved state makes draws independent of call
        boundaries and interleaved streams. seed is optional on the first call
        only. Explicit noise bypasses the generator and does not advance it.
        State and model settings must stay fixed; this method disables autograd.
        """
        if noise_mags.ndim != 3 or noise_mags.shape[-1] != self.n_bands:
            raise ValueError("noise_mags must have shape (batch, frames, n_bands)")
        B, count, _ = noise_mags.shape
        fft_size = _next_pow2(self.frame_hop + self.ir_size - 1)
        tail_size = fft_size - self.frame_hop
        rng = torch.Generator(device="cpu")
        if state is None:
            if seed is None:
                rng.seed()
            else:
                rng.manual_seed(seed)
            overlap = noise_mags.new_zeros(B, tail_size, dtype=torch.float32)
            skip, seen = self.latency_samples, 0
        else:
            if seed is not None:
                raise ValueError("seed can only be set when starting a stream")
            overlap = state.overlap
            if (overlap.shape != (B, tail_size) or overlap.device != noise_mags.device
                    or overlap.dtype != torch.float32
                    or not 0 <= state.skip <= self.latency_samples):
                raise ValueError("state does not match this noise stream's batch, device, or filter")
            rng.set_state(state.rng_state)
            skip, seen = state.skip, state.samples_seen
        length = count * self.frame_hop
        if noise is not None and noise.shape != (B, length):
            raise ValueError("noise must have shape (batch, frames * frame_hop)")
        if count == 0:
            return overlap.new_empty(B, 0), NoiseState(overlap, skip, seen, rng.get_state())
        if noise is None:
            noise = torch.empty(count, B, self.frame_hop).uniform_(-1.0, 1.0, generator=rng)
            noise = noise.permute(1, 0, 2).reshape(B, length).to(noise_mags.device)
        oa = self._overlap_add(self._filter_frames(noise_mags, noise))
        oa[:, :tail_size] += overlap
        dropped = min(skip, length)
        output = oa[:, dropped:length].clone()
        next_state = NoiseState(
            oa[:, length:].clone(), skip - dropped, seen + length, rng.get_state(),
        )
        return output, next_state

    @torch.no_grad()
    def flush(self, state: NoiseState, *, include_tail: bool = False) -> torch.Tensor:
        """Finish a stream, then discard its state.

        By default emit only the outstanding samples needed to match forward's
        input-length output. include_tail=True also retains the physical FIR
        decay after that interval. Flushing does not draw additional noise.
        """
        if state.samples_seen == 0:
            return state.overlap.new_empty(state.overlap.shape[0], 0)
        if include_tail:
            end = self.ir_size - 1
        else:
            end = state.skip + min(self.latency_samples, state.samples_seen)
        return state.overlap[:, state.skip:end].clone()
