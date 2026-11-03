"""Single learnable IR reverb (frequency-domain convolution)."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ReverbState:
    """Per-stream partition spectra, FFT delay line, overlap, and pending input."""

    ir_fft: torch.Tensor
    history: torch.Tensor
    overlap: torch.Tensor
    pending: torch.Tensor
    position: int
    has_audio: bool


class Reverb(nn.Module):
    """One learnable IR shared across the dataset.

    Returns dry + wet. The first IR sample is masked to zero so the dry leak
    from the IR doesn't double-count the dry signal added back at the end.
    """

    def __init__(self, reverb_len: int = 48_000, block_size: int = 64) -> None:
        super().__init__()
        if reverb_len < 1 or block_size < 1:
            raise ValueError("reverb_len and block_size must be positive")
        self.reverb_len = reverb_len
        self.block_size = block_size
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

    @torch.no_grad()
    def stream(
        self, audio: torch.Tensor, state: ReverbState | None = None,
    ) -> tuple[torch.Tensor, ReverbState]:
        """Partitioned FFT convolution of new (B, samples) audio, returning state.

        Each completed block emits block_size dry+wet samples. Partial blocks
        remain pending; the default block is 64 samples (4 ms at 16 kHz).
        The IR is partitioned and transformed once per stream, not once per tiny
        input chunk. Weights must remain fixed until the state is discarded.
        Pass None to reset. Use flush at EOF to finish the partial block and,
        optionally, preserve the remaining reverb decay. Autograd is disabled.
        """
        if audio.ndim != 2:
            raise ValueError("audio must have shape (batch, samples)")
        if audio.device != self.ir.device or audio.dtype != self.ir.dtype:
            raise ValueError("audio must match the reverb weights' device and dtype")
        B = audio.shape[0]
        size = self.block_size
        partitions = (self.reverb_len + size - 1) // size
        if state is None:
            ir = torch.cat([self.ir.new_zeros(1), self.ir[1:]])
            ir = F.pad(ir, (0, partitions * size - self.reverb_len)).reshape(partitions, size)
            ir_fft = torch.fft.rfft(ir, n=2 * size, dim=-1)
            history = ir_fft.new_zeros(B, partitions, size + 1)
            overlap = audio.new_zeros(B, size)
            pending = audio.new_empty(B, 0)
            position, has_audio = 0, False
        else:
            if (state.history.shape != (B, partitions, size + 1)
                    or state.ir_fft.shape != (partitions, size + 1)
                    or state.overlap.shape != (B, size)
                    or state.pending.ndim != 2 or state.pending.shape[0] != B
                    or state.pending.shape[-1] >= size
                    or state.pending.device != audio.device
                    or state.pending.dtype != audio.dtype
                    or not 0 <= state.position < partitions):
                raise ValueError("state does not match this reverb stream's batch, device, or partitions")
            ir_fft = state.ir_fft
            history = state.history.clone()
            overlap, pending = state.overlap, state.pending
            position, has_audio = state.position, state.has_audio

        buffered = torch.cat([pending, audio], dim=-1)
        count = buffered.shape[-1] // size
        consumed = count * size
        blocks = buffered[:, :consumed].reshape(B, count, size)
        outputs = []
        if count:
            spectra = torch.fft.rfft(blocks, n=2 * size, dim=-1)
            offsets = torch.arange(partitions, device=audio.device)
            for index in range(count):
                history[:, position] = spectra[:, index]
                # H[p] multiplies the input block from p blocks ago.
                order = (position - offsets) % partitions
                wet_fft = (history.index_select(1, order) * ir_fft.unsqueeze(0)).sum(dim=1)
                wet = torch.fft.irfft(wet_fft, n=2 * size, dim=-1)
                outputs.append(blocks[:, index] + wet[:, :size] + overlap)
                overlap = wet[:, size:]
                position = (position + 1) % partitions
        output = torch.cat(outputs, dim=-1) if outputs else audio.new_empty(B, 0)
        next_state = ReverbState(
            ir_fft, history, overlap.clone(), buffered[:, consumed:].clone(),
            position, has_audio or audio.shape[-1] > 0,
        )
        return output, next_state

    @torch.no_grad()
    def flush(self, state: ReverbState, *, include_tail: bool = False) -> torch.Tensor:
        """Finish the partial block, optionally emitting the full reverb tail.

        Default output completes forward's original input-length interval.
        include_tail=True appends reverb_len - 1 decay samples after the input.
        Discard the stream state after flushing; it is not mutated by this call.
        """
        length = state.pending.shape[-1]
        if include_tail and state.has_audio:
            length += self.reverb_len - 1
        if length == 0:
            return state.pending.new_empty(state.pending.shape[0], 0)
        padded_length = ((length + self.block_size - 1) // self.block_size) * self.block_size
        audio = F.pad(state.pending, (0, padded_length - state.pending.shape[-1]))
        output, _ = self.stream(audio, replace(state, pending=state.pending[:, :0]))
        return output[:, :length].clone()

