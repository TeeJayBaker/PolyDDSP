"""Trailing A-weighted loudness with DDSP's weighting and static scaling."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _a_weighting_db(frequencies: torch.Tensor, min_db: float = -45.0) -> torch.Tensor:
    """A-weighting curve in dB (mirrors librosa.A_weighting)."""
    f_sq = frequencies.pow(2)
    const = torch.tensor(
        [12194.217, 20.598997, 107.65265, 737.86223],
        dtype=frequencies.dtype,
        device=frequencies.device,
    ).pow(2)
    weights = 2.0 + 20.0 * (
        torch.log10(const[0])
        + 2 * torch.log10(f_sq + 1e-10)
        - torch.log10(f_sq + const[0])
        - torch.log10(f_sq + const[1])
        - 0.5 * torch.log10(f_sq + const[2])
        - 0.5 * torch.log10(f_sq + const[3])
    )
    return torch.maximum(weights, weights.new_tensor(min_db))


@dataclass(frozen=True)
class LoudnessState:
    """Audio history plus a pending partial hop, owned by one stream."""

    audio_buffer: torch.Tensor


class LoudnessExtractor(nn.Module):
    """Compute per-frame A-weighted log-loudness from raw audio.

    When `normalise=True`, applies DDSP's static `(db/80) + 1.0` mapping
    [-80, 0] dB → [0, 1] (matches `ddsp.training.preprocessing.scale_db`).

    At 16 kHz the default Hann window spans the preceding 32 ms. Like ZEncoder,
    frame k ends at sample (k + 1) * frame_hop - 1. Missing startup history is
    zero-padded and incomplete final hops are omitted, with no future lookahead.
    """

    def __init__(
        self,
        sr: int = 16_000,
        frame_hop: int = 64,
        n_fft: int = 512,
    ) -> None:
        super().__init__()
        if not 0 < frame_hop <= n_fft:
            raise ValueError("frame_hop must be positive and no larger than n_fft")
        self.sr = sr
        self.frame_hop = frame_hop
        self.n_fft = n_fft
        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, audio: torch.Tensor, normalise: bool = True) -> torch.Tensor:
        buffered = F.pad(audio, (self.n_fft - self.frame_hop, 0))
        return self._extract_frames(buffered, normalise)

    def _extract_frames(self, buffered: torch.Tensor, normalise: bool) -> torch.Tensor:
        """Analyze completed windows in audio that already includes its history."""
        if buffered.shape[-1] < self.n_fft:
            return buffered.new_empty(buffered.shape[0], 0)
        spec = torch.stft(
            buffered,
            n_fft=self.n_fft,
            hop_length=self.frame_hop,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = spec.abs().pow(2)

        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.sr).to(buffered.device)
        a_db = _a_weighting_db(freqs)
        a_lin = 10 ** (a_db / 10.0)
        weighted = power * a_lin.unsqueeze(0).unsqueeze(-1)

        avg_power = weighted.mean(dim=1).clamp_min(1e-8)
        loudness_db = 10.0 * torch.log10(avg_power)

        if normalise:
            loudness_db = (loudness_db / 80.0) + 1.0
        return loudness_db

    @torch.no_grad()
    def stream(
        self,
        audio: torch.Tensor,
        state: LoudnessState | None = None,
        normalise: bool = True,
    ) -> tuple[torch.Tensor, LoudnessState]:
        """Return newly completed loudness frames and the next stream state.

        Input is (batch, samples) and may contain any number of samples, including
        zero. Pass the returned state on the next call; None starts a fresh stream.
        A partial hop remains pending. Discard it at the end, matching forward,
        or explicitly supply padding if a final padded frame is desired.
        Keep batch size, device, dtype, and extractor settings fixed per stream.
        Autograd is disabled for streaming inference; forward remains trainable.
        """
        if audio.ndim != 2:
            raise ValueError("audio must have shape (batch, samples)")
        history_samples = self.n_fft - self.frame_hop
        if state is None:
            history = audio.new_zeros(audio.shape[0], history_samples)
        else:
            history = state.audio_buffer
            if (history.ndim != 2 or history.shape[0] != audio.shape[0]
                    or not history_samples <= history.shape[-1] < self.n_fft):
                raise ValueError("state audio buffer does not match this stream's batch or window")
            if history.device != audio.device or history.dtype != audio.dtype:
                raise ValueError("audio device and dtype must match the stream state")
        buffered = torch.cat([history, audio], dim=-1)
        loudness = self._extract_frames(buffered, normalise)
        consumed = loudness.shape[-1] * self.frame_hop
        # Copy the small tail so it cannot retain an entire previous chunk's storage.
        next_state = LoudnessState(buffered[:, consumed:].clone())
        return loudness, next_state
