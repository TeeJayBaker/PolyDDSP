"""Timbre encoder: MFCC → per-frame LayerNorm → GRU → Linear → Z.

Normalization uses only the current frame's coefficients. At 16 kHz the default
2048-sample trailing Hann window spans 128 ms, with a 4 ms hop. Frame k uses
audio through sample (k + 1) * frame_hop - 1, the end of its control block.
Missing history is zero-padded; no future audio or right padding is used.
`stream` carries the rolling audio buffer and GRU state between incremental
inference calls; `forward` processes independent training/evaluation clips.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


@dataclass(frozen=True)
class ZEncoderState:
    """Caller-owned state for one stream, never stored on the model.

    audio_buffer holds n_fft - frame_hop history samples plus fewer than one
    hop of pending audio. hidden is None until the first full hop is processed.
    """

    audio_buffer: torch.Tensor
    hidden: torch.Tensor | None


class ZEncoder(nn.Module):
    def __init__(
        self,
        sr: int = 16_000,
        frame_hop: int = 64,
        z_dim: int = 16,
        n_mfcc: int = 30,
        n_fft: int = 2048,
        n_mels: int = 128,
        gru_hidden: int = 512,
    ) -> None:
        super().__init__()
        if not 0 < frame_hop <= n_fft:
            raise ValueError("frame_hop must be positive and no larger than n_fft")
        self.sr = sr
        self.frame_hop = frame_hop
        self.z_dim = z_dim
        self.n_fft = n_fft

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=n_fft,
                hop_length=frame_hop,
                n_mels=n_mels,
                f_min=20.0,
                f_max=8_000.0,
                center=False,
            ),
        )
        self.norm = nn.LayerNorm(n_mfcc)
        self.gru = nn.GRU(
            input_size=n_mfcc,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        # Match TF Keras GRU defaults: orthogonal recurrent kernel + zero bias.
        # PyTorch's `nn.GRU` defaults the recurrent kernel to a uniform, which
        # hurts gradient flow through the time dimension.
        for name, p in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        self.proj = nn.Linear(gru_hidden, z_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def mfcc_frames(self, audio: torch.Tensor) -> torch.Tensor:
        """Return (B, n_mfcc, T // hop) features on a fixed trailing frame grid.

        The first window contains zero history plus the first hop of real audio.
        Subsequent windows advance by exactly one hop, including during startup.
        A partial final hop is omitted, never padded with unavailable samples.
        Appending audio therefore cannot change any existing frame.
        """
        if audio.shape[-1] < self.frame_hop:
            return audio.new_empty(audio.shape[0], self.mfcc.n_mfcc, 0)
        return self.mfcc(F.pad(audio, (self.n_fft - self.frame_hop, 0)))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        coeffs = self.mfcc_frames(audio)  # (B, n_mfcc, T_frames)
        z, _ = self._encode_frames(coeffs)
        return z

    def _encode_frames(
        self,
        coeffs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if coeffs.shape[-1] == 0:
            return coeffs.new_empty(coeffs.shape[0], 0, self.z_dim), hidden
        # Normalize across coefficients within each frame, never across time.
        seq = self.norm(coeffs.transpose(1, 2))  # (B, T_frames, n_mfcc)
        out, hidden = self.gru(seq, hidden)
        return self.proj(out), hidden

    @torch.no_grad()
    def stream(
        self,
        audio: torch.Tensor,
        state: ZEncoderState | None = None,
    ) -> tuple[torch.Tensor, ZEncoderState]:
        """Encode new (B, samples) audio, returning (new_z, next_state).

        Pass next_state into the following call, or None to start/reset a stream.
        Only newly completed hops enter the GRU. Input chunks can have any size,
        including zero; incomplete hops are retained without emitting a frame.
        Discard state at the end to omit the partial final hop, as forward does.

        This inference method disables autograd so retained state cannot keep a
        growing computation graph. Use forward for differentiable clip training.
        Set the model's device/dtype before starting; keep the batch size, audio
        device/dtype, and encoder configuration fixed for the stream's lifetime.
        """
        if audio.ndim != 2:
            raise ValueError("audio must have shape (batch, samples)")
        history_samples = self.n_fft - self.frame_hop
        if state is None:
            history = audio.new_zeros(audio.shape[0], history_samples)
            hidden = None
        else:
            history = state.audio_buffer
            hidden = state.hidden
            if (history.ndim != 2 or history.shape[0] != audio.shape[0]
                    or not history_samples <= history.shape[-1] < self.n_fft):
                raise ValueError("state audio buffer does not match this stream's batch or window")
            if history.device != audio.device or history.dtype != audio.dtype:
                raise ValueError("audio device and dtype must match the stream state")
            if hidden is not None and (
                hidden.shape != (1, audio.shape[0], self.gru.hidden_size)
                or hidden.device != audio.device
            ):
                raise ValueError("state hidden tensor does not match this stream's batch or GRU")

        buffered = torch.cat([history, audio], dim=-1)
        n_frames = (buffered.shape[-1] - history_samples) // self.frame_hop
        if n_frames:
            coeffs = self.mfcc(buffered)
        else:
            coeffs = audio.new_empty(audio.shape[0], self.mfcc.n_mfcc, 0)
        z, hidden = self._encode_frames(coeffs, hidden)

        # Clone the small tail: a slice alone would retain the entire chunk's
        # storage, making retained memory depend on the preceding input size.
        tail = buffered[:, n_frames * self.frame_hop:].clone()
        next_state = ZEncoderState(tail, hidden.clone() if hidden is not None else None)
        return z, next_state
