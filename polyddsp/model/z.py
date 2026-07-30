"""Timbre encoder: MFCC → InstanceNorm → GRU → Linear → Z."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


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
        self.sr = sr
        self.frame_hop = frame_hop
        self.z_dim = z_dim

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
                center=True,
            ),
        )
        self.norm = nn.InstanceNorm1d(n_mfcc, affine=True)
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

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        coeffs = self.mfcc(audio)  # (B, n_mfcc, T_mfcc)
        coeffs = self.norm(coeffs)
        target_frames = audio.shape[-1] // self.frame_hop
        if coeffs.shape[-1] != target_frames:
            coeffs = F.interpolate(coeffs, size=target_frames, mode="linear", align_corners=True)
        seq = coeffs.transpose(1, 2)  # (B, T_frames, n_mfcc)
        out, _ = self.gru(seq)
        return self.proj(out)
