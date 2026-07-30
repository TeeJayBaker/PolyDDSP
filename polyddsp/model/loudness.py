"""A-weighted loudness extractor — Hantrakul et al. recipe (DDSP-compatible)."""
from __future__ import annotations

import torch
import torch.nn as nn


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


class LoudnessExtractor(nn.Module):
    """Compute per-frame A-weighted log-loudness from raw audio.

    When `normalise=True`, applies DDSP's static `(db/80) + 1.0` mapping
    [-80, 0] dB → [0, 1] (matches `ddsp.training.preprocessing.scale_db`).
    """

    def __init__(
        self,
        sr: int = 16_000,
        frame_hop: int = 64,
        n_fft: int = 512,
    ) -> None:
        super().__init__()
        self.sr = sr
        self.frame_hop = frame_hop
        self.n_fft = n_fft
        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, audio: torch.Tensor, normalise: bool = True) -> torch.Tensor:
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.frame_hop,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        power = spec.abs().pow(2)

        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.sr).to(audio.device)
        a_db = _a_weighting_db(freqs)
        a_lin = 10 ** (a_db / 10.0)
        weighted = power * a_lin.unsqueeze(0).unsqueeze(-1)

        avg_power = weighted.mean(dim=1).clamp_min(1e-8)
        loudness_db = 10.0 * torch.log10(avg_power)

        target_frames = audio.shape[-1] // self.frame_hop
        loudness_db = loudness_db[..., :target_frames]

        if normalise:
            loudness_db = (loudness_db / 80.0) + 1.0
        return loudness_db
