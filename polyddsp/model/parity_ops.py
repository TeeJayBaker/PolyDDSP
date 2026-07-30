"""Pure-Torch ports of DDSP primitives required for solo-instrument parity.

Mirrors the subset of `ddsp.core` and `ddsp.training.preprocessing` used in the
autoencoder forward pass, so that a PolyDDSP forward pass can be compared
numerically against upstream DDSP without depending on TensorFlow.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

F0_RANGE = 127.0
DB_RANGE = 80.0


def safe_divide(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Division that swaps `eps` in for `den` exactly where `den == 0`.

    Mirrors `ddsp.core.safe_divide`: only true zeros are replaced with `eps`;
    values smaller than `eps` are NOT clamped, matching upstream's permissiveness.
    """
    safe_den = torch.where(den == 0.0, torch.full_like(den, eps), den)
    return num / safe_den


def hz_to_midi(hz: torch.Tensor) -> torch.Tensor:
    """DDSP `core.hz_to_midi`. Clamps so hz<=0 maps to 0 (silence)."""
    hz = torch.as_tensor(hz)
    notes = 12.0 * (torch.log2(hz.clamp_min(1e-10)) - math.log2(440.0)) + 69.0
    return torch.where(hz <= 0.0, torch.zeros_like(notes), notes)


def midi_to_hz(notes: torch.Tensor) -> torch.Tensor:
    notes = torch.as_tensor(notes)
    return 440.0 * (2.0 ** ((notes - 69.0) / 12.0))


def scale_f0_hz(f0_hz: torch.Tensor) -> torch.Tensor:
    """[0, Nyquist] Hz → [0, ~1] MIDI-scaled (DDSP convention)."""
    return hz_to_midi(f0_hz) / F0_RANGE


def scale_db(db: torch.Tensor) -> torch.Tensor:
    """[-DB_RANGE, 0] dB → [0, 1] linear (DDSP `scale_db`)."""
    return (db / DB_RANGE) + 1.0


def get_harmonic_frequencies(f0_hz: torch.Tensor, n_harmonics: int) -> torch.Tensor:
    """f0 (..., 1) → harmonic freqs (..., H). Mirrors DDSP `core.get_harmonic_frequencies`."""
    ratios = torch.arange(1, n_harmonics + 1, device=f0_hz.device, dtype=f0_hz.dtype)
    return f0_hz * ratios


def normalize_harmonics(
    harmonic_distribution: torch.Tensor,
    f0_hz: torch.Tensor | None = None,
    sample_rate: int | None = None,
) -> torch.Tensor:
    """Mask harmonics above Nyquist (frame-rate) then sum-normalize.

    Mirrors `ddsp.core.normalize_harmonics`.
    """
    if sample_rate is not None and f0_hz is not None:
        n_h = harmonic_distribution.shape[-1]
        harm_f = get_harmonic_frequencies(f0_hz, n_h)
        harmonic_distribution = torch.where(
            harm_f >= sample_rate / 2.0,
            torch.zeros_like(harmonic_distribution),
            harmonic_distribution,
        )
    return safe_divide(
        harmonic_distribution,
        harmonic_distribution.sum(dim=-1, keepdim=True),
    )


def upsample_with_windows(
    inputs: torch.Tensor,
    n_timesteps: int,
    add_endpoint: bool = True,
) -> torch.Tensor:
    """Constant-overlap-add Hann window upsampling.

    Mirrors `ddsp.core.upsample_with_windows`.

    Args:
        inputs: (B, T_frames, C).
        n_timesteps: target time length.
        add_endpoint: if True, repeat the last frame once (n_timesteps must be
            divisible by `n_frames`). If False, n_timesteps must be divisible
            by `n_frames - 1`.

    Returns:
        (B, n_timesteps, C)
    """
    if inputs.dim() != 3:
        raise ValueError(f"upsample_with_windows expects 3-D input, got {inputs.shape}")
    if add_endpoint:
        inputs = torch.cat([inputs, inputs[:, -1:, :]], dim=1)
    n_frames = inputs.shape[1]
    n_intervals = n_frames - 1
    if n_frames >= n_timesteps:
        raise ValueError(
            f"upsample_with_windows can't downsample (frames={n_frames}, target={n_timesteps})"
        )
    if n_timesteps % n_intervals != 0:
        raise ValueError(
            f"n_timesteps ({n_timesteps}) must be divisible by n_intervals ({n_intervals})"
        )

    hop_size = n_timesteps // n_intervals
    window_length = 2 * hop_size
    window = torch.hann_window(window_length, device=inputs.device, dtype=inputs.dtype)

    B, T, C = inputs.shape
    # (B, T, C) -> (B, C, T) -> (B*C, 1, T) — treat each channel independently.
    x = inputs.permute(0, 2, 1).reshape(B * C, 1, T)
    # Build a kernel (1, 1, W) from the window so conv_transpose1d performs
    # constant-overlap-add: y[t] = sum_k x[k] * w[t - k*hop].
    kernel = window.view(1, 1, window_length)
    out = F.conv_transpose1d(x, kernel, stride=hop_size, padding=0)
    # `out`: (B*C, 1, (T-1)*hop + W). Drop the first/last hop_size samples to
    # mirror DDSP's "trim the rise and fall of the first/last window".
    out = out[..., hop_size:-hop_size]
    out = out.reshape(B, C, -1).permute(0, 2, 1).contiguous()
    return out
