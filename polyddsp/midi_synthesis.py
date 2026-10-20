"""Audible preview of the pitch and velocity controls sent to the decoder.

This is deliberately a small, deterministic synth rather than a MIDI-file
renderer.  PolyDDSP's conditioning is already richer than a MIDI event stream:
pitch is frame-aligned Hz (including bends) and velocity is a frame-aligned
amplitude.  Rendering those tensors directly avoids quantising them back to
MIDI or requiring a platform SoundFont installation just to inspect a run.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from polyddsp.model.additive import angular_cumsum


# A bright but unobtrusive four-partial patch.  The octave-heavy spectrum makes
# quiet/wrong notes easier to identify than a sine while remaining clean enough
# to hear pitch allocation errors in dense chords.
_PARTIAL_WEIGHTS = (1.0, 0.5, 0.25, 0.125)

# Distinct colours for the usual 6-10 allocated voices.  More voices cycle the
# palette; overlap is still visible because pixels are combined channel-wise.
_VOICE_COLOURS = (
    (0.95, 0.35, 0.30),
    (0.30, 0.70, 1.00),
    (0.35, 0.90, 0.45),
    (1.00, 0.75, 0.25),
    (0.75, 0.45, 1.00),
    (0.20, 0.90, 0.85),
    (1.00, 0.45, 0.75),
    (0.65, 0.75, 0.25),
    (0.55, 0.55, 1.00),
    (1.00, 0.60, 0.30),
)


def render_pitch_image(
    pitch_hz: torch.Tensor,
    velocity: torch.Tensor | None = None,
    *,
    height: int = 256,
) -> torch.Tensor:
    """Draw frame-aligned pitches as a TensorBoard-ready piano-roll image.

    Args:
        pitch_hz: ``(V, T_frames)`` frequencies in Hz; zero is silence.
        velocity: Optional matching tensor used to vary line brightness.
        height: Image height.  The fixed vertical scale spans MIDI 0-127 so
            plots remain directly comparable across checkpoints and datasets.

    Returns:
        Float RGB tensor ``(3, height, T_frames)`` in ``[0, 1]``.
    """
    if pitch_hz.ndim != 2:
        raise ValueError(f"pitch_hz must have shape (V, T); got {tuple(pitch_hz.shape)}")
    if velocity is not None and velocity.shape != pitch_hz.shape:
        raise ValueError(
            f"velocity must match pitch_hz shape {tuple(pitch_hz.shape)}; "
            f"got {tuple(velocity.shape)}"
        )
    if height < 128:
        raise ValueError(f"height must be at least 128, got {height}")

    pitch = torch.nan_to_num(pitch_hz.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    vel = (
        torch.ones_like(pitch)
        if velocity is None
        else torch.nan_to_num(
            velocity.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=0.0,
        ).clamp(0.0, 1.0)
    )
    voices, frames = pitch.shape
    image = torch.zeros(3, height, frames, dtype=torch.float32)
    if frames == 0:
        return image

    # Octave guide lines, with middle C slightly brighter.  Pitch itself uses
    # the full continuous MIDI axis, so bends are visible rather than quantised.
    def row_for_midi(midi: torch.Tensor | float) -> torch.Tensor:
        midi_tensor = torch.as_tensor(midi, dtype=torch.float32)
        return ((127.0 - midi_tensor.clamp(0.0, 127.0)) * (height - 1) / 127.0).round().long()

    for midi in range(0, 128, 12):
        row = int(row_for_midi(float(midi)))
        image[:, row, :] = 0.12 if midi != 60 else 0.20

    colours = torch.tensor(_VOICE_COLOURS, dtype=torch.float32)
    for voice in range(voices):
        active = pitch[voice] > 0.0
        if not active.any():
            continue
        midi = 69.0 + 12.0 * torch.log2(pitch[voice].clamp_min(1e-7) / 440.0)
        rows = row_for_midi(midi)
        colour = colours[voice % len(colours)]
        active_frames = active.nonzero(as_tuple=False).flatten()
        for frame in active_frames.tolist():
            intensity = 0.30 + 0.70 * float(vel[voice, frame])
            row = int(rows[frame])
            # Three-pixel stroke keeps single-note lines legible after the
            # TensorBoard UI scales a 256px-high image down.
            lo = max(0, row - 1)
            hi = min(height, row + 2)
            stroke = colour[:, None] * intensity
            image[:, lo:hi, frame] = torch.maximum(image[:, lo:hi, frame], stroke)

            # Fill small between-frame pitch changes so bends look continuous;
            # large jumps remain separate notes instead of vertical bars.
            if frame > 0 and active[frame - 1] and abs(float(midi[frame] - midi[frame - 1])) <= 2.0:
                previous = int(rows[frame - 1])
                bend_lo, bend_hi = sorted((previous, row))
                image[:, bend_lo:bend_hi + 1, frame] = torch.maximum(
                    image[:, bend_lo:bend_hi + 1, frame], colour[:, None] * intensity,
                )
    return image


def synthesize_midi(
    pitch_hz: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sample_rate: int,
    frame_hop: int,
    n_samples: int | None = None,
) -> torch.Tensor:
    """Render decoder pitch/velocity conditioning with a basic additive patch.

    Args:
        pitch_hz: ``(B, V, T_frames)`` fundamental frequency in Hz; zero is
            silence.  Values may vary continuously to represent pitch bends.
        velocity: ``(B, V, T_frames)`` amplitudes, normally in ``[0, 1]``.
        sample_rate: Output sample rate in Hz.
        frame_hop: Number of output samples represented by each control frame.
        n_samples: Exact output length.  Defaults to ``T_frames * frame_hop``.

    Returns:
        A peak-normalised mono tensor of shape ``(B, n_samples)``.  Voices are
        mixed together; batches remain separate.
    """
    if pitch_hz.ndim != 3:
        raise ValueError(f"pitch_hz must have shape (B, V, T); got {tuple(pitch_hz.shape)}")
    if velocity.shape != pitch_hz.shape:
        raise ValueError(
            f"velocity must match pitch_hz shape {tuple(pitch_hz.shape)}; "
            f"got {tuple(velocity.shape)}"
        )
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if frame_hop <= 0:
        raise ValueError(f"frame_hop must be positive, got {frame_hop}")

    batch, voices, frames = pitch_hz.shape
    if n_samples is None:
        n_samples = frames * frame_hop
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    if n_samples == 0 or frames == 0:
        return pitch_hz.new_zeros(batch, n_samples)

    # Work in float32 even if training uses a lower precision: this preview is
    # outside autograd and stable oscillator phase matters more than throughput.
    pitch = torch.nan_to_num(pitch_hz.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    pitch = pitch.clamp_min(0.0)
    vel = torch.nan_to_num(velocity.detach().float(), nan=0.0, posinf=1.0, neginf=0.0)
    vel = vel.clamp(0.0, 1.0) * pitch.gt(0.0)

    # Linear control interpolation follows the decoder/additive-synth convention
    # and retains sub-semitone bends.  Folding voices into the channel dimension
    # lets interpolate handle every voice in one operation.
    pitch_samples = F.interpolate(
        pitch, size=n_samples, mode="linear", align_corners=False,
    )
    velocity_samples = F.interpolate(
        vel, size=n_samples, mode="linear", align_corners=False,
    )

    weights = pitch.new_tensor(_PARTIAL_WEIGHTS)
    ratios = torch.arange(1, len(_PARTIAL_WEIGHTS) + 1, device=pitch.device, dtype=pitch.dtype)
    frequencies = pitch_samples.unsqueeze(-1) * ratios
    audible = frequencies.lt(sample_rate / 2.0)
    omega = frequencies * (2.0 * math.pi / sample_rate)

    # angular_cumsum expects time on dimension 1.  Treat B*V as its batch and
    # partials as channels, then restore the voice dimension after oscillation.
    omega = omega.reshape(batch * voices, n_samples, len(_PARTIAL_WEIGHTS))
    phase = angular_cumsum(omega, chunk_size=frame_hop)
    partial_weights = weights * audible.reshape_as(phase)
    weight_sum = partial_weights.sum(dim=-1).clamp_min(1e-8)
    voice_audio = (torch.sin(phase) * partial_weights).sum(dim=-1) / weight_sum
    voice_audio = voice_audio.reshape(batch, voices, n_samples) * velocity_samples
    audio = voice_audio.sum(dim=1)

    # This is a diagnostic preview, so make quiet conditioning audible while
    # retaining all relative velocity and voice balance within each clip.
    peak = audio.abs().amax(dim=-1, keepdim=True)
    gain = torch.where(peak > 0, 0.95 / peak.clamp_min(1e-8), torch.ones_like(peak))
    return audio * gain
