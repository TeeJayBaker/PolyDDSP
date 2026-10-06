"""Voice allocation: list of NoteEvent → per-voice pitch + velocity tensors.

FIFO + lowest-mean-velocity eviction. Operates over a whole-file timeline at
the model's target frame rate. Voice slot IDs persist across notes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from polyddsp.model.note_extraction import (
    CONTOURS_BINS_PER_SEMITONE,
    MIDI_OFFSET,
    NoteEvent,
)


def _midi_to_hz(midi: torch.Tensor) -> torch.Tensor:
    return 440.0 * 2 ** ((midi - 69.0) / 12.0)


def _resample_bends_linear(bends: list[int], n_target: int) -> torch.Tensor:
    """Linearly interpolate a 1-D integer bend sequence (length n_bp) to n_target."""
    n_bp = len(bends)
    if n_bp == n_target:
        return torch.tensor(bends, dtype=torch.float32)
    if n_bp == 1:
        return torch.full((n_target,), float(bends[0]))
    src = torch.tensor(bends, dtype=torch.float32).view(1, 1, n_bp)
    out = F.interpolate(src, size=n_target, mode="linear", align_corners=True)
    return out.view(n_target)


@dataclass(frozen=True)
class FrameNoteEvent:
    """One note already projected onto the target frame grid."""

    start: int
    end: int
    midi: int
    pitch_hz: torch.Tensor
    velocity: torch.Tensor


def allocate_frame_events(
    events: list[FrameNoteEvent],
    n_voices: int,
    target_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate target-rate note trajectories into fixed monophonic voice slots."""
    pitch = torch.zeros(n_voices, target_frames)
    velocity = torch.zeros(n_voices, target_frames)

    for event in sorted(events, key=lambda item: (item.start, item.end, item.midi)):
        start = max(0, event.start)
        end = min(target_frames, event.end)
        if end <= start:
            continue
        source_start = start - event.start
        source_end = source_start + end - start
        if event.pitch_hz.numel() < source_end or event.velocity.numel() < source_end:
            raise ValueError("frame note trajectory is shorter than its declared span")

        chosen = next((voice for voice in range(n_voices) if pitch[voice, start] == 0), -1)
        if chosen < 0:
            slot_amps = velocity[:, start:end].mean(dim=-1)
            chosen = int(torch.argmin(slot_amps).item())
            old_end = end
            for frame in range(end, target_frames):
                if pitch[chosen, frame] != 0:
                    old_end = frame + 1
                else:
                    break
            pitch[chosen, start:old_end] = 0
            velocity[chosen, start:old_end] = 0

        pitch[chosen, start:end] = event.pitch_hz[source_start:source_end]
        velocity[chosen, start:end] = event.velocity[source_start:source_end]

    return pitch, velocity


def allocate_to_voices(
    note_events: list[NoteEvent],
    note_grid_target: torch.Tensor,   # (F, T_target) at target rate
    n_voices: int,
    target_frames: int,
    bp_to_target_ratio: float,        # = T_target / T_bp for this file
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pitch, velocity) of shape (V, T_target) each."""
    frame_events = []
    for ev in note_events:
        s = max(0, int(round(ev.start_bp * bp_to_target_ratio)))
        e = min(target_frames, int(round(ev.end_bp * bp_to_target_ratio)))
        if e <= s:
            continue
        n_span = e - s

        if ev.bends is None:
            midi_per_frame = torch.full((n_span,), float(ev.midi))
        else:
            bend_target = _resample_bends_linear(ev.bends, n_span)
            midi_per_frame = float(ev.midi) + bend_target / CONTOURS_BINS_PER_SEMITONE
        freq_idx = ev.midi - MIDI_OFFSET
        if 0 <= freq_idx < note_grid_target.shape[0]:
            note_velocity = note_grid_target[freq_idx, s:e]
        else:
            note_velocity = torch.full((n_span,), ev.mean_amplitude)
        frame_events.append(FrameNoteEvent(
            start=s,
            end=e,
            midi=ev.midi,
            pitch_hz=_midi_to_hz(midi_per_frame),
            velocity=note_velocity,
        ))

    return allocate_frame_events(frame_events, n_voices, target_frames)
