"""Voice allocation: list of NoteEvent → per-voice pitch + velocity tensors.

FIFO + lowest-mean-velocity eviction. Operates over a whole-file timeline at
the model's target frame rate. Voice slot IDs persist across notes.
"""
from __future__ import annotations

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


def allocate_to_voices(
    note_events: list[NoteEvent],
    note_grid_target: torch.Tensor,   # (F, T_target) at target rate
    n_voices: int,
    target_frames: int,
    bp_to_target_ratio: float,        # = T_target / T_bp for this file
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pitch, velocity) of shape (V, T_target) each."""
    pitch = torch.zeros(n_voices, target_frames)
    velocity = torch.zeros(n_voices, target_frames)

    # Sort by start time so allocation is causal (FIFO is meaningful only with order).
    events_sorted = sorted(note_events, key=lambda e: (e.start_bp, e.end_bp, e.midi))

    for ev in events_sorted:
        s = max(0, int(round(ev.start_bp * bp_to_target_ratio)))
        e = min(target_frames, int(round(ev.end_bp * bp_to_target_ratio)))
        if e <= s:
            continue
        n_span = e - s

        # Pick free voice; else evict lowest-mean-velocity.
        chosen = -1
        for v in range(n_voices):
            if pitch[v, s] == 0:
                chosen = v
                break
        if chosen < 0:
            slot_amps = velocity[:, s:e].mean(dim=-1)
            chosen = int(torch.argmin(slot_amps).item())
            # Erase the evicted note's full extent past `e` so its tail doesn't
            # reappear after the new (shorter) note ends — otherwise the slot
            # contains [old_head | new | leftover_old_tail], a Frankensteined
            # trajectory the synth has no audio basis to model.
            old_end = e
            for t in range(e, target_frames):
                if pitch[chosen, t] != 0:
                    old_end = t + 1
                else:
                    break
            pitch[chosen, s:old_end] = 0
            velocity[chosen, s:old_end] = 0

        # Compute per-frame midi and Hz.
        if ev.bends is None:
            midi_per_frame = torch.full((n_span,), float(ev.midi))
        else:
            bend_target = _resample_bends_linear(ev.bends, n_span)
            midi_per_frame = float(ev.midi) + bend_target / CONTOURS_BINS_PER_SEMITONE
        pitch[chosen, s:e] = _midi_to_hz(midi_per_frame)

        # Per-frame velocity from note grid at target rate.
        freq_idx = ev.midi - MIDI_OFFSET
        if 0 <= freq_idx < note_grid_target.shape[0]:
            velocity[chosen, s:e] = note_grid_target[freq_idx, s:e]
        else:
            velocity[chosen, s:e] = ev.mean_amplitude

    return pitch, velocity
