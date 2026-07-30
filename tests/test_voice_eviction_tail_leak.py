"""Eviction invariant for `voice_allocation.allocate_to_voices`.

When an incoming note evicts an older, longer note from a voice slot, the
evicted note's *full remaining extent* is zeroed — not merely the span that
overlaps the new note. Zeroing only the overlap would leave the old note's tail
in the slot, so once the new (shorter) note ended the slot would read
[old_head | new | leftover_old_tail]: one contiguous non-zero segment holding
two distinct MIDIs, a trajectory no single monophonic voice could have played.

Worked example (old midi 60 over frames [0, 50), new midi 70 over [10, 20),
V=1 so eviction is forced):
    guaranteed:  60×10, 70×10, 0×30, 0×30
    forbidden:   60×10, 70×10, 60×30, 0×30   (tail leak)
"""
from __future__ import annotations

import math

import torch

from polyddsp.model.note_extraction import NoteEvent
from polyddsp.model.voice_allocation import allocate_to_voices


def _hz_to_midi(hz: float) -> int:
    if hz <= 0.0:
        return 0
    return round(69.0 + 12.0 * math.log2(hz / 440.0))


def _voice_to_midi(pitch_row: torch.Tensor) -> list[int]:
    return [_hz_to_midi(float(p)) for p in pitch_row.tolist()]


def test_eviction_does_not_leak_old_note_tail():
    """Old midi-60 [0,50), new midi-70 [10,20), V=1.

    After the new short note ends at frame 20, voice 0 must NOT contain the
    leftover old midi-60 in frames [20, 50). A single contiguous non-zero
    segment in any voice must contain at most one MIDI value (modulo bends,
    which are not used here).
    """
    events = [
        NoteEvent(start_bp=0, end_bp=50, midi=60, mean_amplitude=0.5, bends=None),
        NoteEvent(start_bp=10, end_bp=20, midi=70, mean_amplitude=0.5, bends=None),
    ]
    target_frames = 80
    note_grid = torch.zeros(88, target_frames)

    pitch, _ = allocate_to_voices(
        events,
        note_grid_target=note_grid,
        n_voices=1,
        target_frames=target_frames,
        bp_to_target_ratio=1.0,
    )

    midi_seq = _voice_to_midi(pitch[0])

    # Frames [20, 50) must be silent: the evicted midi-60 tail is gone.
    leaked = [t for t in range(20, 50) if midi_seq[t] == 60]
    assert leaked == [], (
        f"Voice 0 frames [20, 50) leaked {len(leaked)} frames of midi-60 "
        f"from the evicted note. Sequence: {midi_seq[:55]}"
    )
