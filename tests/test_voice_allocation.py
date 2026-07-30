"""Voice allocation: NoteEvent list → (V, T) pitch + velocity tensors."""
from __future__ import annotations

import pytest
import torch

from polyddsp.model.note_extraction import NoteEvent
from polyddsp.model.voice_allocation import allocate_to_voices


def _midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12.0)


def test_three_sequential_notes_fill_one_voice_then_next() -> None:
    # Two non-overlapping notes should both go to voice 0 (FIFO refill).
    events = [
        NoteEvent(start_bp=0, end_bp=10, midi=60, mean_amplitude=0.6, bends=[0]*10),
        NoteEvent(start_bp=20, end_bp=30, midi=62, mean_amplitude=0.6, bends=[0]*10),
    ]
    note_grid = torch.zeros(88, 100)
    note_grid[39, 0:30] = 0.6  # midi 60
    note_grid[41, 60:90] = 0.6  # midi 62 — at target rate (after projection)
    pitch, velocity = allocate_to_voices(
        events, note_grid, n_voices=4, target_frames=100,
        bp_to_target_ratio=3.0,
    )
    assert pitch.shape == (4, 100)
    assert velocity.shape == (4, 100)
    # Voice 0 holds both notes.
    assert pitch[0, 5] == pytest.approx(_midi_to_hz(60), rel=1e-3)
    assert pitch[0, 75] == pytest.approx(_midi_to_hz(62), rel=1e-3)
    # Voices 1-3 stay empty.
    assert (pitch[1:] == 0).all()


def test_simultaneous_notes_fill_distinct_voices() -> None:
    events = [
        NoteEvent(start_bp=0, end_bp=20, midi=60, mean_amplitude=0.7, bends=[0]*20),
        NoteEvent(start_bp=0, end_bp=20, midi=64, mean_amplitude=0.6, bends=[0]*20),
        NoteEvent(start_bp=0, end_bp=20, midi=67, mean_amplitude=0.5, bends=[0]*20),
    ]
    note_grid = torch.zeros(88, 60)
    pitch, velocity = allocate_to_voices(
        events, note_grid, n_voices=4, target_frames=60,
        bp_to_target_ratio=3.0,
    )
    midis_at_t10 = sorted(
        {round(12 * (torch.log2(pitch[v, 10] / 440.0) + 69 / 12).item())
         for v in range(4) if pitch[v, 10] > 0}
    )
    assert midis_at_t10 == [60, 64, 67]


def test_eviction_when_all_voices_taken() -> None:
    # 5 simultaneous notes with V=4. Fifth replaces the lowest-mean-velocity slot.
    events = [
        NoteEvent(start_bp=0, end_bp=20, midi=60+i, mean_amplitude=0.1*(i+1), bends=[0]*20)
        for i in range(5)
    ]
    note_grid = torch.zeros(88, 60)
    for i in range(5):
        note_grid[60 - 21 + i, 0:60] = 0.1 * (i + 1)
    pitch, velocity = allocate_to_voices(
        events, note_grid, n_voices=4, target_frames=60,
        bp_to_target_ratio=3.0,
    )
    midis_at_t10 = sorted(
        {round(12 * (torch.log2(pitch[v, 10] / 440.0) + 69 / 12).item())
         for v in range(4) if pitch[v, 10] > 0}
    )
    # Lowest-amp note (midi 60, amp 0.1) should have been evicted by note (midi 64, amp 0.5).
    assert 60 not in midis_at_t10
    assert 64 in midis_at_t10


def test_bend_offsets_translate_to_per_frame_hz() -> None:
    # 10-frame BP note with bends [0,1,2,...,9] — positive sharp.
    events = [NoteEvent(start_bp=0, end_bp=10, midi=60, mean_amplitude=0.5, bends=list(range(10)))]
    note_grid = torch.zeros(88, 30)
    note_grid[39, 0:30] = 0.5
    pitch, velocity = allocate_to_voices(
        events, note_grid, n_voices=2, target_frames=30,
        bp_to_target_ratio=3.0,
    )
    # Note spans target frames [0, 30). Hz at frame 0 = midi 60 + 0/3 semitones.
    # Hz near end ≈ midi 60 + 9/3 = 63 semitones.
    assert pitch[0, 0] == pytest.approx(_midi_to_hz(60.0), rel=1e-3)
    assert pitch[0, 28] == pytest.approx(_midi_to_hz(60 + 9 / 3.0), rel=5e-2)


def test_bends_none_emits_constant_midi() -> None:
    events = [NoteEvent(start_bp=0, end_bp=10, midi=60, mean_amplitude=0.5, bends=None)]
    note_grid = torch.zeros(88, 30)
    note_grid[39, 0:30] = 0.5
    pitch, _ = allocate_to_voices(
        events, note_grid, n_voices=2, target_frames=30,
        bp_to_target_ratio=3.0,
    )
    # Constant midi 60 across the note span.
    span = pitch[0, 0:30]
    assert torch.allclose(span, torch.full_like(span, _midi_to_hz(60.0)), rtol=1e-3)
