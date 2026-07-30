"""Pure-function tests for the BP note extraction module."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from polyddsp.model.note_extraction import (
    MIDI_OFFSET,
    N_FREQ_BINS_CONTOURS,
    NoteEvent,
    constrain_frequency,
    drop_overlapping_pitch_bends,
    get_infered_onsets,
    get_pitch_bends,
    midi_pitch_to_contour_bin,
    output_to_notes_polyphonic,
)


def test_note_event_dataclass_fields() -> None:
    ev = NoteEvent(start_bp=10, end_bp=20, midi=60, mean_amplitude=0.7, bends=None)
    assert ev.start_bp == 10
    assert ev.end_bp == 20
    assert ev.midi == 60
    assert ev.mean_amplitude == pytest.approx(0.7)
    assert ev.bends is None


def test_midi_pitch_to_contour_bin_a4() -> None:
    # A4 = midi 69 = 440 Hz. Annotation base 27.5 Hz, 3 bins/semitone.
    # bin = 12 * 3 * log2(440 / 27.5) = 36 * 4 = 144.
    assert midi_pitch_to_contour_bin(69) == 144


def test_constrain_frequency_zeros_below_min_and_above_max() -> None:
    # Frame layout: (T, F) = (5, 88).
    onsets = np.ones((5, 88), dtype=np.float32)
    frames = np.ones((5, 88), dtype=np.float32)
    # midi 60 (C4) is freq_idx 60 - 21 = 39.
    o, f = constrain_frequency(onsets, frames, max_freq=261.63, min_freq=130.81)
    # min_freq C3 = midi 48 = idx 27; max_freq C4 = midi 60 = idx 39.
    assert (o[:, :27] == 0).all()
    assert (o[:, 39:] == 0).all()
    assert (o[:, 27:39] == 1).all()
    assert (f[:, :27] == 0).all()
    assert (f[:, 39:] == 0).all()


def test_get_infered_onsets_recovers_step_in_frames() -> None:
    # 10 frames, 3 freq bins. Step from 0→1 in column 1 at t=5.
    # Set a small non-zero onset value elsewhere so the rescale factor
    # (onsets.max()) is non-zero — matches upstream behavior.
    onsets = np.zeros((10, 3), dtype=np.float32)
    onsets[0, 0] = 0.5  # rescale target
    frames = np.zeros((10, 3), dtype=np.float32)
    frames[5:, 1] = 1.0
    out = get_infered_onsets(onsets, frames, n_diff=2)
    # Inferred onset peak should fire at the step in column 1, rescaled to
    # onsets.max() = 0.5.
    assert out[5, 1] == pytest.approx(0.5, rel=1e-5)
    # Other rows in column 1 should be zero (no step there).
    assert out[6, 1] == 0  # frames flat at 1.0 → zero diff
    # Other columns untouched.
    assert out[:, 2].sum() == 0


def test_get_infered_onsets_returns_zeros_when_onsets_all_zero() -> None:
    """Upstream parity: rescale uses onsets.max(); when it's 0, output is 0."""
    onsets = np.zeros((10, 3), dtype=np.float32)
    frames = np.zeros((10, 3), dtype=np.float32)
    frames[5:, 1] = 1.0
    out = get_infered_onsets(onsets, frames, n_diff=2)
    assert out.sum() == 0


# ---------------------------------------------------------------------------
# output_to_notes_polyphonic tests
# ---------------------------------------------------------------------------


def _make_posteriorgram(
    n_frames: int = 200, n_freqs: int = 88
) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((n_frames, n_freqs), dtype=np.float32), np.zeros(
        (n_frames, n_freqs), dtype=np.float32
    )


def test_output_to_notes_single_onset_yields_one_note() -> None:
    frames, onsets = _make_posteriorgram()
    # Onset at t=20, freq_idx=39 (midi 60). Note rings 30 frames > min_note_len=11.
    onsets[20, 39] = 0.9
    frames[20:55, 39] = 0.6
    events = output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=0.5,
        frame_thresh=0.3,
        min_note_len=11,
        infer_onsets=True,
        melodia_trick=False,
        energy_tol=11,
        max_freq=None,
        min_freq=None,
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.start_bp == 20
    assert ev.midi == 60  # 39 + MIDI_OFFSET
    assert ev.mean_amplitude == pytest.approx(0.6, abs=1e-3)
    assert 53 <= ev.end_bp <= 55  # depends on melodia padding behavior


def test_output_to_notes_drops_short_notes() -> None:
    frames, onsets = _make_posteriorgram()
    onsets[20, 39] = 0.9
    frames[20:25, 39] = 0.6  # 5-frame note, below min_note_len=11
    events = output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=0.5,
        frame_thresh=0.3,
        min_note_len=11,
        infer_onsets=False,
        melodia_trick=False,
        energy_tol=11,
        max_freq=None,
        min_freq=None,
    )
    assert events == []


def test_output_to_notes_melodia_recovers_missed_onset() -> None:
    frames, onsets = _make_posteriorgram()
    # Sustained energy with NO onset. Melodia trick should pick it up.
    frames[30:80, 50] = 0.8
    events = output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=0.5,
        frame_thresh=0.3,
        min_note_len=11,
        infer_onsets=False,
        melodia_trick=True,
        energy_tol=11,
        max_freq=None,
        min_freq=None,
    )
    assert len(events) == 1
    assert events[0].midi == 50 + MIDI_OFFSET


# ---------------------------------------------------------------------------
# get_pitch_bends tests
# ---------------------------------------------------------------------------


def test_get_pitch_bends_centers_on_note_when_contour_at_note_bin() -> None:
    # Contour matrix shape (T, F=264). Place sharp peak at the note's centre bin.
    contours = np.zeros((100, N_FREQ_BINS_CONTOURS), dtype=np.float32)
    midi = 60
    centre_bin = midi_pitch_to_contour_bin(midi)
    contours[20:50, centre_bin] = 1.0
    events = [NoteEvent(start_bp=20, end_bp=50, midi=midi, mean_amplitude=0.5)]
    enriched = get_pitch_bends(contours, events, n_bins_tolerance=25)
    assert len(enriched) == 1
    bends = enriched[0].bends
    assert bends is not None
    assert len(bends) == 30  # one per BP frame in the note span
    # All bends should be 0 (centre).
    assert all(b == 0 for b in bends)


def test_get_pitch_bends_offsets_when_contour_above_note() -> None:
    contours = np.zeros((100, N_FREQ_BINS_CONTOURS), dtype=np.float32)
    midi = 60
    centre_bin = midi_pitch_to_contour_bin(midi)
    contours[20:30, centre_bin + 2] = 1.0  # 2 bins sharp = +2/3 semitone
    events = [NoteEvent(start_bp=20, end_bp=30, midi=midi, mean_amplitude=0.5)]
    enriched = get_pitch_bends(contours, events, n_bins_tolerance=25)
    assert all(b == 2 for b in enriched[0].bends)


# ---------------------------------------------------------------------------
# drop_overlapping_pitch_bends tests
# ---------------------------------------------------------------------------


def test_drop_overlapping_pitch_bends_clears_bends_for_overlap_pair() -> None:
    a = NoteEvent(start_bp=0, end_bp=20, midi=60, mean_amplitude=0.5, bends=[0]*20)
    b = NoteEvent(start_bp=10, end_bp=30, midi=64, mean_amplitude=0.5, bends=[1]*20)
    c = NoteEvent(start_bp=40, end_bp=60, midi=67, mean_amplitude=0.5, bends=[0]*20)
    out = drop_overlapping_pitch_bends([a, b, c])
    assert out[0].bends is None  # overlaps b
    assert out[1].bends is None  # overlaps a
    assert out[2].bends == [0]*20  # standalone


def test_drop_overlapping_pitch_bends_preserves_when_disjoint() -> None:
    a = NoteEvent(start_bp=0, end_bp=20, midi=60, mean_amplitude=0.5, bends=[0]*20)
    b = NoteEvent(start_bp=20, end_bp=40, midi=64, mean_amplitude=0.5, bends=[1]*20)
    out = drop_overlapping_pitch_bends([a, b])
    assert out[0].bends == [0]*20
    assert out[1].bends == [1]*20
