"""BP note extraction (port of upstream basic_pitch/note_creation.py).

All functions operate at BP-native rate (22050 Hz / 256 hop = 86.13 fps).
Frame-major layout: posteriorgrams are (T, F). Tensors are accepted as
either numpy arrays or torch tensors; internally we use numpy for parity
with upstream (the algorithm is sequential, not amenable to GPU).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.signal

# Constants — match upstream basic_pitch/constants.py + note_creation.py.
MIDI_OFFSET = 21
MAX_FREQ_IDX = 87
ANNOTATIONS_BASE_FREQUENCY = 27.5
ANNOTATIONS_N_SEMITONES = 88
CONTOURS_BINS_PER_SEMITONE = 3
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE


@dataclass
class NoteEvent:
    start_bp: int
    end_bp: int
    midi: int
    mean_amplitude: float
    bends: Optional[list[int]] = None


def midi_pitch_to_contour_bin(pitch_midi: int) -> int:
    pitch_hz = 440.0 * 2.0 ** ((pitch_midi - 69) / 12.0)
    bin_f = 12.0 * CONTOURS_BINS_PER_SEMITONE * np.log2(pitch_hz / ANNOTATIONS_BASE_FREQUENCY)
    return int(round(bin_f))


def _hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * float(np.log2(hz / 440.0))


def constrain_frequency(
    onsets: np.ndarray,
    frames: np.ndarray,
    max_freq: Optional[float],
    min_freq: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Zero out activations below min_freq or at/above max_freq. (T, F) layout."""
    if max_freq is not None:
        max_idx = int(round(_hz_to_midi(max_freq) - MIDI_OFFSET))
        onsets[:, max_idx:] = 0
        frames[:, max_idx:] = 0
    if min_freq is not None:
        min_idx = int(round(_hz_to_midi(min_freq) - MIDI_OFFSET))
        onsets[:, :min_idx] = 0
        frames[:, :min_idx] = 0
    return onsets, frames


def output_to_notes_polyphonic(
    frames: np.ndarray,
    onsets: np.ndarray,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    infer_onsets: bool,
    melodia_trick: bool,
    energy_tol: int,
    max_freq: Optional[float],
    min_freq: Optional[float],
) -> list[NoteEvent]:
    """Decode (T, F) posteriorgrams to a list of NoteEvent.

    Direct port of upstream `basic_pitch.note_creation.output_to_notes_polyphonic`.
    Algorithm is sequential and runs in numpy.
    """
    onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    if infer_onsets:
        onsets = get_infered_onsets(onsets, frames)

    n_frames = frames.shape[0]
    peak_thresh_mat = np.zeros_like(onsets)
    peaks = scipy.signal.argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    # iterate backwards in time (matches upstream)
    onset_time_idx = onset_idx[0][::-1]
    onset_freq_idx = onset_idx[1][::-1]

    remaining = frames.copy()
    events: list[NoteEvent] = []

    for note_start, freq_idx in zip(onset_time_idx, onset_freq_idx):
        if note_start >= n_frames - 1:
            continue
        i = note_start + 1
        k = 0
        while i < n_frames - 1 and k < energy_tol:
            if remaining[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1
        i -= k
        if i - note_start <= min_note_len:
            continue
        remaining[note_start:i, freq_idx] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining[note_start:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining[note_start:i, freq_idx - 1] = 0
        amplitude = float(np.mean(frames[note_start:i, freq_idx]))
        events.append(NoteEvent(
            start_bp=int(note_start), end_bp=int(i),
            midi=int(freq_idx + MIDI_OFFSET), mean_amplitude=amplitude,
        ))

    if melodia_trick:
        while remaining.max() > frame_thresh:
            i_mid, freq_idx = np.unravel_index(np.argmax(remaining), remaining.shape)
            remaining[i_mid, freq_idx] = 0
            # forward pass
            i = i_mid + 1
            k = 0
            while i < n_frames - 1 and k < energy_tol:
                if remaining[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining[i, freq_idx - 1] = 0
                i += 1
            i_end = i - 1 - k
            # backward pass
            i = i_mid - 1
            k = 0
            while i > 0 and k < energy_tol:
                if remaining[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining[i, freq_idx - 1] = 0
                i -= 1
            i_start = i + 1 + k
            assert i_start >= 0
            assert i_end < n_frames
            if i_end - i_start <= min_note_len:
                continue
            amplitude = float(np.mean(frames[i_start:i_end, freq_idx]))
            events.append(NoteEvent(
                start_bp=int(i_start), end_bp=int(i_end),
                midi=int(freq_idx + MIDI_OFFSET), mean_amplitude=amplitude,
            ))

    return events


def get_pitch_bends(
    contours: np.ndarray,
    note_events: list[NoteEvent],
    n_bins_tolerance: int = 25,
) -> list[NoteEvent]:
    """Estimate per-frame pitch bends in 1/CONTOURS_BINS_PER_SEMITONE-semitone units.

    Direct port of upstream `basic_pitch.note_creation.get_pitch_bends`.
    Returns new NoteEvent list with `bends` filled in.
    """
    window_length = n_bins_tolerance * 2 + 1
    freq_gaussian = scipy.signal.windows.gaussian(window_length, std=5)
    out: list[NoteEvent] = []
    for ev in note_events:
        freq_idx = int(np.round(midi_pitch_to_contour_bin(ev.midi)))
        freq_start = np.max([freq_idx - n_bins_tolerance, 0])
        freq_end = np.min([N_FREQ_BINS_CONTOURS, freq_idx + n_bins_tolerance + 1])
        gauss_lo = np.max([0, n_bins_tolerance - freq_idx])
        gauss_hi = window_length - np.max(
            [0, freq_idx - (N_FREQ_BINS_CONTOURS - n_bins_tolerance - 1)]
        )
        submat = (
            contours[ev.start_bp:ev.end_bp, freq_start:freq_end]
            * freq_gaussian[gauss_lo:gauss_hi]
        )
        pb_shift = n_bins_tolerance - np.max([0, n_bins_tolerance - freq_idx])
        bends: Optional[list[int]] = list(
            (np.argmax(submat, axis=1) - pb_shift).astype(int)
        )
        out.append(NoteEvent(
            start_bp=ev.start_bp, end_bp=ev.end_bp,
            midi=ev.midi, mean_amplitude=ev.mean_amplitude, bends=bends,
        ))
    return out


def drop_overlapping_pitch_bends(events: list[NoteEvent]) -> list[NoteEvent]:
    """Set `bends=None` on any pair of events that overlap in time.

    Mirrors upstream `basic_pitch.note_creation.drop_overlapping_pitch_bends`,
    which sorts by (start, end, ...) then nulls overlapping pairs.
    """
    sorted_events = sorted(events, key=lambda e: (e.start_bp, e.end_bp, e.midi))
    n = len(sorted_events)
    drop = [False] * n
    for i in range(n - 1):
        for j in range(i + 1, n):
            if sorted_events[j].start_bp >= sorted_events[i].end_bp:
                break
            drop[i] = True
            drop[j] = True
    return [
        NoteEvent(
            start_bp=e.start_bp, end_bp=e.end_bp, midi=e.midi,
            mean_amplitude=e.mean_amplitude,
            bends=None if drop[i] else e.bends,
        )
        for i, e in enumerate(sorted_events)
    ]


def get_infered_onsets(
    onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2
) -> np.ndarray:
    """Add inferred onsets where `frames` rises sharply.

    Matches upstream `basic_pitch.note_creation.get_infered_onsets`. (T, F).
    Returns all zeros when `onsets.max() == 0` (matches upstream — when the
    onset head produces no activations, inferred onsets are not synthesized).
    """
    diffs = []
    for n in range(1, n_diff + 1):
        frames_appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    # Upstream: `frame_diff = onsets.max() * frame_diff / frame_diff.max()`.
    # Guard the divisor against zero (numpy yields nan for 0/0); upstream
    # crashes in that case, we return zeros.
    denom = max(float(frame_diff.max()), 1e-12)
    frame_diff = float(onsets.max()) * frame_diff / denom
    return np.maximum(onsets, frame_diff)
