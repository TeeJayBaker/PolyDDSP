"""Aligned ground-truth MIDI to PolyDDSP pitch-cache conversion."""
from __future__ import annotations

from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import soundfile as sf
import torch
from symusic import Score

from polyddsp.model.voice_allocation import FrameNoteEvent, allocate_frame_events
from polyddsp.preprocess.cache import cache_path_for, save_pitch_cache


def cache_suffix(n_voices: int, sample_rate: int, target_hop: int) -> str:
    return f"ground_truth_v{n_voices}_sr{sample_rate}_hop{target_hop}"


def _pairing_key(path: Path) -> str:
    """Return the recording ID shared by an audio file and its MIDI annotation."""
    stem = path.stem
    if stem.endswith("-fine-aligned"):
        stem = stem.removesuffix("-fine-aligned")
    for marker in ("_solo", "_comp"):
        marker_end = stem.find(marker)
        if marker_end >= 0:
            end = marker_end + len(marker)
            if end == len(stem) or stem[end] == "_":
                return stem[:end]
    return stem


@lru_cache(maxsize=None)
def _midi_files_by_key(search_root: Path, midi_glob: str) -> dict[str, tuple[Path, ...]]:
    """Index the configured MIDI set once per process by recording ID."""
    matches: dict[str, list[Path]] = {}
    for path in sorted(search_root.glob(midi_glob)):
        if path.is_file():
            matches.setdefault(_pairing_key(path), []).append(path)
    return {key: tuple(paths) for key, paths in matches.items()}


def midi_path_for(
    audio_path: Path,
    *,
    dataset_root: Path,
    midi_glob: str = "**/*.midi",
) -> Path:
    """Find the unique configured MIDI file corresponding to an audio file."""
    search_root = dataset_root.resolve()
    key = _pairing_key(audio_path)
    matches = _midi_files_by_key(search_root, midi_glob).get(key, ())
    if not matches:
        # Three files in the original GAPS release use this suffix, which does
        # not match a glob ending in ``*wc.mid``. Search for those exact names
        # only after ordinary matching fails.
        fallback_names = {
            f"{audio_path.stem}-fine-aligned.mid",
            f"{audio_path.stem}-fine-aligned.midi",
        }
        matches = tuple(
            path
            for path in search_root.rglob("*")
            if path.is_file() and path.name in fallback_names
        )
    if not matches:
        raise FileNotFoundError(
            f"no ground-truth MIDI for {audio_path.name}: no MIDI with pairing key "
            f"{key!r} under {search_root} matching {midi_glob!r}"
        )
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"ambiguous ground-truth MIDI for {audio_path.name}; "
            f"multiple files have pairing key {key!r}: {rendered}"
        )
    return matches[0]


def _pedal_intervals(track, timeline_end: float) -> list[tuple[float, float]]:
    intervals = [(float(pedal.time), float(pedal.end)) for pedal in track.pedals]
    pedal_down = None
    for control in sorted(track.controls, key=lambda event: float(event.time)):
        if int(control.number) != 64:
            continue
        if int(control.value) >= 64 and pedal_down is None:
            pedal_down = float(control.time)
        elif int(control.value) < 64 and pedal_down is not None:
            intervals.append((pedal_down, float(control.time)))
            pedal_down = None
    if pedal_down is not None:
        intervals.append((pedal_down, timeline_end))
    return sorted(intervals)


def _sustained_end(note, notes, pedal_intervals, apply_sustain: bool) -> float:
    end = float(note.end)
    if apply_sustain:
        for pedal_start, pedal_end in pedal_intervals:
            if pedal_start <= end < pedal_end:
                end = pedal_end
                break
    # A repeated key terminates the previous trajectory even while the pedal is down.
    next_onset = min(
        (
            float(other.time)
            for other in notes
            if other is not note
            and int(other.pitch) == int(note.pitch)
            and float(other.time) > float(note.time)
        ),
        default=end,
    )
    return min(end, next_onset)


def _bend_semitones_at_frames(
    pitch_bends,
    frame_times: torch.Tensor,
    pitch_bend_range: float,
) -> torch.Tensor:
    if not pitch_bends:
        return torch.zeros_like(frame_times)
    bends = sorted((float(bend.time), int(bend.value)) for bend in pitch_bends)
    times = [time for time, _ in bends]
    values = [value for _, value in bends]
    offsets = []
    for frame_time in frame_times.tolist():
        index = bisect_right(times, frame_time) - 1
        value = values[index] if index >= 0 else 0
        offsets.append(value / 8192.0 * pitch_bend_range)
    return torch.tensor(offsets, dtype=torch.float32)


def midi_to_voices(
    midi_path: Path,
    *,
    target_frames: int,
    sample_rate: int,
    target_hop: int,
    n_voices: int,
    apply_sustain: bool = True,
    pitch_bend_range: float = 2.0,
    min_freq: float | None = None,
    max_freq: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize aligned MIDI into per-voice Hz and normalized-velocity tracks."""
    score = Score(midi_path).to("second")
    frame_events: list[FrameNoteEvent] = []
    frames_per_second = sample_rate / target_hop

    for track in score.tracks:
        if track.is_drum:
            continue
        notes = list(track.notes)
        pedal_intervals = _pedal_intervals(track, target_frames / frames_per_second)
        bends = list(track.pitch_bends)
        for note in notes:
            midi = int(note.pitch)
            nominal_hz = 440.0 * 2.0 ** ((midi - 69) / 12.0)
            if min_freq is not None and nominal_hz < min_freq:
                continue
            if max_freq is not None and nominal_hz >= max_freq:
                continue

            start = max(0, int(round(float(note.time) * frames_per_second)))
            end_seconds = _sustained_end(note, notes, pedal_intervals, apply_sustain)
            end = min(target_frames, int(round(end_seconds * frames_per_second)))
            if end <= start:
                continue

            frame_times = torch.arange(start, end, dtype=torch.float32) / frames_per_second
            bend = _bend_semitones_at_frames(bends, frame_times, pitch_bend_range)
            midi_per_frame = torch.full_like(bend, float(midi)) + bend
            pitch_hz = 440.0 * 2.0 ** ((midi_per_frame - 69.0) / 12.0)
            velocity = torch.full((end - start,), float(note.velocity) / 127.0)
            frame_events.append(FrameNoteEvent(start, end, midi, pitch_hz, velocity))

    return allocate_frame_events(frame_events, n_voices, target_frames)


def precompute_one(
    audio_path: Path,
    sample_rate: int,
    target_hop: int,
    n_voices: int,
    *,
    dataset_root: Path,
    midi_glob: str = "**/*.midi",
    apply_sustain: bool = True,
    pitch_bend_range: float = 2.0,
    min_freq: float | None = None,
    max_freq: float | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, bool]:
    """Convert the MIDI paired with one audio file and write its cache."""
    midi_path = midi_path_for(
        audio_path, dataset_root=dataset_root, midi_glob=midi_glob,
    )
    if not midi_path.is_file():
        raise FileNotFoundError(f"ground-truth MIDI file does not exist: {midi_path}")

    suffix = cache_suffix(n_voices, sample_rate, target_hop)
    cache = cache_path_for(audio_path, suffix, root=dataset_root, output_dir=output_dir)
    newest_input = max(audio_path.stat().st_mtime, midi_path.stat().st_mtime)
    if cache.exists() and cache.stat().st_mtime >= newest_input:
        return cache, False

    info = sf.info(audio_path)
    target_samples = int(round(info.frames * sample_rate / info.samplerate))
    target_frames = target_samples // target_hop
    pitch, velocity = midi_to_voices(
        midi_path,
        target_frames=target_frames,
        sample_rate=sample_rate,
        target_hop=target_hop,
        n_voices=n_voices,
        apply_sustain=apply_sustain,
        pitch_bend_range=pitch_bend_range,
        min_freq=min_freq,
        max_freq=max_freq,
    )
    save_pitch_cache(cache, pitch, velocity, "ground-truth")
    return cache, True
