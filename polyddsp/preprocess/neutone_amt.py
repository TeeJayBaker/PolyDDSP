"""Neutone AMT audio transcription backend."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf
import torch
from tqdm.auto import tqdm

from polyddsp.preprocess.cache import cache_path_for, save_pitch_cache
from polyddsp.preprocess.audio import load_resample

if TYPE_CHECKING:
    from neutone_amt.model import AMTModel
    from polyddsp.model.neutone_onnx import NeutoneONNXModel

NATIVE_SAMPLE_RATE = 44_100
MIDI_VELOCITY_64 = 64.0 / 127.0


def variant(source: str) -> str:
    if source == "neutone-amt-streaming":
        return "streaming"
    if source == "neutone-amt-nonstreaming":
        return "nonstreaming"
    raise ValueError(
        "Neutone model source must be neutone-amt-streaming or "
        "neutone-amt-nonstreaming"
    )


def cache_suffix(
    source: str, n_voices: int, sample_rate: int, target_hop: int,
) -> str:
    return f"neutone_{variant(source)}_v{n_voices}_sr{sample_rate}_hop{target_hop}"


def audio_within_duration_limit(
    audio_path: Path, *, max_samples: int | None, sample_rate: int,
) -> bool:
    """Check only the audio header against a model input-duration limit."""
    if max_samples is None:
        return True
    info = sf.info(audio_path)
    if info.frames * sample_rate <= max_samples * info.samplerate:
        return True
    tqdm.write(
        f"[skip] {audio_path}: {info.duration:.2f}s exceeds the nonstreaming "
        f"ONNX CUDA duration limit of {max_samples / sample_rate:.2f}s; "
        "use the streaming export or device=cpu for this recording.",
        file=sys.stderr,
    )
    return False


@torch.no_grad()
def to_voices(
    audio_path: Path,
    sample_rate: int,
    target_hop: int,
    n_voices: int,
    device: str,
    model: "AMTModel | NeutoneONNXModel",
    min_freq: float | None = None,
    max_freq: float | None = None,
    target_frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transcribe one file into PolyDDSP voice tracks."""
    native_sr = int(getattr(model.spec, "sample_rate", getattr(model.spec, "sr", NATIVE_SAMPLE_RATE)))
    from neutone_amt.model import select_delay, unshift_predictions
    from neutone_amt.pianoroll import pianoroll_to_midi
    from polyddsp.model.note_extraction import NoteEvent
    from polyddsp.model.voice_allocation import allocate_to_voices

    audio_amt = load_resample(audio_path, native_sr).to(getattr(model, "input_device", device))
    if target_frames is None:
        target_samples = int(round(audio_amt.shape[-1] * sample_rate / native_sr))
        target_frames = target_samples // target_hop

    outputs = model(audio_amt.unsqueeze(0))
    delays = list(getattr(model, "delays", []) or [])
    outputs, selected_delay = select_delay(outputs, delays)
    shift = selected_delay if delays else int(getattr(model, "target_shift", 0) or 0)
    onset = unshift_predictions(torch.sigmoid(outputs["onset"][0]), shift)
    frame = unshift_predictions(torch.sigmoid(outputs["frame"][0]), shift)
    offset = unshift_predictions(torch.sigmoid(outputs["offset"][0]), shift)

    model_hop = int(getattr(model.spec, "hop_length", 512))
    score = pianoroll_to_midi(
        onset,
        frame,
        offset_tensor=offset,
        velocity_tensor=None,
        fps=native_sr / model_hop,
        velocity=64,
    )
    events: list[NoteEvent] = []
    for track in score.tracks:
        for note in track.notes:
            midi = int(note.pitch)
            hz = 440.0 * 2.0 ** ((midi - 69) / 12.0)
            if min_freq is not None and hz < min_freq:
                continue
            if max_freq is not None and hz >= max_freq:
                continue
            start = max(0, int(round(float(note.time) * sample_rate / target_hop)))
            end = min(
                target_frames,
                int(round((float(note.time) + float(note.duration)) * sample_rate / target_hop)),
            )
            if end > start:
                events.append(NoteEvent(start, end, midi, MIDI_VELOCITY_64))

    velocity_grid = torch.full((88, target_frames), MIDI_VELOCITY_64)
    return allocate_to_voices(
        events, velocity_grid, n_voices, target_frames, bp_to_target_ratio=1.0,
    )


@torch.no_grad()
def precompute_one(
    audio_path: Path,
    sample_rate: int,
    target_hop: int,
    n_voices: int,
    device: str,
    model: "AMTModel | NeutoneONNXModel",
    pitch_cache_source: str | None = None,
    min_freq: float | None = None,
    max_freq: float | None = None,
    root: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, bool]:
    """Transcribe one file and cache constant-velocity voice tracks."""
    source = pitch_cache_source or source_for_model(model)
    validate_source(model, source)
    suffix = cache_suffix(source, n_voices, sample_rate, target_hop)
    cache = cache_path_for(audio_path, suffix, root=root, output_dir=output_dir)
    if cache.exists() and cache.stat().st_mtime > audio_path.stat().st_mtime:
        return cache, False

    native_sr = int(getattr(model.spec, "sample_rate", getattr(model.spec, "sr", NATIVE_SAMPLE_RATE)))
    if not audio_within_duration_limit(
        audio_path, max_samples=getattr(model, "max_input_samples", None), sample_rate=native_sr,
    ):
        return cache, False
    pitch, velocity = to_voices(
        audio_path,
        sample_rate,
        target_hop,
        n_voices,
        device,
        model,
        min_freq=min_freq,
        max_freq=max_freq,
    )
    save_pitch_cache(cache, pitch, velocity, source)
    return cache, True


def load_model(model_path: str | Path, device: str, *, num_threads: int | None = None):
    """Load a Neutone AMT Lightning checkpoint or ONNX export."""
    model_path = Path(model_path)
    if model_path.suffix.lower() == ".onnx":
        from polyddsp.model.neutone_onnx import NeutoneONNXModel
        return NeutoneONNXModel(model_path, device, num_threads=num_threads)
    from neutone_amt.model import AMTModel
    return AMTModel.load_from_checkpoint(
        model_path, map_location=device, strict=False, weights_only=False,
    ).to(device).eval()


def source_for_model(model) -> str:
    mode = "streaming" if bool(getattr(model, "streaming", False)) else "nonstreaming"
    return f"neutone-amt-{mode}"


def validate_source(model, expected_source: str) -> None:
    variant(expected_source)
    actual_source = source_for_model(model)
    if actual_source != expected_source:
        raise ValueError(
            f"model={expected_source} does not match the selected model file; "
            f"detected {actual_source}"
        )
