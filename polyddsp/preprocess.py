"""Precompute pitch annotations for each audio file in a dataset.

For each file in the configured experiment dataset, runs Basic Pitch or
neutoneAMT, extracts note events, and allocates them to voice slots. The
resulting `{pitch, velocity}` tensors (each (V, T) at `audio_len // hop` frames
of the target sample rate) are cached beside the audio by default. With a
separate pitch cache root, paths relative to the dataset root are preserved
beneath that directory. Idempotent: skips caches newer than their source audio.

Usage:
    polyddsp-preprocess experiment=guitarset
    polyddsp-preprocess experiment=maestro \
        model=neutone-amt \
        model_path=/path/to/amt.onnx parallel=true

Basic Pitch transcription lives in
`polyddsp.model.pitch.basic_pitch_to_voices`; neutoneAMT uses Lightning
checkpoints or ONNX exports with its packaged piano-roll decoder.
"""
from __future__ import annotations

from concurrent.futures import as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import hydra
import soundfile as sf
import torch
import torchaudio.functional as AF
from omegaconf import DictConfig
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from neutone_amt.model import AMTModel
    from polyddsp.model.neutone_onnx import NeutoneONNXModel
    from polyddsp.model.pitch import BasicPitchModel


NEUTONE_NATIVE_SR = 44_100
MIDI_VELOCITY_64 = 64.0 / 127.0


@dataclass(frozen=True)
class PreprocessOptions:
    """Resolved, spawn-safe preprocessing settings derived from Hydra config."""

    root: Path
    backend: Literal["basic_pitch", "neutone"]
    model_path: Path | None
    output_dir: Path | None
    file_glob: str
    sample_rate: int
    hop: int
    n_voices: int
    min_freq: float | None
    max_freq: float | None
    device: str
    parallel: bool
    files_per_task: int
    memory_headroom: int


def preprocess_options_from_cfg(cfg: DictConfig) -> PreprocessOptions:
    """Resolve shared experiment settings without passing Hydra into workers."""
    execution = cfg
    backends = {"basic-pitch": "basic_pitch", "neutone-amt": "neutone"}
    if execution.model not in backends:
        raise ValueError("model must be basic-pitch or neutone-amt")
    device = str(execution.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(str(execution.model_path)) if execution.model_path is not None else None
    output_dir = (
        Path(str(execution.output_dir))
        if execution.output_dir is not None else None
    )
    options = PreprocessOptions(
        root=Path(str(execution.data_root)),
        backend=backends[execution.model],
        model_path=model_path,
        output_dir=output_dir,
        file_glob=str(execution.file_glob),
        sample_rate=int(execution.sample_rate),
        hop=int(execution.hop),
        n_voices=int(execution.n_voices),
        min_freq=(float(execution.min_freq) if execution.min_freq is not None else None),
        max_freq=(float(execution.max_freq) if execution.max_freq is not None else None),
        device=device,
        parallel=bool(execution.parallel),
        files_per_task=int(execution.files_per_task),
        memory_headroom=int(execution.memory_headroom),
    )
    validate_preprocess_options(options)
    return options


def validate_preprocess_options(options: PreprocessOptions) -> None:
    if options.backend not in {"basic_pitch", "neutone"}:
        raise ValueError(
            "preprocessing backend must be basic_pitch or neutone"
        )
    if options.sample_rate < 1 or options.hop < 1 or options.n_voices < 1:
        raise ValueError(
            "sample_rate, hop, and n_voices must be positive"
        )
    if options.files_per_task < 1:
        raise ValueError("files_per_task must be at least 1")
    if options.memory_headroom < 0:
        raise ValueError("memory_headroom must be nonnegative")
    if options.parallel:
        try:
            parallel_device = torch.device(options.device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "parallel preprocessing requires device=cpu, cuda, or cuda:N"
            ) from exc
        if parallel_device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "parallel preprocessing requires device=cpu, cuda, or cuda:N"
            )
    if options.backend == "neutone" and options.model_path is None:
        raise ValueError(
            "model_path is required when model=neutone-amt"
        )
    if options.backend == "neutone" and options.model_path is not None:
        if options.model_path.suffix.lower() == ".data":
            raise ValueError(
                "model_path must point to the .onnx file, not its .onnx.data weights"
            )
        if not options.model_path.is_file():
            raise ValueError(f"model file does not exist: {options.model_path}")


def _save_pitch_cache(cache: Path, pitch: torch.Tensor, velocity: torch.Tensor) -> None:
    """Publish a complete cache atomically, including when Memex retries a task."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache.parent, prefix=f".{cache.name}.", delete=False) as f:
            temporary = Path(f.name)
            torch.save({"pitch": pitch.contiguous(), "velocity": velocity.contiguous()}, f)
        os.replace(temporary, cache)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cache_path_for(
    audio_path: Path,
    suffix: str,
    *,
    root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Return a cache path, optionally mirrored beneath a separate output root."""
    target = audio_path
    if output_dir is not None:
        if root is None:
            raise ValueError("root is required when output_dir is provided")
        try:
            relative = audio_path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"audio path {audio_path} is outside dataset root {root}") from exc
        target = output_dir / relative
    return target.with_suffix(target.suffix + f".{suffix}.f0.pt")


def load_resample(path: Path, target_sr: int) -> torch.Tensor:
    """Read an audio file as mono float32 and resample to `target_sr`."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    audio = torch.from_numpy(wav)
    if sr != target_sr:
        audio = AF.resample(audio, orig_freq=sr, new_freq=target_sr)
    return audio


def bp_cache_suffix(n_voices: int, sample_rate: int, target_hop: int) -> str:
    return f"bp_v{n_voices}_sr{sample_rate}_hop{target_hop}"


def neutone_cache_suffix(n_voices: int, sample_rate: int, target_hop: int) -> str:
    return f"neutone_v{n_voices}_sr{sample_rate}_hop{target_hop}"


def resolve_pitch_cache(
    cfg, source: str = "basic-pitch",
) -> tuple[str | None, str | None, int | None]:
    """Map a Hydra config to `RawAudioDataset`'s (kind, suffix, n_voices) cache args.

    Returns `(None, None, None)` when the experiment's pitch source runs in-loop
    rather than reading a precomputed cache.
    """
    if cfg.experiment.model.get("pitch_source") != "cached_basic_pitch":
        return None, None, None
    cache_for_source = {
        "basic-pitch": ("basic_pitch", bp_cache_suffix),
        "neutone-amt": ("neutone", neutone_cache_suffix),
    }
    if source not in cache_for_source:
        raise ValueError(
            f"Unknown pitch cache source {source!r}; expected basic-pitch or neutone-amt"
        )
    cache_kind, suffix_fn = cache_for_source[source]
    n_voices = cfg.experiment.model.n_voices
    suffix = suffix_fn(n_voices, cfg.model.sr, cfg.model.frame_hop)
    return cache_kind, suffix, n_voices


def precompute_one_bp(
    audio_path: Path,
    sample_rate: int,
    target_hop: int,
    n_voices: int,
    device: str = "cpu",
    bp_model: Optional["BasicPitchModel"] = None,
    min_freq: float | None = None,
    max_freq: float | None = None,
    root: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, bool]:
    """Cache (pitch, velocity) tensors at target rate. Returns (cache_path, recomputed).

    Thin I/O wrapper around `polyddsp.model.pitch.basic_pitch_to_voices` — the
    same function the in-loop `PitchEncoder(source="basic_pitch")` calls, so the
    cache and the live encoder cannot diverge. Pass `bp_model` to reuse one
    loaded CNN (and its CQT kernels) across files.
    """
    from polyddsp.model.pitch import BP_NATIVE_SR, basic_pitch_to_voices

    suffix = bp_cache_suffix(n_voices, sample_rate, target_hop)
    cache = cache_path_for(audio_path, suffix, root=root, output_dir=output_dir)
    if cache.exists() and cache.stat().st_mtime > audio_path.stat().st_mtime:
        return cache, False

    # Resample the source file *directly* to BP's native rate: going via the
    # model rate would discard everything above 8 kHz that BP's CQT reaches.
    audio_22k = load_resample(audio_path, BP_NATIVE_SR).to(device)

    # Frame-count budget at target rate.
    n_target_samples = int(round(audio_22k.shape[-1] * sample_rate / BP_NATIVE_SR))
    target_frames = n_target_samples // target_hop

    pitch, velocity = basic_pitch_to_voices(
        audio_22k,
        n_voices=n_voices,
        target_frames=target_frames,
        bp_model=bp_model,
        min_freq=min_freq,
        max_freq=max_freq,
    )

    _save_pitch_cache(cache, pitch, velocity)
    return cache, True


def audio_within_duration_limit(
    audio_path: Path, *, max_samples: int | None, sample_rate: int,
) -> bool:
    """Check only the audio header against a limit at the model's sample rate."""
    if max_samples is None:
        return True
    info = sf.info(audio_path)
    # Compare integer counts to account for resampling without rounding a
    # just-over-limit recording down to the accepted duration.
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
def neutone_amt_to_voices(
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
    """Transcribe one file with Neutone AMT into PolyDDSP voice tracks."""
    native_sr = int(getattr(model.spec, "sample_rate", getattr(model.spec, "sr", NEUTONE_NATIVE_SR)))
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

    # The allocator copies the chosen MIDI bin into every active output frame.
    velocity_grid = torch.full((88, target_frames), MIDI_VELOCITY_64)
    pitch, velocity = allocate_to_voices(
        events, velocity_grid, n_voices, target_frames, bp_to_target_ratio=1.0,
    )
    return pitch, velocity


@torch.no_grad()
def precompute_one_neutone_amt(
    audio_path: Path,
    sample_rate: int,
    target_hop: int,
    n_voices: int,
    device: str,
    model: "AMTModel | NeutoneONNXModel",
    min_freq: float | None = None,
    max_freq: float | None = None,
    root: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, bool]:
    """Cache neutoneAMT notes as PolyDDSP pitch and constant-velocity tracks."""
    suffix = neutone_cache_suffix(n_voices, sample_rate, target_hop)
    cache = cache_path_for(audio_path, suffix, root=root, output_dir=output_dir)
    if cache.exists() and cache.stat().st_mtime > audio_path.stat().st_mtime:
        return cache, False

    native_sr = int(getattr(model.spec, "sample_rate", getattr(model.spec, "sr", NEUTONE_NATIVE_SR)))
    if not audio_within_duration_limit(
        audio_path, max_samples=getattr(model, "max_input_samples", None), sample_rate=native_sr,
    ):
        return cache, False

    pitch, velocity = neutone_amt_to_voices(
        audio_path,
        sample_rate,
        target_hop,
        n_voices,
        device,
        model,
        min_freq=min_freq,
        max_freq=max_freq,
    )
    _save_pitch_cache(cache, pitch, velocity)
    return cache, True


def load_neutone_amt_model(
    model_path: str | Path, device: str, *, num_threads: int | None = None,
):
    """Load a Neutone AMT Lightning checkpoint or ONNX export."""
    model_path = Path(model_path)
    if model_path.suffix.lower() == ".onnx":
        from polyddsp.model.neutone_onnx import NeutoneONNXModel
        return NeutoneONNXModel(model_path, device, num_threads=num_threads)
    from neutone_amt.model import AMTModel
    return AMTModel.load_from_checkpoint(
        model_path, map_location=device, strict=False, weights_only=False,
    ).to(device).eval()


def _load_pitch_model(
    options: PreprocessOptions, device: str, *, num_threads: int | None = None,
):
    if options.backend == "basic_pitch":
        from polyddsp.model.pitch import load_basic_pitch
        return load_basic_pitch().to(device)
    assert options.model_path is not None
    return load_neutone_amt_model(options.model_path, device, num_threads=num_threads)


def _precompute_file(
    audio_path: Path, options: PreprocessOptions, model, device: str,
) -> tuple[Path, bool]:
    common = dict(
        min_freq=options.min_freq, max_freq=options.max_freq,
        root=options.root, output_dir=options.output_dir,
    )
    if options.backend == "basic_pitch":
        return precompute_one_bp(
            audio_path, options.sample_rate, options.hop, options.n_voices, device,
            bp_model=model, **common,
        )
    return precompute_one_neutone_amt(
        audio_path, options.sample_rate, options.hop, options.n_voices, device,
        model=model, **common,
    )


def _precompute_batch(
    files: list[Path], options: PreprocessOptions, *, device: str,
) -> list[tuple[Path, bool]]:
    """Spawn-safe Memex task; keep one model loaded for the entire batch."""
    # Every process otherwise competes for all host cores, especially with ORT.
    torch.set_num_threads(1)
    model = _load_pitch_model(options, device, num_threads=1)
    results = []
    for audio_path in files:
        try:
            _, written = _precompute_file(audio_path, options, model, device)
            results.append((audio_path, written))
        except Exception as exc:
            # Preserve exception type as well as text for Memex's OOM detection.
            exc.add_note(f"Failed preprocessing {audio_path}")
            raise
    return results


@contextmanager
def _parallel_device_scope(device: str):
    """Make Memex account for exactly the GPUs exposed to its spawned tasks."""
    target = torch.device(device)
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    pinned = target.type == "cuda" and target.index is not None
    if pinned:
        if previous is None:
            selected = str(target.index)
        else:
            visible = [value.strip() for value in previous.split(",") if value.strip()]
            if target.index >= len(visible) or visible[target.index] == "-1":
                raise ValueError(f"{device} is not exposed by CUDA_VISIBLE_DEVICES={previous!r}")
            selected = visible[target.index]
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
    try:
        yield target.type
    finally:
        if pinned:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _parallel_results(files: list[Path], options: PreprocessOptions, suffix: str):
    """Queue stale files and let Memex control task admission and concurrency."""
    todo = []
    for audio_path in files:
        cache = cache_path_for(
            audio_path, suffix, root=options.root, output_dir=options.output_dir,
        )
        if cache.exists() and cache.stat().st_mtime > audio_path.stat().st_mtime:
            yield audio_path, False
        else:
            todo.append(audio_path)
    if not todo:
        return

    from memex import MemoryExecutor

    # Memex calibrates one task before admitting concurrency. Use one file for
    # that task, then amortize model startup across batches.
    batch_size = options.files_per_task
    batches = [todo[:1], *[todo[i:i + batch_size] for i in range(1, len(todo), batch_size)]]
    with _parallel_device_scope(options.device) as backend:
        executor = MemoryExecutor(backend=backend, headroom=options.memory_headroom)
        try:
            # submit() queues work; Memex decides when resources allow it to run.
            pending = [
                executor.submit(_precompute_batch, batch, options) for batch in batches
            ]
            for future in as_completed(pending):
                yield from future.result()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def run_preprocess(options: PreprocessOptions) -> int:
    """Precompute one configured dataset. Returns the number of caches written."""
    validate_preprocess_options(options)
    files = sorted(options.root.glob(options.file_glob))
    if not files:
        raise FileNotFoundError(
            f"no files in {options.root} matching {options.file_glob}"
        )
    suffix_fn = bp_cache_suffix if options.backend == "basic_pitch" else neutone_cache_suffix
    suffix = suffix_fn(options.n_voices, options.sample_rate, options.hop)
    print(f"found {len(files)} files; caching {options.backend} allocation with suffix={suffix}")

    t0 = time.perf_counter()
    if not options.parallel:
        pitch_model = _load_pitch_model(options, options.device)
        print(
            f"loaded {options.backend} in {time.perf_counter() - t0:.2f}s "
            "(reused for all files)"
        )
        results = (
            (f, _precompute_file(f, options, pitch_model, options.device)[1])
            for f in files
        )
    else:
        print(
            f"Memex: automatic concurrency on {options.device}, "
            f"up to {options.files_per_task} files/task, "
            f"{options.memory_headroom} MiB headroom; "
            "calibrating on one file before increasing concurrency"
        )
        results = _parallel_results(files, options, suffix)

    t0 = time.perf_counter()
    n_written = 0
    progress = tqdm(total=len(files), desc="preprocess", unit="file", dynamic_ncols=True, position=0)
    # A second, text-only line keeps long filenames from changing the main
    # bar's width. Unlike tqdm.write(), it is replaced in place and erased when
    # preprocessing finishes instead of leaving one permanent line per file.
    status = tqdm(total=0, bar_format="{desc}", leave=False, dynamic_ncols=True, position=1)
    try:
        for f, recomputed in results:
            n_written += int(recomputed)
            progress.update(1)
            marker = "wrote" if recomputed else "skip "
            status.set_description_str(
                f"[{marker}] {f.relative_to(options.root)}", refresh=True,
            )
    finally:
        results.close()
        status.close()
        progress.close()
    print(f"done: {n_written}/{len(files)} recomputed in {time.perf_counter() - t0:.1f}s")
    return n_written


@hydra.main(config_path="../configs", config_name="preprocess", version_base=None)
def main(cfg: DictConfig) -> None:
    run_preprocess(preprocess_options_from_cfg(cfg))


if __name__ == "__main__":
    main()
