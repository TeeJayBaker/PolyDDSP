"""Hydra CLI and execution orchestration for pitch-cache preprocessing."""
from __future__ import annotations

from concurrent.futures import as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import os
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from polyddsp.preprocess import basic_pitch, ground_truth, neutone_amt
from polyddsp.preprocess.cache import cache_path_for, save_pitch_cache


@dataclass(frozen=True)
class PreprocessOptions:
    root: Path
    pitch_cache_source: str
    backend: str
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
    midi_glob: str = "**/*.midi"
    apply_sustain: bool = True
    pitch_bend_range: float = 2.0


def _transcription_backend(source: str) -> str:
    if source == "basic-pitch":
        return "basic-pitch"
    if source in ("neutone-amt-streaming", "neutone-amt-nonstreaming"):
        return "neutone"
    if source == "ground-truth":
        return "ground-truth"
    raise ValueError(
        "model must be basic-pitch, neutone-amt-streaming, or "
        "neutone-amt-nonstreaming, or ground-truth"
    )


def preprocess_options_from_cfg(cfg: DictConfig) -> PreprocessOptions:
    source = str(cfg.model)
    backend = _transcription_backend(source)
    device = str(cfg.device)
    if device == "auto":
        device = (
            "cpu"
            if backend == "ground-truth"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    options = PreprocessOptions(
        root=Path(str(cfg.data_root)),
        pitch_cache_source=source,
        backend=backend,
        model_path=Path(str(cfg.model_path)) if cfg.model_path is not None else None,
        output_dir=Path(str(cfg.output_dir)) if cfg.output_dir is not None else None,
        file_glob=str(cfg.file_glob),
        midi_glob=str(cfg.midi_glob),
        sample_rate=int(cfg.sample_rate),
        hop=int(cfg.hop),
        n_voices=int(cfg.n_voices),
        min_freq=float(cfg.min_freq) if cfg.min_freq is not None else None,
        max_freq=float(cfg.max_freq) if cfg.max_freq is not None else None,
        device=device,
        parallel=bool(cfg.parallel),
        files_per_task=int(cfg.files_per_task),
        memory_headroom=int(cfg.memory_headroom),
        apply_sustain=bool(cfg.get("apply_sustain", True)),
        pitch_bend_range=float(cfg.get("pitch_bend_range", 2.0)),
    )
    validate_preprocess_options(options)
    return options


def validate_preprocess_options(options: PreprocessOptions) -> None:
    expected_backend = _transcription_backend(options.pitch_cache_source)
    if options.backend != expected_backend:
        raise ValueError(
            f"model={options.pitch_cache_source} requires backend={expected_backend}"
        )
    if options.sample_rate < 1 or options.hop < 1 or options.n_voices < 1:
        raise ValueError("sample_rate, hop, and n_voices must be positive")
    if options.files_per_task < 1:
        raise ValueError("files_per_task must be at least 1")
    if options.memory_headroom < 0:
        raise ValueError("memory_headroom must be nonnegative")
    if not options.midi_glob:
        raise ValueError("midi_glob must not be empty")
    if options.pitch_bend_range < 0:
        raise ValueError("pitch_bend_range must be nonnegative")
    if options.parallel:
        try:
            parallel_device = torch.device(options.device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError("parallel preprocessing requires device=cpu, cuda, or cuda:N") from exc
        if parallel_device.type not in {"cpu", "cuda"}:
            raise ValueError("parallel preprocessing requires device=cpu, cuda, or cuda:N")
    if options.backend == "neutone" and options.model_path is None:
        raise ValueError(f"model_path is required when model={options.pitch_cache_source}")
    if options.backend == "neutone" and options.model_path is not None:
        if options.model_path.suffix.lower() == ".data":
            raise ValueError("model_path must point to the .onnx file, not its .onnx.data weights")
        if not options.model_path.is_file():
            raise ValueError(f"model file does not exist: {options.model_path}")


def _load_pitch_model(options: PreprocessOptions, device: str, *, num_threads: int | None = None):
    if options.backend == "basic-pitch":
        from polyddsp.model.pitch import load_basic_pitch
        return load_basic_pitch().to(device)
    if options.backend == "ground-truth":
        return None
    assert options.model_path is not None
    model = neutone_amt.load_model(options.model_path, device, num_threads=num_threads)
    neutone_amt.validate_source(model, options.pitch_cache_source)
    return model


def _precompute_file(audio_path: Path, options: PreprocessOptions, model, device: str):
    audio_options = dict(
        min_freq=options.min_freq,
        max_freq=options.max_freq,
        root=options.root,
        output_dir=options.output_dir,
    )
    if options.backend == "basic-pitch":
        return basic_pitch.precompute_one(
            audio_path, options.sample_rate, options.hop, options.n_voices, device,
            bp_model=model, **audio_options,
        )
    if options.backend == "ground-truth":
        return ground_truth.precompute_one(
            audio_path,
            options.sample_rate,
            options.hop,
            options.n_voices,
            dataset_root=options.root,
            midi_glob=options.midi_glob,
            apply_sustain=options.apply_sustain,
            pitch_bend_range=options.pitch_bend_range,
            min_freq=options.min_freq,
            max_freq=options.max_freq,
            output_dir=options.output_dir,
        )
    return neutone_amt.precompute_one(
        audio_path, options.sample_rate, options.hop, options.n_voices, device,
        model=model, pitch_cache_source=options.pitch_cache_source, **audio_options,
    )


def _precompute_batch(files: list[Path], options: PreprocessOptions, *, device: str):
    torch.set_num_threads(1)
    model = _load_pitch_model(options, device, num_threads=1)
    results = []
    for audio_path in files:
        try:
            _, written = _precompute_file(audio_path, options, model, device)
            results.append((audio_path, written))
        except Exception as exc:
            exc.add_note(f"Failed preprocessing {audio_path}")
            raise
    return results


@contextmanager
def _parallel_device_scope(device: str):
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
    todo = []
    for audio_path in files:
        cache = cache_path_for(audio_path, suffix, root=options.root, output_dir=options.output_dir)
        newest_input = audio_path.stat().st_mtime
        if options.backend == "ground-truth":
            midi_path = ground_truth.midi_path_for(
                audio_path,
                dataset_root=options.root,
                midi_glob=options.midi_glob,
            )
            if not midi_path.is_file():
                raise FileNotFoundError(f"ground-truth MIDI file does not exist: {midi_path}")
            newest_input = max(newest_input, midi_path.stat().st_mtime)
        if cache.exists() and cache.stat().st_mtime >= newest_input:
            yield audio_path, False
        else:
            todo.append(audio_path)
    if not todo:
        return

    from memex import MemoryExecutor
    size = options.files_per_task
    batches = [todo[:1], *[todo[i:i + size] for i in range(1, len(todo), size)]]
    with _parallel_device_scope(options.device) as backend:
        executor = MemoryExecutor(backend=backend, headroom=options.memory_headroom)
        try:
            pending = [executor.submit(_precompute_batch, batch, options) for batch in batches]
            for future in as_completed(pending):
                yield from future.result()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def run_preprocess(options: PreprocessOptions) -> int:
    validate_preprocess_options(options)
    files = sorted(options.root.glob(options.file_glob))
    if not files:
        raise FileNotFoundError(f"no files in {options.root} matching {options.file_glob}")
    if options.backend == "basic-pitch":
        suffix = basic_pitch.cache_suffix(options.n_voices, options.sample_rate, options.hop)
    elif options.backend == "neutone":
        suffix = neutone_amt.cache_suffix(
            options.pitch_cache_source, options.n_voices, options.sample_rate, options.hop,
        )
    else:
        suffix = ground_truth.cache_suffix(options.n_voices, options.sample_rate, options.hop)
    print(f"found {len(files)} files; caching {options.backend} allocation with suffix={suffix}")

    started = time.perf_counter()
    if not options.parallel:
        model = _load_pitch_model(options, options.device)
        if model is not None:
            print(
                f"loaded {options.backend} in {time.perf_counter() - started:.2f}s "
                "(reused for all files)"
            )
        results = ((path, _precompute_file(path, options, model, options.device)[1]) for path in files)
    else:
        print(
            f"Memex: automatic concurrency on {options.device}, up to "
            f"{options.files_per_task} files/task, {options.memory_headroom} MiB headroom; "
            "calibrating on one file before increasing concurrency"
        )
        results = _parallel_results(files, options, suffix)

    started = time.perf_counter()
    n_written = 0
    progress = tqdm(total=len(files), desc="preprocess", unit="file", dynamic_ncols=True, position=0)
    status = tqdm(total=0, bar_format="{desc}", leave=False, dynamic_ncols=True, position=1)
    try:
        for path, recomputed in results:
            n_written += int(recomputed)
            progress.update(1)
            marker = "wrote" if recomputed else "skip "
            status.set_description_str(f"[{marker}] {path.relative_to(options.root)}", refresh=True)
    finally:
        results.close()
        status.close()
        progress.close()
    print(f"done: {n_written}/{len(files)} recomputed in {time.perf_counter() - started:.1f}s")
    return n_written


# Compatibility aliases for private tests while orchestration lives in this module.
_save_pitch_cache = save_pitch_cache


@hydra.main(config_path="../../configs", config_name="preprocess", version_base=None)
def main(cfg: DictConfig) -> None:
    run_preprocess(preprocess_options_from_cfg(cfg))
