"""Parallel admission, device selection, retries, and cache publication."""
from concurrent.futures import Future
from dataclasses import replace
import os
from pathlib import Path

import memex
import pytest
import torch

from polyddsp import preprocess


def options_for(tmp_path, **overrides):
    values = dict(
        root=tmp_path,
        backend="basic_pitch",
        model_path=None,
        output_dir=tmp_path / "cache",
        file_glob="*.wav",
        sample_rate=16_000,
        hop=64,
        n_voices=6,
        min_freq=None,
        max_freq=None,
        device="cpu",
        parallel=True,
        files_per_task=4,
        memory_headroom=2048,
    )
    return preprocess.PreprocessOptions(**(values | overrides))


def test_parallel_queues_all_work_for_memex_and_reports_each_file(monkeypatch, tmp_path):
    files = [tmp_path / f"{index:02}.wav" for index in range(13)]
    for path in files:
        path.touch()
    batches = []
    shutdowns = []

    class Executor:
        def __init__(self, *, backend, headroom):
            assert (backend, headroom) == ("cpu", 2048)

        def submit(self, fn, batch, args):
            assert fn is preprocess._precompute_batch
            batches.append(list(batch))
            future = Future()
            future.set_result([(path, path != files[-1]) for path in batch])
            return future

        def shutdown(self, *, wait, cancel_futures):
            shutdowns.append((wait, cancel_futures))

    monkeypatch.setattr(memex, "MemoryExecutor", Executor)
    iterator = preprocess._parallel_results(files, options_for(tmp_path), "test")
    first = next(iterator)
    # All tasks must reach Memex before consuming results: no caller worker cap.
    assert [path for batch in batches for path in batch] == files
    results = [first, *iterator]
    assert sorted(path for path, written in results if written) == files[:-1]
    assert (files[-1], False) in results  # Skipped files must not count as written.
    assert len(results) == len(files)
    assert len(batches[0]) == 1  # Short initial task for Memex calibration.
    assert all(len(batch) <= 4 for batch in batches)
    assert shutdowns == [(True, True)]


def test_parallel_skips_fresh_cache_without_starting_executor(monkeypatch, tmp_path):
    audio = tmp_path / "note.wav"
    audio.touch()
    options = options_for(tmp_path)
    cache = preprocess.cache_path_for(
        audio, "test", root=options.root, output_dir=options.output_dir,
    )
    preprocess._save_pitch_cache(cache, torch.zeros(1, 3), torch.zeros(1, 3))
    os.utime(audio, (1, 1))

    def unexpected(**kwargs):
        raise AssertionError("all-cached runs must not start Memex or load a model")

    monkeypatch.setattr(memex, "MemoryExecutor", unexpected)
    assert list(preprocess._parallel_results([audio], options, "test")) == [(audio, False)]


@pytest.mark.parametrize("parallel", [False, True])
def test_runner_selects_parallel_only_when_requested(monkeypatch, tmp_path, parallel):
    audio = tmp_path / "note.wav"
    audio.touch()
    calls = []

    def load(args, device):
        calls.append("load")
        return object()

    def compute(path, args, model, device):
        calls.append("sequential")
        return path, True

    def parallel_results(files, args, suffix):
        calls.append("parallel")
        assert args.parallel
        yield from ((path, True) for path in files)

    monkeypatch.setattr(preprocess, "_load_pitch_model", load)
    monkeypatch.setattr(preprocess, "_precompute_file", compute)
    monkeypatch.setattr(preprocess, "_parallel_results", parallel_results)
    preprocess.run_preprocess(options_for(tmp_path, parallel=parallel))
    assert calls == (["parallel"] if parallel else ["load", "sequential"])


def test_batch_reuses_model_and_injected_device(monkeypatch, tmp_path):
    files = [tmp_path / "a.wav", tmp_path / "b.wav"]
    loads, calls = [], []
    model = object()

    def load(args, device, *, num_threads):
        loads.append((device, num_threads))
        return model

    def compute(path, args, pitch_model, device):
        assert pitch_model is model
        calls.append((path, device))
        return path, True

    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(preprocess, "_load_pitch_model", load)
    monkeypatch.setattr(preprocess, "_precompute_file", compute)
    assert preprocess._precompute_batch(files, options_for(tmp_path), device="cuda:1") == [
        (path, True) for path in files
    ]
    assert loads == [("cuda:1", 1)]
    assert calls == [(path, "cuda:1") for path in files]


def test_batch_preserves_oom_type_for_memex_retry(monkeypatch, tmp_path):
    audio = tmp_path / "note.wav"
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(preprocess, "_load_pitch_model", lambda *args, **kwargs: object())

    def fail(*args):
        raise MemoryError()

    monkeypatch.setattr(preprocess, "_precompute_file", fail)
    with pytest.raises(MemoryError) as exc:
        preprocess._precompute_batch([audio], options_for(tmp_path), device="cpu")
    assert str(audio) in exc.value.__notes__[0]


@pytest.mark.parametrize("previous,device,inside", [
    (None, "cuda:1", "1"), ("3,7", "cuda:1", "7"),
    ("GPU-a,GPU-b", "cuda:0", "GPU-a"), ("3,7", "cuda", "3,7"),
])
def test_pinned_gpu_scope_honors_visibility_and_restores_on_error(monkeypatch, previous, device, inside):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if previous is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", previous)
    with pytest.raises(RuntimeError, match="worker failure"):
        with preprocess._parallel_device_scope(device) as backend:
            assert backend == "cuda"
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == inside
            raise RuntimeError("worker failure")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == previous


def test_pinned_gpu_must_be_visible(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(ValueError, match="not exposed"):
        with preprocess._parallel_device_scope("cuda:1"):
            pytest.fail("invalid GPU should be rejected")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_failed_cache_write_keeps_previous_cache_and_cleans_temporary(monkeypatch, tmp_path):
    cache = tmp_path / "note.f0.pt"
    zeros = torch.zeros(2, 10)
    preprocess._save_pitch_cache(cache, zeros, zeros)
    original = cache.read_bytes()

    def fail_save(blob, f):
        f.write(b"incomplete checkpoint")
        raise MemoryError("out of memory")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(MemoryError):
        preprocess._save_pitch_cache(cache, zeros, zeros)
    assert cache.read_bytes() == original
    assert list(tmp_path.iterdir()) == [cache]


@pytest.mark.parametrize("overrides", [
    {"files_per_task": 0},
    {"memory_headroom": -1},
    {"parallel": True, "device": "mps"},
])
def test_invalid_parallel_options_fail_before_loading(overrides, tmp_path):
    options = replace(options_for(tmp_path), **overrides)
    with pytest.raises(ValueError):
        preprocess.run_preprocess(options)
