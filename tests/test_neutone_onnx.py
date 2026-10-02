"""ONNX format, streaming state, and prediction timing regressions."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import soundfile as sf
import torch

from polyddsp.model.neutone_onnx import NeutoneONNXModel
from polyddsp import preprocess
from polyddsp.preprocess import (
    MIDI_VELOCITY_64,
    PreprocessOptions,
    precompute_one_neutone_amt,
    run_preprocess,
    validate_preprocess_options,
)


class FakeSession:
    def __init__(self, metadata=None, streaming=False):
        self.metadata = metadata or {}
        self.streaming = streaming
        self.calls = []
        self.providers = ["CPUExecutionProvider"]

    def get_inputs(self):
        inputs = [SimpleNamespace(
            name="input_audio", type="tensor(float)",
            shape=[1, 4] if self.streaming else ["batch", "samples"],
        )]
        if self.streaming:
            inputs.append(SimpleNamespace(name="cache", type="tensor(float)", shape=[1, 2]))
        return inputs

    def get_outputs(self):
        names = ["velocity", "offset", "onset", "frame"]
        return [SimpleNamespace(name=name) for name in names + (["next_cache"] if self.streaming else [])]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self.metadata)

    def get_providers(self):
        return self.providers

    def run(self, names, inputs):
        self.calls.append({key: value.copy() for key, value in inputs.items()})
        if self.streaming:
            # The first delay encodes the step counter; the other is distinct so
            # selecting the wrong delay cannot accidentally pass the assertions.
            step = inputs["cache"][0, 0]
            roll = np.stack([
                np.full((1, 88, 1), step, dtype=np.float32),
                np.full((1, 88, 1), step + 100, dtype=np.float32),
            ], axis=1)
            result = {key: roll for key in ("onset", "frame", "offset")}
            result["next_cache"] = inputs["cache"] + 1
        else:
            result = {key: np.full((1, 88, 100), -20, dtype=np.float32)
                      for key in ("onset", "frame", "offset")}
            result["onset"][0, 60 - 21, 10] = 20
            result["frame"][0, 60 - 21, 10:30] = 20
            result["offset"][0, 60 - 21, 30] = 20
        return [result[name] for name in names]


def install_session(monkeypatch, session):
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: session)


@pytest.mark.parametrize("baked,start,end", [("1", 25, 75), ("0", 15, 65)])
def test_offline_metadata_controls_resampling_and_note_alignment(monkeypatch, tmp_path, baked, start, end):
    session = FakeSession({
        "sample_rate": "8000", "hop_length": "80",
        "target_shift_frames": "4", "target_shift_baked": baked,
    })
    install_session(monkeypatch, session)
    model = NeutoneONNXModel(tmp_path / "amt.onnx")
    audio = tmp_path / "source" / "nested" / "note.wav"
    audio.parent.mkdir(parents=True)
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    # Cache validity uses strictly newer mtimes; avoid filesystem clock granularity.
    source_time = audio.stat().st_mtime - 2
    os.utime(audio, (source_time, source_time))
    kwargs = dict(
        sample_rate=16000, target_hop=64, n_voices=6, device="cpu", model=model,
        root=tmp_path / "source", output_dir=tmp_path / "cache",
    )
    cache, written = precompute_one_neutone_amt(audio, **kwargs)
    assert written and cache.parent == tmp_path / "cache" / "nested"
    assert session.calls[0]["input_audio"].shape == (1, 8000)
    assert session.calls[0]["input_audio"].dtype == np.float32
    blob = torch.load(cache, weights_only=True)
    assert blob["pitch"].shape == (6, 250)
    expected = torch.zeros(6, 250, dtype=torch.bool)
    expected[0, start:end] = True
    assert torch.equal(blob["pitch"] != 0, expected)
    assert torch.allclose(blob["pitch"][expected], torch.tensor(261.625565))
    assert torch.all(blob["velocity"][expected] == MIDI_VELOCITY_64)
    assert torch.all(blob["velocity"][~expected] == 0)
    assert precompute_one_neutone_amt(audio, **kwargs) == (cache, False)
    assert len(session.calls) == 1


def test_legacy_offline_metadata_defaults_and_raw_logits(monkeypatch):
    session = FakeSession()
    install_session(monkeypatch, session)
    model = NeutoneONNXModel(Path("amt.onnx"))
    assert (model.spec.sample_rate, model.spec.hop_length, model.target_shift) == (44100, 512, 0)
    outputs = model(torch.zeros(1, 44100, dtype=torch.float64))
    assert outputs["onset"][0, 39, 10] == 20  # Sigmoid belongs to the shared decoder.
    assert outputs["offset"][0, 39, 30] == 20


@pytest.mark.parametrize("device,streaming,limited", [
    ("cuda:0", False, True), ("cuda:0", True, False), ("cpu", False, False),
])
def test_duration_limit_applies_only_to_offline_cuda(monkeypatch, device, streaming, limited):
    session = FakeSession({"sample_rate": "8000", "hop_length": "4"}, streaming=streaming)
    session.providers.insert(0, "CUDAExecutionProvider")
    monkeypatch.setattr(ort, "get_available_providers", lambda: session.providers)
    install_session(monkeypatch, session)
    model = NeutoneONNXModel(Path("amt.onnx"), device)
    assert model.max_input_samples == (65000 * 4 if limited else None)


@pytest.mark.parametrize("source_frames,allowed", [(15999, True), (16000, True), (16001, False)])
def test_header_duration_limit_accounts_for_resampling(tmp_path, capsys, source_frames, allowed):
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(source_frames, dtype=np.float32), 16000)
    assert preprocess.audio_within_duration_limit(
        path, max_samples=8000, sample_rate=8000,
    ) is allowed
    message = capsys.readouterr().err
    if not allowed:
        assert str(path) in message and "duration limit" in message
    else:
        assert not message


def test_oversized_audio_skips_before_loading_or_inference(monkeypatch, tmp_path, capsys):
    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(16001, dtype=np.float32), 16000)
    session = FakeSession({"sample_rate": "8000", "hop_length": "80"})
    install_session(monkeypatch, session)
    model = NeutoneONNXModel(Path("amt.onnx"))
    model.max_input_samples = 8000

    def unexpected(*args, **kwargs):
        pytest.fail("oversized audio must be skipped before reading waveform samples")

    monkeypatch.setattr(preprocess, "load_resample", unexpected)
    cache, written = precompute_one_neutone_amt(
        path, sample_rate=16000, target_hop=64, n_voices=6, device="cuda:0", model=model,
    )
    assert not written and not cache.exists()
    assert not session.calls
    assert "[skip]" in capsys.readouterr().err


def test_streaming_preserves_state_pads_tail_selects_delay_and_resets(monkeypatch):
    session = FakeSession({
        "sample_rate": "8000", "hop_length": "4", "delays_frames": "1,3",
        "lookahead_frames": "1", "latency_frames": "2",
    }, streaming=True)
    install_session(monkeypatch, session)
    model = NeutoneONNXModel(Path("amt.onnx"))
    audio = torch.arange(10, dtype=torch.float32).reshape(1, -1)
    first = model(audio)
    assert model.target_shift == 0  # The adapter has already realigned the rolls.
    assert len(session.calls) == 5  # Three audio hops, then two latency-flush hops.
    np.testing.assert_array_equal(session.calls[2]["input_audio"], [[8, 9, 0, 0]])
    np.testing.assert_array_equal(session.calls[3]["input_audio"], [[0, 0, 0, 0]])
    for index, call in enumerate(session.calls):
        np.testing.assert_array_equal(call["cache"], [[index, index]])
    assert first["onset"].shape == (1, 88, 3)
    torch.testing.assert_close(first["onset"][0, 0], torch.tensor([2., 3., 4.]))
    second = model(audio)
    np.testing.assert_array_equal(session.calls[5]["cache"], [[0, 0]])
    for name in first:
        torch.testing.assert_close(first[name], second[name])


def test_cuda_provider_uses_requested_device_and_detects_fallback(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    requested = []

    def create(*args, providers, **kwargs):
        requested.extend(providers)
        return session

    monkeypatch.setattr(ort, "InferenceSession", create)
    with pytest.raises(RuntimeError, match="could not initialize CUDA"):
        NeutoneONNXModel(Path("amt.onnx"), "cuda:2")
    assert requested[0] == ("CUDAExecutionProvider", {"device_id": 2})
    session.providers.insert(0, "CUDAExecutionProvider")
    model = NeutoneONNXModel(Path("amt.onnx"), "cuda:2")
    assert model.input_device == "cpu"


def test_runner_loads_onnx_once_for_multiple_files(monkeypatch, tmp_path):
    session = FakeSession()
    loads = []

    def create(path, **kwargs):
        loads.append(path)
        return session

    monkeypatch.setattr(ort, "InferenceSession", create)
    path = tmp_path / "amt.ONNX"
    path.touch()
    for name in ("a.wav", "b.wav"):
        sf.write(tmp_path / name, np.zeros(44100, dtype=np.float32), 44100)
    run_preprocess(PreprocessOptions(
        root=tmp_path,
        backend="neutone",
        model_path=path,
        output_dir=tmp_path / "cache",
        file_glob="*.wav",
        sample_rate=16_000,
        hop=64,
        n_voices=6,
        min_freq=None,
        max_freq=None,
        device="cpu",
        parallel=False,
        files_per_task=4,
        memory_headroom=2048,
    ))
    assert loads == [str(path)]
    assert len(list((tmp_path / "cache").glob("*.f0.pt"))) == 2


def test_config_explains_external_weights_path(tmp_path):
    options = PreprocessOptions(
        root=tmp_path,
        backend="neutone",
        model_path=Path("amt.onnx.data"),
        output_dir=None,
        file_glob="*.wav",
        sample_rate=16_000,
        hop=64,
        n_voices=6,
        min_freq=None,
        max_freq=None,
        device="cpu",
        parallel=False,
        files_per_task=4,
        memory_headroom=2048,
    )
    with pytest.raises(ValueError, match="not its .onnx.data weights"):
        validate_preprocess_options(options)
