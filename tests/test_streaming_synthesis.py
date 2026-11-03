"""Source-independent streaming synthesis and its offline/training boundary."""
from __future__ import annotations

from itertools import cycle

import pytest
import torch
import torch.nn.functional as F

from polyddsp.model.polyddsp import PolyDDSP
from polyddsp.model.streaming import StreamingSynthesizer


def _model(voices=2, use_z=True, noise=True, reverb=True, bands=65):
    torch.manual_seed(61)
    return PolyDDSP(
        n_voices=voices, use_z=use_z, use_noise=noise, use_reverb=reverb,
        n_harmonics=4, z_dim=4, noise_bands=bands, reverb_len=257,
        gru_hidden=16, mlp_hidden=16, mlp_layers=1,
        pitch_source="cached_basic_pitch",
    ).eval()


def _controls(model, frames=37):
    pitch = torch.full((2, model.n_voices, frames), 233.08)
    pitch[:, :, frames // 3:2 * frames // 3] = 277.18
    if model.n_voices > 1:
        pitch[:, 1, :frames // 2] = 0
    velocity = (pitch > 0).float() * 0.7
    loudness = torch.linspace(0.3, 0.8, frames).expand(2, frames)
    z = torch.randn(2, frames, 4) if model.use_z else None
    return pitch, velocity, loudness, z


def _stream(synth, controls, sizes, excitation=None):
    pitch, velocity, loudness, z = controls
    state = None
    outputs = []
    consumed = 0
    for size in cycle(sizes):
        if consumed == pitch.shape[-1]:
            break
        end = min(consumed + size, pitch.shape[-1])
        block, state = synth.process(
            pitch[..., consumed:end], velocity[..., consumed:end],
            loudness[..., consumed:end], None if z is None else z[:, consumed:end], state,
            noise=None if excitation is None else excitation[..., consumed * 64:end * 64],
        )
        outputs.append(block)
        assert state.pitch.shape[-1] <= 2
        assert state.harmonic_queue.shape[-1] <= max(
            64, synth.noise_synth.latency_samples if synth.noise_synth is not None else 0,
        )
        assert state.noise_queue.shape[-1] <= 64
        consumed = end
    return torch.cat([*outputs, synth.flush(state)], dim=-1), outputs, state


@pytest.mark.parametrize("sizes", [(1,), (7, 0, 3), (99,)])
@pytest.mark.parametrize("voices,use_z,noise,reverb,bands", [
    (1, False, False, False, 9),
    (2, True, False, False, 9),
    (2, True, True, False, 65),
    (2, False, False, True, 9),
    (2, True, True, True, 65),
    (2, True, True, True, 257),  # noise compensation exceeds a control hop
])
def test_streaming_core_matches_offline_controls(voices, use_z, noise, reverb, bands, sizes):
    model = _model(voices, use_z, noise, reverb, bands)
    controls = _controls(model)
    excitation = torch.rand(2, voices, 37 * 64) * 2 - 1 if noise else None
    model.inference_z_scale = 0.5
    model.inference_noise_scale = 0.7
    synth = StreamingSynthesizer.from_model(model).eval()
    with torch.no_grad():
        expected, _ = model.synthesize(*controls, noise=excitation)
    actual, parts, state = _stream(synth, controls, sizes, excitation)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    if reverb:
        expected_dry, _ = model._synthesise(*controls, 0.5, 0.7, excitation)
        expected_full = model.reverb(F.pad(expected_dry, (0, model.reverb.reverb_len - 1)))
        full = torch.cat([*parts, synth.flush(state, include_tail=True)], dim=-1)
        torch.testing.assert_close(full, expected_full, rtol=1e-4, atol=1e-5)


def test_core_is_independent_of_analysis_and_trainable_path_has_gradients(monkeypatch):
    model = _model()
    def unexpected(*args, **kwargs):
        raise AssertionError("synthesis must not call any audio/pitch encoder")
    for component in (model.pitch_encoder, model.loudness, model.z_encoder):
        monkeypatch.setattr(component, "forward", unexpected)
    controls = _controls(model, frames=11)
    audio, _ = model.synthesize(*controls)
    audio.square().mean().backward()
    assert model.decoder.gru.weight_hh_l0.grad.abs().sum() > 0
    assert model.reverb.ir.grad.abs().sum() > 0
    synth = StreamingSynthesizer.from_model(model, noise_seed=67).eval()
    assert synth.decoder is model.decoder
    assert not hasattr(synth, "pitch_encoder")
    assert not hasattr(synth, "z_encoder")
    one, _, _ = _stream(synth, controls, (99,))
    many, _, _ = _stream(synth, controls, (1, 0, 3))
    torch.testing.assert_close(one, many, rtol=1e-4, atol=1e-5)


def test_one_frame_empty_calls_and_reset():
    model = _model()
    synth = StreamingSynthesizer.from_model(model, noise_seed=71)
    controls = _controls(model, frames=1)
    pitch, velocity, loudness, z = controls
    empty, empty_state = synth.process(pitch[..., :0], velocity[..., :0], loudness[..., :0], z[:, :0])
    assert empty.shape == (2, 0)
    assert synth.flush(empty_state).shape == (2, 0)
    first, state = synth.process(*controls, empty_state)
    assert first.shape == (2, 0)  # interpolation waits for one neighboring frame
    no_output, same = synth.process(pitch[..., :0], velocity[..., :0], loudness[..., :0], z[:, :0], state)
    assert no_output.shape == (2, 0)
    assert same is state
    tail = synth.flush(state)
    assert tail.shape == (2, 64)
    _, reset = synth.process(*controls, None)
    torch.testing.assert_close(synth.flush(reset), tail, rtol=0, atol=0)


def test_control_state_is_small_and_independent_between_streams():
    model = _model()
    synth = StreamingSynthesizer.from_model(model, noise_seed=73)
    controls = _controls(model, frames=100)
    _, state = synth.process(*controls)
    snapshots = [t.clone() for t in (state.hidden, state.phase, state.pitch, state.harmonics, state.amplitudes)]
    for tensor in (state.hidden, state.phase, state.pitch, state.harmonics, state.amplitudes,
                   state.harmonic_queue, state.noise_queue):
        assert not tensor.requires_grad
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    other = _controls(model, frames=5)
    synth.process(*other, None)
    synth.process(*other, state)
    for actual, saved in zip((state.hidden, state.phase, state.pitch, state.harmonics, state.amplitudes), snapshots):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)


def test_stream_rejects_missing_conditioning_or_changed_voice_slots():
    model = _model()
    synth = StreamingSynthesizer.from_model(model)
    pitch, velocity, loudness, z = _controls(model, frames=3)
    with pytest.raises(ValueError, match="require z"):
        synth.process(pitch, velocity, loudness)
    _, state = synth.process(pitch, velocity, loudness, z)
    with pytest.raises(ValueError, match="voice slots"):
        synth.process(pitch[:, :1], velocity[:, :1], loudness, z, state)


def test_checkpoint_benchmark_reports_the_control_block_budget():
    from polyddsp.benchmark_stream import benchmark

    model = _model()
    synth = StreamingSynthesizer.from_model(model)
    result = benchmark(synth, _controls(model, frames=4), iterations=3, warmup=1)
    assert result["block_ms"] == 16.0
    assert result["voices"] == 2
    assert 0 <= result["deadline_misses"] <= 3
    assert 0 < result["median_ms"] <= result["max_ms"]
