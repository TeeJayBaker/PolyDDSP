"""Filtered noise across call boundaries, including delay and RNG state."""
from __future__ import annotations

from itertools import cycle

import pytest
import torch
import torch.nn.functional as F

from polyddsp.model.noise import FilteredNoise, _apply_hann_window_to_ir, exp_sigmoid


def _render_stream(synth, mags, chunks, *, noise=None, seed=123):
    state = None
    outputs = []
    consumed = emitted = 0
    for size in cycle(chunks):
        if consumed == mags.shape[1]:
            break
        end = min(consumed + size, mags.shape[1])
        excitation = None if noise is None else noise[:, consumed * synth.frame_hop:end * synth.frame_hop]
        block, state = synth.stream(
            mags[:, consumed:end], state, noise=excitation,
            seed=seed if state is None else None,
        )
        outputs.append(block)
        consumed = end
        emitted += block.shape[-1]
        assert emitted == max(0, consumed * synth.frame_hop - synth.latency_samples)
        assert state.overlap.shape[-1] < 2 * (synth.frame_hop + synth.ir_size)
    outputs.append(synth.flush(state))
    return torch.cat(outputs, dim=-1), state


@pytest.mark.parametrize("hop,bands", [(64, 65), (8, 65), (128, 9), (64, 2)])
@pytest.mark.parametrize("window", [0, 31])
@pytest.mark.parametrize("chunks", [(1,), (3, 0, 7, 2), (99,)])
def test_noise_stream_matches_forward_with_identical_excitation(hop, bands, window, chunks):
    torch.manual_seed(31)
    synth = FilteredNoise(frame_hop=hop, n_bands=bands, window_size=window)
    mags = torch.randn(2, 37, bands) + 2
    noise = torch.rand(2, 37 * hop) * 2 - 1
    expected = synth(mags, noise=noise)
    actual, _ = _render_stream(synth, mags, chunks, noise=noise)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_noise_flush_matches_direct_time_domain_convolution():
    """A clip shorter than compensation must still flush correctly, with its tail."""
    torch.manual_seed(37)
    synth = FilteredNoise(frame_hop=8, n_bands=65)
    mags = torch.randn(2, 4, 65) + 3
    noise = torch.rand(2, 32) * 2 - 1
    ir = torch.fft.irfft(exp_sigmoid(mags - 5), dim=-1)
    ir = _apply_hann_window_to_ir(ir, ir.shape[-1])
    raw = torch.zeros(2, noise.shape[-1] + synth.ir_size - 1)
    for batch in range(2):
        for frame in range(4):
            start = frame * synth.frame_hop
            excitation = noise[batch:batch + 1, start:start + synth.frame_hop].unsqueeze(1)
            filtered = F.conv1d(
                F.pad(excitation, (synth.ir_size - 1, synth.ir_size - 1)),
                ir[batch, frame].flip(0).reshape(1, 1, -1),
            )[0, 0]
            raw[batch, start:start + filtered.numel()] += filtered
    expected = raw[:, synth.latency_samples:]
    first, state = synth.stream(mags, noise=noise, seed=1)
    assert first.shape[-1] == 0
    complete = torch.cat([first, synth.flush(state, include_tail=True)], dim=-1)
    torch.testing.assert_close(complete, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(synth(mags, noise=noise), expected[:, :32], rtol=1e-5, atol=1e-6)


def test_noise_seeded_draws_do_not_depend_on_chunk_size():
    mags = torch.randn(3, 37, 65) + 2
    synth = FilteredNoise()
    one, _ = _render_stream(synth, mags, (99,), seed=41)
    many, _ = _render_stream(synth, mags, (1,), seed=41)
    uneven, _ = _render_stream(synth, mags, (3, 0, 7, 2), seed=41)
    other, _ = _render_stream(synth, mags, (1,), seed=42)
    torch.testing.assert_close(one, many, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(one, uneven, rtol=1e-5, atol=1e-6)
    assert (one - other).abs().max().item() > 1e-3


def test_noise_empty_calls_and_interleaved_streams_preserve_state():
    synth = FilteredNoise()
    mags = torch.randn(2, 12, 65) + 2
    first, state = synth.stream(mags[:, :3], seed=43)
    saved_overlap, saved_rng = state.overlap.clone(), state.rng_state.clone()
    empty, same = synth.stream(mags[:, :0], state)
    assert empty.shape == (2, 0)
    torch.testing.assert_close(same.overlap, saved_overlap, rtol=0, atol=0)
    torch.testing.assert_close(same.rng_state, saved_rng, rtol=0, atol=0)
    synth.stream(mags, seed=99)  # another stream must not change our generator
    rest, final = synth.stream(mags[:, 3:], same)
    actual = torch.cat([first, rest, synth.flush(final)], dim=-1)
    expected, _ = _render_stream(synth, mags, (12,), seed=43)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state.overlap, saved_overlap, rtol=0, atol=0)
    torch.testing.assert_close(state.rng_state, saved_rng, rtol=0, atol=0)
    _, empty_state = synth.stream(mags[:, :0], seed=43)
    assert synth.flush(empty_state, include_tail=True).shape == (2, 0)


def test_noise_state_is_bounded_and_forward_remains_differentiable():
    synth = FilteredNoise()
    mags = (torch.randn(2, 257, 65) + 2).requires_grad_()
    noise = (torch.rand(2, 257 * 64) * 2 - 1).requires_grad_()
    output = synth(mags, noise=noise)
    output.square().mean().backward()
    assert mags.grad.abs().sum() > 0
    assert noise.grad.abs().sum() > 0
    streamed, state = synth.stream(mags, noise=noise, seed=1)
    assert not streamed.requires_grad
    for tensor in (state.overlap, state.rng_state):
        assert not tensor.requires_grad
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
