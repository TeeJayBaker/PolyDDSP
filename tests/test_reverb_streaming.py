"""Partitioned reverb: full-IR parity, tail retention, and bounded stream state."""
from __future__ import annotations

from itertools import cycle

import pytest
import torch
import torch.nn.functional as F

from polyddsp.model.noise import FilteredNoise
from polyddsp.model.reverb import Reverb


@pytest.mark.parametrize("block_size", [17, 64])
@pytest.mark.parametrize("ir_length", [1, 37, 257])
@pytest.mark.parametrize("chunks", [(64,), (197,), (0, 13, 5, 67)])
def test_reverb_stream_matches_whole_file_and_full_tail(block_size, ir_length, chunks):
    torch.manual_seed(47)
    rev = Reverb(reverb_len=ir_length, block_size=block_size)
    with torch.no_grad():
        rev.ir.normal_(std=0.1)
    audio = torch.randn(2, 1_037) * 0.2
    expected = rev(audio)
    expected_tail = rev(F.pad(audio, (0, ir_length - 1)))
    state = None
    outputs = []
    consumed = emitted = 0
    for size in cycle(chunks):
        if consumed == audio.shape[-1]:
            break
        chunk = audio[:, consumed:consumed + size]
        block, state = rev.stream(chunk, state)
        outputs.append(block)
        consumed += chunk.shape[-1]
        emitted += block.shape[-1]
        assert emitted == consumed // block_size * block_size
        assert state.pending.shape == (2, consumed % block_size)
    actual = torch.cat([*outputs, rev.flush(state)], dim=-1)
    full_tail = torch.cat([*outputs, rev.flush(state, include_tail=True)], dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(full_tail, expected_tail, rtol=1e-5, atol=2e-6)


def test_default_length_reverb_keeps_echoes_across_hundreds_of_partitions():
    rev = Reverb()
    with torch.no_grad():
        rev.ir.zero_()
        rev.ir[0] = 9.0  # must still be masked
        rev.ir[1] = 0.25
        rev.ir[129] = -0.5
        rev.ir[-1] = 0.125
    audio = torch.zeros(1, 64)
    audio[:, -1] = 1
    first, state = rev.stream(audio)
    assert first.shape == audio.shape
    torch.testing.assert_close(first, audio, rtol=0, atol=1e-6)
    output = torch.cat([first, rev.flush(state, include_tail=True)], dim=-1)
    expected = torch.zeros(1, 64 + rev.reverb_len - 1)
    expected[:, 63] = 1
    expected[:, 64] = 0.25
    expected[:, 192] = -0.5
    expected[:, -1] = 0.125
    torch.testing.assert_close(output, expected, rtol=0, atol=1e-6)


def test_reverb_empty_calls_reset_and_interleaving():
    rev = Reverb(reverb_len=133)
    with torch.no_grad():
        rev.ir.normal_(std=0.1)
    a, b = torch.randn(1, 257), torch.randn(1, 513)
    a_first, a_state = rev.stream(a[:, :97])
    saved_history = a_state.history.clone()
    b_first, b_state = rev.stream(b[:, :133])
    empty, same = rev.stream(a[:, :0], a_state)
    assert empty.shape == (1, 0)
    assert same.position == a_state.position
    torch.testing.assert_close(same.history, saved_history, rtol=0, atol=0)
    a_rest, a_final = rev.stream(a[:, 97:], same)
    b_rest, b_final = rev.stream(b[:, 133:], b_state)
    torch.testing.assert_close(
        torch.cat([a_first, a_rest, rev.flush(a_final)], dim=-1), rev(a),
        rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        torch.cat([b_first, b_rest, rev.flush(b_final)], dim=-1), rev(b),
        rtol=1e-5, atol=1e-6,
    )
    reset, _ = rev.stream(a[:, :97], None)
    torch.testing.assert_close(reset, a_first, rtol=0, atol=0)
    torch.testing.assert_close(a_state.history, saved_history, rtol=0, atol=0)
    _, empty_state = rev.stream(a[:, :0])
    assert rev.flush(empty_state, include_tail=True).shape == (1, 0)


def test_reverb_retains_only_fixed_state_and_forward_gradients():
    rev = Reverb(reverb_len=257)
    audio = torch.randn(2, 10_013, requires_grad=True)
    rev(audio).square().mean().backward()
    assert audio.grad.abs().sum() > 0
    assert rev.ir.grad.abs().sum() > 0
    assert rev.ir.grad[0].item() == 0
    output, state = rev.stream(audio)
    assert not output.requires_grad
    for tensor in (state.ir_fft, state.history, state.overlap, state.pending):
        assert not tensor.requires_grad
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    assert state.pending.shape[-1] < rev.block_size


def test_reverb_stream_rejects_batch_and_dtype_changes():
    rev = Reverb(reverb_len=257)
    _, state = rev.stream(torch.zeros(1, 64))
    with pytest.raises(ValueError, match="batch"):
        rev.stream(torch.zeros(2, 64), state)
    with pytest.raises(ValueError, match="dtype"):
        rev.stream(torch.zeros(1, 64, dtype=torch.float64), state)


def test_noise_then_reverb_matches_offline_with_tail():
    """Noise's shortened first chunk must not lose samples at the reverb boundary."""
    torch.manual_seed(53)
    synth = FilteredNoise()
    rev = Reverb(reverb_len=257)
    with torch.no_grad():
        rev.ir.normal_(std=0.1)
    mags = torch.randn(2, 19, 65) + 2
    excitation = torch.rand(2, 19 * 64) * 2 - 1
    expected_dry = synth(mags, noise=excitation)
    expected = rev(F.pad(expected_dry, (0, rev.reverb_len - 1)))
    noise_state = reverb_state = None
    parts = []
    consumed = 0
    for size in cycle([1, 3, 0, 2]):
        if consumed == mags.shape[1]:
            break
        end = min(consumed + size, mags.shape[1])
        dry, noise_state = synth.stream(
            mags[:, consumed:end], noise_state,
            noise=excitation[:, consumed * 64:end * 64],
        )
        wet, reverb_state = rev.stream(dry, reverb_state)
        parts.append(wet)
        consumed = end
    wet, reverb_state = rev.stream(synth.flush(noise_state), reverb_state)
    parts.extend([wet, rev.flush(reverb_state, include_tail=True)])
    torch.testing.assert_close(torch.cat(parts, dim=-1), expected, rtol=1e-5, atol=2e-6)
