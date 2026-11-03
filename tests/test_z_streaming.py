"""Incremental timbre encoding: parity, bounded state, and stream isolation."""
from __future__ import annotations

from itertools import cycle

import pytest
import torch

from polyddsp.model.z import ZEncoder


@pytest.mark.parametrize("chunk_sizes", [(64,), (197,), (1, 63, 0, 257, 127)])
@pytest.mark.parametrize("window", [(2048, 128, 30), (64, 8, 8)])
def test_stream_matches_forward_and_only_decodes_new_frames(
    chunk_sizes: tuple[int, ...], window: tuple[int, int, int],
) -> None:
    torch.manual_seed(17)
    n_fft, n_mels, n_mfcc = window
    enc = ZEncoder(n_fft=n_fft, n_mels=n_mels, n_mfcc=n_mfcc, gru_hidden=16).eval()
    audio = torch.randn(2, 4_133) * 0.1
    with torch.no_grad():
        expected = enc(audio)

    decoded_frames = []
    handle = enc.gru.register_forward_pre_hook(
        lambda module, args: decoded_frames.append(args[0].shape[1])
    )
    state = None
    outputs = []
    consumed = 0
    try:
        for size in cycle(chunk_sizes):
            if consumed == audio.shape[-1]:
                break
            chunk = audio[:, consumed:consumed + size]
            z, state = enc.stream(chunk, state)
            outputs.append(z)
            consumed += chunk.shape[-1]
            assert state.audio_buffer.shape == (
                2, n_fft - enc.frame_hop + consumed % enc.frame_hop,
            )
            if state.hidden is not None:
                assert state.hidden.shape == (1, 2, enc.gru.hidden_size)
    finally:
        handle.remove()

    actual = torch.cat(outputs, dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert sum(decoded_frames) == audio.shape[-1] // enc.frame_hop


def test_stream_retains_only_small_state_without_autograd_history() -> None:
    enc = ZEncoder(gru_hidden=16).eval()
    # A large chunk must not survive indirectly through a small tail view.
    audio = torch.randn(2, 100_013, requires_grad=True)
    z, state = enc.stream(audio)
    assert not z.requires_grad
    assert state.hidden is not None
    for tensor in (state.audio_buffer, state.hidden):
        assert tensor.grad_fn is None
        assert not tensor.requires_grad
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    assert state.audio_buffer.shape[-1] < enc.n_fft


def test_partial_and_empty_chunks_do_not_advance_gru() -> None:
    enc = ZEncoder(gru_hidden=16).eval()
    audio = torch.randn(1, 128)
    state = None
    for lo, hi in [(0, 0), (0, 17), (17, 63)]:
        z, state = enc.stream(audio[:, lo:hi], state)
        assert z.shape == (1, 0, enc.z_dim)
        assert state.hidden is None

    first_z, state = enc.stream(audio[:, 63:64], state)
    original_buffer = state.audio_buffer.clone()
    original_hidden = state.hidden.clone()
    empty_z, unchanged = enc.stream(audio[:, :0], state)
    assert empty_z.shape == (1, 0, enc.z_dim)
    torch.testing.assert_close(unchanged.audio_buffer, original_buffer, rtol=0, atol=0)
    torch.testing.assert_close(unchanged.hidden, original_hidden, rtol=0, atol=0)
    second_z, _ = enc.stream(audio[:, 64:], unchanged)

    with torch.no_grad():
        expected = enc(audio)
    torch.testing.assert_close(torch.cat([first_z, second_z], dim=1), expected)


def test_streams_are_independent_and_none_resets_state() -> None:
    torch.manual_seed(23)
    enc = ZEncoder(gru_hidden=16).eval()
    a = torch.randn(1, 1_024) * 0.02
    b = torch.randn(1, 1_024) * 0.5
    a_first, a_state = enc.stream(a[:, :133])
    saved_audio = a_state.audio_buffer.clone()
    saved_hidden = a_state.hidden.clone()
    b_first, b_state = enc.stream(b[:, :257])
    a_rest, _ = enc.stream(a[:, 133:], a_state)
    b_rest, _ = enc.stream(b[:, 257:], b_state)
    # Processing more audio must not mutate the caller's previous state.
    torch.testing.assert_close(a_state.audio_buffer, saved_audio, rtol=0, atol=0)
    torch.testing.assert_close(a_state.hidden, saved_hidden, rtol=0, atol=0)
    reset_first, _ = enc.stream(a[:, :133], None)

    with torch.no_grad():
        expected_a, expected_b = enc(a), enc(b)
    torch.testing.assert_close(torch.cat([a_first, a_rest], dim=1), expected_a)
    torch.testing.assert_close(torch.cat([b_first, b_rest], dim=1), expected_b)
    torch.testing.assert_close(reset_first, a_first, rtol=0, atol=0)


def test_stream_rejects_batch_or_dtype_changes() -> None:
    enc = ZEncoder(gru_hidden=16).eval()
    _, state = enc.stream(torch.randn(1, 64))
    with pytest.raises(ValueError, match="batch"):
        enc.stream(torch.randn(2, 64), state)
    with pytest.raises(ValueError, match="dtype"):
        enc.stream(torch.randn(1, 64, dtype=torch.float64), state)
