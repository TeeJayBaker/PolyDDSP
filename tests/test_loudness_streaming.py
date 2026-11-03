"""Trailing loudness: timing, chunk parity, and alignment with timbre frames."""
from __future__ import annotations

from itertools import cycle

import pytest
import torch

from polyddsp.model.loudness import LoudnessExtractor
from polyddsp.model.z import ZEncoder


@pytest.mark.parametrize("normalise", [False, True])
@pytest.mark.parametrize("n_fft", [512, 64])
@pytest.mark.parametrize("chunk_sizes", [(64,), (197,), (1, 63, 0, 257, 127)])
def test_loudness_stream_matches_forward(
    normalise: bool, n_fft: int, chunk_sizes: tuple[int, ...],
) -> None:
    torch.manual_seed(29)
    extractor = LoudnessExtractor(n_fft=n_fft)
    audio = torch.randn(2, 4_133) * 0.1
    expected = extractor(audio, normalise=normalise)
    state = None
    outputs = []
    consumed = 0
    for size in cycle(chunk_sizes):
        if consumed == audio.shape[-1]:
            break
        chunk = audio[:, consumed:consumed + size]
        loudness, state = extractor.stream(chunk, state, normalise=normalise)
        outputs.append(loudness)
        consumed += chunk.shape[-1]
        assert state.audio_buffer.shape == (
            2, n_fft - extractor.frame_hop + consumed % extractor.frame_hop,
        )
    torch.testing.assert_close(torch.cat(outputs, dim=-1), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_samples", [0, 1, 63, 64, 65, 511, 512, 513])
def test_loudness_startup_and_partial_hops(n_samples: int) -> None:
    extractor = LoudnessExtractor()
    audio = torch.zeros(2, n_samples)
    expected = extractor(audio)
    loudness, _ = extractor.stream(audio)
    assert loudness.shape == (2, n_samples // extractor.frame_hop)
    assert torch.isfinite(loudness).all()
    torch.testing.assert_close(loudness, expected)


def test_loudness_trailing_hann_impulse_timing() -> None:
    extractor = LoudnessExtractor()
    audio = torch.zeros(1, 2_048)
    audio[:, 512] = 1.0  # impulse at 32 ms
    loudness = extractor(audio, normalise=False)
    # First eight blocks precede the impulse and must remain silent.
    torch.testing.assert_close(loudness[:, :8], torch.full((1, 8), -80.0))
    # It reaches the Hann peak in the window ending at sample 767 (frame 11).
    assert loudness.argmax().item() == 11


def test_loudness_prefix_and_gradient_do_not_depend_on_future_audio() -> None:
    extractor = LoudnessExtractor()
    audio = (torch.randn(1, 4_133) * 0.1).requires_grad_()
    full = extractor(audio)
    prefix = extractor(audio[:, :2_081])
    torch.testing.assert_close(prefix, full[:, :32], rtol=0, atol=0)
    grad, = torch.autograd.grad(full[:, 31].sum(), audio)
    assert torch.count_nonzero(grad[:, :2_048]) > 0
    assert torch.count_nonzero(grad[:, 2_048:]) == 0


def test_loudness_and_z_emit_the_same_control_blocks() -> None:
    extractor = LoudnessExtractor()
    encoder = ZEncoder(gru_hidden=16).eval()
    audio = torch.randn(1, 4_133) * 0.1
    loudness_state = z_state = None
    loudness_parts, z_parts = [], []
    consumed = frames = 0
    for size in cycle([13, 51, 0, 1, 127, 389]):
        if consumed == audio.shape[-1]:
            break
        chunk = audio[:, consumed:consumed + size]
        loudness, loudness_state = extractor.stream(chunk, loudness_state)
        z, z_state = encoder.stream(chunk, z_state)
        assert loudness.shape[-1] == z.shape[1]
        frames += loudness.shape[-1]
        consumed += chunk.shape[-1]
        assert frames == consumed // 64
        loudness_parts.append(loudness)
        z_parts.append(z)
    with torch.no_grad():
        expected_z = encoder(audio)
    torch.testing.assert_close(torch.cat(loudness_parts, dim=-1), extractor(audio))
    torch.testing.assert_close(torch.cat(z_parts, dim=1), expected_z, rtol=1e-5, atol=1e-6)


def test_loudness_state_is_small_independent_and_resettable() -> None:
    extractor = LoudnessExtractor()
    a = torch.randn(1, 10_013, requires_grad=True)
    loudness, a_state = extractor.stream(a)
    assert not loudness.requires_grad
    assert a_state.audio_buffer.shape[-1] < extractor.n_fft
    buffer = a_state.audio_buffer
    assert not buffer.requires_grad
    assert buffer.untyped_storage().nbytes() == buffer.numel() * buffer.element_size()
    saved = buffer.clone()
    b = torch.randn(1, 1_024)
    b_first, b_state = extractor.stream(b[:, :133])
    extractor.stream(torch.randn(1, 64), a_state)
    b_rest, _ = extractor.stream(b[:, 133:], b_state)
    torch.testing.assert_close(a_state.audio_buffer, saved, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([b_first, b_rest], dim=-1), extractor(b))
    reset_first, _ = extractor.stream(b[:, :133], None)
    torch.testing.assert_close(reset_first, b_first, rtol=0, atol=0)


def test_loudness_stream_rejects_batch_or_dtype_changes() -> None:
    extractor = LoudnessExtractor()
    _, state = extractor.stream(torch.randn(1, 64))
    with pytest.raises(ValueError, match="batch"):
        extractor.stream(torch.randn(2, 64), state)
    with pytest.raises(ValueError, match="dtype"):
        extractor.stream(torch.randn(1, 64, dtype=torch.float64), state)
