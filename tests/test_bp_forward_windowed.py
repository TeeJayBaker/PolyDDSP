"""Verify BasicPitchModel.forward_windowed matches forward_raw on a single window
and produces correct frame counts on multi-window inputs.

Frame contract: matches upstream `basic_pitch.inference.unwrap_output` —
target_frames = int(audio_len / hop * n_frames_per_window) where
n_frames_per_window = (AUDIO_N_SAMPLES // BP_FFT_HOP + 1) - BP_OVERLAP_FRAMES = 142
and hop = AUDIO_N_SAMPLES - BP_OVERLAP_FRAMES * BP_FFT_HOP = 36164.
"""
from __future__ import annotations

import torch

from polyddsp.model.pitch import (
    AUDIO_N_SAMPLES,
    BP_FFT_HOP,
    BP_NATIVE_SR,
    BP_OVERLAP_FRAMES,
    BasicPitchModel,
    load_basic_pitch,
)


_HOP = AUDIO_N_SAMPLES - BP_OVERLAP_FRAMES * BP_FFT_HOP  # 36164
_N_FRAMES_PER_WINDOW = AUDIO_N_SAMPLES // BP_FFT_HOP + 1 - BP_OVERLAP_FRAMES  # 142


def _expected_frames(audio_len: int) -> int:
    return int(audio_len / _HOP * _N_FRAMES_PER_WINDOW)


def test_forward_windowed_single_window_produces_correct_frame_count() -> None:
    torch.manual_seed(0)
    audio = torch.randn(AUDIO_N_SAMPLES) * 0.05
    model = load_basic_pitch()
    full = model.forward_windowed(audio)
    raw = model.forward_raw(audio.unsqueeze(0))
    expected = _expected_frames(AUDIO_N_SAMPLES)  # 172
    assert full["onset"].shape[-1] == expected
    assert full["contour"].shape[-1] == expected
    # And the frequency dim is preserved.
    assert full["onset"].shape[-2] == raw["onset"].shape[-2]


def test_forward_windowed_handles_multi_window_audio() -> None:
    # 4 BP windows worth of audio.
    audio = torch.randn(4 * AUDIO_N_SAMPLES) * 0.05
    model = load_basic_pitch()
    out = model.forward_windowed(audio)
    expected_frames = _expected_frames(audio.shape[0])
    assert out["onset"].shape[-1] == expected_frames
    assert out["contour"].shape[-1] == expected_frames
    assert out["note"].shape[-1] == expected_frames


def test_forward_windowed_returns_2d_tensors_per_key() -> None:
    audio = torch.randn(AUDIO_N_SAMPLES) * 0.05
    out = load_basic_pitch().forward_windowed(audio)
    for k in ("onset", "contour", "note"):
        assert out[k].ndim == 2  # (F, T)


def test_forward_windowed_handles_mid_band_audio_lengths() -> None:
    """Regression: audio lengths in the mid-band (~hop after a window boundary)
    used to silently truncate output before the n_windows fix."""
    audio = torch.randn(40004) * 0.05
    out = load_basic_pitch().forward_windowed(audio)
    expected = _expected_frames(40004)
    assert out["onset"].shape[-1] == expected
    assert out["contour"].shape[-1] == expected
    assert out["note"].shape[-1] == expected


def test_forward_windowed_handles_audio_shorter_than_one_window() -> None:
    """Audio shorter than AUDIO_N_SAMPLES still produces correct frame count."""
    short_len = 10000  # ~0.45 s, less than one BP window
    audio = torch.randn(short_len) * 0.05
    out = load_basic_pitch().forward_windowed(audio)
    assert out["onset"].shape[-1] == _expected_frames(short_len)
