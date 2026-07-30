"""Per-frame timing properties of `BasicPitchModel.forward_windowed`.

Both tests here are **expected failures** (`xfail(strict=True)`). They assert a
stricter frame-timing guarantee than upstream Spotify basic-pitch provides, and
`forward_windowed` deliberately reproduces upstream's frame contract rather than
diverging from it — the precomputed pitch caches must line up with what
`basic_pitch.predict()` produces.

The drift arithmetic
--------------------
- The BP CNN emits `AUDIO_N_SAMPLES // BP_FFT_HOP + 1 = 172` frames per
  43844-sample window (CQT2010v2 left-pads, hence `n_samples // hop + 1`).
- `BP_OVERLAP_FRAMES // 2 = 15` frames are dropped from each side, so every
  window contributes `172 - 30 = 142` "inner" frames to the concatenation —
  `142 * 256 = 36352` samples of advertised time.
- The window itself advances only
  `AUDIO_N_SAMPLES - BP_OVERLAP_FRAMES * BP_FFT_HOP = 36164` samples.
- So each window over-advertises time by `36352 - 36164 = 188` samples
  (0.7344 BP frames). Over a 60-s clip (37 windows) that accumulates to roughly
  27 BP frames. The trailing `[:target_frames]` truncation hides it at the clip
  end, but interior frame timestamps stay skewed by a length-dependent amount.

This is an accepted, documented limitation inherited from upstream's windowing,
not an outstanding bug. The tests are kept rather than deleted so the arithmetic
stays executable: `strict=True` turns an unexpected pass into an error, which is
how we would learn that the upstream frame contract had changed.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torchaudio.functional as AF

from polyddsp.model.pitch import (
    AUDIO_N_SAMPLES,
    BP_FFT_HOP,
    BP_NATIVE_SR,
    BP_OVERLAP_FRAMES,
    load_basic_pitch,
)


def _sine_22k(frequency_hz: float, duration_s: float) -> torch.Tensor:
    """Pure sine generated at 16 kHz then resampled to BP_NATIVE_SR (matches data path)."""
    sr = 16_000
    t = np.arange(int(sr * duration_s)) / sr
    sig = (0.5 * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)
    audio16 = torch.from_numpy(sig)
    return AF.resample(audio16, orig_freq=sr, new_freq=BP_NATIVE_SR)


@pytest.mark.xfail(
    strict=True,
    reason="inherited upstream basic-pitch windowing drift: each 2-s window "
    "advertises 142 inner frames (36352 samples) but advances only 36164 samples, "
    "so a 60-s clip accumulates ~27 BP frames of frame-axis skew",
)
def test_forward_windowed_no_timing_drift_on_60s_sine() -> None:
    """A 60-s constant 440 Hz sine peaks at one stable contour bin in EVERY frame.

    The property asserted: the output frame count matches the clip's true frame
    count (`audio_22k_samples // BP_FFT_HOP`) and the peak bin is locked from the
    first frame to the last. The cross-window seam skew of 0.73 frames per window
    breaks this on clips long enough for the drift to accumulate.
    """
    audio22 = _sine_22k(440.0, 60.0)
    target_bp_frames = audio22.shape[0] // BP_FFT_HOP  # 1323000 // 256 = 5167
    expected_secs = 60.0
    expected_frames_true = expected_secs * BP_NATIVE_SR / BP_FFT_HOP  # 5167.96875

    model = load_basic_pitch()
    post = model.forward_windowed(audio22)
    contour = post["contour"].cpu().numpy()  # (264, T_bp)

    # First, the output must cover ~the entire clip (no missing frames, no drift-truncation).
    # Allow the trailing fractional frame (at most 1) to be missing.
    n_frames = contour.shape[1]
    assert abs(n_frames - expected_frames_true) <= 1, (
        f"expected ~{expected_frames_true:.2f} frames; got {n_frames}"
    )

    # The peak bin must be the same on every frame (constant 440 Hz throughout).
    # 440 Hz: midi 69, contour bin = (69-21)*3 = 144 (145 is equally acceptable — the
    # half-bin offset is resolved downstream by the pitch bends; here we only want stability).
    peaks = contour.argmax(axis=0)
    unique, counts = np.unique(peaks, return_counts=True)
    mode_bin = int(unique[np.argmax(counts)])
    locked = peaks == mode_bin

    # The trajectory must lock onto the mode bin from frame 0 to frame n_frames-1 (no
    # internal seam drift). Allow a 1-frame onset/offset tolerance for BP CNN edge effects.
    first_locked = int(np.flatnonzero(locked)[0])
    last_locked = int(np.flatnonzero(locked)[-1])
    assert first_locked <= 1, f"trajectory starts at frame {first_locked}; should be 0 or 1"
    assert last_locked >= n_frames - 2, (
        f"trajectory ends at frame {last_locked}; should be >= {n_frames - 2}"
    )

    # The locked count must equal n_frames (minus at most 2 edge frames).
    assert locked.sum() >= n_frames - 2, (
        f"only {locked.sum()}/{n_frames} frames at peak bin {mode_bin}; "
        "interior drift is leaking into peak detection"
    )


@pytest.mark.xfail(
    strict=True,
    reason="inherited upstream basic-pitch windowing drift: advertised inner-frame "
    "advance (142 * 256 = 36352 samples) exceeds the window hop (36164 samples) by 188",
)
def test_forward_windowed_seam_sample_alignment() -> None:
    """The inner-frame axis is uniformly time-spaced (HOP=256 samples per frame).

    Pure arithmetic on the windowing constants, independent of the BP CNN: the
    samples advertised by one window's inner frames should equal the samples the
    window actually advances. Upstream's constants differ by 188 samples.
    """
    overlap_samples = BP_OVERLAP_FRAMES * BP_FFT_HOP  # 7680
    hop_samples = AUDIO_N_SAMPLES - overlap_samples   # 36164
    # 172 frames per window minus 30 overlap = 142 inner frames advertised per window,
    # i.e. 142 * 256 = 36352 samples, while the window steps only 36164 samples.
    advertised_inner_frames = (AUDIO_N_SAMPLES // BP_FFT_HOP + 1) - BP_OVERLAP_FRAMES
    advertised_advance = advertised_inner_frames * BP_FFT_HOP
    drift_per_window_samples = advertised_advance - hop_samples
    assert drift_per_window_samples == 0, (
        f"each window over-advertises by {drift_per_window_samples} samples "
        f"({drift_per_window_samples / BP_FFT_HOP:.4f} BP frames)"
    )
