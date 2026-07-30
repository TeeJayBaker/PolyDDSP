"""Filtered-noise must compensate for the linear-phase Hann FIR group delay so it
stays aligned with the harmonic path.

Without compensation the overlap-add output is delayed by ~ir_size/2 samples
(~4 ms at ir_size=128). `ddsp.core.crop_and_compensate_delay(padding='same',
delay_compensation=-1)` shifts left by `(ir_size-1)//2 - 1` to undo it. This test
fires an impulse-like band gain at a chosen frame and checks that the resulting
noise energy peaks within that frame, not the next one.
"""
from __future__ import annotations

import torch

from polyddsp.model.noise import FilteredNoise


def test_noise_energy_peak_aligned_with_band_impulse() -> None:
    torch.manual_seed(0)
    sr = 16_000
    hop = 64
    n_bands = 65
    n_frames = 16
    impulse_frame = 8

    synth = FilteredNoise(frame_hop=hop, n_bands=n_bands, window_size=0)

    # Wide-band gain only at `impulse_frame`; silence elsewhere.
    noise_mags = torch.full((1, n_frames, n_bands), -10.0)  # exp_sigmoid → near zero
    noise_mags[0, impulse_frame, :] = 5.0  # exp_sigmoid → near max

    out = synth(noise_mags)  # (1, n_frames * hop)
    assert out.shape == (1, n_frames * hop)

    # Energy per frame: split T_samples into n_frames bins of `hop` samples and
    # sum power. The peak should land in the impulse frame, not the one after.
    per_frame_energy = (out[0].pow(2)).reshape(n_frames, hop).sum(dim=-1)
    peak_frame = int(per_frame_energy.argmax().item())
    assert abs(peak_frame - impulse_frame) <= 1, (
        f"peak energy at frame {peak_frame}, expected ~{impulse_frame}; "
        f"per-frame energy = {per_frame_energy.tolist()}"
    )
