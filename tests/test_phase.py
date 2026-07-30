"""Phase accumulator numerics.

`angular_cumsum` is the accumulator `AdditiveSynth` actually uses (with
`chunk_size=frame_hop`), not a side helper: its chunked mod-2π accumulation is
what keeps fp32 phase resolution constant over a long file *and* what makes the
phase independent of where the signal is cut into blocks. `PolyDDSP.render`
depends on the second property, so it is tested here directly.

Checks: (a) an inclusive cumsum of a constant angular frequency lands on the
analytical phase, (b) `angular_cumsum` stays on the same point of the unit circle
as a plain `torch.cumsum` over 100 k samples, (c) it is split-invariant when the
previous slice's `final_phase` is carried forward, (d) `AdditiveSynth` block
rendering reproduces a whole-file pass sample for sample, and (e) the
end-to-end synth renders a single 440 Hz harmonic as a clean spectral peak.
"""
from __future__ import annotations

import math

import torch

from polyddsp.model.additive import AdditiveSynth, angular_cumsum


SR = 16_000
T_SAMPLES = 64_000
TWO_PI = 2 * math.pi


def test_constant_f0_phase_matches_analytical() -> None:
    """Cumulative phase of constant 440 Hz over 4 s matches 2π·440·4 mod 2π
    (testing the standalone primitive — inclusive cumsum)."""
    f0 = 440.0
    omegas = torch.full((1, T_SAMPLES), 2 * math.pi * f0 / SR)
    phase = torch.cumsum(omegas, dim=1)
    expected = (2 * math.pi * f0 * (T_SAMPLES / SR)) % (2 * math.pi)
    actual = phase[0, -1].item() % (2 * math.pi)
    assert abs(actual - expected) < 1e-3


def test_angular_cumsum_matches_plain_cumsum_mod_2pi() -> None:
    rng = torch.Generator().manual_seed(0)
    omegas = torch.rand(2, 100_000, 1, generator=rng) * 0.1
    plain = torch.cumsum(omegas, dim=1)
    chunked = angular_cumsum(omegas, chunk_size=1000)
    # Compare on the unit circle: equivalent phases give cos(diff) ≈ 1.
    # Linear subtraction of `% 2π` values misreports a tiny circular drift
    # as a full ~2π gap whenever the two paths straddle the 2π boundary.
    cos_diff = torch.cos(plain - chunked)
    err = (1.0 - cos_diff).abs().max().item()
    assert err < 1e-3, f"angular_cumsum diverges from plain cumsum (1-cos err {err:.6f})"


def test_angular_cumsum_is_split_invariant() -> None:
    """The property `PolyDDSP.render` is built on: cutting the signal on chunk
    boundaries and carrying `final_phase` reproduces the whole-signal phase."""
    rng = torch.Generator().manual_seed(0)
    n_time, chunk = 64_000, 64
    omegas = torch.rand(2, n_time, 3, generator=rng) * 3.0  # up to ~Nyquist per sample
    whole = angular_cumsum(omegas, chunk_size=chunk)

    parts: list[torch.Tensor] = []
    phase: torch.Tensor | None = None
    step = chunk * 237  # blocks are chunk-aligned but not equal to the whole
    for start in range(0, n_time, step):
        block, phase = angular_cumsum(
            omegas[:, start : start + step],
            chunk_size=chunk,
            initial_phase=phase,
            return_final=True,
        )
        parts.append(block)
        assert phase.shape == (2, 1, 3)
        assert (phase >= 0).all() and (phase < TWO_PI).all()
    joined = torch.cat(parts, dim=1)

    err = (joined - whole).abs().max().item()
    assert err < 1e-4, f"angular_cumsum is not split-invariant (max |Δphase| {err:.2e})"
    # Non-vacuous: without the carry the blocks each restart at 0.
    naive = torch.cat(
        [angular_cumsum(omegas[:, s : s + step], chunk_size=chunk) for s in range(0, n_time, step)],
        dim=1,
    )
    assert (naive - whole).abs().max().item() > 1.0


def test_additive_block_render_matches_whole_file() -> None:
    """`AdditiveSynth(initial_phase=, return_phase=, keep_frames=)` block protocol.

    Renders the same controls whole-file and in 137-frame blocks, each block
    handed one frame of look-behind / look-ahead. 233.08 Hz over 137 frames is
    127.6 cycles — deliberately not a whole number, so a synth that restarted its
    phase per block could not pass.
    """
    torch.manual_seed(0)
    hop, n_harmonics, n_voices, n_frames = 64, 5, 2, 600
    synth = AdditiveSynth(sr=SR, frame_hop=hop, n_harmonics=n_harmonics)

    pitch = torch.zeros(1, n_voices, n_frames)
    pitch[0, 0] = 233.08
    pitch[0, 0, 200:400] = 466.16
    pitch[0, 1, 100:] = 311.13
    harm_dist = torch.randn(1, n_voices, n_frames, n_harmonics)
    amp_v = torch.rand(1, n_voices, n_frames)

    whole = synth(pitch, harm_dist, amp_v)
    assert whole.abs().max().item() > 0.1

    blocks: list[torch.Tensor] = []
    phase: torch.Tensor | None = None
    chunk = 137
    for start in range(0, n_frames, chunk):
        end = min(start + chunk, n_frames)
        lo, hi = max(0, start - 1), min(end + 1, n_frames)  # look-behind / look-ahead
        block, phase = synth(
            pitch[:, :, lo:hi],
            harm_dist[:, :, lo:hi],
            amp_v[:, :, lo:hi],
            initial_phase=phase,
            return_phase=True,
            keep_frames=(start - lo, start - lo + end - start),
        )
        assert block.shape == (1, (end - start) * hop)
        assert phase.shape == (1, 1, n_voices * n_harmonics)
        blocks.append(block)

    joined = torch.cat(blocks, dim=-1)
    err = (joined - whole).abs().max().item()
    assert err < 1e-5, f"block render differs from whole-file by {err:.2e}"


def test_additive_final_phase_matches_analytical() -> None:
    """`return_phase` really is the phase at the last rendered sample.

    Constant 233.08 Hz for 100 frames (6400 samples = 0.4 s) is 93.232 cycles, so
    the expected phase is 0.232·2π — nowhere near 0, which is what makes the
    check meaningful.
    """
    hop, n_frames, f0 = 64, 100, 233.08
    synth = AdditiveSynth(sr=SR, frame_hop=hop, n_harmonics=2)
    pitch = torch.full((1, 1, n_frames), f0)
    _, phase = synth(
        pitch,
        torch.zeros(1, 1, n_frames, 2),
        torch.ones(1, 1, n_frames),
        return_phase=True,
    )
    cycles = f0 * n_frames * hop / SR  # 93.232
    for h, expected in enumerate([(cycles % 1.0) * TWO_PI, ((2 * cycles) % 1.0) * TWO_PI]):
        got = phase[0, 0, h].item()
        assert abs(got - expected) < 1e-3, f"harmonic {h + 1}: phase {got} != {expected}"


def test_synthesised_sine_pure_tone_spectrum() -> None:
    """End-to-end synth: V=1, H=1, F0=440 Hz → spectral magnitude matches analytical.

    DDSP-compatible synth uses inclusive cumsum + overlap-add window upsample,
    which introduces an arbitrary phase reference and a soft endpoint envelope.
    A pure-tone reconstruction is therefore checked spectrally (phase-invariant).
    """
    synth = AdditiveSynth(sr=SR, frame_hop=64, n_harmonics=1)
    B, V, T_frames = 1, 1, T_SAMPLES // 64
    pitch = torch.full((B, V, T_frames), 440.0)
    harm_dist = torch.ones(B, V, T_frames, 1)
    amp_v = torch.ones(B, V, T_frames)
    audio = synth(pitch=pitch, harm_dist=harm_dist, amp_v=amp_v)
    assert audio.shape == (B, T_SAMPLES)

    spec = torch.fft.rfft(audio[0]).abs()
    # 440 Hz at 64 k samples / 16 kHz = bin 1760.
    bin_440 = int(round(440.0 * T_SAMPLES / SR))
    peak = spec.argmax().item()
    assert abs(peak - bin_440) <= 1, f"peak bin {peak} expected near {bin_440}"
    energy_in_peak = spec[bin_440 - 2: bin_440 + 3].pow(2).sum()
    total_energy = spec.pow(2).sum()
    concentration = (energy_in_peak / total_energy).item()
    assert concentration > 0.95, f"only {concentration:.3f} energy in 440 Hz peak"
