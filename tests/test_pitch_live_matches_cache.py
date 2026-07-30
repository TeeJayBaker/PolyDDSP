"""Anti-divergence guard: the in-loop pitch encoder and the precompute cache
must run the *same* Basic Pitch pipeline.

This repo previously carried two transcription algorithms — a windowed-inference
+ upstream-note-pipeline version in `preprocess`, and a crude single-shot
peak-picker in `PitchEncoder`. They disagreed, and only the cached one produced
published results. Both now call `polyddsp.model.pitch.basic_pitch_to_voices`;
this test fails if anyone reintroduces a second code path.

The fixture wav is written at 16 kHz so both callers perform the identical
16 kHz -> 22.05 kHz resample. (`precompute_one_bp` resamples the *source file*
to 22.05 kHz, so a higher-rate fixture would legitimately give the cache more
high-frequency content than the 16 kHz model audio the encoder sees.)
"""
from __future__ import annotations

import math
from pathlib import Path

import soundfile as sf
import torch

from polyddsp.model.pitch import BP_NATIVE_SR, basic_pitch_to_voices
from polyddsp.preprocess import load_resample, precompute_one_bp

SR = 16_000
HOP = 64
N_VOICES = 4


# Discrete, well-separated notes on purpose. A continuous glissando makes BP's
# onset inference chaotically sensitive to sub-1e-4 input noise (it re-segments
# the glide), which would make this test flaky for reasons unrelated to code
# divergence. Steady notes with silent gaps give stable onsets.
_NOTES: tuple[tuple[float, float, tuple[float, ...]], ...] = (
    # (start_s, dur_s, freqs_hz)
    (0.3, 0.9, (220.0,)),                  # A3
    (1.4, 0.9, (293.66, 440.0)),           # D4 + A4 dyad
    (2.6, 0.9, (329.63,)),                 # E4
    (3.8, 1.4, (261.63, 329.63, 392.0)),   # C major triad
)


def _write_multitone_wav(path: Path, duration_s: float = 6.0, sr: int = SR) -> None:
    """A few discrete sustained notes/chords — enough onsets to populate >1 voice."""
    audio = torch.zeros(int(duration_s * sr), dtype=torch.float64)
    for start_s, dur_s, freqs in _NOTES:
        s = int(start_s * sr)
        n = min(int(dur_s * sr), audio.shape[0] - s)
        if n <= 0:
            continue
        t = torch.arange(n, dtype=torch.float64) / sr
        # 5 ms raised-cosine fades so each note has a clean attack and no click.
        env = torch.ones(n, dtype=torch.float64)
        ramp = max(1, int(0.005 * sr))
        env[:ramp] = torch.linspace(0.0, 1.0, ramp, dtype=torch.float64)
        env[-ramp:] = torch.linspace(1.0, 0.0, ramp, dtype=torch.float64)
        for f0 in freqs:
            audio[s : s + n] += 0.35 * env * torch.sin(2 * math.pi * f0 * t)
    sf.write(str(path), audio.to(torch.float32).numpy(), sr)


def _note_segments(
    pitch: torch.Tensor, velocity: torch.Tensor
) -> list[tuple[int, int, float, float]]:
    """Flatten (V, T) pitch/velocity into a voice-order-independent note list.

    Each entry is `(onset_frame, offset_frame, mean_midi, mean_velocity)`,
    ordered by `(rounded_midi, onset)`. Voice *identity* is deliberately
    discarded: which slot a chord tone lands in depends on the allocator's note
    sort key `(start, end, midi)`, so a one-frame difference in a note's offset
    can permute two slots without the transcription itself changing. Ordering by
    pitch-then-onset (rather than by the raw tuple) keeps the pairing stable
    under exactly that permutation.
    """
    segments: list[tuple[int, int, float, float]] = []
    for v in range(pitch.shape[0]):
        active = (pitch[v] > 0).tolist()
        start = None
        for t, on in enumerate(active + [False]):
            if on and start is None:
                start = t
            elif not on and start is not None:
                span = pitch[v, start:t]
                midi = float((69.0 + 12.0 * torch.log2(span / 440.0)).mean())
                segments.append(
                    (start, t, midi, float(velocity[v, start:t].mean()))
                )
                start = None
    return sorted(segments, key=lambda s: (round(s[2]), s[0]))


def test_live_basic_pitch_matches_precomputed_cache(tmp_path: Path) -> None:
    """The live encoder and the cache must transcribe the same notes.

    Compared at note level rather than tensor level: the two callers reach
    22.05 kHz by different routes (`torchaudio.functional.resample` in
    `load_resample` vs the `Resample` transform the encoder holds as a
    submodule), which differ by ~5e-5 per sample. That noise is inaudible but
    can move a note boundary by one BP frame, which in turn permutes voice
    slots. Note onsets, pitches and velocities are stable, so those are what we
    pin. A genuinely different algorithm — e.g. the integer-MIDI peak-picker
    this repo used to run in-loop — fails these assertions loudly.
    """
    from polyddsp.model.pitch import PitchEncoder

    wav = tmp_path / "multitone.wav"
    _write_multitone_wav(wav)

    cache_path, recomputed = precompute_one_bp(
        wav, sample_rate=SR, target_hop=HOP, n_voices=N_VOICES, device="cpu",
    )
    assert recomputed
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)

    enc = PitchEncoder(sr=SR, n_voices=N_VOICES, target_frame_hop=HOP, source="basic_pitch")
    out = enc(load_resample(wav, SR).unsqueeze(0))

    assert out["bp_post"] == {}
    assert out["pitch"].shape == (1, N_VOICES, cached["pitch"].shape[-1])
    assert out["velocity"].shape == out["pitch"].shape

    live_notes = _note_segments(out["pitch"][0], out["velocity"][0])
    cache_notes = _note_segments(cached["pitch"], cached["velocity"])

    # Sanity: the fixture really did transcribe polyphony, so the comparison
    # below is not vacuously matching two empty note lists.
    assert len(cache_notes) >= 4, cache_notes
    assert (cached["pitch"] > 0).sum(dim=-1).gt(0).sum() >= 2, "expected >1 live voice"

    assert len(live_notes) == len(cache_notes), (
        f"live produced {len(live_notes)} notes, cache {len(cache_notes)}; the two "
        f"Basic Pitch code paths have drifted apart.\nlive={live_notes}\ncache={cache_notes}"
    )
    # 1 BP frame (11.6 ms) is 2.9 target frames at sr=16000/hop=64.
    for (ls, le, lm, lv), (cs, ce, cm, cv) in zip(live_notes, cache_notes):
        assert abs(ls - cs) <= 1, f"onset {ls} vs {cs} (live vs cache)"
        assert abs(le - ce) <= 3, f"offset {le} vs {ce} (live vs cache)"
        assert abs(lm - cm) < 0.2, f"midi {lm:.3f} vs {cm:.3f} (live vs cache)"
        assert abs(lv - cv) < 0.05, f"velocity {lv:.3f} vs {cv:.3f} (live vs cache)"

    # Both paths must carry sub-semitone pitch bends. The superseded in-loop
    # allocator emitted exact integer MIDI, so this pins the shared pipeline.
    for name, p in (("live", out["pitch"][0]), ("cache", cached["pitch"])):
        midi = 69.0 + 12.0 * torch.log2(p[p > 0] / 440.0)
        frac = (midi - midi.round()).abs()
        assert frac.max() > 0.05, f"{name} pitch is integer-MIDI — pitch bends are missing"


def test_basic_pitch_to_voices_shape_dtype_device(tmp_path: Path) -> None:
    """Contract: mono 22.05 kHz in -> (V, target_frames) float32 CPU tensors out."""
    wav = tmp_path / "short.wav"
    _write_multitone_wav(wav, duration_s=3.0)
    audio_bp = load_resample(wav, BP_NATIVE_SR)
    assert audio_bp.ndim == 1

    target_frames = 700
    pitch, velocity = basic_pitch_to_voices(
        audio_bp, n_voices=3, target_frames=target_frames,
    )
    for name, x in (("pitch", pitch), ("velocity", velocity)):
        assert x.shape == (3, target_frames), name
        assert x.dtype == torch.float32, name
        assert x.device.type == "cpu", name
    # pitch is Hz with 0 == silence; velocity is a posteriorgram amplitude.
    assert (pitch >= 0).all()
    assert ((velocity >= 0) & (velocity <= 1)).all()


def test_basic_pitch_to_voices_rejects_batched_input(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="mono"):
        basic_pitch_to_voices(torch.zeros(2, 4096), n_voices=1, target_frames=10)
