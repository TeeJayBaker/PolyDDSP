"""Smoke + idempotence tests for the Basic Pitch precompute cache."""
from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf
import torch
from symusic import ControlChange, Note, PitchBend, Score, Tempo, Track

from polyddsp.preprocess import (
    MIDI_VELOCITY_64,
    bp_cache_suffix,
    cache_path_for,
    ground_truth_cache_suffix,
    precompute_one_ground_truth,
    neutone_cache_suffix,
    precompute_one_bp,
    precompute_one_neutone_amt,
)
from polyddsp.preprocess.ground_truth import midi_path_for


def _make_chirp_wav(path: Path, duration_s: float = 6.0, sr: int = 16000) -> None:
    t = torch.linspace(0, duration_s, int(duration_s * sr))
    # Slow chirp 220→440 Hz; should give some BP onsets.
    audio = torch.sin(2 * 3.14159 * (220 + (440 - 220) * (t / duration_s)) * t).numpy()
    sf.write(str(path), audio, sr)


def test_precompute_one_bp_writes_cache_with_correct_shape(tmp_path: Path) -> None:
    audio = tmp_path / "chirp.wav"
    _make_chirp_wav(audio, duration_s=6.0, sr=16000)
    cache, recomputed = precompute_one_bp(
        audio, sample_rate=16000, target_hop=64, n_voices=4, device="cpu",
    )
    assert recomputed
    assert cache.exists()
    blob = torch.load(cache, weights_only=True)
    assert set(blob) == {"pitch", "velocity", "pitch_cache_source"}
    assert blob["pitch_cache_source"] == "basic-pitch"
    expected_target_frames = (16000 * 6) // 64
    assert blob["pitch"].shape == (4, expected_target_frames)
    assert blob["velocity"].shape == (4, expected_target_frames)


def test_precompute_one_bp_is_idempotent(tmp_path: Path) -> None:
    audio = tmp_path / "chirp.wav"
    _make_chirp_wav(audio, duration_s=4.0, sr=16000)
    _, first = precompute_one_bp(audio, 16000, 64, n_voices=4, device="cpu")
    _, second = precompute_one_bp(audio, 16000, 64, n_voices=4, device="cpu")
    assert first is True
    assert second is False


def test_bp_cache_suffix_includes_voices_sr_hop() -> None:
    assert bp_cache_suffix(n_voices=6, sample_rate=16000, target_hop=64) == "bp_v6_sr16000_hop64"


def test_neutone_cache_suffix_includes_voices_sr_hop() -> None:
    assert (
        neutone_cache_suffix("neutone-amt-streaming", 6, 16000, 64)
        == "neutone_streaming_v6_sr16000_hop64"
    )
    assert (
        neutone_cache_suffix("neutone-amt-nonstreaming", 6, 16000, 64)
        == "neutone_nonstreaming_v6_sr16000_hop64"
    )


def test_ground_truth_cache_suffix_includes_voices_sr_hop() -> None:
    assert (
        ground_truth_cache_suffix(10, 16000, 64)
        == "ground_truth_v10_sr16000_hop64"
    )


def test_ground_truth_glob_pairs_separate_directories_by_stem(tmp_path: Path) -> None:
    audio = tmp_path / "audio" / "003_HSswc.wav"
    midi = tmp_path / "midi" / "003_HSswc.mid"
    audio.parent.mkdir()
    midi.parent.mkdir()
    audio.touch()
    midi.touch()

    assert midi_path_for(
        audio,
        dataset_root=tmp_path,
        midi_glob="midi/**/*wc.mid",
    ) == midi


def test_ground_truth_finds_gaps_fine_aligned_file_excluded_by_glob(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio" / "Bc1wc.wav"
    midi = tmp_path / "midi" / "Bc1wc-fine-aligned.mid"
    audio.parent.mkdir()
    midi.parent.mkdir()
    audio.touch()
    midi.touch()

    # This is the GAPS glob, which does not itself match `-fine-aligned.mid`.
    assert not list(tmp_path.glob("**/*wc.mid"))
    assert midi_path_for(
        audio,
        dataset_root=tmp_path,
        midi_glob="**/*wc.mid",
    ) == midi


@pytest.mark.parametrize("audio_suffix", ["mix", "mic", "hex", "hex_cln"])
def test_ground_truth_glob_ignores_guitarset_audio_suffix(
    tmp_path: Path, audio_suffix: str,
) -> None:
    audio = tmp_path / "audio" / f"04_BN3-154-E_solo_{audio_suffix}.wav"
    midi = tmp_path / "midi" / "04_BN3-154-E_solo.mid"
    audio.parent.mkdir()
    midi.parent.mkdir()
    audio.touch()
    midi.touch()

    assert midi_path_for(audio, dataset_root=tmp_path, midi_glob="**/*.mid") == midi


def test_ground_truth_preserves_overlap_sustain_velocity_and_bends(tmp_path: Path) -> None:
    audio = tmp_path / "piece.wav"
    sf.write(audio, torch.zeros(16_000).numpy(), 16_000)

    score = Score(480)
    score.tempos.append(Tempo(0, 120))
    track = Track()
    track.notes.extend([
        Note(0, 480, 60, 127),      # C4: 0.0–0.5 s
        Note(240, 240, 64, 64),     # E4: 0.25–0.5 s, overlapping C4
    ])
    track.controls.extend([
        ControlChange(240, 64, 127),  # sustain down at 0.25 s
        ControlChange(720, 64, 0),    # sustain up at 0.75 s
    ])
    track.pitch_bends.extend([
        PitchBend(240, 4096),       # +1 semitone with the default ±2 range
        PitchBend(480, 0),
    ])
    score.tracks.append(track)
    score.dump_midi(tmp_path / "piece.midi")

    cache, recomputed = precompute_one_ground_truth(
        audio, 16_000, 64, 4, dataset_root=tmp_path,
    )
    assert recomputed
    blob = torch.load(cache, weights_only=True)
    assert blob["pitch_cache_source"] == "ground-truth"
    assert blob["pitch"].shape == (4, 250)

    # Both notes occupy distinct voices during their overlap; no contour is dropped.
    assert blob["pitch"][0, 63] == pytest.approx(277.1826, rel=1e-4)  # C#4 after bend
    assert blob["pitch"][1, 63] == pytest.approx(349.2282, rel=1e-4)  # F4 after bend
    assert blob["velocity"][0, 63] == pytest.approx(1.0)
    assert blob["velocity"][1, 63] == pytest.approx(64 / 127)

    # Sustain extends both through 0.75 s, while pitch bend resets at 0.5 s.
    assert blob["pitch"][0, 150] == pytest.approx(261.6256, rel=1e-4)
    assert blob["pitch"][1, 150] == pytest.approx(329.6276, rel=1e-4)
    assert torch.all(blob["pitch"][:, 188:] == 0)
    assert precompute_one_ground_truth(
        audio, 16_000, 64, 4, dataset_root=tmp_path,
    )[1] is False


def test_cache_path_for_mirrors_nested_path_in_output_dir(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    audio = root / "player_00" / "take.wav"
    output_dir = tmp_path / "pitch-cache"

    cache = cache_path_for(
        audio,
        "bp_v6_sr16000_hop64",
        root=root,
        output_dir=output_dir,
    )

    assert cache == output_dir / "player_00" / "take.wav.bp_v6_sr16000_hop64.f0.pt"


def test_precompute_one_neutone_amt_uses_constant_midi_velocity(tmp_path: Path) -> None:
    class FakeSpec:
        hop_length = 512

    class FakeModel:
        spec = FakeSpec()
        delays: list[int] = []
        target_shift = 0

        def __call__(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
            shape = (1, 88, 90)
            onset = torch.full(shape, -20.0)
            frame = torch.full(shape, -20.0)
            offset = torch.full(shape, -20.0)
            pitch_idx = 60 - 21
            onset[0, pitch_idx, 10] = 20.0
            frame[0, pitch_idx, 10:30] = 20.0
            offset[0, pitch_idx, 30] = 20.0
            return {"onset": onset, "frame": frame, "offset": offset}

    audio = tmp_path / "note.wav"
    sf.write(str(audio), torch.zeros(44_100).numpy(), 44_100)
    cache, recomputed = precompute_one_neutone_amt(
        audio,
        sample_rate=16_000,
        target_hop=64,
        n_voices=4,
        device="cpu",
        model=FakeModel(),
    )

    assert recomputed
    blob = torch.load(cache, weights_only=True)
    assert blob["pitch_cache_source"] == "neutone-amt-nonstreaming"
    active = blob["pitch"] != 0
    assert active.any()
    assert torch.all(blob["velocity"][active] == MIDI_VELOCITY_64)
    assert torch.all(blob["velocity"][~active] == 0)
    assert blob["pitch"].shape == (4, 250)
