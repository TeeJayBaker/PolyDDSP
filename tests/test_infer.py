"""Chunked rendering (`PolyDDSP.render`) and the `polyddsp.infer` CLI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from polyddsp import preprocess
from polyddsp.infer import _normalise_argv, main, pitch_features
from polyddsp.model.additive import AdditiveSynth
from polyddsp.model.parity_ops import scale_f0_hz
from polyddsp.model.polyddsp import PolyDDSP

SR = 16_000
HOP = 64
V = 2


def _tiny_kwargs(use_noise: bool = False, use_reverb: bool = False) -> dict:
    """Deliberately tiny model: the point is the chunking logic, not fidelity."""
    return dict(
        sr=SR,
        frame_hop=HOP,
        n_harmonics=4,
        z_dim=16,
        noise_bands=65,
        noise_window=0,
        reverb_len=4_000,
        gru_hidden=16,
        mlp_hidden=16,
        mlp_layers=1,
        n_voices=V,
        use_z=False,
        use_reverb=use_reverb,
        use_noise=use_noise,
        pitch_source="cached_basic_pitch",
    )


def _tiny_model(**over) -> PolyDDSP:
    kw = _tiny_kwargs()
    kw.update(over)
    torch.manual_seed(0)
    return PolyDDSP(**kw).eval()


def _cond(
    n_samples: int, n_voices: int = V
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(audio, pitch, velocity) with a couple of pitch changes so blocks differ.

    The pitches are deliberately non-integer (Bb3 / Db4 / Eb4) and the block
    lengths used below are deliberately not a whole second: with e.g. f0=220 Hz
    and 250 frames (exactly 1 s) every block spans a whole number of cycles, so
    an oscillator that restarted its phase at each block would land back on the
    same point of the unit circle and every exactness test would pass vacuously.
    """
    n_frames = n_samples // HOP
    t = torch.arange(n_samples) / SR
    audio = (0.3 * torch.sin(2 * torch.pi * 233.08 * t)).unsqueeze(0)
    pitch = torch.zeros(1, n_voices, n_frames)
    pitch[0, 0] = 233.08
    pitch[0, 0, n_frames // 3 : 2 * n_frames // 3] = 277.18
    if n_voices > 1:
        pitch[0, 1, n_frames // 2 :] = 311.13
    velocity = (pitch > 0).float() * 0.7
    return audio, pitch, velocity


def test_normalise_argv_accepts_both_spellings() -> None:
    assert _normalise_argv(["ckpt=a/best.pt", "input=x.wav"]) == [
        "--ckpt", "a/best.pt", "--input", "x.wav",
    ]
    # Already-flag tokens pass through; only the first `=` splits.
    assert _normalise_argv(["--out", "y.wav", "device=cuda:0"]) == [
        "--out", "y.wav", "--device", "cuda:0",
    ]
    assert _normalise_argv(["out=a=b.wav"]) == ["--out", "a=b.wav"]


def test_main_passes_pitch_encoder_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_render_file(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)

    monkeypatch.setattr("polyddsp.infer.render_file", fake_render_file)
    main([
        "--ckpt", "run/best.pt", "--input", "in.wav", "--out", "out.wav",
        "--pitch-encoder", "neutone-amt",
        "--pitch-encoder-checkpoint", "amt.onnx",
    ])

    assert captured["pitch_encoder"] == "neutone-amt"
    assert captured["pitch_encoder_checkpoint"] == "amt.onnx"


def test_neutone_pitch_features_use_shared_preprocess_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    model_path = tmp_path / "amt.onnx"
    model_path.touch()
    cfg = OmegaConf.create({
        "model": {"sr": SR, "frame_hop": HOP},
        "experiment": {"model": {
            "n_voices": V,
        }},
    })

    class FakeAMT:
        spec = type("Spec", (), {"sample_rate": 44_100})()
        max_input_samples = None

    amt_model = FakeAMT()
    calls = {}

    def fake_load(path, device):  # type: ignore[no-untyped-def]
        calls["load"] = (path, device)
        return amt_model

    def fake_transcribe(audio_path, **kwargs):  # type: ignore[no-untyped-def]
        calls["transcribe"] = (audio_path, kwargs)
        shape = (kwargs["n_voices"], kwargs["target_frames"])
        return torch.full(shape, 220.0), torch.full(shape, 0.5)

    monkeypatch.setattr(preprocess, "load_neutone_amt_model", fake_load)
    monkeypatch.setattr(preprocess, "neutone_amt_to_voices", fake_transcribe)

    feats = pitch_features(
        tmp_path / "in.wav",
        cfg,
        target_frames=17,
        device="cpu",
        pitch_encoder="neutone-amt",
        pitch_encoder_checkpoint=model_path,
    )

    assert calls["load"] == (model_path, "cpu")
    _, kwargs = calls["transcribe"]
    assert kwargs["sample_rate"] == SR
    assert kwargs["target_hop"] == HOP
    assert kwargs["model"] is amt_model
    assert feats["pitch"].shape == (1, V, 17)
    assert torch.all(feats["velocity"] == 0.5)


def test_neutone_pitch_features_require_encoder_checkpoint() -> None:
    cfg = OmegaConf.create({
        "model": {"sr": SR, "frame_hop": HOP},
        "experiment": {"model": {
            "n_voices": V,
        }},
    })
    with pytest.raises(SystemExit, match="pitch-encoder-checkpoint is required"):
        pitch_features("in.wav", cfg, 10, pitch_encoder="neutone-amt")


def test_render_single_block_matches_forward() -> None:
    """One block, no noise → render must reproduce forward exactly.

    `use_noise=False` is required: `FilteredNoise` draws fresh uniform noise per
    call, so a noisy model can only ever match statistically.
    """
    model = _tiny_model(use_reverb=True)
    audio, pitch, velocity = _cond(2 * SR)
    n_frames = audio.shape[-1] // HOP

    ref, _ = model(audio, pitch=pitch, velocity=velocity)
    got = model.render(audio, pitch=pitch, velocity=velocity, chunk_frames=n_frames + 10)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize(
    "n_voices,use_z",
    [
        (1, False),  # V=1: MonoDecoder drops the velocity stream (use_velocity=False)
        (V, False),
        (V, True),   # z is whole-file conditioning; blocks must slice it consistently
    ],
)
def test_render_multi_block_matches_forward(n_voices: int, use_z: bool) -> None:
    """Block rendering is exact, not merely seam-free.

    Four blocks of 237 frames over 3 s, reverb on, noise off. Every stage that
    could disagree with a whole-file pass is exercised: the decoder GRU's hidden
    state crosses three block joins, so does the oscillator phase, both
    upsamplers see interior *and* file-edge lattices, and the reverb tail runs
    across every join. `use_noise=False` because `FilteredNoise` redraws its
    uniform noise on each call (see the noise test below for that half).
    """
    model = _tiny_model(use_reverb=True, n_voices=n_voices, use_z=use_z)
    audio, pitch, velocity = _cond(3 * SR, n_voices)
    n_frames = audio.shape[-1] // HOP
    chunk = 237  # not a whole second, and does not divide n_frames — see `_cond`
    assert n_frames // chunk >= 3, "test needs at least 3 blocks"

    ref, _ = model(audio, pitch=pitch, velocity=velocity)
    got = model.render(audio, pitch=pitch, velocity=velocity, chunk_frames=chunk)
    err = (got - ref).abs().max().item()
    assert err < 1e-5, f"block render differs from whole-file forward by {err:.2e}"
    # Non-degenerate: the signal has to be big enough for 1e-5 to mean something.
    assert ref.abs().max().item() > 0.05


def test_render_multi_block_needs_the_carried_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canary for the test above: drop the phase carry and it must fail loudly.

    Without this, `test_render_multi_block_matches_forward` could pass for the
    wrong reason (e.g. block boundaries landing on whole cycles).
    """
    model = _tiny_model(use_reverb=True)
    audio, pitch, velocity = _cond(3 * SR)
    ref, _ = model(audio, pitch=pitch, velocity=velocity)

    real_forward = AdditiveSynth.forward

    def without_carry(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["initial_phase"] = None
        return real_forward(self, *args, **kwargs)

    monkeypatch.setattr(AdditiveSynth, "forward", without_carry)
    got = model.render(audio, pitch=pitch, velocity=velocity, chunk_frames=237)
    assert (got - ref).abs().max().item() > 1e-3


def test_render_multi_block_matches_single_block_with_noise() -> None:
    """With noise on, multi-block render == single-block render bit-for-bit-ish.

    Valid because the filtered-noise synth is no longer inside the block loop:
    both renders call it exactly once, over the whole file, on `noise_mags` that
    agree to fp32 rounding. The one call is the only consumer of the global RNG
    in the whole render, so re-seeding before each render makes the uniform noise
    draw identical — which is what makes the comparison meaningful rather than
    statistical.
    """
    model = _tiny_model(use_noise=True, use_reverb=True)
    audio, pitch, velocity = _cond(3 * SR)
    n_frames = audio.shape[-1] // HOP

    torch.manual_seed(1234)
    one = model.render(audio, pitch=pitch, velocity=velocity, chunk_frames=n_frames + 10)
    torch.manual_seed(1234)
    many = model.render(audio, pitch=pitch, velocity=velocity, chunk_frames=237)
    err = (one - many).abs().max().item()
    assert err < 1e-5, f"noisy multi-block render differs from single-block by {err:.2e}"
    assert one.abs().max().item() > 0.1

    # The seeding is load-bearing, not decorative: without it the noise realisation
    # differs and the two renders are only statistically alike.
    other = model.render(audio, pitch=pitch, velocity=velocity, chunk_frames=237)
    assert (one - other).abs().max().item() > 1e-5


def test_decode_block_matches_mono_decoder() -> None:
    """`PolyDDSP._decode_block` must be `MonoDecoder.forward` plus GRU threading.

    `_decode_block` re-runs the decoder's submodules by hand (to get at the GRU's
    final hidden state, which `MonoDecoder.forward` discards and which
    `decoder.py` is off-limits to change). Run over a whole clip with `h_0=None`
    and no look-ahead frame it must be the same computation.
    """
    for use_z in (False, True):
        model = _tiny_model(use_z=use_z, n_voices=V)
        audio, pitch, velocity = _cond(SR)
        _, _, loudness, z, _ = model._conditioning(audio, pitch, velocity)
        n_frames = pitch.shape[-1]
        pitch_scaled = scale_f0_hz(pitch)

        ref = model.decoder(pitch_scaled, velocity, loudness, z)
        got = model._decode_block(pitch_scaled, velocity, loudness, z, None, n_frames)
        assert got[3].shape == (1, V, model.decoder.gru.hidden_size)
        for a, b, name in zip(ref, got[:3], ("harm_dist", "amp_v", "noise_mags")):
            assert torch.equal(a, b), f"{name} differs between decoder paths (use_z={use_z})"


def test_render_length_matches_non_frame_aligned_input() -> None:
    model = _tiny_model()
    n_samples = 3 * SR + 37  # not divisible by HOP
    audio, pitch, velocity = _cond(n_samples)
    audio = torch.nn.functional.pad(audio, (0, n_samples - audio.shape[-1]))
    out = model.render(audio, pitch=pitch, velocity=velocity, chunk_frames=400)
    assert out.shape == (1, n_samples)
    assert torch.isfinite(out).all()


def _write_run_dir(tmp_path: Path, model: PolyDDSP) -> Path:
    """A minimal run directory: `best.pt` + the `config.yaml` train.py writes."""
    run = tmp_path / "run"
    run.mkdir()
    torch.save({"model": model.state_dict()}, run / "best.pt")
    kw = _tiny_kwargs()
    cfg = OmegaConf.create({
        # `${now:...}` is a Hydra-only resolver: keeping it here proves load_run
        # never resolves the loaded config.
        "run": {"name": "tiny_${now:%Y%m%d}"},
        "model": {
            "sr": kw["sr"], "clip_seconds": 2, "frame_hop": kw["frame_hop"],
            "n_harmonics": kw["n_harmonics"], "z_dim": kw["z_dim"],
            "noise_bands": kw["noise_bands"], "noise_window": kw["noise_window"],
            "reverb_len": kw["reverb_len"],
            "loudness_n_fft": 512, "gru_hidden": kw["gru_hidden"],
            "mlp_hidden": kw["mlp_hidden"], "mlp_layers": kw["mlp_layers"],
        },
        "experiment": {
            "name": "tiny",
            "model": {
                "n_voices": V, "use_z": False, "use_reverb": False, "use_noise": False,
                "pitch_source": "cached_basic_pitch",
            },
        },
    })
    OmegaConf.save(cfg, run / "config.yaml")
    return run


def test_main_end_to_end(tmp_path: Path) -> None:
    """CLI → load_run → real Basic Pitch transcription → render → wav on disk."""
    run = _write_run_dir(tmp_path, _tiny_model())

    n_samples = 2 * SR
    t = np.arange(n_samples) / SR
    wav = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    src = tmp_path / "in.wav"
    sf.write(src, wav, SR)

    out = tmp_path / "nested" / "out.wav"
    main([f"ckpt={run / 'best.pt'}", f"input={src}", f"out={out}", "device=cpu"])

    assert out.exists()
    got, got_sr = sf.read(out, dtype="float32")
    assert got_sr == SR
    assert got.shape[0] == n_samples
    assert np.isfinite(got).all()
    assert np.abs(got).max() == pytest.approx(1.0, abs=1e-3)  # peak-normalised


def test_load_run_errors_without_config(tmp_path: Path) -> None:
    from polyddsp.infer import load_run

    ckpt = tmp_path / "best.pt"
    torch.save({"model": _tiny_model().state_dict()}, ckpt)
    with pytest.raises(SystemExit, match="config.yaml"):
        load_run(ckpt)
