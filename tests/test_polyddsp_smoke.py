"""End-to-end PolyDDSP — shape, finiteness, gradient flow."""
from __future__ import annotations

import torch

from polyddsp.model.polyddsp import PolyDDSP


def _cfg(use_z: bool, use_reverb: bool, n_voices: int) -> dict:
    return dict(
        sr=16_000,
        frame_hop=64,
        n_harmonics=8,        # tiny for the smoke test
        z_dim=16,
        noise_bands=65,
        noise_window=0,
        reverb_len=48_000,
        gru_hidden=64,        # tiny for the smoke test
        mlp_hidden=64,
        mlp_layers=2,
        n_voices=n_voices,
        use_z=use_z,
        use_reverb=use_reverb,
    )


def test_end_to_end_shape_and_finite(four_s_sine: torch.Tensor) -> None:
    model = PolyDDSP(**_cfg(use_z=True, use_reverb=False, n_voices=2))
    audio_pred, aux = model(four_s_sine)
    assert audio_pred.shape == four_s_sine.shape
    assert torch.isfinite(audio_pred).all()
    assert "bp_post" in aux


def test_solo_violin_config(four_s_sine: torch.Tensor) -> None:
    model = PolyDDSP(**_cfg(use_z=False, use_reverb=True, n_voices=1))
    audio_pred, _ = model(four_s_sine)
    assert audio_pred.shape == four_s_sine.shape
    assert torch.isfinite(audio_pred).all()


def test_gradient_flows_through_trainable_params(four_s_sine: torch.Tensor) -> None:
    model = PolyDDSP(**_cfg(use_z=True, use_reverb=False, n_voices=2))
    audio_pred, _ = model(four_s_sine)
    loss = audio_pred.pow(2).mean()
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable), (
        "No gradient reached any trainable parameter"
    )


def test_solo_pitch_is_continuous(four_s_sine: torch.Tensor) -> None:
    """n_voices=1 path must emit continuous (sub-semitone) f0, not integer-MIDI."""
    model = PolyDDSP(**_cfg(use_z=False, use_reverb=True, n_voices=1))
    _, aux = model(four_s_sine)
    p = aux["pitch"][aux["pitch"] > 0]
    if p.numel() > 50:
        midi = 12 * torch.log2(p / 440.0) + 69.0
        non_integer = ((midi - midi.round()).abs() > 1e-3).float().mean().item()
        assert non_integer > 0.5


def test_polyddsp_accepts_pitch_velocity_kwargs_for_cached_bp() -> None:
    import torch
    from polyddsp.model.polyddsp import PolyDDSP

    model = PolyDDSP(
        sr=16000, frame_hop=64, n_harmonics=20, z_dim=16,
        noise_bands=65, n_voices=4, use_z=False, use_reverb=False,
        pitch_source="cached_basic_pitch",
    )
    audio = torch.zeros(2, 64000)
    pitch = torch.full((2, 4, 1000), 220.0)
    velocity = torch.full((2, 4, 1000), 0.5)
    out, aux = model(audio, pitch=pitch, velocity=velocity)
    assert out.shape == audio.shape
    assert torch.isfinite(out).all()
    assert aux["pitch"].shape == (2, 4, 1000)
    assert aux["velocity"].shape == (2, 4, 1000)
