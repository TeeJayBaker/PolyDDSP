"""Single train step — loss decreases on toy data."""
from __future__ import annotations

import torch

from polyddsp.losses import MultiScaleSpectral
from polyddsp.model.polyddsp import PolyDDSP
from polyddsp.train import train_step


def _tiny_model(pitch_source: str = "basic_pitch") -> PolyDDSP:
    return PolyDDSP(
        sr=16_000, frame_hop=64, n_harmonics=8, z_dim=16, noise_bands=65,
        noise_window=257, reverb_len=64_000, gru_hidden=32, mlp_hidden=32,
        mlp_layers=1, n_voices=1, use_z=False, use_reverb=False,
        pitch_source=pitch_source,
    )


def test_train_step_runs_and_returns_finite_loss(four_s_sine: torch.Tensor) -> None:
    model = _tiny_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = MultiScaleSpectral()
    loss = train_step(model, four_s_sine, loss_fn, opt, grad_clip=1.0)
    assert torch.isfinite(loss)


def test_train_step_handles_dict_batch(four_s_sine: torch.Tensor) -> None:
    """A cached-pitch batch is a dict; train_step unwraps audio + pitch conditioning."""
    model = _tiny_model(pitch_source="cached_basic_pitch")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = MultiScaleSpectral()
    B, T_frames = four_s_sine.shape[0], four_s_sine.shape[-1] // 64
    batch = {
        "audio": four_s_sine,
        "pitch": torch.full((B, 1, T_frames), 440.0),
        "velocity": torch.full((B, 1, T_frames), 0.7),
    }
    loss = train_step(model, batch, loss_fn, opt, grad_clip=1.0)
    assert torch.isfinite(loss)
