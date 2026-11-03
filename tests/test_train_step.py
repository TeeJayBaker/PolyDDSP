"""Single train step — loss decreases on toy data."""
from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from polyddsp.losses import MultiScaleSpectral
from polyddsp.model.polyddsp import PolyDDSP
from polyddsp.train import TrainState, ZScaleScheduler, train_step


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


def test_z_scale_scheduler() -> None:
    scheduler = ZScaleScheduler(total_steps=10_000, block_pct=40, ramp_pct=10)
    assert scheduler(0) == 0.0
    assert scheduler(4_000) == 0.0
    assert scheduler(4_500) == 0.5
    assert scheduler(5_000) == 1.0


def test_checkpoint_stores_render_scales(tmp_path: Path) -> None:
    model = _tiny_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
    state = TrainState(step=8_500)
    path = tmp_path / "checkpoint.pt"

    state.save(path, model, opt, sched, z_scale=0.5, noise_scale=0.5)

    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["step"] == 8_500
    assert saved["streaming_version"] == PolyDDSP.STREAMING_VERSION
    assert saved["render_scales"] == {"z_scale": 0.5, "noise_scale": 0.5}


def test_noise_warmup_defaults_to_z_warmup_settings() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=[
            "train.z_warmup.enabled=true",
            "train.z_warmup.block_pct=25",
            "train.z_warmup.ramp_pct=15",
        ])
        separate = compose(config_name="config", overrides=[
            "train.z_warmup.enabled=true",
            "train.noise_warmup.enabled=false",
            "train.noise_warmup.block_pct=5",
            "train.noise_warmup.ramp_pct=20",
        ])
    assert cfg.train.noise_warmup.enabled
    assert cfg.train.noise_warmup.block_pct == 25
    assert cfg.train.noise_warmup.ramp_pct == 15
    assert separate.train.z_warmup.enabled
    assert not separate.train.noise_warmup.enabled
    assert separate.train.noise_warmup.block_pct == 5
    assert separate.train.noise_warmup.ramp_pct == 20


def test_zero_noise_scale_blocks_noise_output_and_head_gradient() -> None:
    model = PolyDDSP(
        sr=16_000, frame_hop=64, n_harmonics=8, z_dim=4, noise_bands=9,
        noise_window=0, reverb_len=32, gru_hidden=16, mlp_hidden=16,
        mlp_layers=1, n_voices=2, use_z=False, use_reverb=False,
        use_noise=True, pitch_source="cached_basic_pitch",
    )
    frames = 8
    pitch = torch.full((1, 2, frames), 220.0)
    velocity = torch.full_like(pitch, 0.5)
    loudness = torch.zeros(1, frames)

    mix, parts = model._synthesise(
        pitch, velocity, loudness, None,
        z_scale=1.0, noise_scale=0.0,
    )
    assert torch.equal(parts["audio_noise"], torch.zeros_like(parts["audio_noise"]))
    assert torch.equal(mix, parts["audio_harm"])

    mix.square().mean().backward()
    head_grad = model.decoder.head.weight.grad
    assert head_grad is not None
    noise_start = 1 + model.decoder.n_harmonics
    assert torch.count_nonzero(head_grad[:noise_start]).item() > 0
    assert torch.count_nonzero(head_grad[noise_start:]).item() == 0
