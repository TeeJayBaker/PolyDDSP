"""Round-trip training summaries through real TensorBoard event files."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import polyddsp.train as training
from polyddsp.train import TensorBoardLogger


def _config(enabled=True):
    return OmegaConf.create({
        "tensorboard": {"enabled": enabled, "flush_secs": 30},
        "model": {"sr": 16000},
        "sample_rate_copy": "${model.sr}",
    })


def _events(out_dir):
    return EventAccumulator(str(out_dir / "tensorboard"), size_guidance={
        "scalars": 0, "audio": 0, "tensors": 0,
    }).Reload()


def test_scalars_config_and_jsonl(tmp_path):
    logger = TensorBoardLogger(_config(), tmp_path)
    logger.log_step(10, torch.tensor(1.25))
    # Both evaluation passes may log at the same training step.
    logger.log_metrics(10, {"mss": 0.5, "mfcc": 0.25})
    logger.log_metrics(10, {"fad": 0.75, "clap_fd_val_only": 0.125})
    logger.log_metrics(11, {"fad": 0.625}, prefix="val/final/")
    logger.close()

    events = _events(tmp_path)
    expected = {
        "loss": (10, 1.25), "val/mss": (10, 0.5), "val/mfcc": (10, 0.25),
        "val/fad": (10, 0.75), "val/clap_fd_val_only": (10, 0.125),
        "val/final/fad": (11, 0.625),
    }
    assert set(events.Tags()["scalars"]) == set(expected)
    for tag, (step, value) in expected.items():
        event, = events.Scalars(tag)
        assert (event.step, event.value) == (step, value)
    config = events.Tensors("config/text_summary")[0].tensor_proto.string_val[0].decode()
    assert "sample_rate_copy: 16000" in config
    assert "${model.sr}" not in config
    assert [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()] == [
        {"step": 10, "loss": 1.25},
        {"step": 10, "val/mss": 0.5, "val/mfcc": 0.25},
        {"step": 10, "val/fad": 0.75, "val/clap_fd_val_only": 0.125},
        {"step": 11, "val/final/fad": 0.625},
    ]


class _AudioModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.0))
        self.seen_kwargs = None

    def forward(self, audio, **kwargs):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert audio.shape[0] == 1
        self.seen_kwargs = kwargs
        return audio * self.gain, {"audio_harm": audio * 0.25, "audio_noise": audio * 0.125}


@pytest.mark.parametrize("dict_batch", [False, True])
@pytest.mark.parametrize("silent", [False, True])
def test_all_audio_previews(tmp_path, dict_batch, silent):
    audio = torch.zeros(2, 160) if silent else torch.linspace(-2, 2, 160).repeat(2, 1)
    batch = {"audio": audio, "pitch": torch.ones(2, 1, 3), "velocity": torch.ones(2, 1, 3)} if dict_batch else audio
    model = _AudioModel()
    logger = TensorBoardLogger(_config(), tmp_path)
    logger.log_audio_sample(7, model, [batch])
    logger.close()
    if dict_batch:
        assert model.seen_kwargs["pitch"].shape == (1, 1, 3)
        assert model.seen_kwargs["velocity"].shape == (1, 1, 3)
    else:
        assert model.seen_kwargs == {}

    ref = audio[0].numpy()
    dry = ref * 0.375
    clips = {
        "val/audio_ref": ref, "val/audio_pred": ref,
        "val/audio_harm": ref * 0.25, "val/audio_noise": ref * 0.125,
        "val/audio_dry_mix_norm": dry / (np.abs(dry).max() + 1e-9) * 0.99,
        "val/audio_wet": ref - dry,
    }
    events = _events(tmp_path)
    assert set(events.Tags()["audio"]) == set(clips)
    for tag, expected in clips.items():
        event, = events.Audio(tag)
        assert event.step == 7
        assert event.sample_rate == 16000
        assert event.length_frames == 160
        decoded, sr = sf.read(io.BytesIO(event.encoded_audio_string))
        assert sr == 16000
        np.testing.assert_allclose(decoded, expected.clip(-1, 1), atol=2 / 32768)


def test_disabled_keeps_jsonl_without_audio_inference(tmp_path):
    logger = TensorBoardLogger(_config(enabled=False), tmp_path)
    logger.log_step(0, torch.tensor(0.5))
    logger.log_metrics(1, {"mss": 0.25})
    # No model or loader should be touched when audio logging is disabled.
    logger.log_audio_sample(1, None, None)
    logger.close()
    assert not (tmp_path / "tensorboard").exists()
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 2


def test_resume_hides_stale_events_and_appends_jsonl(tmp_path):
    logger = TensorBoardLogger(_config(), tmp_path)
    for step in (0, 10, 20):
        logger.log_step(step, torch.tensor(1.0))
    logger.close()

    logger = TensorBoardLogger(_config(), tmp_path, purge_step=10)
    logger.log_step(10, torch.tensor(0.5))
    logger.close()
    assert [(e.step, e.value) for e in _events(tmp_path).Scalars("loss")] == [(0, 1.0), (10, 0.5)]
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 4


def test_training_config_has_tensorboard_overrides():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        default = compose(config_name="config")
        disabled = compose(config_name="config", overrides=[
            "tensorboard.enabled=false", "tensorboard.flush_secs=5",
        ])
    assert default.tensorboard.enabled
    assert default.tensorboard.flush_secs == 30
    assert "wandb" not in default
    assert not disabled.tensorboard.enabled
    assert disabled.tensorboard.flush_secs == 5


def test_training_loop_writes_all_summary_types(tmp_path, monkeypatch):
    """Exercise the logger wiring and evaluation cadence without real datasets/embedders."""
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=[
            f"run.out_dir={tmp_path}", "run.fresh=true", "train.steps=3",
            "train.log_every=1", "train.eval_cheap_every=1", "train.eval_full_every=1",
        ])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training, "RawAudioDataset", lambda **kwargs: object())

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, audio):
            pred = audio * self.gain
            return pred, {"audio_harm": pred, "audio_noise": torch.zeros_like(pred)}

    monkeypatch.setattr(training.PolyDDSP, "from_cfg", lambda cfg: TinyModel())
    monkeypatch.setattr(training, "MultiScaleSpectral", torch.nn.L1Loss)
    loader = [torch.linspace(-1, 1, 160).repeat(2, 1)]
    monkeypatch.setattr(training, "_make_loader", lambda *args, **kwargs: loader)
    monkeypatch.setattr(training, "DataLoader", lambda *args, **kwargs: loader)
    monkeypatch.setattr(training, "_evaluate_cheap", lambda *args: {"mss": 0.5})
    monkeypatch.setattr(training, "_evaluate_full", lambda *args: {"fad": 0.75})

    training.main.__wrapped__(cfg)

    events = _events(tmp_path)
    assert [e.step for e in events.Scalars("loss")] == [0, 1, 2]
    assert [e.step for e in events.Scalars("val/mss")] == [0, 1, 2]
    assert [e.step for e in events.Scalars("val/fad")] == [1, 2]
    assert [e.step for e in events.Scalars("val/final/fad")] == [3]
    assert len(events.Tags()["audio"]) == 6
    for tag in events.Tags()["audio"]:
        assert [e.step for e in events.Audio(tag)] == [0, 1, 2]
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "config.yaml").is_file()
