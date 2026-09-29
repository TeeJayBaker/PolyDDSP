"""Training config selects the matching preprocessed pitch cache."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import soundfile as sf
import torch

from polyddsp.data import RawAudioDataset
from polyddsp.preprocess import (
    cache_path_for,
    preprocess_options_from_cfg,
    resolve_pitch_cache,
)


@pytest.mark.parametrize("experiment", ["guitarset", "maestro"])
@pytest.mark.parametrize(
    "source,expected_kind,prefix",
    [("basic-pitch", "basic_pitch", "bp"), ("neutone-amt", "neutone", "neutone")],
)
def test_config_selects_cache_for_training_and_validation(
    tmp_path, experiment, source, expected_kind, prefix,
):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=[
            f"experiment={experiment}",
            f"experiment.dataset.pitch_cache_source={source}",
        ])
    kind, suffix, voices = resolve_pitch_cache(
        cfg, source=cfg.experiment.dataset.pitch_cache_source,
    )
    assert kind == expected_kind
    assert suffix == f"{prefix}_v{voices}_sr16000_hop64"

    root, cache_root = tmp_path / "audio", tmp_path / "cache"
    root.mkdir()
    frames = cfg.model.sr // cfg.model.frame_hop
    for index in range(5):
        audio = root / f"{index}_mix.wav"
        sf.write(audio, torch.zeros(cfg.model.sr).numpy(), cfg.model.sr)
        cache = cache_path_for(audio, suffix, root=root, output_dir=cache_root)
        cache.parent.mkdir(exist_ok=True)
        torch.save({
            "pitch": torch.full((voices, frames), 220.0),
            "velocity": torch.full((voices, frames), 0.5),
        }, cache)

    for split in ("train", "val"):
        ds = RawAudioDataset(
            root=str(root), split=split, sample_rate=cfg.model.sr, clip_seconds=1,
            file_glob=cfg.experiment.dataset.file_glob,
            pitch_cache_kind=kind, pitch_cache_suffix=suffix,
            pitch_cache_root=str(cache_root), f0_hop=cfg.model.frame_hop,
            n_voices=voices,
        )
        item = ds[0]
        torch.testing.assert_close(item["pitch"], torch.full((voices, frames), 220.0))
        torch.testing.assert_close(item["velocity"], torch.full((voices, frames), 0.5))


def test_legacy_config_defaults_to_basic_pitch():
    cfg = OmegaConf.create({
        "experiment": {"model": {"pitch_source": "cached_basic_pitch", "n_voices": 6}},
        "model": {"sr": 16000, "frame_hop": 64},
    })
    assert resolve_pitch_cache(cfg) == ("basic_pitch", "bp_v6_sr16000_hop64", 6)
    with pytest.raises(ValueError, match="Unknown pitch cache source"):
        resolve_pitch_cache(cfg, source="typo")
    cfg.experiment.model.pitch_source = "basic_pitch"
    assert resolve_pitch_cache(cfg, source="neutone-amt") == (None, None, None)


@pytest.mark.parametrize("experiment,voices,file_glob", [
    ("guitarset", 6, "**/*_mix.wav"), ("maestro", 10, "**/*.wav"),
])
def test_preprocess_hydra_config_uses_experiment_values(tmp_path, experiment, voices, file_glob):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    model_path = tmp_path / "amt.onnx"
    model_path.touch()
    audio_root = tmp_path / "audio"
    cache_root = tmp_path / "cache"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="preprocess", overrides=[
            f"experiment={experiment}",
            f"experiment.dataset.root={audio_root}",
            f"output_dir={cache_root}",
            "model=neutone-amt",
            f"model_path={model_path}",
            "device=cpu",
            "parallel=true",
            "files_per_task=2",
        ])
    options = preprocess_options_from_cfg(cfg)
    assert not {"train", "optim", "schedule", "tensorboard", "ckpt", "preprocess"} & set(cfg)
    assert options.root == audio_root
    assert options.output_dir == cache_root
    assert options.file_glob == file_glob
    assert options.backend == "neutone"
    assert options.model_path == model_path
    assert options.sample_rate == 16_000
    assert options.hop == 64
    assert options.n_voices == voices
    assert options.device == "cpu"
    assert options.parallel
    assert options.files_per_task == 2
    # Training's cache input directory must not redirect preprocessing output.
    cfg.experiment.dataset.pitch_cache_root = str(tmp_path / "training-cache")
    assert preprocess_options_from_cfg(cfg).output_dir == cache_root
    cfg.output_dir = None
    assert preprocess_options_from_cfg(cfg).output_dir is None
    # Every input can be overridden explicitly without editing the experiment.
    cfg.data_root = str(tmp_path / "other-audio")
    cfg.file_glob = "**/*.flac"
    cfg.n_voices = 3
    cfg.sample_rate = 22050
    cfg.hop = 128
    cfg.model = "basic-pitch"
    cfg.experiment.dataset.pitch_cache_source = "neutone-amt"
    overridden = preprocess_options_from_cfg(cfg)
    assert overridden.root == tmp_path / "other-audio"
    assert overridden.file_glob == "**/*.flac"
    assert (overridden.n_voices, overridden.sample_rate, overridden.hop) == (3, 22050, 128)
    assert overridden.backend == "basic_pitch"
    cfg.model = "typo"
    with pytest.raises(ValueError, match="model"):
        preprocess_options_from_cfg(cfg)


def test_missing_neutone_cache_suggests_matching_preprocessing(tmp_path):
    sf.write(tmp_path / "fake.wav", torch.zeros(16000).numpy(), 16000)
    with pytest.raises(FileNotFoundError, match="model_path"):
        RawAudioDataset(
            root=str(tmp_path), split="val", pitch_cache_kind="neutone",
            pitch_cache_suffix="neutone_v6_sr16000_hop64", n_voices=6,
        )
