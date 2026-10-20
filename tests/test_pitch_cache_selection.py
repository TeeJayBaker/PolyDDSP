"""Training config selects the matching preprocessed pitch cache."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import soundfile as sf
import torch

from polyddsp.data import RawAudioDataset, _load_bp_cache_cached
from polyddsp.preprocess import (
    cache_path_for,
    pitch_cache_root_for,
    preprocess_options_from_cfg,
    resolve_pitch_cache,
)


@pytest.mark.parametrize("experiment", ["guitarset", "gaps", "maestro"])
@pytest.mark.parametrize(
    "source,expected_kind,prefix",
    [
        ("basic-pitch", "basic-pitch", "bp"),
        ("neutone-amt-streaming", "neutone-amt-streaming", "neutone_streaming"),
        ("neutone-amt-nonstreaming", "neutone-amt-nonstreaming", "neutone_nonstreaming"),
        ("ground-truth", "ground-truth", "ground_truth"),
    ],
)
def test_config_selects_cache_for_training_and_validation(
    tmp_path, experiment, source, expected_kind, prefix,
):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=[
            f"experiment={experiment}",
            f"run.pitch_cache_source={source}",
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
        filename = f"{index}wc.wav" if experiment == "gaps" else f"{index}_mix.wav"
        audio = root / filename
        sf.write(audio, torch.zeros(cfg.model.sr).numpy(), cfg.model.sr)
        cache = cache_path_for(audio, suffix, root=root, output_dir=cache_root)
        cache.parent.mkdir(exist_ok=True)
        torch.save({
            "pitch": torch.full((voices, frames), 220.0),
            "velocity": torch.full((voices, frames), 0.5),
            "pitch_cache_source": source,
        }, cache)

    for split in ("train", "val"):
        ds = RawAudioDataset(
            root=str(root), split=split, sample_rate=cfg.model.sr, clip_seconds=1,
            file_glob=cfg.experiment.dataset.file_glob,
            pitch_cache_kind=kind, pitch_cache_source=source, pitch_cache_suffix=suffix,
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
    assert resolve_pitch_cache(cfg) == ("basic-pitch", "bp_v6_sr16000_hop64", 6)
    for unsupported in ("neutone-amt", "groundtruth", "typo"):
        with pytest.raises(ValueError, match="Unknown pitch cache source"):
            resolve_pitch_cache(cfg, source=unsupported)
    cfg.experiment.model.pitch_source = "basic_pitch"
    assert resolve_pitch_cache(cfg, source="neutone-amt-streaming") == (None, None, None)


@pytest.mark.parametrize("experiment,voices,file_glob,midi_glob", [
    ("guitarset", 6, "**/*_mix.wav", "**/*.mid"),
    ("gaps", 6, "**/*wc.wav", "**/*wc.mid"),
    ("maestro", 10, "**/*.wav", "**/*.midi"),
])
def test_preprocess_hydra_config_uses_experiment_values(
    tmp_path, experiment, voices, file_glob, midi_glob,
):
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
            "model=neutone-amt-nonstreaming",
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
    assert options.midi_glob == midi_glob
    assert options.backend == "neutone"
    assert options.pitch_cache_source == "neutone-amt-nonstreaming"
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
    cfg.midi_glob = "annotations/**/*.mid"
    cfg.n_voices = 3
    cfg.sample_rate = 22050
    cfg.hop = 128
    cfg.model = "basic-pitch"
    cfg.experiment.dataset.pitch_cache_source = "neutone-amt-streaming"
    overridden = preprocess_options_from_cfg(cfg)
    assert overridden.root == tmp_path / "other-audio"
    assert overridden.file_glob == "**/*.flac"
    assert overridden.midi_glob == "annotations/**/*.mid"
    assert (overridden.n_voices, overridden.sample_rate, overridden.hop) == (3, 22050, 128)
    assert overridden.backend == "basic-pitch"
    cfg.model = "typo"
    with pytest.raises(ValueError, match="model"):
        preprocess_options_from_cfg(cfg)


@pytest.mark.parametrize("experiment", ["guitarset", "gaps", "maestro"])
@pytest.mark.parametrize(
    "source",
    [
        "basic-pitch",
        "neutone-amt-streaming",
        "neutone-amt-nonstreaming",
        "ground-truth",
    ],
)
def test_default_cache_layout_is_source_then_experiment(experiment, source):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        train_cfg = compose(config_name="config", overrides=[
            f"experiment={experiment}",
            f"run.pitch_cache_source={source}",
        ])
        preprocess_cfg = compose(config_name="preprocess", overrides=[
            f"experiment={experiment}",
            f"model={source}",
        ])

    expected = Path(train_cfg.run.cache_root) / source / experiment
    assert Path(train_cfg.experiment.dataset.pitch_cache_root) == expected
    assert Path(preprocess_cfg.output_dir) == expected
    assert pitch_cache_root_for(train_cfg.run.cache_root, source, experiment) == expected


def test_ground_truth_preprocess_config_has_midi_controls(tmp_path):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="preprocess", overrides=[
            "experiment=maestro",
            "model=ground-truth",
            "midi_glob=**/*.mid",
            "apply_sustain=false",
            "pitch_bend_range=12",
        ])
    options = preprocess_options_from_cfg(cfg)
    assert options.backend == "ground-truth"
    assert options.midi_glob == "**/*.mid"
    assert not options.apply_sustain
    assert options.pitch_bend_range == 12


@pytest.mark.parametrize(
    "profile,experiment,dataset_suffix,steps,device_key,voices,prompt",
    [
        (
            "guitarset", "guitarset", "guitarset/audio_mono-pickup_mix",
            10_000, "guitarset", 6, "solo acoustic guitar",
        ),
        (
            "gaps", "gaps", "gaps",
            10_000, "gaps", 6, "solo classical guitar",
        ),
        (
            "maestro", "maestro", "MAESTRO/maestro-v3.0.0",
            20_000, "maestro", 10, "solo grand piano",
        ),
    ],
)
def test_training_profile_uses_local_machine_roots(
    profile, experiment, dataset_suffix, steps, device_key, voices, prompt,
):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=[
            f"train_profile={profile}",
            "run.pitch_cache_source=ground-truth",
        ])
    assert cfg.experiment.name == experiment
    assert Path(cfg.experiment.dataset.root) == Path(cfg.run.data_root) / dataset_suffix
    assert Path(cfg.experiment.dataset.pitch_cache_root) == (
        Path(cfg.run.cache_root) / cfg.run.pitch_cache_source / experiment
    )
    assert cfg.experiment.dataset.pitch_cache_source == cfg.run.pitch_cache_source
    assert cfg.experiment.model.n_voices == voices
    assert cfg.experiment.metrics.clap_prompt == prompt
    assert cfg.device == cfg.run.devices[device_key]
    assert cfg.train.steps == steps
    assert cfg.train.batch_size == 2
    assert cfg.train.eval_cheap_every == 500
    assert cfg.train.eval_full_every == 1_000_000


def test_missing_neutone_cache_suggests_matching_preprocessing(tmp_path):
    sf.write(tmp_path / "fake.wav", torch.zeros(16000).numpy(), 16000)
    with pytest.raises(FileNotFoundError, match="model_path") as error:
        RawAudioDataset(
            root=str(tmp_path), split="val", pitch_cache_kind="neutone-amt-streaming",
            pitch_cache_source="neutone-amt-streaming",
            pitch_cache_suffix="neutone_streaming_v6_sr16000_hop64", n_voices=6,
        )
    expected = tmp_path / "fake.wav.neutone_streaming_v6_sr16000_hop64.f0.pt"
    assert f"First missing cache: {expected}" in str(error.value)
    assert f"(for audio: {tmp_path / 'fake.wav'})" in str(error.value)


def test_cache_metadata_rejects_source_mismatch_but_allows_legacy(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    for index in range(5):
        sf.write(root / f"{index}.wav", torch.zeros(16000).numpy(), 16000)

    suffix = "neutone_streaming_v1_sr16000_hop64"
    source = "neutone-amt-streaming"
    ds_kwargs = dict(
        root=str(root), split="val", sample_rate=16000, clip_seconds=1,
        file_glob="*.wav", pitch_cache_kind="neutone-amt-streaming",
        pitch_cache_source=source, pitch_cache_suffix=suffix,
        f0_hop=64, n_voices=1,
    )
    for audio in root.glob("*.wav"):
        torch.save({
            "pitch": torch.ones(1, 250),
            "velocity": torch.ones(1, 250),
            "pitch_cache_source": "neutone-amt-nonstreaming",
        }, cache_path_for(audio, suffix))
    with pytest.raises(ValueError, match="pitch cache source mismatch"):
        RawAudioDataset(**ds_kwargs)[0]

    # Caches written before source metadata was introduced remain readable.
    for audio in root.glob("*.wav"):
        torch.save({
            "pitch": torch.ones(1, 250),
            "velocity": torch.ones(1, 250),
        }, cache_path_for(audio, suffix))
    _load_bp_cache_cached.cache_clear()
    assert RawAudioDataset(**ds_kwargs)[0]["pitch"].shape == (1, 250)
