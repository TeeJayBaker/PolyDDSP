"""Standalone evaluation driver — runs the full metric battery on a checkpoint."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from polyddsp.data import RawAudioDataset
from polyddsp.metrics.clap import clap_cos, clap_fd
from polyddsp.metrics.fad import fad
from polyddsp.metrics.loudness import loudness_l1
from polyddsp.metrics.mfcc import mfcc_l1
from polyddsp.metrics.mss import mss_l1
from polyddsp.metrics.multipitch import multipitch_mse
from polyddsp.model.polyddsp import PolyDDSP
from polyddsp.preprocess import resolve_pitch_cache
from polyddsp.stats import mean_sem

log = logging.getLogger("polyddsp.eval")


def save_per_example_parquet(per_example: dict[str, np.ndarray], path: Path) -> None:
    cols = {k: pa.array(np.asarray(v)) for k, v in per_example.items()}
    table = pa.table(cols)
    pq.write_table(table, path)


def _all_metrics():
    return {
        "loudness": loudness_l1,
        "multipitch": multipitch_mse,
        "mss": mss_l1,
        "mfcc": mfcc_l1,
        "clap_cos": clap_cos,
        "clap": clap_fd,
        "fad": fad,
    }


def _generate(model: PolyDDSP, loader: DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    from polyddsp.train import _unwrap
    refs, gens = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            audio, model_kwargs = _unwrap(batch)
            audio = audio.to(device)
            for k, v in model_kwargs.items():
                if isinstance(v, torch.Tensor):
                    model_kwargs[k] = v.to(device)
            pred, _ = model(audio, **model_kwargs)
            refs.append(audio.cpu())
            gens.append(pred.cpu())
    return torch.cat(refs, dim=0), torch.cat(gens, dim=0)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise SystemExit("Set `ckpt=path/to/ckpt.pt` on the CLI")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(cfg.ckpt)
    log.info("Loading %s", ckpt_path)

    # Reconstruct the architecture from the *run's* saved config, not the config
    # Hydra just composed from `configs/` + this CLI. Otherwise evaluating a run
    # trained with any non-default model setting silently builds a different
    # model unless every override is retyped here. Dataset and metric selection
    # still come from the CLI config — only the architecture is pinned.
    if (ckpt_path.parent / "config.yaml").exists():
        from polyddsp.infer import load_run

        model, _ = load_run(ckpt_path, device)
    else:
        log.warning(
            "no config.yaml in %s — rebuilding the model from the CLI-composed config. "
            "If this run used non-default model settings, pass them all on the CLI or the "
            "architecture will not match the checkpoint.",
            ckpt_path.parent,
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = PolyDDSP.from_cfg(cfg).to(device)
        model.load_state_dict(state["model"])

    pitch_cache_kind, pitch_cache_suffix, n_voices_for_cache = resolve_pitch_cache(
        cfg, source=cfg.experiment.dataset.get("pitch_cache_source", "basic-pitch"),
    )

    val_ds = RawAudioDataset(
        root=cfg.experiment.dataset.root,
        split="val",
        sample_rate=cfg.model.sr,
        clip_seconds=cfg.model.clip_seconds,
        seed=cfg.run.seed,
        file_glob=cfg.experiment.dataset.file_glob,
        pitch_cache_kind=pitch_cache_kind,
        pitch_cache_suffix=pitch_cache_suffix,
        pitch_cache_root=cfg.experiment.dataset.get("pitch_cache_root"),
        f0_hop=cfg.model.frame_hop,
        n_voices=n_voices_for_cache,
    )
    loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4)
    refs, gens = _generate(model, loader, device)

    per_example: dict[str, np.ndarray] = {}
    metrics = _all_metrics()
    for name in cfg.experiment.metrics.primary:
        fn = metrics[name]
        out = fn(refs, gens, sr=cfg.model.sr)
        per_example.update(out)

    out_dir = ckpt_path.parent
    save_per_example_parquet(per_example, out_dir / "per_example.parquet")

    summary = {k: mean_sem(v) for k, v in per_example.items()}
    (out_dir / "summary.json").write_text(json.dumps({k: {"mean": m, "sem": s} for k, (m, s) in summary.items()}, indent=2))
    for k, (m, s) in summary.items():
        print(f"{k:24s}  {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
