"""Hydra-driven training entry-point for PolyDDSP."""
from __future__ import annotations

import dataclasses as dc
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from polyddsp.data import RawAudioDataset
from polyddsp.losses import MultiScaleSpectral
from polyddsp.midi_synthesis import render_pitch_image, synthesize_midi
from polyddsp.model.polyddsp import PolyDDSP
from polyddsp.preprocess import resolve_pitch_cache

log = logging.getLogger("polyddsp.train")


# --------------------------------------------------------------------- helpers


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    """Resolve one training device from `auto | cpu | cuda | cuda:N`."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"invalid training device {requested!r}; expected auto, cpu, cuda, or cuda:N"
        ) from exc
    if device.type not in {"cpu", "cuda"} or (device.type == "cpu" and device.index is not None):
        raise ValueError(
            f"invalid training device {requested!r}; expected auto, cpu, cuda, or cuda:N"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"training device {requested!r} requested, but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"training device {requested!r} does not exist; "
                f"found {torch.cuda.device_count()} CUDA device(s)"
            )
    return str(device)


def _unwrap(batch: Any) -> tuple[torch.Tensor, dict]:
    """Peel a dataset batch into (audio, model_kwargs).

    Returns:
      raw tensor batch                                 → (audio, {})
      {"audio": ..., "pitch": ..., "velocity": ...}    → (audio, {"pitch": ..., "velocity": ...})
    """
    if isinstance(batch, dict):
        audio = batch["audio"]
        kwargs: dict = {}
        if "pitch" in batch:
            kwargs["pitch"] = batch["pitch"]
        if "velocity" in batch:
            kwargs["velocity"] = batch["velocity"]
        return audio, kwargs
    return batch, {}


@dc.dataclass(frozen=True)
class WarmupScaleScheduler:
    total_steps: int
    block_pct: float
    ramp_pct: float

    def __call__(self, step: int) -> float:
        """Scale one synthesis path from 0 to 1 after an initial block."""
        block_end = self.total_steps * self.block_pct / 100
        ramp_end = block_end + self.total_steps * self.ramp_pct / 100
        if step < block_end:
            return 0.0
        if step >= ramp_end or ramp_end == block_end:
            return 1.0
        return (step - block_end) / (ramp_end - block_end)


ZScaleScheduler = WarmupScaleScheduler  # Backward-compatible import name.


def _scale_at(scheduler: WarmupScaleScheduler | None, step: int) -> float:
    return scheduler(step) if scheduler is not None else 1.0


def train_step(
    model: PolyDDSP,
    batch: Any,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_clip: float = 1.0,
    z_scale: float = 1.0,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    model.train()
    device = next(model.parameters()).device
    audio, model_kwargs = _unwrap(batch)
    audio = audio.to(device)
    for k, v in model_kwargs.items():
        if isinstance(v, torch.Tensor):
            model_kwargs[k] = v.to(device)
    audio_pred, _ = model(
        audio, z_scale=z_scale, noise_scale=noise_scale, **model_kwargs,
    )
    loss = loss_fn(audio_pred, audio)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    if sched is not None:
        sched.step()
    return loss.detach()


# ------------------------------------------------------------------ resumption


@dc.dataclass
class TrainState:
    step: int = 0
    best_metric: float = math.inf
    best_step: int = 0

    @staticmethod
    def out_dir_for(cfg: DictConfig) -> Path:
        if cfg.run.fresh:
            return Path(cfg.run.out_dir)
        # If a prior dir matching this experiment exists, reuse the most recent.
        parent = Path(cfg.run.out_dir).parent
        prefix = cfg.experiment.name + "_"
        suffix = "_" + cfg.experiment.dataset.pitch_cache_source
        if parent.exists():
            candidates = sorted([
                p for p in parent.iterdir()
                if p.is_dir() and p.name.startswith(prefix) and p.name.endswith(suffix)
            ])
            if candidates:
                return candidates[-1]
        return Path(cfg.run.out_dir)

    @classmethod
    def resume_or_init(
        cls,
        cfg: DictConfig,
        model: PolyDDSP,
        opt: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler,
    ) -> "TrainState":
        out_dir = cls.out_dir_for(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        last = out_dir / "last.pt"
        state = cls()
        if last.exists():
            ckpt = torch.load(last, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            sched.load_state_dict(ckpt["sched"])
            state.step = ckpt["step"]
            state.best_metric = ckpt["best_metric"]
            state.best_step = ckpt["best_step"]
            log.info("Resumed from %s at step %d", last, state.step)
        OmegaConf.save(cfg, out_dir / "config.yaml")
        return state

    def save(
        self,
        path: Path,
        model: PolyDDSP,
        opt: torch.optim.Optimizer,
        sched,
        *,
        z_scale: float,
        noise_scale: float,
    ) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "step": self.step,
                "best_metric": self.best_metric,
                "best_step": self.best_step,
                "render_scales": {
                    "z_scale": float(z_scale),
                    "noise_scale": float(noise_scale),
                },
            },
            path,
        )


# --------------------------------------------------------------- top-k checkpoints


class TopKCheckpoint:
    def __init__(self, out_dir: Path, k: int = 3) -> None:
        self.out_dir = out_dir
        self.k = k
        self.entries: list[tuple[float, int, Path]] = []

    def maybe_save(
        self,
        state: TrainState,
        model,
        opt,
        sched,
        metric_value: float,
        *,
        z_scale: float,
        noise_scale: float,
    ) -> bool:
        if any(metric_value >= e[0] for e in self.entries) and len(self.entries) >= self.k:
            return False
        path = self.out_dir / f"top_{state.step:08d}.pt"
        state.save(
            path, model, opt, sched,
            z_scale=z_scale, noise_scale=noise_scale,
        )
        self.entries.append((metric_value, state.step, path))
        self.entries.sort(key=lambda e: e[0])
        if len(self.entries) > self.k:
            _, _, drop = self.entries.pop(-1)
            try:
                drop.unlink()
            except FileNotFoundError:
                pass
        if metric_value < state.best_metric:
            state.best_metric = metric_value
            state.best_step = state.step
            best = self.out_dir / "best.pt"
            state.save(
                best, model, opt, sched,
                z_scale=z_scale, noise_scale=noise_scale,
            )
        return True


# ---------------------------------------------------------------------- logger


class TensorBoardLogger:
    def __init__(self, cfg: DictConfig, out_dir: Path, *, purge_step: int | None = None) -> None:
        self.cfg = cfg
        self.out_dir = out_dir
        self.writer = None
        if cfg.tensorboard.enabled:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(
                log_dir=str(out_dir / "tensorboard"),
                flush_secs=cfg.tensorboard.flush_secs,
                purge_step=purge_step,
            )
            self.writer.add_text(
                "config", f"```yaml\n{OmegaConf.to_yaml(cfg, resolve=True)}```",
                global_step=purge_step or 0,
            )
            self.writer.add_custom_scalars({
                "Training": {
                    "Warmup scales": [
                        "Multiline",
                        ["train/z_scale", "train/noise_scale"],
                    ],
                },
                "Validation": {
                    "Component RMS": [
                        "Multiline",
                        [
                            "val/component_rms/harmonic",
                            "val/component_rms/noise",
                        ],
                    ],
                },
            })
        self.jsonl = (out_dir / "log.jsonl").open("a")

    def log_step(
        self,
        step: int,
        loss: torch.Tensor,
        *,
        z_scale: float | None = None,
        noise_scale: float | None = None,
    ) -> None:
        payload = {"step": step, "loss": float(loss.item())}
        if z_scale is not None:
            payload["train/z_scale"] = float(z_scale)
        if noise_scale is not None:
            payload["train/noise_scale"] = float(noise_scale)
        if self.writer is not None:
            for tag, value in payload.items():
                if tag != "step":
                    self.writer.add_scalar(tag, value, global_step=step)
        self.jsonl.write(json.dumps(payload) + "\n")
        self.jsonl.flush()

    def log_metrics(self, step: int, metrics: dict, prefix: str = "val/") -> None:
        flat = {f"{prefix}{k}": float(v) for k, v in metrics.items()}
        if self.writer is not None:
            for tag, value in flat.items():
                self.writer.add_scalar(tag, value, global_step=step)
        flat["step"] = step
        self.jsonl.write(json.dumps(flat) + "\n")
        self.jsonl.flush()

    def log_audio_sample(
        self,
        step: int,
        model: PolyDDSP,
        val_loader: DataLoader,
        checkpoint_label: str = "last.pt",
        log_conditioning: bool = False,
        log_component_rms: bool = True,
        z_scale: float = 1.0,
        noise_scale: float = 1.0,
    ) -> None:
        if self.writer is None:
            return

        model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            device = next(model.parameters()).device
            audio, model_kwargs = _unwrap(batch)
            audio = audio.to(device)[:1]
            for k, v in model_kwargs.items():
                if isinstance(v, torch.Tensor):
                    model_kwargs[k] = v.to(device)[:1]
            pred, aux = model(
                audio,
                z_scale=z_scale,
                noise_scale=noise_scale,
                **model_kwargs,
            )
        sr = self.cfg.model.sr
        T = pred.shape[-1]
        harm = aux["audio_harm"][0, :T]
        noise = aux["audio_noise"][0, :T]
        dry_mix = (harm + noise).cpu().numpy()
        # Peak-normalize so the pre-reverb signal is audible even when the
        # model has learned a quiet dry mix that the reverb amplifies.
        dry_mix_norm = dry_mix / (max(abs(dry_mix.max()), abs(dry_mix.min())) + 1e-9) * 0.99
        # The reverb returns dry + wet, so pred - dry_mix is the 100% wet tail.
        wet_only = (pred[0].cpu().numpy() - dry_mix)
        if log_component_rms:
            self.writer.add_scalar(
                "val/component_rms/harmonic",
                harm.square().mean().sqrt().item(),
                global_step=step,
            )
            self.writer.add_scalar(
                "val/component_rms/noise",
                noise.square().mean().sqrt().item(),
                global_step=step,
            )
        prefix = f"val/{checkpoint_label}"
        clips = {
            f"{prefix}/audio_ref": audio[0].cpu().numpy(),
            f"{prefix}/audio_pred": pred[0].cpu().numpy(),
            f"{prefix}/audio_harm": harm.cpu().numpy(),
            f"{prefix}/audio_noise": noise.cpu().numpy(),
            f"{prefix}/audio_dry_mix_norm": dry_mix_norm,
            f"{prefix}/audio_wet": wet_only,
        }
        if log_conditioning:
            clips["val/conditioning/synthesized_midi"] = synthesize_midi(
                aux["pitch"][:1],
                aux["velocity"][:1],
                sample_rate=sr,
                frame_hop=self.cfg.model.frame_hop,
                n_samples=T,
            )[0].cpu().numpy()
            self.writer.add_image(
                "val/conditioning/pitches",
                render_pitch_image(aux["pitch"][0], aux["velocity"][0]),
                global_step=step,
            )
        for tag, clip in clips.items():
            # Match the old PCM WAV previews: clip overloads, but retain the
            # relative levels of all stems except the explicitly normalized mix.
            self.writer.add_audio(tag, clip.clip(-1, 1)[None, :], global_step=step, sample_rate=sr)

    def close(self) -> None:
        try:
            self.jsonl.close()
        finally:
            if self.writer is not None:
                self.writer.close()


# ----------------------------------------------------------------- main loop


def _make_loader(ds: RawAudioDataset, batch_size: int, shuffle: bool) -> DataLoader:
    # `persistent_workers=True` is load-bearing for small datasets: with only a
    # handful of batches per epoch the 4 workers would be torn down and respawned
    # ~1×/sec, each time re-decoding + resampling audio on cold LRU caches, and
    # the GPU would starve between bursts.
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=True,
    )


def _infinite(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _evaluate_cheap(
    model,
    val_loader,
    sr: int,
    primary: list[str],
    *,
    z_scale: float = 1.0,
    noise_scale: float = 1.0,
) -> dict[str, float]:
    """Run cheap per-sample metrics across the val loader and return mean values."""
    from polyddsp.metrics.clap import clap_cos
    from polyddsp.metrics.loudness import loudness_l1
    from polyddsp.metrics.mfcc import mfcc_l1
    from polyddsp.metrics.mss import mss_l1
    from polyddsp.metrics.multipitch import multipitch_mse

    cheap_set = {"loudness", "mss", "mfcc"}
    if "multipitch" in primary:
        cheap_set.add("multipitch")
    if "clap_cos" in primary:
        cheap_set.add("clap_cos")

    fns = {
        "loudness": loudness_l1,
        "mss": mss_l1,
        "mfcc": mfcc_l1,
        "multipitch": multipitch_mse,
        "clap_cos": clap_cos,
    }
    accum: dict[str, list[float]] = {}
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in tqdm(
            val_loader,
            desc="validation",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        ):
            audio, model_kwargs = _unwrap(batch)
            audio = audio.to(device)
            for k, v in model_kwargs.items():
                if isinstance(v, torch.Tensor):
                    model_kwargs[k] = v.to(device)
            pred, _ = model(
                audio,
                z_scale=z_scale,
                noise_scale=noise_scale,
                **model_kwargs,
            )
            for k in cheap_set:
                vals = fns[k](audio, pred, sr=sr)
                for name, arr in vals.items():
                    accum.setdefault(name, []).extend(arr.tolist())
    return {k: float(sum(v) / max(len(v), 1)) for k, v in accum.items()}


def _evaluate_full(
    model,
    loaders: list[DataLoader],
    sr: int,
    primary: list[str],
    *,
    z_scale: float = 1.0,
    noise_scale: float = 1.0,
) -> dict[str, float]:
    """Pooled distributional metrics (CLAP-FD, FAD) across all provided loaders.

    With 1 val file we only have ~45 clips; pooling with the train split (~180)
    pushes Fréchet computations onto a less noisy footing. Adds val-only versions
    so the train/val gap is observable as an overfit signal.
    """
    from polyddsp.metrics.clap import clap_fd
    from polyddsp.metrics.fad import fad

    full_set = [k for k in ("clap", "fad") if k in primary]
    if not full_set:
        return {}
    fns = {"clap": clap_fd, "fad": fad}

    model.eval()
    device = next(model.parameters()).device
    pooled_refs: list[torch.Tensor] = []
    pooled_gens: list[torch.Tensor] = []
    val_refs: list[torch.Tensor] = []
    val_gens: list[torch.Tensor] = []

    with torch.no_grad():
        for li, loader in enumerate(loaders):
            for batch in tqdm(
                loader,
                desc=f"full eval {li + 1}/{len(loaders)}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            ):
                audio, model_kwargs = _unwrap(batch)
                audio = audio.to(device)
                for k, v in model_kwargs.items():
                    if isinstance(v, torch.Tensor):
                        model_kwargs[k] = v.to(device)
                pred, _ = model(
                    audio,
                    z_scale=z_scale,
                    noise_scale=noise_scale,
                    **model_kwargs,
                )
                pooled_refs.append(audio.cpu())
                pooled_gens.append(pred.cpu())
                # Convention: last loader = val (see main()).
                if li == len(loaders) - 1:
                    val_refs.append(audio.cpu())
                    val_gens.append(pred.cpu())

    pooled_ref = torch.cat(pooled_refs, dim=0)
    pooled_gen = torch.cat(pooled_gens, dim=0)
    out: dict[str, float] = {}
    for k in full_set:
        vals = fns[k](pooled_ref, pooled_gen, sr=sr)
        for name, arr in vals.items():
            out[name] = float(arr.mean())

    if val_refs:
        val_ref = torch.cat(val_refs, dim=0)
        val_gen = torch.cat(val_gens, dim=0)
        for k in full_set:
            vals = fns[k](val_ref, val_gen, sr=sr)
            for name, arr in vals.items():
                out[f"{name}_val_only"] = float(arr.mean())
    return out


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.run.seed)
    device = resolve_device(str(cfg.get("device", "auto")))

    pitch_cache_kind, pitch_cache_suffix, n_voices_for_cache = resolve_pitch_cache(
        cfg, source=cfg.experiment.dataset.get("pitch_cache_source", "basic-pitch"),
    )
    pitch_cache_source = cfg.experiment.dataset.get("pitch_cache_source", "basic-pitch")

    train_ds = RawAudioDataset(
        root=cfg.experiment.dataset.root,
        split="train",
        sample_rate=cfg.model.sr,
        clip_seconds=cfg.model.clip_seconds,
        seed=cfg.run.seed,
        file_glob=cfg.experiment.dataset.file_glob,
        pitch_cache_kind=pitch_cache_kind,
        pitch_cache_source=pitch_cache_source,
        pitch_cache_suffix=pitch_cache_suffix,
        pitch_cache_root=cfg.experiment.dataset.get("pitch_cache_root"),
        f0_hop=cfg.model.frame_hop,
        n_voices=n_voices_for_cache,
    )
    val_ds = RawAudioDataset(
        root=cfg.experiment.dataset.root,
        split="val",
        sample_rate=cfg.model.sr,
        clip_seconds=cfg.model.clip_seconds,
        seed=cfg.run.seed,
        file_glob=cfg.experiment.dataset.file_glob,
        pitch_cache_kind=pitch_cache_kind,
        pitch_cache_source=pitch_cache_source,
        pitch_cache_suffix=pitch_cache_suffix,
        pitch_cache_root=cfg.experiment.dataset.get("pitch_cache_root"),
        f0_hop=cfg.model.frame_hop,
        n_voices=n_voices_for_cache,
    )
    model = PolyDDSP.from_cfg(cfg).to(device)

    optim_type = cfg.optim.get("type", "adamw")
    if optim_type == "adam":
        opt = Adam(model.parameters(), lr=cfg.optim.lr, eps=1e-7)
    elif optim_type == "adamw":
        opt = AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay, eps=1e-7)
    else:
        raise ValueError(f"unknown optim.type: {optim_type}")

    sched_type = cfg.schedule.get("type", "sgdr")
    if sched_type == "exp_decay":
        sched = ExponentialLR(opt, gamma=cfg.schedule.gamma_per_step)
    elif sched_type == "sgdr":
        sched = CosineAnnealingWarmRestarts(opt, T_0=cfg.schedule.T_0, T_mult=cfg.schedule.T_mult)
    else:
        raise ValueError(f"unknown schedule.type: {sched_type}")
    z_scheduler = None
    if cfg.train.z_warmup.enabled:
        z_scheduler = WarmupScaleScheduler(
            total_steps=cfg.train.steps,
            block_pct=cfg.train.z_warmup.block_pct,
            ramp_pct=cfg.train.z_warmup.ramp_pct,
        )
    noise_scheduler = None
    if cfg.train.noise_warmup.enabled:
        noise_scheduler = WarmupScaleScheduler(
            total_steps=cfg.train.steps,
            block_pct=cfg.train.noise_warmup.block_pct,
            ramp_pct=cfg.train.noise_warmup.ramp_pct,
        )
    loss_fn = MultiScaleSpectral()

    state = TrainState.resume_or_init(cfg, model, opt, sched)
    out_dir = TrainState.out_dir_for(cfg)
    logger = TensorBoardLogger(
        cfg, out_dir, purge_step=state.step if (out_dir / "last.pt").exists() else None,
    )
    ckpt = TopKCheckpoint(out_dir, k=cfg.train.ckpt_keep_top_k)

    train_loader = _make_loader(train_ds, cfg.train.batch_size, shuffle=True)
    val_loader = _make_loader(val_ds, cfg.train.batch_size, shuffle=False)
    # Non-shuffled, non-dropping pass over train files for pooled distributional eval.
    eval_train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=True,
    )

    primary = list(cfg.experiment.metrics.primary)

    log.info("Training %s for up to %d steps on %s", cfg.experiment.name, cfg.train.steps, device)
    start = time.time()

    # Step-0 eval: verifies cached annotations + val pipeline before any gradient
    # step. The last.pt audio_ref + audio_harm pair is the pitch-correctness check
    # — at random init audio_harm should already be a buzzy chord at the audio's
    # pitches; if pitches don't match, the cache is wrong.
    if state.step == 0:
        log.info("running step-0 eval before training")
        z_scale = _scale_at(z_scheduler, state.step)
        noise_scale = _scale_at(noise_scheduler, state.step)
        step0_metrics = _evaluate_cheap(
            model, val_loader, cfg.model.sr, primary,
            z_scale=z_scale, noise_scale=noise_scale,
        )
        logger.log_metrics(state.step, step0_metrics, prefix="val/")
        logger.log_audio_sample(
            state.step, model, val_loader,
            z_scale=z_scale, noise_scale=noise_scale,
        )

    progress = tqdm(
        total=cfg.train.steps,
        initial=min(state.step, cfg.train.steps),
        desc=f"train {cfg.experiment.name}",
        unit="step",
        dynamic_ncols=True,
    )
    try:
        for batch in _infinite(train_loader):
            z_scale = _scale_at(z_scheduler, state.step)
            noise_scale = _scale_at(noise_scheduler, state.step)
            loss = train_step(
                model,
                batch,
                loss_fn,
                opt,
                sched,
                grad_clip=cfg.train.grad_clip,
                z_scale=z_scale,
                noise_scale=noise_scale,
            )
            # Logging every step costs a CUDA sync (`loss.item()`) + summary write
            # + jsonl.flush; serialised against the next iter's H2D copy. Gate on
            # `train.log_every` so the GPU stays pipelined.
            if state.step % cfg.train.log_every == 0:
                logger.log_step(
                    state.step, loss,
                    z_scale=z_scale, noise_scale=noise_scale,
                )
                progress.set_postfix(loss=f"{loss.item():.4f}")

            if state.step > 0 and state.step % cfg.train.eval_cheap_every == 0:
                metrics = _evaluate_cheap(
                    model, val_loader, cfg.model.sr, primary,
                    z_scale=z_scale, noise_scale=noise_scale,
                )
                logger.log_metrics(state.step, metrics, prefix="val/")
                logger.log_audio_sample(
                    state.step, model, val_loader,
                    z_scale=z_scale, noise_scale=noise_scale,
                )
                ckpt.maybe_save(
                    state, model, opt, sched, metrics["mss"],
                    z_scale=z_scale, noise_scale=noise_scale,
                )
                state.save(
                    out_dir / "last.pt", model, opt, sched,
                    z_scale=z_scale, noise_scale=noise_scale,
                )

            if state.step > 0 and state.step % cfg.train.eval_full_every == 0:
                full_metrics = _evaluate_full(
                    model, [eval_train_loader, val_loader], cfg.model.sr, primary,
                    z_scale=z_scale, noise_scale=noise_scale,
                )
                if full_metrics:
                    logger.log_metrics(state.step, full_metrics, prefix="val/")
                # FAD/CLAP_FD load large embedders + pool refs+gens to GPU; the
                # allocator can be left fragmented enough that the next train
                # step OOMs on its first big tensor (~1.7 GB additive). Drop
                # cached blocks before resuming.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            state.step += 1
            progress.update(1)
            if state.step >= cfg.train.steps:
                break
    finally:
        progress.close()
        state.save(
            out_dir / "last.pt", model, opt, sched,
            z_scale=_scale_at(z_scheduler, state.step),
            noise_scale=_scale_at(noise_scheduler, state.step),
        )
        # Always log a final full eval on the last checkpoint so CLAP/FAD aren't
        # tied solely to the eval_full_every cadence.
        try:
            final = _evaluate_full(
                model,
                [eval_train_loader, val_loader],
                cfg.model.sr,
                primary,
                z_scale=_scale_at(z_scheduler, state.step),
                noise_scale=_scale_at(noise_scheduler, state.step),
            )
            if final:
                logger.log_metrics(state.step, final, prefix="val/final/")
        except Exception:  # noqa: BLE001 — never let final eval mask training errors
            log.exception("Final full eval failed")
        try:
            # Explicit end-of-training previews make the final and best checkpoint
            # audible without relying on the periodic evaluation cadence.
            if logger.writer is not None:
                logger.log_audio_sample(
                    state.step,
                    model,
                    val_loader,
                    "last.pt",
                    log_conditioning=True,
                    z_scale=_scale_at(z_scheduler, state.step),
                    noise_scale=_scale_at(noise_scheduler, state.step),
                )
                best_path = out_dir / "best.pt"
                if best_path.exists():
                    best_ckpt = torch.load(best_path, map_location=device, weights_only=True)
                    model.load_state_dict(best_ckpt["model"])
                    saved_scales = best_ckpt.get("render_scales")
                    if isinstance(saved_scales, dict):
                        best_z_scale = float(saved_scales["z_scale"])
                        best_noise_scale = float(saved_scales["noise_scale"])
                    else:
                        # Legacy checkpoints have a step but no explicit scale metadata.
                        best_z_scale = _scale_at(z_scheduler, best_ckpt["step"])
                        best_noise_scale = _scale_at(noise_scheduler, best_ckpt["step"])
                    logger.log_audio_sample(
                        best_ckpt["step"],
                        model,
                        val_loader,
                        "best.pt",
                        log_component_rms=False,
                        z_scale=best_z_scale,
                        noise_scale=best_noise_scale,
                    )
        except Exception:  # noqa: BLE001 — never let audio previews mask training errors
            log.exception("Final checkpoint audio logging failed")
        finally:
            logger.close()
    log.info("Training complete in %.1f min", (time.time() - start) / 60.0)


if __name__ == "__main__":
    main()
