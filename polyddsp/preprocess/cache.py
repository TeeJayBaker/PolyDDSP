"""Shared pitch-cache storage."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch


def save_pitch_cache(
    cache: Path,
    pitch: torch.Tensor,
    velocity: torch.Tensor,
    pitch_cache_source: str | None = None,
) -> None:
    """Publish a complete cache atomically, including when a task is retried."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache.parent, prefix=f".{cache.name}.", delete=False,
        ) as file:
            temporary = Path(file.name)
            blob = {"pitch": pitch.contiguous(), "velocity": velocity.contiguous()}
            if pitch_cache_source is not None:
                blob["pitch_cache_source"] = pitch_cache_source
            torch.save(blob, file)
        os.replace(temporary, cache)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cache_path_for(
    audio_path: Path,
    suffix: str,
    *,
    root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Return a cache path, optionally mirrored beneath a separate output root."""
    target = audio_path
    if output_dir is not None:
        if root is None:
            raise ValueError("root is required when output_dir is provided")
        try:
            relative = audio_path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"audio path {audio_path} is outside dataset root {root}") from exc
        target = output_dir / relative
    return target.with_suffix(target.suffix + f".{suffix}.f0.pt")


def resolve_pitch_cache(
    cfg, source: str = "basic-pitch",
) -> tuple[str | None, str | None, int | None]:
    """Return the cache kind, filename suffix, and voice count for a config."""
    if source not in (
        "basic-pitch",
        "neutone-amt-streaming",
        "neutone-amt-nonstreaming",
        "ground-truth",
    ):
        raise ValueError(
            f"Unknown pitch cache source {source!r}; expected basic-pitch, "
            "neutone-amt-streaming, neutone-amt-nonstreaming, or ground-truth"
        )
    if cfg.experiment.model.get("pitch_source") != "cached_basic_pitch":
        return None, None, None

    # Local imports avoid a cycle: backends use this module to write caches.
    from polyddsp.preprocess import basic_pitch, ground_truth, neutone_amt

    n_voices = cfg.experiment.model.n_voices
    sample_rate = cfg.model.sr
    hop = cfg.model.frame_hop
    if source == "basic-pitch":
        return "basic-pitch", basic_pitch.cache_suffix(n_voices, sample_rate, hop), n_voices
    if source in ("neutone-amt-streaming", "neutone-amt-nonstreaming"):
        return "neutone", neutone_amt.cache_suffix(source, n_voices, sample_rate, hop), n_voices
    return "ground_truth", ground_truth.cache_suffix(n_voices, sample_rate, hop), n_voices
