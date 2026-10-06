"""Basic Pitch audio transcription backend."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polyddsp.model.pitch import BasicPitchModel

from polyddsp.preprocess.cache import cache_path_for, save_pitch_cache
from polyddsp.preprocess.audio import load_resample


def cache_suffix(n_voices: int, sample_rate: int, target_hop: int) -> str:
    return f"bp_v{n_voices}_sr{sample_rate}_hop{target_hop}"


def precompute_one(
    audio_path: Path,
    sample_rate: int,
    target_hop: int,
    n_voices: int,
    device: str = "cpu",
    bp_model: "BasicPitchModel | None" = None,
    min_freq: float | None = None,
    max_freq: float | None = None,
    root: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, bool]:
    """Transcribe one audio file and cache target-rate voice tensors."""
    from polyddsp.model.pitch import BP_NATIVE_SR, basic_pitch_to_voices

    suffix = cache_suffix(n_voices, sample_rate, target_hop)
    cache = cache_path_for(audio_path, suffix, root=root, output_dir=output_dir)
    if cache.exists() and cache.stat().st_mtime > audio_path.stat().st_mtime:
        return cache, False

    # Resample directly to BP's native rate so its CQT retains content above 8 kHz.
    audio_22k = load_resample(audio_path, BP_NATIVE_SR).to(device)
    n_target_samples = int(round(audio_22k.shape[-1] * sample_rate / BP_NATIVE_SR))
    target_frames = n_target_samples // target_hop
    pitch, velocity = basic_pitch_to_voices(
        audio_22k,
        n_voices=n_voices,
        target_frames=target_frames,
        bp_model=bp_model,
        min_freq=min_freq,
        max_freq=max_freq,
    )
    save_pitch_cache(cache, pitch, velocity, "basic-pitch")
    return cache, True
