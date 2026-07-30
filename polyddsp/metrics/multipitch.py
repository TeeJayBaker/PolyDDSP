"""Basic Pitch posteriorgram MSE — polyphonic analogue of DDSP's CREPE-on-output."""
from __future__ import annotations

import functools

import numpy as np
import torch
import torchaudio.functional as AF

from polyddsp.model.pitch import BP_NATIVE_SR, load_basic_pitch


@functools.lru_cache(maxsize=1)
def _bp():
    return load_basic_pitch()


def _bp_post(audio: torch.Tensor, sr: int) -> dict[str, torch.Tensor]:
    if sr != BP_NATIVE_SR:
        audio = AF.resample(audio, orig_freq=sr, new_freq=BP_NATIVE_SR)
    model = _bp().to(audio.device)
    with torch.no_grad():
        return model.forward_raw(audio)


def multipitch_mse(ref: torch.Tensor, gen: torch.Tensor, sr: int = 16_000) -> dict[str, np.ndarray]:
    pr = _bp_post(ref, sr)
    pg = _bp_post(gen, sr)
    keys = ("onset", "contour", "note")
    per_metric = {}
    for k in keys:
        a, b = pr[k], pg[k]
        n = min(a.shape[-1], b.shape[-1])
        diff = (a[..., :n] - b[..., :n]).pow(2).mean(dim=(-2, -1))
        per_metric[f"multipitch_{k}"] = diff.detach().cpu().numpy().astype(np.float64)
    per_metric["multipitch_mean"] = (
        (per_metric["multipitch_onset"] + per_metric["multipitch_contour"] + per_metric["multipitch_note"]) / 3.0
    )
    return per_metric
