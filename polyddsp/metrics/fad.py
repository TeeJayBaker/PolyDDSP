"""Fréchet Audio Distance in VGGish embedding space.

Uses `audio-metrics`' AudioMetrics with the VGGish embedder (PyTorch port
via torch.hub). No TensorFlow.
"""
from __future__ import annotations

import functools

import numpy as np
import torch


@functools.lru_cache(maxsize=1)
def _make_metric():
    from audio_metrics import AudioMetrics
    from audio_metrics.embedders.vggish import VGGish

    embedder = VGGish()
    # win_dur gates minimum clip duration (empty ref set if clips < win_dur);
    # our pipeline runs on 4s clips, so match.
    return AudioMetrics(metrics=["fad"], embedder=embedder, input_sr=16_000, win_dur=4.0)


def fad(ref: torch.Tensor, gen: torch.Tensor, sr: int = 16_000) -> dict[str, np.ndarray]:
    metric = _make_metric()
    metric.reset_reference()
    metric.add_reference(ref.detach().cpu().numpy())
    # evaluate() returns a flat dict {"fad": value}, not nested by embedder name
    result = metric.evaluate(gen.detach().cpu().numpy())
    score = float(result["fad"])
    return {"fad": np.asarray([score], dtype=np.float64)}
