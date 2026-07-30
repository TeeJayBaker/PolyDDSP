"""LAION CLAP-based audio-similarity metrics.

- `clap_fd`  set-vs-set Fréchet distance in CLAP embedding space (distributional).
- `clap_cos` per-sample cosine similarity between ref and pred CLAP embeddings.

Both share the same LaionCLAP backbone (music-trained checkpoint), reached two
different ways: `clap_fd` via `audio-metrics`' AudioMetrics wrapper, `clap_cos`
via `laion_clap` directly (we need raw embeddings, AudioMetrics only exposes
distributional outputs).
"""
from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF


@functools.lru_cache(maxsize=1)
def _get_embedder():
    """LaionCLAP loaded once from audio_metrics' cached music checkpoint."""
    from audio_metrics.embedders.clap import LaionCLAP

    return LaionCLAP()


@functools.lru_cache(maxsize=1)
def _make_metric():
    from audio_metrics import AudioMetrics

    # win_dur gates minimum clip duration (empty ref set if clips < win_dur);
    # our pipeline runs on 4s clips, so match.
    return AudioMetrics(metrics=["fad"], embedder=_get_embedder(), input_sr=16_000, win_dur=4.0)


def clap_fd(ref: torch.Tensor, gen: torch.Tensor, sr: int = 16_000) -> dict[str, np.ndarray]:
    metric = _make_metric()
    metric.reset_reference()
    metric.add_reference(ref.detach().cpu().numpy())
    # evaluate() returns a flat dict {"fad": value}, not nested by embedder name
    result = metric.evaluate(gen.detach().cpu().numpy())
    score = float(result["fad"])
    return {"clap_fd": np.asarray([score], dtype=np.float64)}


def _clap_embed(audio: torch.Tensor, sr: int) -> torch.Tensor:
    """Run the CLAP audio encoder on a (B, T) waveform; return (B, D) embeddings.

    Reuses the same LaionCLAP instance as `clap_fd` — no second checkpoint
    download, no second copy of weights on GPU.
    """
    embedder = _get_embedder()
    clap = embedder.clap
    device = next(clap.parameters()).device
    if sr != 48_000:
        audio = AF.resample(audio, orig_freq=sr, new_freq=48_000)
    audio = audio.to(device)
    with torch.no_grad():
        emb = clap.get_audio_embedding_from_data(x=audio, use_tensor=True)
    return emb


def clap_cos(ref: torch.Tensor, gen: torch.Tensor, sr: int = 16_000) -> dict[str, np.ndarray]:
    """Per-sample cosine similarity between CLAP audio embeddings of ref and gen."""
    ref_e = F.normalize(_clap_embed(ref, sr), dim=-1)
    gen_e = F.normalize(_clap_embed(gen, sr), dim=-1)
    cos = (ref_e * gen_e).sum(dim=-1)
    return {"clap_cos": cos.detach().cpu().numpy().astype(np.float64)}
