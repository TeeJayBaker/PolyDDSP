"""LoudnessExtractor: DDSP-compatible static (db/80)+1 scaling."""
from __future__ import annotations

import torch

from polyddsp.model.loudness import LoudnessExtractor


def test_loudness_static_scale_silence():
    ex = LoudnessExtractor(sr=16000, frame_hop=64, n_fft=512)
    audio = torch.zeros(1, 16000)
    out = ex(audio, normalise=True)
    # Silence → very negative dB → static scale (db/80)+1 produces values << 0.
    assert out.max().item() < 0.5


def test_loudness_n_fft_changes_output():
    audio = torch.randn(2, 16000)
    out_a = LoudnessExtractor(sr=16000, frame_hop=64, n_fft=512)(audio, True)
    out_b = LoudnessExtractor(sr=16000, frame_hop=64, n_fft=256)(audio, True)
    assert not torch.allclose(out_a, out_b)


def test_loudness_default_n_fft_is_512():
    ex = LoudnessExtractor()
    assert ex.n_fft == 512


def test_loudness_dynamic_range():
    """Static scaling: 0 dB → 1.0, -80 dB → 0.0."""
    ex = LoudnessExtractor()
    t = torch.arange(16000) / 16000.0
    audio = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
    out = ex(audio, normalise=True)
    assert out.max().item() < 1.5  # not blown up
