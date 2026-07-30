"""Shared test fixtures."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch


SR = 16_000
T_SAMPLES = 64_000  # 4 s @ 16 kHz
T_FRAMES = 1_000  # T_SAMPLES // 64


@pytest.fixture
def tiny_audio() -> torch.Tensor:
    """1 s of 440 Hz sine at 16 kHz, batch=2."""
    t = torch.arange(SR) / SR
    sine = torch.sin(2 * torch.pi * 440 * t).float()
    return sine.unsqueeze(0).repeat(2, 1)


@pytest.fixture
def four_s_sine() -> torch.Tensor:
    """4 s of 440 Hz sine, batch=2 (matches T_SAMPLES)."""
    t = torch.arange(T_SAMPLES) / SR
    sine = torch.sin(2 * torch.pi * 440 * t).float()
    return sine.unsqueeze(0).repeat(2, 1)


@pytest.fixture
def fake_features() -> dict:
    """Pre-built feature dict at correct shapes for V=2."""
    B, V, T = 2, 2, T_FRAMES
    return {
        "pitch": torch.full((B, V, T), 440.0),
        "velocity": torch.full((B, V, T), 0.5),
        "loudness": torch.zeros(B, T),
        "z": torch.zeros(B, T, 16),
    }


@pytest.fixture
def tmp_dataset(tmp_path: Path) -> Path:
    """Write 11 short .wav files with varying contents."""
    rng = np.random.default_rng(0)
    for i in range(11):
        wav = rng.standard_normal(int(2.5 * SR)).astype(np.float32) * 0.1
        sf.write(tmp_path / f"clip_{i:02d}.wav", wav, SR)
    return tmp_path
