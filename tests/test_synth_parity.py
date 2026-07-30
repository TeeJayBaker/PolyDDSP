"""AdditiveSynth: DDSP-compatible (exp_sigmoid + normalize + sample-rate Nyquist mask)."""
from __future__ import annotations

import torch

from polyddsp.model.additive import AdditiveSynth


def test_additive_normalises_and_masks_above_nyquist():
    sr, hop, H = 16000, 64, 4
    synth = AdditiveSynth(sr=sr, frame_hop=hop, n_harmonics=H)
    B, V, T = 1, 1, 4
    # pitch=4000 → harmonics 4000, 8000, 12000, 16000. Only h=1 keeps (>= nyquist drops).
    pitch = torch.full((B, V, T), 4000.0)
    raw_h = torch.zeros(B, V, T, H)  # exp_sigmoid → uniform → only h=1 survives mask
    amp_v = torch.full((B, V, T), 0.5)
    audio = synth(pitch=pitch, harm_dist=raw_h, amp_v=amp_v)
    assert audio.shape == (B, T * hop)
    # After normalize_harmonics + Nyquist mask, only h=1 active and sums to 1 →
    # full envelope ≈ amp_v · 1.0 = 0.5; sine peak ≤ 0.5.
    assert audio.abs().max().item() < 0.6


def test_additive_silence_when_pitch_zero():
    sr, hop, H = 16000, 64, 4
    synth = AdditiveSynth(sr=sr, frame_hop=hop, n_harmonics=H)
    B, V, T = 1, 1, 4
    # f0=0 → omegas=0 → phase=0 → sin(0)=0.
    pitch = torch.zeros(B, V, T)
    raw_h = torch.zeros(B, V, T, H)
    amp_v = torch.full((B, V, T), 0.5)
    audio = synth(pitch=pitch, harm_dist=raw_h, amp_v=amp_v)
    assert audio.abs().max().item() == 0.0
