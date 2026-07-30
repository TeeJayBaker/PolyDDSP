"""Sign checks for `MultiScaleSpectral`.

Asserts the log-magnitude term is *added* to the magnitude term: two signals
differing only in scale must give a strictly positive loss (a negative α would
let the log gap cancel the magnitude gap), and identical inputs must give zero.
"""
from __future__ import annotations

import torch

from polyddsp.losses import MultiScaleSpectral


def test_loss_is_positive_when_only_log_term_differs() -> None:
    """Spectrograms equal up to scale: mag term tiny, log term = log 2."""
    loss = MultiScaleSpectral()
    sr = 16_000
    t = torch.arange(sr) / sr
    a = torch.sin(2 * torch.pi * 440 * t).float().unsqueeze(0)
    b = a * 2.0
    value = loss(a, b)
    assert value.item() > 0.0, (
        "Loss must use +α on the log term; -α would cancel the log gap."
    )


def test_loss_is_zero_for_identical_inputs() -> None:
    loss = MultiScaleSpectral()
    sr = 16_000
    t = torch.arange(sr) / sr
    a = torch.sin(2 * torch.pi * 440 * t).float().unsqueeze(0)
    value = loss(a, a)
    assert value.item() == 0.0
