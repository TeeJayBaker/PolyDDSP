"""Stats helpers — mean and SEM."""
from __future__ import annotations

import math

import numpy as np

from polyddsp.stats import format_cell, mean_sem


def test_mean_sem_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500).astype(np.float64)
    m, s = mean_sem(x)
    assert math.isclose(m, float(x.mean()), rel_tol=1e-12, abs_tol=1e-12)
    expected_sem = float(x.std(ddof=1) / math.sqrt(len(x)))
    assert math.isclose(s, expected_sem, rel_tol=1e-12, abs_tol=1e-12)


def test_format_cell_uses_three_decimals() -> None:
    x = np.array([1.0, 2.0, 3.0])
    cell = format_cell(x)
    assert "±" in cell
    assert cell.startswith("2.000")
