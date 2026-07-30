"""Mean ± SEM helpers for table reporting."""
from __future__ import annotations

import math

import numpy as np


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    """Return (mean, standard error of the mean) for a 1-D array."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    n = arr.size
    if n < 2:
        return float(arr.mean()) if n else float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(n))


def format_cell(values: np.ndarray, fmt: str = "{:.3f}") -> str:
    """Render `mean ± sem` for a table cell."""
    m, s = mean_sem(values)
    return f"{fmt.format(m)} ± {fmt.format(s)}"
