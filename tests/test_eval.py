"""eval: aggregate metrics + per-example parquet writer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from polyddsp.eval import save_per_example_parquet


def test_save_per_example_parquet_roundtrip(tmp_path: Path) -> None:
    per_ex = {
        "loudness_l1": np.array([0.1, 0.2, 0.3]),
        "mss": np.array([1.0, 1.5, 2.0]),
    }
    out = tmp_path / "per_example.parquet"
    save_per_example_parquet(per_ex, out)
    table = pq.read_table(out)
    assert table.column_names == ["loudness_l1", "mss"]
    np.testing.assert_array_equal(table.column("loudness_l1").to_numpy(), per_ex["loudness_l1"])
    np.testing.assert_array_equal(table.column("mss").to_numpy(), per_ex["mss"])
