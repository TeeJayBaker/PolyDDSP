"""RawAudioDataset — split, reproducibility, crop shape."""
from __future__ import annotations

from pathlib import Path

import torch

from polyddsp.data import RawAudioDataset


def test_train_val_disjoint_and_sized(tmp_dataset: Path) -> None:
    train = RawAudioDataset(root=str(tmp_dataset), split="train", seed=0)
    val = RawAudioDataset(root=str(tmp_dataset), split="val", seed=0)
    # File-level split: 80/20 over 11 files = 8 train / 3 val.
    assert len(train.active) == 8 and len(val.active) == 3
    assert set(train.active).isdisjoint(set(val.active))
    # Virtual epoch = floor(total_duration / clip_seconds). 2.5s files, 4s clips
    # → 8 * 2.5 / 4 = 5 train clips, 3 * 2.5 / 4 = 1 val clip (floor, min 1).
    assert len(train) == 5 and len(val) == 1


def test_split_reproducible_across_seeds(tmp_dataset: Path) -> None:
    a = RawAudioDataset(root=str(tmp_dataset), split="train", seed=0)
    b = RawAudioDataset(root=str(tmp_dataset), split="train", seed=0)
    assert a.active == b.active


def test_getitem_shape(tmp_dataset: Path) -> None:
    ds = RawAudioDataset(root=str(tmp_dataset), split="train", seed=0)
    x = ds[0]
    assert isinstance(x, torch.Tensor) and x.shape == (64_000,) and x.dtype == torch.float32
