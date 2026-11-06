from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def make_split(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42) -> tuple[list[int], list[int]]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr, va = next(sss.split(df["id"], df["class_id"]))
    return sorted(df["id"].iloc[tr].tolist()), sorted(df["id"].iloc[va].tolist())


def assert_disjoint(train_ids: Iterable[int], val_ids: Iterable[int]) -> None:
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise AssertionError(f"{len(overlap)} ids are in both train and val")


def read_ids(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().split()]


def write_ids(path: Path, ids: Iterable[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(i) for i in ids) + "\n")
