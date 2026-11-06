from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import scipy.io as sio


def find_root(start: Path) -> Path:
    if (start / "devkit" / "cars_meta.mat").exists():
        return start
    hits = sorted(start.rglob("cars_meta.mat"))
    if not hits:
        raise FileNotFoundError(f"cars_meta.mat not found under {start}")
    return hits[0].parent.parent


def class_names(root: Path) -> list[str]:
    meta = sio.loadmat(root / "devkit" / "cars_meta.mat", squeeze_me=True)["class_names"]
    return [str(n) for n in meta]


def load(root: Path, split: Literal["train", "test"], allow_test_labels: bool = False) -> pd.DataFrame:
    if split == "train":
        ann, img_dir = root / "devkit" / "cars_train_annos.mat", root / "cars_train"
    else:
        ann, img_dir = root / "cars_test_annos_withlabels.mat", root / "cars_test"
    rows = sio.loadmat(ann, squeeze_me=True)["annotations"]
    df = pd.DataFrame({
        "id": [int(Path(str(r["fname"])).stem) for r in rows],
        "path": [str(img_dir / str(r["fname"])) for r in rows],
        "class_id": [int(r["class"]) - 1 for r in rows],
        "x1": [int(r["bbox_x1"]) - 1 for r in rows],
        "y1": [int(r["bbox_y1"]) - 1 for r in rows],
        "x2": [int(r["bbox_x2"]) for r in rows],
        "y2": [int(r["bbox_y2"]) for r in rows],
    })
    if split == "test" and not allow_test_labels:
        df = df.drop(columns=["class_id"])
    return df.sort_values("id").reset_index(drop=True)
