#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from PIL import Image

from p5.classical.features import blocks, extract
from p5.classical.svm import fit
from p5.confidence.temperature import fit_temperature
from p5.data.split import read_ids
from p5.eval.classification import summary
from p5.taxonomy import Taxonomy


def one(job: tuple[Path, tuple]) -> np.ndarray:
    import cv2
    cv2.setNumThreads(1)
    path, box = job
    rgb = np.asarray(Image.open(path).convert("RGB"))
    vec, roi, geo = extract(rgb, box)
    return vec


def features(index: pd.DataFrame, crop_dir: Path, out: Path, workers: int) -> np.ndarray:
    if out.exists():
        return np.load(out)["x"]
    jobs = [(crop_dir / f"{int(r.id):05d}.jpg", (r.bx1, r.by1, r.bx2, r.by2)) for r in index.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        x = np.stack(list(ex.map(one, jobs, chunksize=32)))
    np.savez_compressed(out, x=x, ids=index.id.to_numpy())
    return x


def main() -> int:
    p = argparse.ArgumentParser(description="E08: classical colour/texture/shape features with a linear SVM")
    p.add_argument("--cache", type=Path, default=ROOT / "cache")
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "e08_classical")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--components", type=int, default=256)
    p.add_argument("--c", type=float, default=0.05)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tax = Taxonomy.load(args.assets / "taxonomy")
    index = pd.read_csv(args.cache / "index_train.csv")
    test_index = pd.read_csv(args.cache / "index_test.csv")
    feat_dir = args.cache / "classical"
    feat_dir.mkdir(parents=True, exist_ok=True)
    x_all = features(index, args.cache / "crops" / "train", feat_dir / "feats_train_gt.npz", args.workers)
    features(test_index, args.cache / "crops" / "test", feat_dir / "feats_test_gt.npz", args.workers)

    val_ids = set(read_ids(args.assets / "splits" / "val_ids.txt"))
    is_val = index.id.isin(val_ids).to_numpy()
    y = index.class_id.to_numpy()
    xtr, ytr, xva, yva = x_all[~is_val], y[~is_val], x_all[is_val], y[is_val]

    grid = {}
    for c in (0.005, 0.02, 0.05, 0.2):
        grid[c] = summary(fit(xtr, ytr, args.components, c).scores(xva), yva, tax)["top1"]
        print(json.dumps({"C": c, "val_top1": grid[c]}), flush=True)
    args.c = max(grid, key=grid.get)
    model = fit(xtr, ytr, args.components, args.c)
    model.temperature = fit_temperature(model.scores(xva), yva)
    model.save(args.out / "classical_global.npz")
    s = model.scores(xva)
    np.savez_compressed(args.out / "val_scores.npz", scores=s, labels=yva, ids=index.id.to_numpy()[is_val])
    result = {"split": "val", "input": "gt", "all": summary(s, yva, tax), "temperature": model.temperature,
              "c": args.c, "c_grid_val_top1": grid}

    ablation = {}
    b = blocks()
    for name, sl in b.items():
        keep = np.ones(x_all.shape[1], bool)
        keep[sl] = False
        m = fit(xtr[:, keep], ytr, args.components, args.c)
        ablation[f"without_{name}"] = summary(m.scores(xva[:, keep]), yva, tax)["top1"]
        only = fit(xtr[:, sl], ytr, min(args.components, sl.stop - sl.start), args.c)
        ablation[f"only_{name}"] = summary(only.scores(xva[:, sl]), yva, tax)["top1"]
        print(json.dumps({name: {"without": ablation[f"without_{name}"], "only": ablation[f"only_{name}"]}}), flush=True)
    result["block_ablation_top1"] = ablation
    (args.out / "metrics.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["all"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
