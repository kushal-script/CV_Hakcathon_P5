#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from PIL import Image

from p5.classical.features import extract
from p5.classical.saliency import classical_roi
from p5.data.split import read_ids


def one(job):
    import cv2
    cv2.setNumThreads(1)
    path, box = job
    rgb = np.asarray(Image.open(path).convert("RGB"))
    if box is None:
        roi = classical_roi(rgb)
        vec, _, _ = extract(rgb, roi.box, roi)
        return vec, list(roi.box)
    vec, roi, _ = extract(rgb, box)
    return vec, list(box)


def main() -> int:
    p = argparse.ArgumentParser(description="Classical features on detector boxes (pred) or a saliency ROI (roi)")
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--input", choices=["pred", "roi"], required=True)
    p.add_argument("--det", type=Path, default=None, help="det_ft_{split}.csv for --input pred")
    p.add_argument("--cache", type=Path, default=ROOT / "cache")
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    folder = "train" if args.split == "val" else "test"
    if args.split == "val":
        ids = sorted(read_ids(args.assets / "splits" / "val_ids.txt"))
    else:
        ids = sorted(pd.read_csv(args.cache / "index_test.csv").id.tolist())
    full = args.cache / "full" / folder
    boxes: dict[int, tuple | None] = {i: None for i in ids}
    if args.input == "pred":
        df = pd.read_csv(args.det).sort_values("score", ascending=False).drop_duplicates("id")
        top = {int(r.id): (r.x1, r.y1, r.x2, r.y2) for r in df.itertuples()}
        for i in ids:
            boxes[i] = top.get(i)
    jobs = [(full / f"{i:05d}.jpg", boxes[i]) for i in ids]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, jobs, chunksize=32))
    out = args.cache / "classical" / f"feats_{args.split}_{args.input}.npz"
    np.savez_compressed(out, x=np.stack([r[0] for r in res]), boxes=np.array([r[1] for r in res]), ids=np.array(ids))
    print(f"wrote {out} ({len(ids)} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
