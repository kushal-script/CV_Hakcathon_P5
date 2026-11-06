#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from PIL import Image, ImageOps

from p5.classification.crop import cache_crop
from p5.data import stanford

PAD, MAX_SIDE, FULL_SIDE = 0.16, 448, 512


def work(job: tuple[str, int, tuple, Path, Path]) -> tuple[int, float, float, float, float, int, int]:
    path, iid, box, crop_dir, full_dir = job
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        crop, inner = cache_crop(im, box, PAD, MAX_SIDE)
        crop.save(crop_dir / f"{iid:05d}.jpg", quality=95)
        s = min(1.0, FULL_SIDE / max(im.size))
        full = im if s == 1.0 else im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
        full.save(full_dir / f"{iid:05d}.jpg", quality=92)
        return (iid, *inner, im.width, im.height)


def main() -> int:
    p = argparse.ArgumentParser(description="Cache padded GT crops and downscaled full images")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "stanford_cars")
    p.add_argument("--out", type=Path, default=ROOT / "cache")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    root = stanford.find_root(args.data)
    for split in ("train", "test"):
        df = stanford.load(root, split)
        crop_dir, full_dir = args.out / "crops" / split, args.out / "full" / split
        crop_dir.mkdir(parents=True, exist_ok=True)
        full_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(r.path, r.id, (r.x1, r.y1, r.x2, r.y2), crop_dir, full_dir) for r in df.itertuples()]
        with ProcessPoolExecutor(args.workers) as ex:
            rows = list(ex.map(work, jobs, chunksize=64))
        index = pd.DataFrame(rows, columns=["id", "bx1", "by1", "bx2", "by2", "orig_w", "orig_h"])
        index = df.drop(columns=["path"]).merge(index, on="id")
        index.to_csv(args.out / f"index_{split}.csv", index=False)
        print(f"{split}: {len(index)} crops cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
