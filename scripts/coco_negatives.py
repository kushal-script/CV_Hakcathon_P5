#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VEHICLE_CATS = {3, 4, 6, 8}
SIZES = {"train": 500, "dev": 100, "test": 100}


def main() -> int:
    p = argparse.ArgumentParser(description="Sample COCO val2017 images without vehicles as detector negatives")
    p.add_argument("--coco", type=Path, required=True, help="folder with val2017/ and annotations/instances_val2017.json")
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--out", type=Path, required=True, help="where to copy the sampled images")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ann = json.loads((args.coco / "annotations" / "instances_val2017.json").read_text())
    with_vehicle = {a["image_id"] for a in ann["annotations"] if a["category_id"] in VEHICLE_CATS}
    candidates = sorted(im["file_name"] for im in ann["images"] if im["id"] not in with_vehicle)
    random.Random(args.seed).shuffle(candidates)
    start = 0
    for name, n in SIZES.items():
        files = sorted(candidates[start:start + n])
        start += n
        (args.assets / "splits").mkdir(parents=True, exist_ok=True)
        (args.assets / "splits" / f"coco_neg_{name}.txt").write_text("\n".join(files) + "\n")
        dest = args.out / name
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(args.coco / "val2017" / f, dest / f)
        print(f"{name}: {len(files)} negatives")
    return 0


if __name__ == "__main__":
    sys.exit(main())
