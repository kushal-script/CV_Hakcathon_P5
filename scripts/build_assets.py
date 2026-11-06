#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from p5.data import split, stanford
from p5.taxonomy import Taxonomy


def check_boxes(df) -> int:
    bad = 0
    for r in df.itertuples():
        with Image.open(r.path) as im:
            w, h = im.size
        if not (0 <= r.x1 < r.x2 <= w and 0 <= r.y1 < r.y2 <= h):
            bad += 1
    return bad


def main() -> int:
    p = argparse.ArgumentParser(description="E01: build taxonomy and split assets and validate the dataset")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "stanford_cars")
    p.add_argument("--out", type=Path, default=ROOT / "assets")
    p.add_argument("--skip-box-check", action="store_true")
    args = p.parse_args()

    root = stanford.find_root(args.data)
    names = stanford.class_names(root)
    tax = Taxonomy.from_names(names)
    train = stanford.load(root, "train")
    test = stanford.load(root, "test")

    checks = {
        "train_count": (len(train), 8144),
        "test_count": (len(test), 8041),
        "classes": (len(names), 196),
        "train_classes": (train["class_id"].nunique(), 196),
        "first_class": (names[0], "AM General Hummer SUV 2000"),
        "makes": (len(tax.makes), 49),
        "display_makes": (len(tax.display_makes), 47),
        "families": (len(tax.family_names), 189),
    }
    for a, b in tax.year_twins:
        checks[f"year_twin_{a}_{b}"] = (tax.family_of[a] == tax.family_of[b], True)
    if not args.skip_box_check:
        checks["train_bad_boxes"] = (check_boxes(train), 0)
        checks["test_bad_boxes"] = (check_boxes(test), 0)

    failed = {k: v for k, v in checks.items() if v[0] != v[1]}
    for k, (got, want) in checks.items():
        print(f"{'ok  ' if got == want else 'FAIL'} {k}: {got} (expected {want})")
    if failed:
        return 1

    tr_ids, va_ids = split.make_split(train)
    split.assert_disjoint(tr_ids, va_ids)
    tax.save(args.out / "taxonomy")
    split.write_ids(args.out / "splits" / "val_ids.txt", va_ids)
    summary = {"train": len(tr_ids), "val": len(va_ids), "val_classes": int(train[train.id.isin(va_ids)].class_id.nunique())}
    (args.out / "splits" / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"split: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
