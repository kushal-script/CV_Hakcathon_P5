#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from PIL import Image

from p5 import render
from p5.classical.quality import analyse
from p5.classification.cam import class_activation_map, overlay
from p5.classification.classifier import Classifier
from p5.data import stanford


def main() -> int:
    p = argparse.ArgumentParser(description="Redraw failure-case figures from failures.json (no labels read)")
    p.add_argument("--failures", type=Path, default=ROOT / "runs" / "e15_test" / "failures" / "failures.json")
    p.add_argument("--pipeline", type=Path, default=ROOT / "runs" / "remote" / "work-runs" / "e15_pipeline" / "test_pipeline.jsonl")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "stanford_cars")
    p.add_argument("--models", type=Path, default=ROOT / "models")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    f = json.loads(args.failures.read_text())
    recs = {r["id"]: r for r in map(json.loads, args.pipeline.read_text().splitlines())}
    boxes = stanford.load(stanford.find_root(args.data), "test").set_index("id")
    clf = Classifier(args.models / "classifier.onnx", args.models / "classifier_meta.json", threads=4)
    fc = np.load(args.models / "head.npz")["fc_weight"]
    out = args.out or args.failures.parent
    for key, items in f["examples"].items():
        for e in items:
            i, rec, row = e["id"], recs[e["id"]], boxes.loc[e["id"]]
            img = Image.open(row.path).convert("RGB")
            rgb = np.asarray(img).copy()
            render.draw_box(rgb, (row.x1, row.y1, row.x2, row.y2), "truth: " + e["truth"], (46, 160, 67), below=True)
            panels = []
            if rec["box"] is not None:
                s = min(1.0, 1600 / max(img.size))
                if analyse(np.asarray(img)).flags and max(img.size) * s > 1024:
                    s = 1024 / max(img.size)
                box = [v / s for v in rec["box"]]
                render.draw_box(rgb, box, f"pred: {rec['label']} {rec['confidence']:.0%}", (207, 34, 46))
                d = clf(img, box)
                cam = class_activation_map(d.features, fc, int(d.logits.argmax()), d.crop.shape[1::-1])
                panels.append(cv2.resize(overlay(d.crop, cam), (260, 260)))
            panels.insert(0, render.fit(rgb, 560))
            h = max(x.shape[0] for x in panels)
            canvas = np.full((h, sum(x.shape[1] for x in panels) + 10 * (len(panels) - 1), 3), 255, np.uint8)
            x0 = 0
            for pnl in panels:
                canvas[:pnl.shape[0], x0:x0 + pnl.shape[1]] = pnl
                x0 += pnl.shape[1] + 10
            Image.fromarray(canvas).save(out / f"{key}_{i:05d}.jpg", quality=90)
    print(f"figures written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
