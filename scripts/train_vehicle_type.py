#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import scipy.io
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from p5.classification.classifier import Classifier
from p5.detection.detector import Detector
from p5.video.analyse import embedding

VEHICLES = {2, 3, 5, 7}


def video_crop(rgb: np.ndarray, box, pad: float, size: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    c = rgb[int(max(0, y1 - pad * bh)):int(min(rgb.shape[0], y2 + pad * bh)), int(max(0, x1 - pad * bw)):int(min(rgb.shape[1], x2 + pad * bw))]
    return np.asarray(Image.fromarray(c).resize((size, size), Image.BILINEAR))


def features(clf: Classifier, items: list[tuple[np.ndarray, tuple]], pad: float, batch: int = 16) -> np.ndarray:
    out = []
    for k in range(0, len(items), batch):
        crops = []
        for rgb, box in items[k:k + batch]:
            crops.append(clf.prepare(Image.fromarray(rgb), box))
            crops.append(video_crop(rgb, box, pad, clf.size))
        _, f = clf.run(crops)
        out += [embedding(x) for x in f]
    return np.asarray(out, dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description="Train the auto-rickshaw check on classifier embeddings")
    p.add_argument("--autos", type=Path, default=ROOT / "data/autorickshaw")
    p.add_argument("--models", type=Path, default=ROOT / "models")
    p.add_argument("--out", type=Path, default=ROOT / "models/vehicle_type.json")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--india", type=int, default=2000)
    p.add_argument("--stanford", type=int, default=1000)
    p.add_argument("--max_fpr", type=float, default=0.005)
    p.add_argument("--pad", type=float, default=0.10)
    args = p.parse_args()
    rng = random.Random(22)
    t0 = time.time()
    det = Detector(args.models / "detector_coco.onnx", threads=6)
    clf = Classifier(args.models / "classifier.onnx", args.models / "classifier_meta.json", 6)

    rows = list(csv.DictReader((args.autos / "manifest.csv").open()))
    autos, groups, kept = [], [], []
    for r in rows:
        path = args.autos / "raw" / r["file"]
        if not path.exists():
            continue
        rgb = np.asarray(Image.open(path).convert("RGB"))
        dets = det(rgb, conf=0.25, classes=VEHICLES)
        if not dets:
            continue
        d = max(dets, key=lambda x: (x.box[2] - x.box[0]) * (x.box[3] - x.box[1]))
        if (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) < 0.05 * rgb.shape[0] * rgb.shape[1]:
            continue
        autos.append((rgb, tuple(d.box)))
        groups.append(r["artist"] or r["file"])
        kept.append(r["file"])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=22)
    tr, rest = next(gss.split(autos, groups=groups))
    rest_groups = [groups[i] for i in rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=22)
    va_i, te_i = next(gss2.split(rest, groups=rest_groups))
    auto_split = {"train": list(tr), "val": [rest[i] for i in va_i], "test": [rest[i] for i in te_i]}

    india = [r for r in csv.DictReader((ROOT / "assets/india/split.csv").open())]
    rng.shuffle(india)
    india_by = {s: [r for r in india if r["split"] == s] for s in ("train", "val", "test")}
    n = {"train": int(args.india * 0.7), "val": int(args.india * 0.15), "test": int(args.india * 0.15)}

    def india_items(rs):
        out = []
        for r in rs:
            f = ROOT / "data/india/crops" / r["file"]
            if f.exists():
                rgb = np.asarray(Image.open(f).convert("RGB"))
                out.append((rgb, (0, 0, rgb.shape[1], rgb.shape[0])))
        return out

    sc = ROOT / "data/stanford_cars"
    annos = scipy.io.loadmat(sc / "devkit/cars_train_annos.mat")["annotations"][0]
    box = {int(Path(str(a["fname"][0])).stem): tuple(int(a[k][0][0]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")) for a in annos}
    val_ids = [int(x) for x in (ROOT / "assets/splits/val_ids.txt").read_text().split()]
    train_ids = sorted(set(box) - set(val_ids))
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    st = {"train": train_ids[:int(args.stanford * 0.7)], "val": val_ids[:int(args.stanford * 0.15)],
          "test": val_ids[int(args.stanford * 0.15):int(args.stanford * 0.30)]}

    def stanford_items(ids):
        return [(np.asarray(Image.open(sc / "cars_train" / f"{i:05d}.jpg").convert("RGB")), box[i]) for i in ids]

    X, y, src = {}, {}, {}
    for s in ("train", "val", "test"):
        a = features(clf, [autos[i] for i in auto_split[s]], args.pad)
        ni = features(clf, india_items(india_by[s][:n[s]]), args.pad)
        ns = features(clf, stanford_items(st[s]), args.pad)
        X[s] = np.concatenate([a, ni, ns])
        y[s] = np.concatenate([np.ones(len(a)), np.zeros(len(ni) + len(ns))])
        src[s] = np.array(["auto"] * len(a) + ["india"] * len(ni) + ["stanford"] * len(ns))
        print(s, {k: int((src[s] == k).sum()) for k in ("auto", "india", "stanford")}, f"{time.time() - t0:.0f} s", flush=True)

    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
    lr.fit(X["train"], y["train"])
    sv = lr.predict_proba(X["val"])[:, 1]
    neg = np.sort(sv[y["val"] == 0])[::-1]
    k = int(np.floor(args.max_fpr * len(neg)))
    threshold = float(min(0.99, max(0.5, neg[k] + 1e-6)))

    def rates(s):
        pr = lr.predict_proba(X[s])[:, 1] >= threshold
        out = {"autos_found": float(pr[src[s] == "auto"].mean()), "indian_cars_called_auto": float(pr[src[s] == "india"].mean()),
               "us_cars_called_auto": float(pr[src[s] == "stanford"].mean()),
               "n": {k: int((src[s] == k).sum()) for k in ("auto", "india", "stanford")}}
        return out

    metrics = {"threshold": threshold, "val": rates("val"), "test": rates("test"),
               "autos_kept": len(autos), "autos_listed": len(rows), "photographers": len(set(groups)), "note": "two crop styles per image (photo and video)"}
    args.out.write_text(json.dumps({"kind": "logistic regression on the Stanford Cars classifier embedding", "positive": "auto-rickshaw",
                                    "coef": [round(float(v), 7) for v in lr.coef_[0]], "intercept": float(lr.intercept_[0]),
                                    "threshold": threshold, "metrics": metrics}, indent=1))
    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (args.report / "autos_used.txt").write_text("\n".join(kept) + "\n")
    print(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
