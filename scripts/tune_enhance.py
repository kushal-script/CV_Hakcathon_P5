#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from PIL import Image

from p5.classical import preprocess as P
from p5.classical.quality import analyse, noise_sigma
from p5.classification.classifier import Classifier
from p5.data.corrupt import corrupt
from p5.data.split import read_ids
from p5.detection.detector import Detector

import cv2


def variants(rgb: np.ndarray, q) -> dict[str, np.ndarray]:
    out = {"raw": rgb}
    g = P.auto_gamma(rgb) if "dark" in q.flags else rgb
    out["gamma"] = g
    out["gamma_clahe"] = P.clahe_l(g, 1.5) if q.flags else rgb
    s = noise_sigma(cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)) / 255.0
    out["gamma_lightnlm_clahe"] = P.clahe_l(P.denoise(g, s, 0.4), 1.5) if q.flags else rgb
    out["current"] = P.enhance(rgb, q).rgb
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="E12: choose the quality-gated enhancement on corrupted val images")
    p.add_argument("--models", type=Path, default=ROOT / "models")
    p.add_argument("--cache", type=Path, default=ROOT / "cache")
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "e12_enhance")
    args = p.parse_args()

    det = Detector(args.models / "detector.onnx", threads=8)
    clf = Classifier(args.models / "classifier.onnx", args.models / "classifier_meta.json", threads=8)
    index = pd.read_csv(args.cache / "index_train.csv").set_index("id")
    ids = sorted(pd.Series(read_ids(args.assets / "splits" / "val_ids.txt")).sample(args.n, random_state=7).tolist())
    conds = [("clean", 0), ("lowlight", 1), ("lowlight", 3), ("lowlight", 5), ("gaussian_noise", 3), ("contrast", 3), ("contrast", 5)]
    names = ["raw", "gamma", "gamma_clahe", "gamma_lightnlm_clahe", "current", "select_raw_or_current"]
    res = {f"{c}_s{s}": {n: [0, 0] for n in names} for c, s in conds}
    for i in ids:
        y = int(index.loc[i, "class_id"])
        base = np.asarray(Image.open(args.cache / "full" / "train" / f"{i:05d}.jpg").convert("RGB"))
        for c, s in conds:
            rgb = base if c == "clean" else corrupt(base, c, s, seed=i)
            q = analyse(rgb)
            vs = variants(rgb, q)
            scores = {}
            for name, im in vs.items():
                d = det(im, conf=0.10)
                scores[name] = (d[0].score if d else 0.0, d[0].box if d else None, im)
            raw_s, cur_s = scores["raw"][0], scores["current"][0]
            scores["select_raw_or_current"] = scores["current"] if cur_s > raw_s else scores["raw"]
            for name in names:
                sc, box, im = scores[name]
                r = res[f"{c}_s{s}"][name]
                if box is not None:
                    r[0] += 1
                    r[1] += int(clf(Image.fromarray(im), box).logits.argmax() == y)
    table = {k: {n: {"detected": v[0] / args.n, "top1": v[1] / args.n} for n, v in d.items()} for k, d in res.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "enhance_tuning.json").write_text(json.dumps(table, indent=1))
    for k, d in table.items():
        print(k, " ".join(f"{n}={v['top1']:.3f}/{v['detected']:.2f}" for n, v in d.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
