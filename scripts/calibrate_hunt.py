#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from p5.classical.colour import vehicle_colour
from p5.classification.classifier import Classifier
from p5.classification.crop import jitter_box, make_crop
from p5.confidence.policy import Thresholds
from p5.confidence.temperature import softmax_t
from p5.data.split import read_ids
from p5.investigate.hunt import colour_match, features
from p5.video.analyse import embedding

FEATURES = ("appearance", "model", "colour")


def camera_view(img: Image.Image, box, size: int, rng: np.random.Generator) -> Image.Image:
    crop = make_crop(img, jitter_box(box, 0.12, 0.06, rng), size)
    if rng.random() < 0.5:
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
    f = rng.uniform(0.2, 0.6)
    crop = crop.resize((max(16, int(size * f)), max(16, int(size * f))), Image.BILINEAR).resize((size, size), Image.BILINEAR)
    a = np.asarray(crop).astype(np.float32) / 255.0
    a = a ** rng.uniform(0.6, 1.6) * rng.uniform(0.85, 1.15, 3)
    crop = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        crop = crop.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=int(rng.integers(25, 70)))
    return Image.open(buf).convert("RGB")


def profile(clf: Classifier, th: Thresholds, crops: list[np.ndarray]) -> list[dict]:
    out = []
    for k in range(0, len(crops), 16):
        z, f = clf.run(crops[k:k + 16])
        for zi, fi, c in zip(z, f, crops[k:k + 16]):
            out.append({"embedding": embedding(fi).tolist(), "probs": softmax_t(zi, th.t_d).tolist(), "colour": vehicle_colour(c)})
    return out


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 100) -> tuple[np.ndarray, float]:
    xb = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(xb.shape[1])
    reg = l2 * np.eye(xb.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-xb @ w))
        g = xb.T @ (p - y) / len(y) + reg @ w
        hess = (xb * (p * (1 - p))[:, None]).T @ xb / len(y) + reg
        step = np.linalg.solve(hess + 1e-9 * np.eye(len(w)), g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w[:-1], float(w[-1])


def auc(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def main() -> int:
    p = argparse.ArgumentParser(description="Learn vehicle-hunt match weights from simulated second-camera views of validation cars")
    p.add_argument("--cache", type=Path, default=ROOT / "cache")
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--models", type=Path, default=ROOT / "models")
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--fpr", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=ROOT / "configs" / "hunt_model.json")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    index = pd.read_csv(args.cache / "index_train.csv").set_index("id")
    val = sorted(read_ids(args.assets / "splits" / "val_ids.txt"))
    ids = rng.choice(val, size=min(args.n, len(val)), replace=False)
    clf = Classifier(args.models / "classifier.onnx", args.models / "classifier_meta.json", threads=8)
    th = Thresholds.load(args.models / "thresholds.yaml")
    ref, cam, labels = [], [], []
    for i in ids:
        r = index.loc[i]
        img = Image.open(args.cache / "crops" / "train" / f"{i:05d}.jpg").convert("RGB")
        box = (r.bx1, r.by1, r.bx2, r.by2)
        ref.append(np.asarray(make_crop(img, box, clf.size)))
        cam.append(np.asarray(camera_view(img, box, clf.size, rng)))
        labels.append(int(r.class_id))
    labels = np.asarray(labels)
    a, b = profile(clf, th, ref), profile(clf, th, cam)

    half = np.zeros(len(ids), bool)
    half[rng.permutation(len(ids))[: len(ids) // 2]] = True
    by_class = {c: np.flatnonzero(labels == c) for c in np.unique(labels)}

    def pairs(mask: np.ndarray):
        rows, ys, kinds = [], [], []
        idx = np.flatnonzero(mask)
        allowed = set(idx.tolist())
        for i in idx:
            rows.append(features(a[i], b[i])); ys.append(1); kinds.append("same vehicle")
            same = [j for j in by_class[labels[i]] if j != i and j in allowed]
            if same:
                j = int(rng.choice(same))
                rows.append(features(a[i], b[j])); ys.append(0); kinds.append("same model, other vehicle")
            for j in rng.choice(idx[labels[idx] != labels[i]], size=3, replace=False):
                rows.append(features(a[i], b[j])); ys.append(0); kinds.append("other model")
        x = np.array([[r[k] for k in FEATURES] for r in rows])
        return x, np.array(ys), np.array(kinds)

    x_fit, y_fit, _ = pairs(half)
    x_ev, y_ev, k_ev = pairs(~half)
    w, bias = fit_logistic(x_fit, y_fit)
    prob = lambda x: 1 / (1 + np.exp(-(x @ w + bias)))
    p_fit, p_ev = prob(x_fit), prob(x_ev)
    threshold = float(np.quantile(p_fit[y_fit == 0], 1 - args.fpr))
    report = {
        "pairs_fit": int(len(y_fit)), "pairs_eval": int(len(y_ev)), "fpr_target": args.fpr,
        "auc_eval": auc(p_ev, y_ev),
        "auc_eval_appearance_only": auc(x_ev[:, 0], y_ev),
        "tpr_eval_at_threshold": float((p_ev[y_ev == 1] >= threshold).mean()),
        "fpr_eval_at_threshold": float((p_ev[y_ev == 0] >= threshold).mean()),
        "fpr_eval_by_kind": {k: float((p_ev[(y_ev == 0) & (k_ev == k)] >= threshold).mean()) for k in np.unique(k_ev[y_ev == 0])},
        "colour_agreement_same_vehicle": float(x_ev[y_ev == 1, 2].mean()),
        "tpr_eval_at_fpr": {str(f): float((p_ev[y_ev == 1] >= np.quantile(p_fit[y_fit == 0], 1 - f)).mean()) for f in (0.01, 0.05, 0.10)},
        "auc_eval_without_colour": auc(x_ev[:, :2] @ fit_logistic(x_fit[:, :2], y_fit)[0], y_ev),
    }
    model = {"features": list(FEATURES), "weights": {k: float(v) for k, v in zip(FEATURES, w)}, "bias": bias,
             "threshold": threshold, "source": "scripts/calibrate_hunt.py", "seed": args.seed, "n_vehicles": int(len(ids)),
             "report": report}
    args.out.write_text(json.dumps(model, indent=1))
    print(json.dumps(model, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
