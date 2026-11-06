#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from p5.confidence.policy import Thresholds, decide
from p5.confidence.selective import accuracy_at_coverage, min_threshold, risk_coverage
from p5.confidence.temperature import ece, fit_temperature, softmax_t
from p5.taxonomy import Taxonomy

TARGETS = {"class": 0.97, "family": 0.97, "make": 0.95, "ood_tpr": 0.95}


def group_conf(p: np.ndarray, group_of: list) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(group_of)
    top = p.argmax(1)
    conf = np.array([p[i, g == g[t]].sum() for i, t in enumerate(top)])
    return conf, g[top]


def thresholds(z: np.ndarray, y: np.ndarray, tax: Taxonomy, t: float) -> dict[str, float]:
    p = softmax_t(z, t)
    top = p.argmax(1)
    fam_conf, fam_pred = group_conf(p, tax.family_of)
    make_conf, make_pred = group_conf(p, tax.display_make_of)
    fam, mk = np.asarray(tax.family_of), np.asarray(tax.display_make_of)
    return {
        "tau_hi": min_threshold(p.max(1), top == y, TARGETS["class"]),
        "tau_fam": min_threshold(fam_conf, fam_pred == fam[y], TARGETS["family"]),
        "tau_make": min_threshold(make_conf, make_pred == mk[y], TARGETS["make"]),
        "tau_ood": float(np.percentile(z.max(1), 100 * (1 - TARGETS["ood_tpr"]))),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="E13: temperature scaling, selective thresholds and confidence report")
    p.add_argument("--logits", type=Path, required=True, help="npz with logits, labels (val split)")
    p.add_argument("--key", default="logits")
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--tau-lo", type=float, default=0.10)
    p.add_argument("--tau-det", type=float, default=0.25)
    args = p.parse_args()

    tax = Taxonomy.load(args.assets / "taxonomy")
    d = np.load(args.logits)
    z, y = d[args.key], d["labels"]
    t = fit_temperature(z, y)
    e_before, m_before = ece(z, y, 1.0)
    e_after, m_after = ece(z, y, t)
    th = thresholds(z, y, tax, t)

    rng = np.random.default_rng(42)
    boots = []
    for _ in range(args.boot):
        i = rng.integers(0, len(y), len(y))
        tb = fit_temperature(z[i], y[i])
        boots.append(thresholds(z[i], y[i], tax, tb))
    ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))]
          for k in th}

    probs = softmax_t(z, t)
    conf, correct = probs.max(1), probs.argmax(1) == y
    _, _, aurc = risk_coverage(conf, correct)
    final = Thresholds(tau_lo=args.tau_lo, tau_det=args.tau_det, t_d=t, **th)
    levels = Counter()
    level_correct = Counter()
    fam, mk = np.asarray(tax.family_of), np.asarray(tax.display_make_of)
    for zi, yi in zip(z, y):
        dec = decide(zi, tax, final)
        levels[dec.level] += 1
        ok = {"class": dec.class_id == yi, "family": fam[dec.class_id] == fam[yi],
              "make": mk[dec.class_id] == mk[yi], "ood_make": mk[dec.class_id] == mk[yi]}.get(dec.level)
        if ok is not None:
            level_correct[dec.level] += int(ok)

    report = {
        "n": int(len(y)), "temperature": t, "ece_before": e_before, "ece_after": e_after,
        "mce_before": m_before, "mce_after": m_after, "aurc": aurc, "targets": TARGETS,
        "thresholds": th, "bootstrap_ci_95": ci,
        "accuracy_at_coverage": {str(c): accuracy_at_coverage(conf, correct, c) for c in (0.8, 0.9, 0.95, 1.0)},
        "levels": dict(levels),
        "level_accuracy": {k: level_correct[k] / levels[k] for k in level_correct if levels[k]},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    final.save(args.out / "thresholds.yaml")
    (args.out / "calibration.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
