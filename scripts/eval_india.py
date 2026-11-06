#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from eval_test import log_call
from p5.confidence.policy import Thresholds, decide
from p5.confidence.selective import accuracy_at_coverage, risk_coverage
from p5.confidence.temperature import ece, softmax_t
from p5.eval.classification import summary
from p5.eval.stats import mcnemar
from p5.taxonomy import Taxonomy


def boot_ci(correct: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def evaluate(run: Path, tax: Taxonomy, th: Thresholds) -> tuple[dict, np.ndarray, pd.DataFrame]:
    d = np.load(run / "logits" / "test_gt.npz", allow_pickle=True)
    z, y = d["logits"], d["labels"]
    p = softmax_t(z, th.t_d)
    conf, correct = p.max(1), p.argmax(1) == y
    _, _, aurc = risk_coverage(conf, correct)
    per_class = pd.DataFrame({"class_id": y, "correct": correct}).groupby("class_id").correct.agg(["mean", "count"])
    per_class["name"] = [tax.names[i] for i in per_class.index]
    levels, level_ok = Counter(), Counter()
    mk = np.asarray(tax.display_make_of)
    for zi, yi in zip(z, y):
        dec = decide(zi, tax, th)
        levels[dec.level] += 1
        ok = {"class": dec.class_id == yi, "family": dec.class_id == yi, "make": mk[dec.class_id] == mk[yi],
              "ood_make": mk[dec.class_id] == mk[yi]}.get(dec.level)
        if ok is not None:
            level_ok[dec.level] += int(ok)
    e, m = ece(z, y, th.t_d)
    report = {
        "n": int(len(y)), "classes_in_test": int(len(per_class)),
        **{k: v for k, v in summary(z, y, tax).items() if k in ("top1", "top5", "make_top1")},
        "top1_ci95": boot_ci(correct.astype(float)),
        "macro_top1": float(per_class["mean"].mean()),
        "flip_tta_top1": float(((d["logits"] + d["logits_flip"]).argmax(1) == y).mean()),
        "ece": e, "mce": m, "aurc": aurc,
        "accuracy_at_coverage": {str(c): accuracy_at_coverage(conf, correct, c) for c in (0.5, 0.8, 0.9, 1.0)},
        "levels": dict(levels), "level_accuracy": {k: level_ok[k] / levels[k] for k in level_ok if levels[k]},
    }
    return report, correct, per_class.sort_values("mean")


def main() -> int:
    p = argparse.ArgumentParser(description="E18 Indian-market classifier: one-time evaluation on the photographer-disjoint test split")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--thresholds", type=Path, nargs="+", required=True)
    p.add_argument("--assets", type=Path, default=ROOT / "assets" / "india")
    p.add_argument("--us-transfer", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    log_call(args)
    tax = Taxonomy.load(args.assets / "taxonomy")
    args.out.mkdir(parents=True, exist_ok=True)
    reports, corrects = {}, {}
    for run, thp in zip(args.runs, args.thresholds):
        rep, corr, per_class = evaluate(run, tax, Thresholds.load(thp))
        reports[run.name], corrects[run.name] = rep, corr
        per_class.to_csv(args.out / f"per_class_{run.name}.csv")
    names = list(corrects)
    if len(names) == 2:
        a, b = corrects[names[0]], corrects[names[1]]
        reports["mcnemar"] = {"arms": names, **mcnemar(a, b)}
    if args.us_transfer and args.us_transfer.exists():
        reports["us_model_on_india_test"] = json.loads(args.us_transfer.read_text())
    (args.out / "test_report.json").write_text(json.dumps(reports, indent=1))
    print(json.dumps(reports, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
