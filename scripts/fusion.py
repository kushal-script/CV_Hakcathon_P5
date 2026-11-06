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

from p5.classical.svm import ClassicalModel
from p5.confidence.temperature import fit_temperature, nll
from p5.eval.classification import summary
from p5.eval.stats import mcnemar
from p5.hybrid.fusion import GRID, fuse
from p5.taxonomy import Taxonomy


def aligned(npz: Path, ids: np.ndarray, key: str) -> np.ndarray:
    d = np.load(npz)
    pos = {int(i): k for k, i in enumerate(d["ids"])}
    return d[key][[pos[int(i)] for i in ids]]


def main() -> int:
    p = argparse.ArgumentParser(description="E12: late fusion of deep logits with classical SVM probabilities (val only)")
    p.add_argument("--deep", type=Path, required=True, help="npz with val logits on detector crops")
    p.add_argument("--classical", type=Path, default=ROOT / "runs" / "e08_classical" / "classical_global.npz")
    p.add_argument("--feats", type=Path, default=ROOT / "cache" / "classical" / "feats_val_pred.npz")
    p.add_argument("--cache", type=Path, default=ROOT / "cache")
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    tax = Taxonomy.load(args.assets / "taxonomy")
    f = np.load(args.feats)
    ids = f["ids"]
    y = pd.read_csv(args.cache / "index_train.csv").set_index("id").loc[ids, "class_id"].to_numpy()
    z = aligned(args.deep, ids, "logits")
    cm = ClassicalModel.load(args.classical)
    pc = cm.proba(f["x"])
    t = fit_temperature(z, y)

    rows = {}
    for b in GRID:
        zf = fuse(z, pc, b)
        rows[str(b)] = {"nll": nll(zf, y, t), **{k: v for k, v in summary(zf, y, tax).items() if k in ("top1", "make_top1")}}
    best = min(GRID, key=lambda b: rows[str(b)]["nll"])
    zf = fuse(z, pc, best)
    test = mcnemar(zf.argmax(1) == y, z.argmax(1) == y)
    accept = best > 0 and test["p_value"] < 0.05 and rows[str(best)]["top1"] > rows["0.0"]["top1"]
    report = {"n": int(len(y)), "temperature": t, "beta_grid": rows, "best_beta": best,
              "mcnemar_fused_vs_deep": test, "fusion_accepted": bool(accept),
              "deep": summary(z, y, tax), "classical": summary(np.log(pc + 1e-9), y, tax), "fused": summary(zf, y, tax)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fusion.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("best_beta", "mcnemar_fused_vs_deep", "fusion_accepted")}))
    print(json.dumps({b: {k: round(v, 4) for k, v in r.items()} for b, r in rows.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
