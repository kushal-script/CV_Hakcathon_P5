#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from p5.classical.svm import ClassicalModel
from p5.confidence.policy import Thresholds, decide
from p5.confidence.selective import accuracy_at_coverage, risk_coverage
from p5.confidence.temperature import ece, softmax_t
from p5.data import stanford
from p5.eval.classification import summary
from p5.eval.detection import detection_report
from p5.hybrid.fusion import fuse
from p5.taxonomy import Taxonomy


def log_call(args: argparse.Namespace) -> None:
    path = ROOT / "experiments" / "test_log.csv"
    new = not path.exists()
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    cfg = hashlib.sha256(json.dumps(vars(args), default=str, sort_keys=True).encode()).hexdigest()[:12]
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "commit", "config_hash", "user", "tag"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), commit, cfg, getpass.getuser(), args.tag])


def aligned(npz: Path, ids: np.ndarray) -> dict[str, np.ndarray]:
    d = np.load(npz)
    pos = {int(i): k for k, i in enumerate(d["ids"])}
    order = [pos[int(i)] for i in ids]
    return {k: d[k][order] for k in d.files if k != "ids"}


def preds_from_csv(path: Path, scale: dict[int, float] | None = None) -> dict[int, list]:
    df = pd.read_csv(path)
    out: dict[int, list] = {}
    for r in df.itertuples():
        out.setdefault(int(r.id), []).append(([r.x1, r.y1, r.x2, r.y2], float(r.score)))
    return out


def failure_analysis(args, tax: Taxonomy, test: pd.DataFrame, recs: dict) -> dict:
    import cv2
    from PIL import Image

    from p5 import render
    from p5.classification.cam import class_activation_map, overlay
    from p5.classification.classifier import Classifier

    out = args.out / "failures"
    out.mkdir(parents=True, exist_ok=True)
    gt = test.set_index("id")
    pairs = Counter()
    wrong = []
    for i, r in gt.iterrows():
        rec = recs[int(i)]
        c = rec["class_id"]
        if c is None or c != r.class_id:
            if c is not None:
                pairs[(int(r.class_id), int(c))] += 1
            wrong.append((int(i), rec))
    top_pairs = [{"true": tax.names[a], "predicted": tax.names[b], "count": n,
                  "same_family": tax.family_of[a] == tax.family_of[b],
                  "same_display_make": tax.display_make_of[a] == tax.display_make_of[b]}
                 for (a, b), n in pairs.most_common(20)]
    cats = Counter()
    for i, rec in wrong:
        a, c = int(gt.loc[i, "class_id"]), rec["class_id"]
        if c is None:
            cats["no vehicle or rejected input"] += 1
        elif tax.family_of[a] == tax.family_of[c]:
            cats["same model, different year"] += 1
        elif tax.display_make_of[a] == tax.display_make_of[c]:
            cats["same make, different model"] += 1
        elif any(a in g and c in g for g in tax.rebadge_twins.values()):
            cats["rebadged twin"] += 1
        else:
            cats["different make"] += 1

    clf = Classifier(args.models / "classifier.onnx", args.models / "classifier_meta.json", threads=4)
    fc = np.load(args.models / "head.npz")["fc_weight"]
    chosen = {}
    for i, rec in sorted(wrong, key=lambda t: -(t[1]["confidence"] or 0)):
        a, c = int(gt.loc[i, "class_id"]), rec["class_id"]
        if c is None:
            key = "no_detection"
        elif tax.family_of[a] == tax.family_of[c]:
            key = "year_twin"
        elif any(a in g and c in g for g in tax.rebadge_twins.values()):
            key = "rebadge_twin"
        elif tax.display_make_of[a] == tax.display_make_of[c]:
            key = "same_make"
        else:
            key = "confident_cross_make"
        chosen.setdefault(key, [])
        if len(chosen[key]) < 3:
            chosen[key].append(i)
    figures = {}
    for key, ids_ in chosen.items():
        for i in ids_:
            rec, row = recs[i], gt.loc[i]
            img = Image.open(row.path).convert("RGB")
            rgb = np.asarray(img).copy()
            render.draw_box(rgb, (row.x1, row.y1, row.x2, row.y2), "truth: " + tax.names[int(row.class_id)], (46, 160, 67))
            panels = [render.fit(rgb, 480)]
            if rec["box"] is not None:
                from p5.classical.quality import analyse
                s = min(1.0, 1600 / max(img.size))
                if analyse(np.asarray(img)).flags and max(img.size) * s > 1024:
                    s = 1024 / max(img.size)
                box = [v / s for v in rec["box"]]
                out_d = clf(img, box)
                cam = class_activation_map(out_d.features, fc, int(out_d.logits.argmax()), out_d.crop.shape[1::-1])
                pred_rgb = np.asarray(img).copy()
                render.draw_box(pred_rgb, box, f"pred: {rec['label']} {rec['confidence']:.0%}", (207, 34, 46))
                panels = [render.fit(pred_rgb, 480), cv2.resize(overlay(out_d.crop, cam), (240, 240))]
            h = max(p.shape[0] for p in panels)
            canvas = np.full((h, sum(p.shape[1] for p in panels) + 10 * (len(panels) - 1), 3), 255, np.uint8)
            x = 0
            for pnl in panels:
                canvas[:pnl.shape[0], x:x + pnl.shape[1]] = pnl
                x += pnl.shape[1] + 10
            path = out / f"{key}_{i:05d}.jpg"
            Image.fromarray(canvas).save(path, quality=90)
            figures.setdefault(key, []).append({"id": i, "truth": tax.names[int(row.class_id)], "level": rec["level"],
                                                "prediction": rec["label"], "confidence": rec["confidence"], "figure": path.name})
    result = {"n_wrong_top1": len(wrong), "categories": dict(cats), "top_confusions": top_pairs, "examples": figures}
    (out / "failures.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="E15: the only script that reads Stanford Cars test labels")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "stanford_cars")
    p.add_argument("--cache", type=Path, default=ROOT / "cache")
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--logits", type=Path, required=True, help="folder with logits_test_{gt,pred,full}.npz")
    p.add_argument("--int8", type=Path, default=None, help="npz with INT8 logits on test pred crops")
    p.add_argument("--det", type=Path, required=True, help="folder with det_ft_test.csv and det_zs_test.csv")
    p.add_argument("--thresholds", type=Path, required=True)
    p.add_argument("--classical", type=Path, default=None, help="classical_global.npz")
    p.add_argument("--classical-feats", type=Path, default=None, help="folder with feats_test_{gt,pred,roi}.npz")
    p.add_argument("--robust", type=Path, default=None, help="E14 folder with plain/ and m2/ condition npz files")
    p.add_argument("--pipeline", type=Path, default=None, help="jsonl from scripts/run_pipeline.py on the test images")
    p.add_argument("--extra", nargs="*", default=[], help="name=path.npz logits on test pred crops for extra rows")
    p.add_argument("--failures", action="store_true", help="write confusion pairs and failure-case figures")
    p.add_argument("--models", type=Path, default=ROOT / "models")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", default="e15")
    args = p.parse_args()
    log_call(args)

    tax = Taxonomy.load(args.assets / "taxonomy")
    root = stanford.find_root(args.data)
    test = stanford.load(root, "test", allow_test_labels=True)
    index = pd.read_csv(args.cache / "index_test.csv").set_index("id")
    ids = test.id.to_numpy()
    y = test.class_id.to_numpy()
    th = Thresholds.load(args.thresholds)
    report: dict = {"n": int(len(y)), "split": "test", "thresholds": str(args.thresholds)}

    cls = {}
    for name in ("gt", "pred", "full"):
        f = args.logits / f"logits_test_{name}.npz"
        if f.exists():
            d = aligned(f, ids)
            cls[name] = d
            report[f"deep_{name}"] = {"single": summary(d["logits"], y, tax),
                                      "flip_tta": summary(d["logits"] + d["logits_flip"], y, tax)}
    if args.int8 and args.int8.exists():
        d8 = aligned(args.int8, ids)
        cls["int8_pred"] = d8
        report["deep_int8_pred"] = {"single": summary(d8["logits"], y, tax)}

    gt_boxes = {}
    for r in test.itertuples():
        s = min(1.0, 512 / max(index.loc[r.id, "orig_w"], index.loc[r.id, "orig_h"]))
        gt_boxes[int(r.id)] = [r.x1 * s, r.y1 * s, r.x2 * s, r.y2 * s]
    for tag in ("ft", "zs"):
        f = args.det / f"det_{tag}_test.csv"
        if f.exists():
            report[f"detection_{tag}"] = detection_report(preds_from_csv(f), gt_boxes)

    z_main = cls.get("int8_pred", cls.get("pred", {})).get("logits")
    if z_main is not None:
        e0, m0 = ece(z_main, y, 1.0)
        e1, m1 = ece(z_main, y, th.t_d)
        prob = softmax_t(z_main, th.t_d)
        conf, correct = prob.max(1), prob.argmax(1) == y
        _, _, aurc = risk_coverage(conf, correct)
        levels, ok = Counter(), Counter()
        fam, mk = np.asarray(tax.family_of), np.asarray(tax.display_make_of)
        for zi, yi in zip(z_main, y):
            dec = decide(zi, tax, th)
            levels[dec.level] += 1
            hit = {"class": dec.class_id == yi, "family": fam[dec.class_id] == fam[yi], "make": mk[dec.class_id] == mk[yi],
                   "ood_make": mk[dec.class_id] == mk[yi]}.get(dec.level)
            if hit is not None:
                ok[dec.level] += int(hit)
        report["confidence"] = {"ece_before": e0, "ece_after": e1, "mce_before": m0, "mce_after": m1, "aurc": aurc,
                                "accuracy_at_coverage": {str(c): accuracy_at_coverage(conf, correct, c) for c in (0.8, 0.9, 0.95, 1.0)},
                                "levels": {k: v / len(y) for k, v in levels.items()},
                                "level_accuracy": {k: ok[k] / levels[k] for k in ok if levels[k]}}

    if args.classical and args.classical_feats:
        cm = ClassicalModel.load(args.classical)
        for name in ("gt", "pred", "roi"):
            f = args.classical_feats / f"feats_test_{name}.npz"
            if f.exists():
                d = np.load(f)
                pos = {int(i): k for k, i in enumerate(d["ids"])}
                x = d["x"][[pos[int(i)] for i in ids]]
                s = cm.scores(x)
                report[f"classical_{name}"] = summary(s, y, tax)
                if name == "pred" and "pred" in cls and th.beta > 0:
                    zf = fuse(cls["pred"]["logits"], cm.proba(x), th.beta)
                    report["hybrid_fused_pred"] = summary(zf, y, tax)

    for spec in args.extra:
        name, path = spec.split("=", 1)
        d = aligned(Path(path), ids)
        report[f"extra_{name}"] = summary(d["logits"], y, tax)

    if args.pipeline and args.pipeline.exists():
        recs = {r["id"]: r for r in map(json.loads, args.pipeline.read_text().splitlines())}
        fam, mk = np.asarray(tax.family_of), np.asarray(tax.display_make_of)
        z = np.stack([np.array(recs[int(i)]["logits"]) if recs[int(i)]["logits"] else np.full(196, -1e3) for i in ids])
        levels, ok = Counter(), Counter()
        for i, yi in zip(ids, y):
            r = recs[int(i)]
            lv = r["level"]
            levels[lv] += 1
            c = r["class_id"]
            hit = {"class": c == yi, "family": c is not None and fam[c] == fam[yi], "make": c is not None and mk[c] == mk[yi],
                   "ood_make": c is not None and mk[c] == mk[yi]}.get(lv)
            if hit is not None:
                ok[lv] += int(hit)
        totals = np.array([recs[int(i)]["timings_ms"].get("total", np.nan) for i in ids], dtype=float)
        enh = sum(bool(recs[int(i)]["enhancement"]) for i in ids)
        report["deployed_pipeline"] = {
            **summary(z, y, tax),
            "no_vehicle_rate": levels["no_vehicle"] / len(y),
            "levels": {k: v / len(y) for k, v in levels.items()},
            "level_accuracy": {k: ok[k] / levels[k] for k in ok if levels[k]},
            "answered_correct_rate": sum(ok.values()) / len(y),
            "enhancement_triggered": enh / len(y),
            "latency_ms_single_thread": {"p50": float(np.nanpercentile(totals, 50)), "p95": float(np.nanpercentile(totals, 95))},
        }

    if args.robust and args.robust.exists():
        label = dict(zip(ids, y))
        table = {}
        for f in sorted((args.robust / "plain").glob("*.npz")):
            row = {}
            for tag, folder in (("", "plain"), ("_m2", "m2")):
                g = args.robust / folder / f.name.replace(".npz", ".npz" if tag == "" else "_m2.npz")
                if not g.exists():
                    continue
                d = np.load(g)
                yy = np.array([label[int(i)] for i in d["ids"]])
                row[f"e2e{tag}"] = float((d["logits_e2e"].argmax(1) == yy).mean())
                row[f"gt{tag}"] = float((d["logits_gt"].argmax(1) == yy).mean())
                row[f"detected{tag}"] = float((d["det_score"] >= th.tau_lo).mean())
                if tag == "_m2":
                    row["m2_applied"] = float(np.mean([bool(a) for a in d["applied"]]))
            g = args.robust / "m2" / f.name.replace(".npz", "_m2.npz")
            if g.exists():
                a, b = np.load(f), np.load(g)
                assert (a["ids"] == b["ids"]).all()
                yy = np.array([label[int(i)] for i in a["ids"]])
                use_m2 = b["det_score"] > a["det_score"]
                pick = np.where(use_m2[:, None], b["logits_e2e"], a["logits_e2e"])
                row["e2e_select"] = float((pick.argmax(1) == yy).mean())
                row["detected_select"] = float((np.maximum(a["det_score"], b["det_score"]) >= th.tau_lo).mean())
                row["select_used_m2"] = float(use_m2.mean())
            table[f.stem] = row
        report["robustness"] = table
    if args.failures and args.pipeline and args.pipeline.exists():
        report["failures"] = failure_analysis(args, tax, test, recs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "test_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
