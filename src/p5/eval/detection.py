from __future__ import annotations

import numpy as np

THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ap_at(preds: dict[int, list[tuple[list[float], float]]], gts: dict[int, list[list[float]]], thr: float) -> float:
    flat = [(s, i, b) for i, ps in preds.items() for b, s in ps]
    flat.sort(key=lambda t: -t[0])
    n_gt = sum(len(v) for v in gts.values())
    used = {i: np.zeros(len(v), bool) for i, v in gts.items()}
    tp = np.zeros(len(flat))
    for k, (_, i, b) in enumerate(flat):
        g = gts.get(i, [])
        if not g:
            continue
        ious = np.array([iou(b, x) for x in g])
        j = int(ious.argmax())
        if ious[j] >= thr and not used[i][j]:
            used[i][j] = True
            tp[k] = 1
    if n_gt == 0 or not len(flat):
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall, precision = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def localisation(preds: dict[int, list[tuple[list[float], float]]], gts: dict[int, list[float]]) -> dict[str, float]:
    ious, missed = [], 0
    for i, g in gts.items():
        ps = preds.get(i, [])
        if not ps:
            missed += 1
            ious.append(0.0)
            continue
        best = max(ps, key=lambda t: t[1])[0]
        ious.append(iou(best, g))
    a = np.array(ious)
    return {"mean_iou": float(a.mean()), "iou_ge_0_5": float((a >= 0.5).mean()), "iou_ge_0_75": float((a >= 0.75).mean()),
            "missed": missed, "n": len(gts)}


def detection_report(preds: dict, gts_single: dict[int, list[float]]) -> dict[str, float]:
    gts = {i: [g] for i, g in gts_single.items()}
    aps = [ap_at(preds, gts, t) for t in THRESHOLDS]
    return {**localisation(preds, gts_single), "map_50": aps[0], "map_75": aps[5], "map_50_95": float(np.mean(aps))}
