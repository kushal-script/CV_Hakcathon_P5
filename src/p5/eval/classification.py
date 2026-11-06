from __future__ import annotations

import numpy as np

from p5.taxonomy import Taxonomy


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def topk(logits: np.ndarray, y: np.ndarray, k: int) -> float:
    pred = np.argsort(-logits, axis=1)[:, :k]
    return float((pred == y[:, None]).any(axis=1).mean())


def grouped_top1(logits: np.ndarray, y: np.ndarray, group_of: list) -> float:
    g = np.asarray(group_of)
    return float((g[logits.argmax(1)] == g[y]).mean())


def twin_confusion(logits: np.ndarray, y: np.ndarray, groups: list[list[int]]) -> float | None:
    pred = logits.argmax(1)
    hits = total = 0
    for members in groups:
        mask = np.isin(y, members)
        total += int(mask.sum())
        hits += int(sum(p in members and p != t for p, t in zip(pred[mask], y[mask])))
    return hits / total if total else None


def summary(logits: np.ndarray, y: np.ndarray, tax: Taxonomy) -> dict[str, float]:
    return {
        "top1": topk(logits, y, 1),
        "top5": topk(logits, y, 5),
        "make_top1": grouped_top1(logits, y, tax.display_make_of),
        "family_top1": grouped_top1(logits, y, tax.family_of),
        "year_twin_confusion": twin_confusion(logits, y, [list(p) for p in tax.year_twins]),
        "rebadge_twin_confusion": twin_confusion(logits, y, list(tax.rebadge_twins.values())),
    }
