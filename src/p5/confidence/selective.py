from __future__ import annotations

import numpy as np


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-conf)
    c = correct[order].astype(float)
    n = len(c)
    coverage = np.arange(1, n + 1) / n
    risk = 1 - np.cumsum(c) / np.arange(1, n + 1)
    return coverage, risk, float(risk.mean())


def min_threshold(conf: np.ndarray, correct: np.ndarray, target_acc: float, grid: np.ndarray | None = None) -> float:
    grid = np.linspace(0.05, 0.99, 95) if grid is None else grid
    for t in grid:
        sel = conf >= t
        if sel.sum() >= 20 and correct[sel].mean() >= target_acc:
            return float(t)
    return float(grid[-1])


def accuracy_at_coverage(conf: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    k = max(1, int(round(coverage * len(conf))))
    return float(correct[np.argsort(-conf)[:k]].mean())
