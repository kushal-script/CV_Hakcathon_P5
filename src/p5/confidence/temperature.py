from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax_t(z: np.ndarray, t: float) -> np.ndarray:
    s = z / t
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def nll(z: np.ndarray, y: np.ndarray, t: float) -> float:
    p = softmax_t(z, t)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(lambda lt: nll(z, y, float(np.exp(lt))), bounds=(np.log(0.01), np.log(100)), method="bounded")
    return float(np.exp(res.x))


def ece(z: np.ndarray, y: np.ndarray, t: float = 1.0, bins: int = 15) -> tuple[float, float]:
    p = softmax_t(z, t)
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = m = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            gap = abs(acc[sel].mean() - conf[sel].mean())
            e += sel.mean() * gap
            m = max(m, gap)
    return float(e), float(m)
