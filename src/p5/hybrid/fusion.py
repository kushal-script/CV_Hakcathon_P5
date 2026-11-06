from __future__ import annotations

import numpy as np

from p5.confidence.temperature import nll

GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def fuse(z_deep: np.ndarray, p_cls: np.ndarray, beta: float, eps: float = 1e-6) -> np.ndarray:
    return z_deep + beta * np.log(p_cls + eps)


def fit_beta(z_deep: np.ndarray, p_cls: np.ndarray, y: np.ndarray, t_d: float, grid=GRID) -> tuple[float, dict]:
    scores = {b: nll(fuse(z_deep, p_cls, b), y, t_d) for b in grid}
    return min(scores, key=scores.get), scores
