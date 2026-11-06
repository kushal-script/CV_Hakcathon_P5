from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"a_only": b, "b_only": c, "p_value": float(p)}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for k, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - k) * p))
        out[name] = running
    return out
