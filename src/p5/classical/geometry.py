from __future__ import annotations

import cv2
import numpy as np

from p5.types import Geometry, Roi

RULES = {"sym_front": 0.80, "aspect_front": 1.45, "aspect_side": 2.0, "sym_side": 0.70}


def symmetry(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0
    flipped = crop[:, ::-1]
    inter, union = np.logical_and(crop, flipped).sum(), np.logical_or(crop, flipped).sum()
    return float(inter / union) if union else 0.0


def describe(roi: Roi, rules: dict | None = None) -> Geometry:
    r = {**RULES, **(rules or {})}
    x1, y1, x2, y2 = roi.box
    aspect = float((x2 - x1) / max(y2 - y1, 1e-3))
    if roi.contour is not None and len(roi.contour) >= 3:
        c = roi.contour.astype(np.float32)
        area = float(cv2.contourArea(c))
        hull = float(cv2.contourArea(cv2.convexHull(c)))
        extent = area / max((x2 - x1) * (y2 - y1), 1e-3)
        solidity = area / max(hull, 1e-3)
        hu = cv2.HuMoments(cv2.moments(c)).flatten()
        hu = [float(-np.sign(v) * np.log10(abs(v) + 1e-30)) for v in hu]
    else:
        extent, solidity, hu = 1.0, 1.0, [0.0] * 7
    sym = symmetry(roi.mask)
    if sym >= r["sym_front"] and aspect <= r["aspect_front"]:
        view = "front_or_rear"
    elif aspect >= r["aspect_side"] and sym < r["sym_side"]:
        view = "side"
    else:
        view = "three_quarter"
    return Geometry(aspect, float(extent), float(solidity), sym, hu, view)
