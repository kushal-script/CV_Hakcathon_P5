from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from p5.classical.geometry import describe
from p5.classical.refine import refine
from p5.types import Box, Geometry, Roi

H, W = 128, 256


def hsv_hist(rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = None if mask is None or not mask.any() else mask
    hist = cv2.calcHist([hsv], [0, 1, 2], m, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return hist / max(hist.sum(), 1e-6)


def colour_moments(rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32) / 255.0
    if mask is not None and mask.any():
        lab = lab[mask.reshape(-1) > 0]
    mean, std = lab.mean(0), lab.std(0)
    skew = np.cbrt(((lab - mean) ** 3).mean(0))
    return np.concatenate([mean, std, skew])


def lbp_hist(gray: np.ndarray) -> np.ndarray:
    out = []
    for p, r in ((8, 1), (16, 2)):
        codes = local_binary_pattern(gray, p, r, method="uniform")
        hist, _ = np.histogram(codes, bins=p + 2, range=(0, p + 2))
        out.append(hist / max(hist.sum(), 1))
    return np.concatenate(out)


def hog_vec(gray: np.ndarray) -> np.ndarray:
    return hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys",
               feature_vector=True)


def edge_grid(edges: np.ndarray, gx: int = 8, gy: int = 4) -> np.ndarray:
    h, w = edges.shape
    return np.array([edges[i * h // gy:(i + 1) * h // gy, j * w // gx:(j + 1) * w // gx].mean() / 255.0
                     for i in range(gy) for j in range(gx)])


def extract(rgb: np.ndarray, box: Box, roi: Roi | None = None) -> tuple[np.ndarray, Roi, Geometry]:
    roi = roi or refine(rgb, box)
    geo = describe(roi)
    hgt, wid = rgb.shape[:2]
    x1, y1, x2, y2 = roi.box
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(wid, x2)), int(min(hgt, y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        x1, y1, x2, y2 = 0, 0, wid, hgt
    crop = cv2.resize(rgb[y1:y2, x1:x2], (W, H), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(roi.mask[y1:y2, x1:x2], (W, H), interpolation=cv2.INTER_NEAREST)
    edges = cv2.resize(roi.edges[y1:y2, x1:x2], (W, H), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    vec = np.concatenate([hsv_hist(crop, mask), colour_moments(crop, mask), lbp_hist(gray), hog_vec(gray),
                          edge_grid(edges), [roi.edge_density], geo.vector()]).astype(np.float32)
    return vec, roi, geo


def blocks() -> dict[str, slice]:
    sizes = {"hsv": 128, "moments": 9, "lbp": 28, "hog": 7 * 15 * 36, "edges": 33, "geometry": 11}
    out, start = {}, 0
    for k, n in sizes.items():
        out[k] = slice(start, start + n)
        start += n
    return out
