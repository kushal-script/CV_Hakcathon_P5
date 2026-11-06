from __future__ import annotations

import cv2
import numpy as np

from p5.types import Box, Roi


def canny_auto(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    m = float(np.median(gray))
    return cv2.Canny(gray, int(max(0, (1 - sigma) * m)), int(min(255, (1 + sigma) * m)))


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def grabcut_mask(rgb: np.ndarray, rect: tuple[int, int, int, int], iters: int = 5) -> np.ndarray:
    mask = np.zeros(rgb.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        x, y, w, h = rect
        mask[y:y + h, x:x + w] = cv2.GC_PR_FGD
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def refine(rgb: np.ndarray, box: Box, work_side: int = 320, iters: int = 5, source: str = "detector") -> Roi:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px1, py1 = int(max(0, x1 - 0.05 * bw)), int(max(0, y1 - 0.05 * bh))
    px2, py2 = int(min(w, x2 + 0.05 * bw)), int(min(h, y2 + 0.05 * bh))
    region = rgb[py1:py2, px1:px2]
    s = min(1.0, work_side / max(region.shape[:2]))
    small = cv2.resize(region, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else region
    sh, sw = small.shape[:2]
    rx1, ry1 = int((x1 - px1) * s), int((y1 - py1) * s)
    rect = (max(1, rx1), max(1, ry1), max(2, min(sw - 2, int(bw * s)) - 1), max(2, min(sh - 2, int(bh * s)) - 1))
    mask_small = grabcut_mask(small, rect, iters)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contour_small = largest_contour(mask_small)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    edges = canny_auto(cv2.GaussianBlur(gray, (3, 3), 0))
    if contour_small is not None and cv2.contourArea(contour_small) > 0.05 * sw * sh:
        cx, cy, cw, ch = cv2.boundingRect(contour_small)
        tight = (px1 + cx / s, py1 + cy / s, px1 + (cx + cw) / s, py1 + (cy + ch) / s)
        clean = np.zeros_like(mask_small)
        cv2.drawContours(clean, [contour_small], -1, 255, thickness=-1)
        mask_small = clean
    else:
        tight, contour_small = box, None
    fg = mask_small > 0
    density = float(edges[fg].mean() / 255.0) if fg.any() else float(edges.mean() / 255.0)
    contour = None if contour_small is None else (contour_small.astype(np.float32) / s + np.array([px1, py1], np.float32))
    full_mask = np.zeros((h, w), np.uint8)
    full_mask[py1:py2, px1:px2] = cv2.resize(mask_small, (px2 - px1, py2 - py1), interpolation=cv2.INTER_NEAREST)
    full_edges = np.zeros((h, w), np.uint8)
    full_edges[py1:py2, px1:px2] = cv2.resize(edges, (px2 - px1, py2 - py1), interpolation=cv2.INTER_NEAREST)
    return Roi(tight, full_mask, contour, full_edges, density, source)
