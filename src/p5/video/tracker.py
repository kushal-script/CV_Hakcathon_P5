from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from p5.config import section
from p5.detection.detector import Detection


def to_z(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w * h, w / max(h, 1e-6)], dtype=np.float64)


def to_box(x: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, s, r = x[:4]
    s, r = max(s, 1e-6), max(r, 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)


class Kalman:
    def __init__(self, box):
        self.x = np.zeros(7)
        self.x[:4] = to_z(box)
        self.P = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4]).astype(float)
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        self.Q = np.diag([1, 1, 1, 0.01, 0.01, 0.01, 1e-4])
        self.R = np.diag([1, 1, 10, 10]).astype(float)

    def predict(self) -> tuple[float, float, float, float]:
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return to_box(self.x)

    def update(self, box) -> None:
        y = to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P


@dataclass
class Track:
    tid: int
    kf: Kalman
    cls_votes: dict[int, float] = field(default_factory=dict)
    history: list[tuple[int, tuple, float]] = field(default_factory=list)
    hits: int = 1
    misses: int = 0

    @property
    def box(self):
        return to_box(self.kf.x)

    @property
    def cls(self) -> int:
        return max(self.cls_votes, key=self.cls_votes.get)


def iou_matrix(a: list, b: list) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    A, B = np.asarray(a, float), np.asarray(b, float)
    ix1 = np.maximum(A[:, None, 0], B[None, :, 0])
    iy1 = np.maximum(A[:, None, 1], B[None, :, 1])
    ix2 = np.minimum(A[:, None, 2], B[None, :, 2])
    iy2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    return inter / (area(A)[:, None] + area(B)[None, :] - inter + 1e-9)


def suppress_nested(dets: list[Detection], frac: float | None = None) -> list[Detection]:
    frac = section("video")["nested_box_fraction"] if frac is None else frac
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: -(d.box[2] - d.box[0]) * (d.box[3] - d.box[1])):
        x1, y1, x2, y2 = d.box
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        inside = False
        for k in keep:
            ix = max(0.0, min(x2, k.box[2]) - max(x1, k.box[0]))
            iy = max(0.0, min(y2, k.box[3]) - max(y1, k.box[1]))
            if ix * iy / area >= frac and k.score >= d.score:
                inside = True
                break
        if not inside:
            keep.append(d)
    return keep


def assign(tracks: list[Track], dets: list[Detection], thr: float) -> tuple[list, list, list]:
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    m = iou_matrix([t.box for t in tracks], [d.box for d in dets])
    rows, cols = linear_sum_assignment(-m)
    pairs = [(r, c) for r, c in zip(rows, cols) if m[r, c] >= thr]
    used_t, used_d = {r for r, _ in pairs}, {c for _, c in pairs}
    return pairs, [i for i in range(len(tracks)) if i not in used_t], [j for j in range(len(dets)) if j not in used_d]


class Tracker:
    def __init__(self, fps: float = 8.0, min_hits: int | None = None):
        cfg = section("tracker")
        self.high, self.low, self.iou = cfg["high_score"], cfg["low_score"], cfg["match_iou"]
        self.max_age = max(1, int(round(cfg["max_age_seconds"] * fps)))
        self.min_hits = cfg["min_hits"] if min_hits is None else min_hits
        self.dup_iou = cfg["duplicate_iou"]
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self.next_id = 1

    def _touch(self, t: Track, d: Detection, frame: int) -> None:
        t.kf.update(d.box)
        t.hits += 1
        t.misses = 0
        t.cls_votes[d.cls] = t.cls_votes.get(d.cls, 0.0) + d.score
        t.history.append((frame, d.box, d.score))

    def update(self, dets: list[Detection], frame: int) -> list[Track]:
        for t in self.tracks:
            t.kf.predict()
        high = [d for d in dets if d.score >= self.high]
        low = [d for d in dets if self.low <= d.score < self.high]
        pairs, un_t, un_d = assign(self.tracks, high, self.iou)
        for r, c in pairs:
            self._touch(self.tracks[r], high[c], frame)
        rest = [self.tracks[i] for i in un_t]
        pairs2, un_t2, _ = assign(rest, low, self.iou)
        for r, c in pairs2:
            self._touch(rest[r], low[c], frame)
        for i in un_t2:
            rest[i].misses += 1
        boxes_now = [t.box for t in self.tracks]
        for j in un_d:
            d = high[j]
            if boxes_now and iou_matrix([d.box], boxes_now).max() > self.dup_iou:
                continue
            t = Track(self.next_id, Kalman(d.box), {d.cls: d.score}, [(frame, d.box, d.score)])
            boxes_now.append(d.box)
            self.next_id += 1
            self.tracks.append(t)
        alive = []
        for t in self.tracks:
            if t.misses > self.max_age:
                if t.hits >= self.min_hits:
                    self.finished.append(t)
            else:
                alive.append(t)
        self.tracks = alive
        return [t for t in self.tracks if t.hits >= self.min_hits and t.misses == 0]

    def all_tracks(self) -> list[Track]:
        return self.finished + [t for t in self.tracks if t.hits >= self.min_hits]
