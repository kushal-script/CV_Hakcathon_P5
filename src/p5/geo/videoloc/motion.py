from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np


@dataclass
class MotionConfig:
    target_fps: float = 5.0
    max_side: int = 640
    max_seconds: float = 600.0
    max_corners: int = 600
    grid: int = 8
    focal_frac: float = 0.8
    fb_max_px: float = 1.0
    ransac_px: float = 1.0
    still_rate: float = 0.031


def _features(gray, mask, cfg: MotionConfig):
    h, w = gray.shape
    per = max(4, cfg.max_corners // (cfg.grid * cfg.grid))
    pts = []
    for i in range(cfg.grid):
        for j in range(cfg.grid):
            y0, y1, x0, x1 = i * h // cfg.grid, (i + 1) * h // cfg.grid, j * w // cfg.grid, (j + 1) * w // cfg.grid
            c = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], per, 0.01, 7, mask=mask[y0:y1, x0:x1])
            if c is not None:
                pts.append(c.reshape(-1, 2) + [x0, y0])
    return np.concatenate(pts).astype(np.float32) if pts else np.zeros((0, 2), np.float32)


def _track(g0, g1, p0, cfg: MotionConfig):
    if len(p0) == 0:
        return p0, p0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(21, 21), maxLevel=3)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(pb - p0, axis=1) < cfg.fb_max_px)
    return p0[ok], p1[ok]


def video_frames(path: str, cfg: MotionConfig):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("This file is not a video we can read.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / cfg.target_fps))
    limit = int(cfg.max_seconds * fps)
    idx = 0
    while idx < limit:
        if not cap.grab():
            break
        if idx % stride == 0:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            yield idx / fps, bgr
        idx += 1
    cap.release()


def measure(images: Iterable, cfg: MotionConfig | None = None, focal_px: float | None = None,
            progress: Callable[[float], None] | None = None, total_s: float | None = None) -> tuple[list[dict], dict]:
    cfg = cfg or MotionConfig()
    out, prev, prev_t, psi, K, votes, foe = [], None, 0.0, 0.0, None, None, []
    thumbs: list[tuple[float, np.ndarray]] = []
    for t, bgr in images:
        h, w = bgr.shape[:2]
        sc = min(1.0, cfg.max_side / max(h, w))
        if sc < 1.0:
            bgr = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        if K is None:
            f = focal_px * sc if focal_px else cfg.focal_frac * W
            K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
            votes = np.zeros((H, W), np.float32)
        if not thumbs or t - thumbs[-1][0] >= 1.0:
            thumbs.append((t, cv2.resize(bgr, (160, max(1, int(160 * bgr.shape[0] / bgr.shape[1]))), interpolation=cv2.INTER_AREA)))
        rec = {"t": float(t), "dyaw": 0.0, "psi": psi, "ok": False}
        if prev is not None:
            a, b = _track(prev, gray, _features(prev, (votes < 3).astype(np.uint8) * 255, cfg), cfg)
            rec["tracked"] = int(len(a))
            if len(a) >= 12 and t > prev_t:
                flow = np.linalg.norm(b - a, axis=1)
                med = float(np.median(flow))
                if med > 1.5:
                    for x, y in a[flow < 0.15].astype(int):
                        votes[max(0, y - 4):y + 5, max(0, x - 4):x + 5] += 1
                rate = med / W / (t - prev_t)
                if rate < cfg.still_rate:
                    rec.update(ok=True, stationary=True)
                else:
                    E, inl = cv2.findEssentialMat(a, b, K, cv2.RANSAC, 0.999, cfg.ransac_px)
                    if E is not None and E.shape == (3, 3) and inl is not None and inl.sum() >= 10:
                        _, R, tvec, _ = cv2.recoverPose(E, a, b, K, mask=inl.copy())
                        dyaw = -math.atan2(R[0, 2], R[2, 2])
                        sel = inl.ravel() > 0
                        pa = cv2.perspectiveTransform(a[sel].reshape(-1, 1, 2).astype(np.float64), K @ R @ np.linalg.inv(K)).reshape(-1, 2)
                        par = np.linalg.norm(b[sel] - pa, axis=1)
                        if np.median(np.linalg.norm(pa - a[sel], axis=1)) > 2 * med + 2:
                            Hm, _ = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
                            dyaw = 0.0
                            if Hm is not None:
                                Rh = np.linalg.inv(K) @ Hm @ K
                                Rh = Rh / np.cbrt(np.linalg.det(Rh))
                                yh = -math.atan2(Rh[0, 2], Rh[2, 2])
                                if abs(yh) * K[0, 0] <= 2 * med + 2:
                                    dyaw = yh
                        psi += dyaw
                        rec.update(ok=True, dyaw=dyaw, psi=psi, parallax=float(np.median(par)))
                        tv = tvec.ravel()
                        if abs(tv[2]) > 0.7 and np.median(par) > 0.3:
                            foe.append(K[1, 1] * tv[1] / tv[2] + K[1, 2])
                        if len(foe) >= 5:
                            y_h = float(np.median(foe[-60:]))
                            g_sel = (a[sel][:, 1] > y_h + 0.08 * H) & (np.abs(a[sel][:, 0] - K[0, 2]) < 0.35 * W)
                            yy, yy2 = pa[g_sel, 1] - y_h, b[sel][g_sel, 1] - y_h
                            okg = (yy > 2) & (yy2 > 0)
                            if okg.sum() >= 5:
                                rel = K[1, 1] * (yy2[okg] - yy[okg]) / (yy[okg] * yy2[okg])
                                rec["ground_rate"] = float(np.median(rel)) / (t - prev_t)
        out.append(rec)
        prev, prev_t = gray, t
        if progress and total_s:
            progress(min(1.0, t / total_s))
    moving = [r for r in out if r.get("ok") and not r.get("stationary")]
    span = out[-1]["t"] - out[0]["t"] if len(out) > 1 else 0.0
    summary = {"frames": len(out), "duration_s": round(span, 1), "moving_s": round(len(moving) * (span / max(len(out) - 1, 1)), 1),
               "total_turn_deg": round(float(np.degrees(sum(abs(r["dyaw"]) for r in moving))), 1),
               "net_turn_deg": round(float(np.degrees(psi)), 1), "ok_fraction": round(sum(r.get("ok", False) for r in out[1:]) / max(len(out) - 1, 1), 3),
               "speed_fraction": round(sum("ground_rate" in r for r in moving) / max(len(moving), 1), 3)}
    return out, {"summary": summary, "K": K.tolist() if K is not None else None, "thumbs": thumbs}
