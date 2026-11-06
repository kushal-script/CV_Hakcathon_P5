from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .filter import Params
from .graph import RoadGraph, wrap
from .motion import MotionConfig, measure, video_frames
from .search import decoys, search

TAU = 0.99
UNKNOWN_CAMERA = {"k_values": (0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.4, 3.0), "scale_prior": (0.5, 3.0)}
STATED_LENS = {"k_values": (0.9, 0.95, 1.0, 1.05, 1.1), "scale_prior": (0.8, 3.0)}
_GRAPH: dict = {}


def graph(path: str | Path) -> RoadGraph:
    key = str(path)
    if key not in _GRAPH:
        _GRAPH[key] = RoadGraph(path)
    return _GRAPH[key]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bearing_on(g: RoadGraph, d: int, x: float, y: float) -> float:
    a, b = g.d_first_seg[d], g.d_first_seg[d + 1]
    dx, dy = x - g.seg_x0[a:b], y - g.seg_y0[a:b]
    t = np.clip(dx * g.seg_ux[a:b] + dy * g.seg_uy[a:b], 0, g.seg_len[a:b])
    i = int(np.argmin(np.hypot(dx - t * g.seg_ux[a:b], dy - t * g.seg_uy[a:b])))
    return float(g.seg_bearing[a + i])


def _turns(records, path, g, k, min_deg):
    ts = np.array([r["t"] for r in records])
    psi = np.array([r["psi"] for r in records]) * k
    pts = [(t, x, y, d) for t, x, y, d in path if d >= 0]
    if len(pts) < 2:
        return []
    out = []
    for (t1, x1, y1, d1), (t2, x2, y2, d2) in zip(pts, pts[1:]):
        v = math.degrees(float(np.interp(t2 - ts[0], ts - ts[0], psi) - np.interp(t1 - ts[0], ts - ts[0], psi)))
        if abs(v) < min_deg:
            continue
        m = math.degrees(float(wrap(_bearing_on(g, d2, x2, y2) - _bearing_on(g, d1, x1, y1))))
        if abs(v) > 180:
            m = m + 360 * round((v - m) / 360)
        out.append({"from_s": round(t1, 1), "to_s": round(t2, 1), "video_deg": round(v, 1), "map_deg": round(m, 1),
                    "street_from": g.name_at(d1), "street_to": g.name_at(d2)})
    return out


def _thumb(thumbs, t):
    if not thumbs:
        return None
    _, img = min(thumbs, key=lambda p: abs(p[0] - t))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def locate_video(path: str, road_graph: RoadGraph, center_lonlat: tuple[float, float], radius_m: float,
                 hfov_deg: float | None, workers: int = 4, progress=None, max_seconds: float = 600.0, ocr=None, places_path=None) -> dict:
    area = {"search_area": {"lon": center_lonlat[0], "lat": center_lonlat[1], "radius_m": radius_m}, "threshold": TAU}
    if not hfov_deg or not 20 <= hfov_deg <= 170:
        return {**area, "status": "not_applicable", "reason": "The camera's horizontal field of view is needed to measure its turns. Take it from the camera's specification."}
    cfg = MotionConfig(max_seconds=max_seconds)
    cap = cv2.VideoCapture(path)
    fps, total, width = cap.get(cv2.CAP_PROP_FPS) or 25.0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0, cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    cap.release()
    if not width:
        raise ValueError("This file is not a video we can read.")
    focal = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    total_s = min(max_seconds, total / fps if fps else max_seconds)
    records, extra = measure(video_frames(path, cfg), cfg, focal, progress=(lambda p: progress(0.3 * p)) if progress else None, total_s=total_s)
    params = Params(k_values=STATED_LENS["k_values"], scale_prior=STATED_LENS["scale_prior"])
    res = locate_records(records, extra, road_graph, center_lonlat, radius_m, params, f"stated field of view {hfov_deg:.0f}°, camera height unknown",
                         workers=workers, progress=(lambda p: progress(0.3 + 0.65 * p)) if progress else None)
    if ocr is not None and places_path and res.get("route") and res["status"] in ("candidate", "ambiguous"):
        from .signs import conflicting_signs, load_places, match_signs, read_signs
        places = load_places(places_path)
        read = read_signs(path, ocr, max_frames=int(min(90, max_seconds / 2)))
        res["signs"] = {"read": len(read), "texts": sorted({s["text"] for s in read})[:40], "supporting": match_signs(read, res["route"], places),
                        "conflicting": conflicting_signs(read, res["route"], places)}
    return res


def locate_records(records: list[dict], extra: dict, road_graph: RoadGraph, center_lonlat: tuple[float, float], radius_m: float, params: Params,
                   camera_model: str, workers: int = 4, progress=None, decoy_areas: int = 4, decoy_runs: int = 3, **search_kw) -> dict:
    summary = extra["summary"]
    cx, cy = road_graph.proj.to_xy(center_lonlat[0], center_lonlat[1])
    base = {"search_area": {"lon": center_lonlat[0], "lat": center_lonlat[1], "radius_m": radius_m}, "camera_motion": summary,
            "camera_model": camera_model, "threshold": TAU}
    if summary["ok_fraction"] < 0.5:
        return {**base, "status": "not_applicable", "reason": "Camera movement could not be measured in most frames, for example because of darkness, blur or a blocked view."}
    if summary["moving_s"] < 10:
        return {**base, "status": "not_applicable",
                "reason": "The camera moves for less than 10 seconds in this footage, too little to match a route. A fixed camera needs its position entered by hand."}
    res = search(road_graph, records, (float(cx), float(cy)), radius_m, params, workers=workers, progress=progress, **search_kw)
    if "best" not in res:
        return {**base, **res}
    best, other = res["best"], res["other"]
    hx, hy, _ = best["hyps"][0]
    lon, lat = road_graph.proj.to_lonlat(hx, hy)
    path_pts = best.get("path") or []
    route = []
    for t, x, y, d in path_pts:
        plon, plat = road_graph.proj.to_lonlat(x, y)
        route.append({"t": round(float(t), 1), "lat": round(float(plat), 6), "lon": round(float(plon), 6), "street": road_graph.name_at(d) if d >= 0 else ""})
    streets = []
    for p in route:
        if p["street"] and (not streets or streets[-1] != p["street"]):
            streets.append(p["street"])
    alternatives = []
    for h in best["hyps"][1:]:
        alon, alat = road_graph.proj.to_lonlat(h[0], h[1])
        alternatives.append({"lat": round(float(alat), 6), "lon": round(float(alon), 6), "probability": round(res["p_best"] * h[2], 4), "kind": "same search, other spot"})
    if other and other.get("hyps"):
        o = other["hyps"][0]
        olon, olat = road_graph.proj.to_lonlat(o[0], o[1])
        alternatives.append({"lat": round(float(olat), 6), "lon": round(float(olon), 6), "probability": round((1 - res["p_best"]) * o[2], 4),
                             "kind": "best alternative elsewhere"})
    turns = _turns(records, path_pts, road_graph, best.get("k", 1.0), 5 * math.degrees(params.sigma_obs))
    for tr in turns:
        tr["frame_jpeg"] = _thumb(extra.get("thumbs"), tr["to_s"])
    conf = res["confidence"]
    status = "candidate" if conf >= TAU else ("ambiguous" if res["mass"] >= TAU else "not_located")
    reason = {"candidate": "If this footage was recorded inside the search area, the camera's turns, stops and speed point to one route far more strongly than to any other. Movement alone cannot show that the footage is from this area, so confirm it with independent features before use.",
              "ambiguous": "Two different places fit the camera movement almost equally well. More video or another clue is needed.",
              "not_located": "The camera movement does not single out one place in the search area."}[status]
    decoy = None
    if status == "candidate" and decoy_areas > 0:
        area_ll = max([x for x in res["main_lls"] + res["alt_lls"] if math.isfinite(x)])
        found = decoys(road_graph, records, (float(cx), float(cy)), radius_m, params, k=decoy_areas, runs=decoy_runs, workers=workers,
                       seed=int(abs(hash((round(cx), round(cy), len(records)))) % 100000))
        best_decoy = max((d["best_ll"] for d in found), default=-math.inf)
        margin = area_ll - best_decoy
        decoy = {"areas": found, "log10_margin": round(margin / math.log(10), 2) if math.isfinite(margin) else None}
        if not margin >= math.log(1 / (1 - TAU) - 1):
            status = "not_located"
            reason = ("A route in this area fits, but routes in random parts of the city where the footage was not recorded fit about as well. "
                      "The footage may not be from this area. Check where it came from or try another area.")
    return {**base, "status": status, "reason": reason, "confidence": round(conf, 4), "log10_likelihood_ratio": None if res["log10_lr"] is None else round(res["log10_lr"], 2),
            "decoys": decoy,
            "location": {"lat": round(float(lat), 6), "lon": round(float(lon), 6), "radius_m": 50, "street": route[-1]["street"] if route else ""},
            "route": route, "streets": streets, "alternatives": alternatives, "turns": turns,
            "search": {"particles_per_run": res["particles"], "runs": res["runs"], "roads_km": res["area_km"], "camera_scale": best.get("c"),
                       "lens_factor": best.get("k"), "agreeing_runs": res["evidence"]["agreeing_runs"], "main_runs": res["evidence"]["main_runs"]}}
