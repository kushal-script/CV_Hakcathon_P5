from __future__ import annotations

import gzip
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

GENERIC = {"road", "main", "cross", "street", "nagar", "layout", "stage", "phase", "block", "sector", "circle", "junction", "signal", "bus", "stop",
           "bengaluru", "bangalore", "karnataka", "india", "hotel", "restaurant", "bank", "atm", "shop", "store", "medical", "hospital", "school",
           "college", "temple", "police", "station", "traffic", "parking", "welcome", "open", "sale", "offer", "new", "exit", "entry", "one", "way",
           "the", "and", "for", "pvt", "ltd", "limited", "private", "service", "centre", "center"}
_INDEX: dict = {}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def distinctive(text: str) -> bool:
    words = [w for w in text.split() if w not in GENERIC]
    return sum(len(w) for w in words) >= 6 and any(len(w) >= 4 and not w.isdigit() for w in words)


def load_places(path: str | Path) -> dict:
    key = str(path)
    if key not in _INDEX:
        rows = json.loads(gzip.decompress(Path(path).read_bytes()))
        names = [r["n"] for r in rows]
        count: dict[str, int] = {}
        where: dict[str, list[int]] = {}
        for i, n in enumerate(names):
            count[n] = count.get(n, 0) + 1
            where.setdefault(n, []).append(i)
        _INDEX[key] = {"rows": rows, "lat": np.array([r["y"] for r in rows]), "lon": np.array([r["x"] for r in rows]), "count": count, "where": where}
    return _INDEX[key]


def read_signs(path: str, ocr, every_s: float = 2.0, max_frames: int = 90, max_side: int = 1280, min_conf: float = 0.6) -> list[dict]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps * every_s)))
    out, idx, n = [], 0, 0
    while n < max_frames:
        if not cap.grab():
            break
        if idx % stride == 0:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            sc = min(1.0, max_side / max(bgr.shape[:2]))
            if sc < 1.0:
                bgr = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            res = ocr(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            for txt, conf in zip(getattr(res, "txts", None) or (), getattr(res, "scores", None) or ()):
                t = norm(txt)
                if conf >= min_conf and distinctive(t):
                    out.append({"t": round(idx / fps, 1), "text": txt.strip(), "norm": t, "ocr_conf": round(float(conf), 3)})
            n += 1
        idx += 1
    cap.release()
    return out


def _at(route: list[dict], t: float) -> tuple[float, float]:
    ts = [p["t"] for p in route]
    return float(np.interp(t, ts, [p["lat"] for p in route])), float(np.interp(t, ts, [p["lon"] for p in route]))


def match_signs(signs: list[dict], route: list[dict], places: dict, radius_m: float = 100.0, min_ratio: float = 0.85) -> list[dict]:
    if not signs or not route:
        return []
    lat, lon = places["lat"], places["lon"]
    k = 111320 * math.cos(math.radians(route[0]["lat"]))
    out, seen = [], set()
    for s in signs:
        plat, plon = _at(route, s["t"])
        d = np.hypot((lon - plon) * k, (lat - plat) * 110540)
        near = np.nonzero(d <= radius_m)[0]
        best = None
        for i in near:
            r = places["rows"][i]
            name = r["n"]
            ratio = 1.0 if s["norm"] == name or (len(s["norm"]) >= 6 and (s["norm"] in name or name in s["norm"])) else SequenceMatcher(None, s["norm"], name).ratio()
            if ratio >= min_ratio and (best is None or (ratio, -d[i]) > (best[0], -best[1])):
                best = (ratio, float(d[i]), r)
        if best and (s["norm"], best[2]["n"]) not in seen:
            seen.add((s["norm"], best[2]["n"]))
            out.append({"t": s["t"], "text": s["text"], "matched": best[2]["label"], "category": best[2].get("c"), "distance_m": round(best[1]),
                        "same_name_in_city": places["count"].get(best[2]["n"], 1), "osm": best[2].get("o")})
    return out


def conflicting_signs(signs: list[dict], route: list[dict], places: dict, max_count: int = 3, near_m: float = 300.0) -> list[dict]:
    if not signs or not route:
        return []
    k = 111320 * math.cos(math.radians(route[0]["lat"]))
    rl, ro = np.array([p["lat"] for p in route]), np.array([p["lon"] for p in route])
    out, seen = [], set()
    for s in signs:
        idx = places["where"].get(s["norm"])
        if not idx or len(idx) > max_count or s["norm"] in seen:
            continue
        seen.add(s["norm"])
        dists = [float(np.min(np.hypot((ro - places["lon"][i]) * k, (rl - places["lat"][i]) * 110540))) for i in idx]
        if min(dists) > near_m:
            r = places["rows"][idx[int(np.argmin(dists))]]
            out.append({"t": s["t"], "text": s["text"], "matched": r["label"], "category": r.get("c"), "distance_from_route_m": round(min(dists)),
                        "lat": r["y"], "lon": r["x"], "osm": r.get("o")})
    return out
