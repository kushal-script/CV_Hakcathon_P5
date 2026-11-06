from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request

from p5.geo.geocode import USER_AGENT, haversine_m
from p5.geo.route import OSRM_URL
from p5.investigate.hints import nearby_pois

VALHALLA_URL = os.environ.get("P5_VALHALLA_URL", "https://valhalla1.openstreetmap.de")
NEAR_M, CAMERA_M = 300.0, 40.0


def _get(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def osrm_routes(points: list[tuple[float, float]], alternatives: bool = True) -> list[dict]:
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)
    alt = "true" if alternatives and len(points) == 2 else "false"
    res = _get(f"{OSRM_URL}/route/v1/driving/{coords}?alternatives={alt}&geometries=geojson&overview=full&steps=true")
    if not res or res.get("code") != "Ok":
        return []
    out = []
    for r in res.get("routes", []):
        roads = []
        for leg in r.get("legs", []):
            for s in leg.get("steps", []):
                if s.get("name") and (not roads or roads[-1] != s["name"]):
                    roads.append(s["name"])
        out.append({"geometry": r["geometry"]["coordinates"], "duration_s": r["duration"], "distance_m": r["distance"], "roads": roads[:12]})
    return out


def isochrone(lat: float, lon: float, minutes=(5, 10, 15)) -> dict | None:
    q = {"locations": [{"lat": lat, "lon": lon}], "costing": "auto", "contours": [{"time": m} for m in minutes], "polygons": True}
    return _get(f"{VALHALLA_URL}/isochrone?json={urllib.parse.quote(json.dumps(q))}")


def dist_to_line_m(pt: tuple[float, float], line: list[list[float]]) -> float:
    lat0 = math.radians(pt[0])
    kx, ky = 111320 * math.cos(lat0), 110540
    px, py = pt[1] * kx, pt[0] * ky
    best = float("inf")
    for (x1, y1), (x2, y2) in zip(line, line[1:]):
        ax, ay, bx, by = x1 * kx, y1 * ky, x2 * kx, y2 * ky
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-9)))
        best = min(best, math.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def initial_bearing(line: list[list[float]], within_m: float = 600.0) -> float | None:
    if len(line) < 2:
        return None
    start = (line[0][1], line[0][0])
    end = start
    for lon, lat in line[1:]:
        end = (lat, lon)
        if haversine_m(start, end) >= within_m:
            break
    y = math.sin(math.radians(end[1] - start[1])) * math.cos(math.radians(end[0]))
    x = math.cos(math.radians(start[0])) * math.sin(math.radians(end[0])) - math.sin(math.radians(start[0])) * math.cos(math.radians(end[0])) * math.cos(math.radians(end[1] - start[1]))
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _score(route: dict, hints: dict, cameras: list[dict], seen_ids: set, t_start: float | None, gap_s: float | None) -> tuple[float, list[str]]:
    score, reasons = 1.0, []
    line = route["geometry"]
    for p in hints.get("places", []):
        if p.get("lat") is None:
            continue
        d = dist_to_line_m((p["lat"], p["lon"]), line)
        if p["role"] == "via" and d <= NEAR_M:
            score *= 1.6
            reasons.append(f"passes {p['label']} (hint)")
        if p["role"] == "avoid" and d <= NEAR_M:
            score *= 0.2
            reasons.append(f"passes {p['label']}, which the note says was avoided")
    if hints.get("direction"):
        b = initial_bearing(line)
        if b is not None:
            cos = math.cos(math.radians(b - hints["direction"]["bearing"]))
            score *= 0.5 + 0.5 * max(0.0, cos)
            reasons.append(f"initial heading {b:.0f}° vs hinted {hints['direction']['word']}")
    if gap_s is not None:
        slack = gap_s - route["duration_s"]
        if slack < -0.25 * route["duration_s"] - 60:
            score *= 0.05
            reasons.append(f"needs {route['duration_s'] / 60:.1f} min but only {gap_s / 60:.1f} min elapsed")
        else:
            score *= math.exp(-max(0.0, slack) / 1800.0)
            if slack > 300:
                reasons.append(f"{slack / 60:.0f} min unaccounted (possible stop)")
    if t_start is not None:
        for cam in cameras:
            if cam["id"] in seen_ids or cam.get("lat") is None or cam.get("start_epoch") is None:
                continue
            if dist_to_line_m((cam["lat"], cam["lon"]), line) > CAMERA_M:
                continue
            frac = haversine_m((line[0][1], line[0][0]), (cam["lat"], cam["lon"])) / max(route["distance_m"], 1.0)
            eta = t_start + frac * route["duration_s"]
            end = cam["start_epoch"] + (cam.get("duration_s") or 0)
            if cam["start_epoch"] - 120 <= eta <= end + 120:
                score *= 0.35
                reasons.append(f"camera '{cam.get('name')}' on this route was recording and did not see the vehicle")
    return score, reasons


def _dedupe(routes: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in sorted(routes, key=lambda r: -r["score"]):
        if any(r.get("destination") == o.get("destination") and abs(r["distance_m"] - o["distance_m"]) < 0.03 * o["distance_m"]
               and r["roads"][:4] == o["roads"][:4] for o in out):
            continue
        out.append(r)
    return out


def _prefer_fast(routes: list[dict], key: str = "destination") -> None:
    best: dict = {}
    for r in routes:
        best[r.get(key)] = min(best.get(r.get(key), float("inf")), r["duration_s"])
    for r in routes:
        extra = r["duration_s"] - best[r.get(key)]
        r["score"] *= math.exp(-extra / 300.0)
        if extra > 60:
            r["reasons"].append(f"{extra / 60:.1f} min slower than the fastest route to the same place")


def _normalise(routes: list[dict]) -> list[dict]:
    routes = _dedupe(routes)
    total = sum(r["score"] for r in routes) or 1.0
    for r in routes:
        r["probability"] = round(r["score"] / total, 3)
    return sorted(routes, key=lambda r: -r["probability"])


def infer(sightings: list[dict], hints: dict, cameras: list[dict]) -> dict:
    pts = sorted([s for s in sightings if s.get("lat") is not None], key=lambda s: (s.get("t_abs") is None, s.get("t_abs") or 0))
    seen = {s["camera_id"] for s in pts}
    segments = []
    for a, b in zip(pts, pts[1:]):
        gap = (b["t_abs"] - a["t_abs"]) if a.get("t_abs") is not None and b.get("t_abs") is not None else None
        routes = osrm_routes([(a["lat"], a["lon"]), (b["lat"], b["lon"])])
        for r in routes:
            r["score"], r["reasons"] = _score(r, hints, cameras, seen, a.get("t_abs"), gap)
        _prefer_fast(routes)
        segments.append({"from": a["camera_name"], "to": b["camera_name"], "gap_s": gap, "routes": _normalise(routes)})
    onward = []
    if pts:
        last = pts[-1]
        origin = (last["lat"], last["lon"])
        targets = [p for p in hints.get("places", []) if p["role"] == "destination" and p.get("lat") is not None]
        vias = [p for p in hints.get("places", []) if p["role"] == "via" and p.get("lat") is not None
                and haversine_m(origin, (p["lat"], p["lon"])) > 150]
        for act in hints.get("activities", []):
            for poi in nearby_pois(act["osm_tag"], *origin, limit=2):
                targets.append({"text": act["name"], "label": f"{poi['label']} ({act['name']})", "lat": poi["lat"], "lon": poi["lon"], "role": "stop"})
        for tgt in targets[:6]:
            cands = osrm_routes([origin, (tgt["lat"], tgt["lon"])])
            for via in vias[:1]:
                cands += osrm_routes([origin, (via["lat"], via["lon"]), (tgt["lat"], tgt["lon"])], alternatives=False)
            for r in cands:
                r["score"], r["reasons"] = _score(r, hints, cameras, seen, last.get("t_abs"), None)
                r["destination"] = tgt["label"]
                r["destination_role"] = tgt["role"]
                if tgt["role"] == "stop":
                    r["score"] *= 0.6
                    r["reasons"].append("possible stop inferred from an activity in the note")
                if last.get("t_abs") is not None:
                    r["eta_epoch"] = last["t_abs"] + r["duration_s"]
            onward.extend(cands)
        _prefer_fast(onward)
        onward = _normalise(onward)[:8]
        reach = isochrone(*origin)
    else:
        reach = None
    return {"segments": segments, "onward": onward, "reachable": reach, "hints": hints,
            "assumptions": ["Drive times are free-flow estimates from OpenStreetMap; real Bengaluru traffic is slower.",
                            "Routes are hypotheses ranked by consistency with the evidence, not confirmed paths."]}
