from __future__ import annotations

import csv
import io
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

from p5.config import section
from p5.geo.geocode import USER_AGENT, haversine_m

OSRM_URL = section("services")["osrm_url"]


def _iso(t: str | None) -> float | None:
    if not t:
        return None
    try:
        return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def parse_gpx(data: bytes) -> list[dict]:
    root = ET.fromstring(data)
    pts = []
    for el in root.iter():
        if el.tag.endswith("trkpt") or el.tag.endswith("rtept") or el.tag.endswith("wpt"):
            t = next((c.text for c in el if c.tag.endswith("time")), None)
            pts.append({"lat": float(el.get("lat")), "lon": float(el.get("lon")), "epoch": _iso(t)})
    return pts


def parse_csv(data: bytes) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8", "ignore"))))
    pts = []
    for r in rows:
        k = {c.lower().strip(): v for c, v in r.items() if c}
        lat = k.get("lat") or k.get("latitude")
        lon = k.get("lon") or k.get("lng") or k.get("longitude")
        if lat and lon:
            t = k.get("t") or k.get("time") or k.get("seconds")
            try:
                tt = float(t) if t not in (None, "") else None
            except ValueError:
                tt = _iso(t)
            pts.append({"lat": float(lat), "lon": float(lon), "epoch": tt})
    return pts


def parse_nmea(data: bytes) -> list[dict]:
    pts = []
    for line in data.decode("ascii", "ignore").splitlines():
        f = line.strip().split(",")
        if f[0].endswith("RMC") and len(f) > 9 and f[2] == "A":
            def dm(v, h):
                d = float(v[:2 if h in "NS" else 3]) + float(v[2 if h in "NS" else 3:]) / 60
                return -d if h in "SW" else d
            try:
                hh, mm, ss = float(f[1][:2]), float(f[1][2:4]), float(f[1][4:])
                pts.append({"lat": dm(f[3], f[4]), "lon": dm(f[5], f[6]), "epoch": hh * 3600 + mm * 60 + ss})
            except (ValueError, IndexError):
                continue
    return pts


def parse_track(data: bytes, filename: str) -> list[dict]:
    name = filename.lower()
    if name.endswith(".gpx") or data.lstrip().startswith(b"<"):
        pts = parse_gpx(data)
    elif name.endswith((".nmea", ".txt", ".log")) or b"$GP" in data[:2000] or b"$GN" in data[:2000]:
        pts = parse_nmea(data)
    else:
        pts = parse_csv(data)
    times = [p["epoch"] for p in pts]
    if pts and all(t is not None for t in times):
        t0 = times[0]
        for p in pts:
            p["t"] = p["epoch"] - t0
    else:
        for k, p in enumerate(pts):
            p["t"] = None
    return pts


def mp4_location(path: str) -> dict | None:
    with open(path, "rb") as fh:
        head = fh.read(4 * 2**20)
        fh.seek(max(0, os.path.getsize(path) - 4 * 2**20))
        tail = fh.read()
    for blob in (head, tail):
        k = blob.find(b"\xa9xyz")
        if k >= 0:
            m = re.search(rb"([+-]\d+\.\d+)([+-]\d+\.\d+)", blob[k:k + 64])
            if m:
                return {"lat": float(m.group(1)), "lon": float(m.group(2)), "source": "video metadata"}
    return None


def _match_chunk(pts: list[dict]) -> dict | None:
    coords = ";".join(f"{p['lon']:.6f},{p['lat']:.6f}" for p in pts)
    params = ["geometries=geojson", "overview=full", "steps=true", "radiuses=" + ";".join(["35"] * len(pts))]
    if all(p.get("t") is not None for p in pts):
        params.append("timestamps=" + ";".join(str(int(round(p["t"]))) for p in pts))
    req = urllib.request.Request(f"{OSRM_URL}/match/v1/driving/{coords}?{'&'.join(params)}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def map_match(pts: list[dict], chunk: int = 90) -> dict:
    pts = [p for p in pts if p.get("lat") is not None]
    if len(pts) < 2:
        return {"matched": False, "reason": "need at least two points", "raw": pts}
    if len(pts) > 600:
        step = len(pts) / 600
        pts = [pts[int(k * step)] for k in range(600)]
    line, roads, conf, dist = [], [], [], 0.0
    for k in range(0, len(pts) - 1, chunk - 1):
        res = _match_chunk(pts[k:k + chunk])
        if not res or res.get("code") != "Ok":
            continue
        for m in res.get("matchings", []):
            line.extend(m["geometry"]["coordinates"])
            conf.append(m.get("confidence", 0.0))
            dist += m.get("distance", 0.0)
            for leg in m.get("legs", []):
                for s in leg.get("steps", []):
                    n = s.get("name")
                    if n and (not roads or roads[-1] != n):
                        roads.append(n)
    if not line:
        raw_len = sum(haversine_m((a["lat"], a["lon"]), (b["lat"], b["lon"])) for a, b in zip(pts, pts[1:]))
        return {"matched": False, "reason": "map matching service unavailable or no road match", "raw": pts,
                "raw_distance_m": raw_len}
    return {"matched": True, "geometry": line, "distance_m": dist, "confidence": float(sum(conf) / len(conf)),
            "roads": roads, "raw": pts}


def position_at(pts: list[dict], t: float) -> tuple[float, float] | None:
    timed = [p for p in pts if p.get("t") is not None]
    if not timed:
        return None
    if t <= timed[0]["t"]:
        return timed[0]["lat"], timed[0]["lon"]
    for a, b in zip(timed, timed[1:]):
        if a["t"] <= t <= b["t"]:
            w = (t - a["t"]) / max(b["t"] - a["t"], 1e-6)
            return a["lat"] + w * (b["lat"] - a["lat"]), a["lon"] + w * (b["lon"] - a["lon"])
    return timed[-1]["lat"], timed[-1]["lon"]
