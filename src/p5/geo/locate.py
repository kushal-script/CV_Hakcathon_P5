from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

import numpy as np

from p5.config import section
from p5.geo.exif import read_exif
from p5.geo.geocode import haversine_m, photon

NOISE = {"open", "sale", "welcome", "exit", "entry", "stop", "toll", "parking", "no parking", "one way", "push", "pull",
         "offer", "new", "shop", "store", "hotel", "restaurant", "bank", "atm", "police", "traffic", "school", "bus", "auto",
         "taxi", "india", "karnataka", "bengaluru", "bangalore", "city", "road", "street", "main", "cross", "nagar"}


@dataclass
class Evidence:
    text: str
    ocr_conf: float
    box: list[float]


@dataclass
class Candidate:
    lat: float
    lon: float
    label: str
    source: str
    confidence: float
    radius_m: float
    evidence: list[dict] = field(default_factory=list)


def clean(text: str) -> str:
    t = re.sub(r"[^A-Za-z0-9&' .-]", " ", text)
    return re.sub(r"\s+", " ", t).strip(" .-'")


def is_placeish(t: str) -> bool:
    letters = sum(c.isalpha() for c in t)
    digits = sum(c.isdigit() for c in t)
    return letters >= 4 and digits <= letters and t.lower() not in NOISE


def ocr_evidence(rgb: np.ndarray, engine) -> list[Evidence]:
    res = engine(rgb)
    texts = getattr(res, "txts", None) or ()
    scores = getattr(res, "scores", None) or ()
    boxes = getattr(res, "boxes", None)
    out = []
    for k, (t, s) in enumerate(zip(texts, scores)):
        c = clean(t)
        if s >= 0.6 and is_placeish(c):
            b = boxes[k].tolist() if boxes is not None else []
            out.append(Evidence(c, float(s), b))
    return out


def queries(ev: list[Evidence], max_q: int = 10) -> list[tuple[str, list[Evidence]]]:
    ev = sorted(ev, key=lambda e: -(e.ocr_conf * min(1.0, len(e.text) / 12)))
    qs: list[tuple[str, list[Evidence]]] = [(e.text, [e]) for e in ev]
    for a, b in zip(ev, ev[1:]):
        if len(qs) >= max_q * 2:
            break
        qs.append((f"{a.text} {b.text}", [a, b]))
    seen, out = set(), []
    for q, src in qs:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            out.append((q, src))
    return out[:max_q]


def similarity(q: str, name: str | None) -> float:
    if not name:
        return 0.0
    q, n = q.lower(), name.lower()
    if n in q or q in n:
        return 1.0
    return SequenceMatcher(None, q, n).ratio()


def cluster(points: list[dict], radius_m: float | None = None) -> list[dict]:
    radius_m = section("geocoding")["cluster_radius_m"] if radius_m is None else radius_m
    clusters: list[dict] = []
    for p in sorted(points, key=lambda p: -p["score"]):
        for c in clusters:
            if haversine_m((c["lat"], c["lon"]), (p["lat"], p["lon"])) <= radius_m:
                c["members"].append(p)
                break
        else:
            clusters.append({"lat": p["lat"], "lon": p["lon"], "members": [p]})
    for c in clusters:
        w = np.array([m["score"] for m in c["members"]])
        c["lat"] = float(np.average([m["lat"] for m in c["members"]], weights=w))
        c["lon"] = float(np.average([m["lon"] for m in c["members"]], weights=w))
        by_query: dict[str, float] = {}
        for m in c["members"]:
            by_query[m["evidence_key"]] = max(by_query.get(m["evidence_key"], 0.0), m["score"])
        c["score"] = sum(by_query.values())
        c["spread_m"] = max(haversine_m((c["lat"], c["lon"]), (m["lat"], m["lon"])) for m in c["members"])
    return sorted(clusters, key=lambda c: -c["score"])


def locate(data: bytes, rgb: np.ndarray, ocr_engine=None, vpr=None, top_k: int = 5) -> dict:
    exif = read_exif(data)
    out: dict = {"exif": exif, "candidates": [], "ocr": [], "methods": []}
    if "lat" in exif:
        out["candidates"].append(asdict(Candidate(exif["lat"], exif["lon"], "Photo GPS metadata", "exif", 0.99, 15.0,
                                                  [{"type": "exif", "taken_at": exif.get("taken_at")}])))
        out["methods"].append("exif")
    if ocr_engine is not None:
        ev = ocr_evidence(rgb, ocr_engine)
        out["ocr"] = [asdict(e) for e in ev]
        points = []
        for q, src in queries(ev):
            for hit in photon(q):
                sim = similarity(q, hit["name"])
                if sim < section("geocoding")["min_similarity"]:
                    continue
                conf = float(np.mean([e.ocr_conf for e in src]))
                spec = min(1.0, len(q) / 14)
                points.append({**hit, "score": conf * sim * spec / (1 + 0.5 * hit["rank"]), "query": q,
                               "evidence_key": " | ".join(sorted(e.text for e in src))})
        if points:
            out["methods"].append("ocr_geocode")
        for c in cluster(points)[:top_k]:
            best = max(c["members"], key=lambda m: m["score"])
            conf = float(1 - np.exp(-1.5 * c["score"]))
            label = best["name"] + (f", {best['area']}" if best.get("area") else "")
            ev_list = [{"type": "text", "text": m["query"], "matched": m["name"], "kind": m["kind"], "osm": m["osm"]} for m in c["members"][:6]]
            out["candidates"].append(asdict(Candidate(c["lat"], c["lon"], label, "ocr_geocode", round(conf, 3),
                                                      round(max(60.0, c["spread_m"]), 1), ev_list)))
    if vpr is not None:
        hits = vpr(rgb, top_k)
        if hits:
            out["methods"].append("visual_place_recognition")
            out["candidates"].extend(hits)
    out["candidates"].sort(key=lambda c: -c["confidence"])
    return out
