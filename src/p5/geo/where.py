from __future__ import annotations

from typing import Callable

import numpy as np

from p5.geo.exif import read_exif
from p5.geo.videoloc.signs import distinctive, norm

MIN_OCR = 0.6
MAX_PLACES = 3
MAX_WORDS = 6
PLACE_LINK_M = 218.0


def kind(category: str | None) -> str:
    if not category:
        return "place"
    if category.startswith("highway="):
        return "road"
    if category.startswith("public_transport="):
        return "stop or station"
    return category.split("=")[-1].replace("_", " ")


def places_named(idx: list[int], rows: list[dict]) -> list[dict]:
    lat = np.array([rows[i]["y"] for i in idx])
    lon = np.array([rows[i]["x"] for i in idx])
    x, y = lon * 111320 * np.cos(np.radians(lat.mean())), lat * 110540
    near = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) <= PLACE_LINK_M
    group = list(range(len(idx)))

    def root(a: int) -> int:
        while group[a] != a:
            group[a] = group[group[a]]
            a = group[a]
        return a

    for a, b in zip(*np.nonzero(np.triu(near, 1))):
        group[root(int(a))] = root(int(b))
    clusters: dict[int, list[int]] = {}
    for j in range(len(idx)):
        clusters.setdefault(root(j), []).append(j)
    out = []
    for members in clusters.values():
        m = np.array(members)
        dist = np.hypot(x[m] - x[m].mean(), y[m] - y[m].mean())
        r = rows[idx[int(m[np.argmin(dist)])]]
        out.append({"label": r["label"], "lat": r["y"], "lon": r["x"], "extent_m": int(round(float(dist.max()))),
                    "kinds": sorted({kind(rows[idx[j]].get("c")) for j in members})})
    return sorted(out, key=lambda p: (p["lat"], p["lon"]))


def read_text(rgb, ocr) -> list[tuple[str, float]]:
    res = ocr(rgb)
    boxes = getattr(res, "boxes", None)
    items = []
    for i, (t, c) in enumerate(zip(getattr(res, "txts", None) or (), getattr(res, "scores", None) or ())):
        if c >= MIN_OCR and t.strip():
            items.append((t.strip(), float(c), np.asarray(boxes[i], dtype=float) if boxes is not None and len(boxes) > i else None))
    if not items or any(b is None for _, _, b in items):
        return [(t, c) for t, c, _ in items]
    words = sorted(({"t": t, "c": c, "y": (b[:, 1].min() + b[:, 1].max()) / 2, "h": b[:, 1].max() - b[:, 1].min(),
                     "x0": b[:, 0].min(), "x1": b[:, 0].max()} for t, c, b in items), key=lambda w: w["y"])
    rows: list[list[dict]] = []
    for w in words:
        if rows and abs(w["y"] - rows[-1][-1]["y"]) <= 0.5 * max(w["h"], rows[-1][-1]["h"]):
            rows[-1].append(w)
        else:
            rows.append([w])
    lines = []
    for row in rows:
        row.sort(key=lambda w: w["x0"])
        line = [row[0]]
        for w in row[1:]:
            if w["x0"] - line[-1]["x1"] <= 1.5 * max(w["h"], line[-1]["h"]):
                line.append(w)
            else:
                lines.append(line)
                line = [w]
        lines.append(line)
    return [(" ".join(w["t"] for w in line), min(w["c"] for w in line)) for line in lines]


def names_in(text: str, where: dict) -> list[str]:
    words = norm(text).split()
    covered: set[int] = set()
    found = []
    for n in range(min(MAX_WORDS, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            span = set(range(i, i + n))
            name = " ".join(words[i:i + n])
            if span & covered or name not in where or not distinctive(name):
                continue
            covered |= span
            found.append(name)
    return found


def where_taken(data: bytes, image: Callable, ocr, places: dict | None, inside: Callable) -> dict:
    exif = read_exif(data)
    camera = " ".join(x for x in (exif.get("camera_make"), exif.get("camera_model")) if x) or None
    out: dict = {"taken_at": exif.get("taken_at"), "time_zone_recorded": exif.get("time_zone_recorded", False), "camera": camera,
                 "gps": None, "texts": [], "clues": []}
    if exif.get("lat") is not None:
        lat, lon = exif["lat"], exif["lon"]
        out["gps"] = {"lat": round(lat, 6), "lon": round(lon, 6), "inside_bengaluru": inside(lat, lon),
                      **{k: exif[k] for k in ("accuracy_m", "altitude_m", "heading_deg") if exif.get(k) is not None}}
        out["status"] = "gps"
        return out
    if ocr is None or places is None:
        return {**out, "status": "unknown", "reason": "The photo has no GPS data, and text reading is not available on this server."}
    read = read_text(image(), ocr)
    out["texts"] = [t for t, _ in read]
    seen: set[str] = set()
    for text, conf in read:
        for name in names_in(text, places["where"]):
            if name in seen:
                continue
            seen.add(name)
            idx = places["where"][name]
            found = places_named(idx, places["rows"])
            clue = {"text": text, "name": places["rows"][idx[0]]["label"], "places_in_city": len(found), "ocr_score": round(conf, 3)}
            if len(found) <= MAX_PLACES:
                clue["places"] = found
            out["clues"].append(clue)
    if any("places" in c for c in out["clues"]):
        return {**out, "status": "clues"}
    if out["clues"]:
        why = "The names read in it are each shared by more than {} places in Bengaluru, so they do not say where it was taken.".format(MAX_PLACES)
    elif out["texts"]:
        why = "Text was read in it, but none of it is the name of a place in Bengaluru's OpenStreetMap data."
    else:
        why = "No readable signboard or place name was found in it."
    return {**out, "status": "unknown", "reason": f"The photo has no GPS data. {why}"}
