from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS = Path(os.environ.get("P5_REGION_DIR", ROOT / "assets" / "bengaluru"))


def _rings(geom: dict) -> list[list[list[tuple[float, float]]]]:
    if geom.get("type") == "Polygon":
        return [[[tuple(p[:2]) for p in ring] for ring in geom["coordinates"]]]
    if geom.get("type") == "MultiPolygon":
        return [[[tuple(p[:2]) for p in ring] for ring in poly] for poly in geom["coordinates"]]
    return []


def _in_ring(ring, x: float, y: float) -> bool:
    inside, j = False, len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class Boundary:
    def __init__(self, polygons, name: str):
        self.polygons, self.name = polygons, name
        xs = [p[0] for poly in polygons for p in poly[0]]
        ys = [p[1] for poly in polygons for p in poly[0]]
        self.bbox = (min(xs), min(ys), max(xs), max(ys)) if xs else None

    @classmethod
    def load(cls, path: Path) -> "Boundary | None":
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        feats = data["features"] if data.get("type") == "FeatureCollection" else [data]
        polys = [p for f in feats for p in _rings(f.get("geometry", f))]
        name = (feats[0].get("properties") or {}).get("name", "Bengaluru") if feats else "Bengaluru"
        return cls(polys, name) if polys else None

    def contains(self, lat: float, lon: float) -> bool:
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return False
        return any(_in_ring(poly[0], lon, lat) and not any(_in_ring(h, lon, lat) for h in poly[1:]) for poly in self.polygons)


@lru_cache(maxsize=1)
def boundary() -> Boundary | None:
    return Boundary.load(ASSETS / "boundary.geojson")


@lru_cache(maxsize=1)
def registry() -> dict:
    path = ASSETS / "police_stations.json"
    if not path.exists():
        return {"divisions": {}, "stations": {}}
    data = json.loads(path.read_text())
    return {"divisions": {d["id"]: d for d in data.get("divisions", [])},
            "stations": {s["id"]: s for s in data.get("stations", [])}}


def inside(lat: float | None, lon: float | None) -> bool | None:
    if lat is None or lon is None:
        return None
    b = boundary()
    return True if b is None else b.contains(float(lat), float(lon))


def station(sid: str | None) -> dict | None:
    return registry()["stations"].get(sid) if sid else None


def division_stations(division_id: str) -> list[dict]:
    return [s for s in registry()["stations"].values() if s.get("division") == division_id]
