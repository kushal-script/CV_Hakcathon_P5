#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from p5.geo.videoloc.signs import distinctive, norm

KEYS = ("amenity", "shop", "office", "tourism", "leisure", "healthcare", "craft", "public_transport", "railway", "building", "landuse", "place", "historic", "man_made", "highway")
NAME_KEYS = ("name", "name:en", "brand", "brand:en", "alt_name", "short_name", "official_name")


def main() -> int:
    ap = argparse.ArgumentParser(description="Named OpenStreetMap places in Bengaluru, used to check signboards read in footage")
    ap.add_argument("--pbf", type=Path, default=ROOT / "data" / "osm" / "southern-zone-latest.osm.pbf")
    ap.add_argument("--boundary", type=Path, default=ROOT / "assets" / "bengaluru" / "boundary.geojson")
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "bengaluru" / "places.json.gz")
    a = ap.parse_args()
    import osmium
    from shapely.geometry import Point, shape
    from shapely.prepared import prep
    g = json.loads(a.boundary.read_text())
    geom = shape(g["features"][0]["geometry"] if "features" in g else g["geometry"])
    area, (w0, s0, e0, n0) = prep(geom), geom.bounds
    rows: list[dict] = []

    def add(oid: str, tags: dict, lon: float, lat: float) -> None:
        if not (w0 <= lon <= e0 and s0 <= lat <= n0) or not area.contains(Point(lon, lat)):
            return
        cat = next((f"{k}={tags[k]}" for k in KEYS if k in tags), None)
        if cat is None:
            return
        label = tags.get("name") or tags.get("name:en") or tags.get("brand")
        seen = set()
        for k in NAME_KEYS:
            n = norm(tags.get(k) or "")
            if n and n not in seen and distinctive(n):
                seen.add(n)
                rows.append({"n": n, "label": label, "x": round(lon, 6), "y": round(lat, 6), "c": cat, "o": oid})

    class H(osmium.SimpleHandler):
        def node(self, n):
            if n.tags and n.location.valid() and any(k in n.tags for k in NAME_KEYS):
                add(f"node/{n.id}", dict(n.tags), n.location.lon, n.location.lat)

        def way(self, w):
            if not w.tags or not any(k in w.tags for k in NAME_KEYS):
                return
            try:
                pts = [(n.location.lon, n.location.lat) for n in w.nodes]
            except osmium.InvalidLocationError:
                return
            tags = dict(w.tags)
            if "highway" in tags:
                for p in pts[::max(1, len(pts) // 8)]:
                    add(f"way/{w.id}", tags, *p)
            elif pts:
                add(f"way/{w.id}", tags, sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    H().apply_file(str(a.pbf), locations=True, idx="flex_mem")
    a.out.write_bytes(gzip.compress(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()))
    print(json.dumps({"places": len(rows), "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
