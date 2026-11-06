#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXTRACT = "https://download.geofabrik.de/asia/india/southern-zone-latest.osm.pbf"
DRIVABLE = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street",
            "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link", "service", "road"]
SKIP_SERVICE = {"parking_aisle", "driveway", "drive-through", "emergency_access"}
NO_ACCESS = {"no", "private"}


def haversine(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


def read_ways(pbf: Path, area, bounds):
    import osmium
    from shapely.geometry import Point

    class Ways(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ways = []

        def way(self, w):
            hw = w.tags.get("highway")
            if hw not in DRIVABLE or (hw == "service" and w.tags.get("service") in SKIP_SERVICE):
                return
            if w.tags.get("access") in NO_ACCESS or w.tags.get("motor_vehicle") in NO_ACCESS or w.tags.get("area") == "yes":
                return
            try:
                pts = [(n.ref, n.location.lon, n.location.lat) for n in w.nodes]
            except osmium.InvalidLocationError:
                return
            w0, s0, e0, n0 = bounds
            if not any(w0 <= x <= e0 and s0 <= y <= n0 for _, x, y in pts):
                return
            if not any(area.contains(Point(p[1], p[2])) for p in (pts[0], pts[len(pts) // 2], pts[-1])):
                return
            ow = w.tags.get("oneway", "")
            oneway = 1 if ow in ("yes", "true", "1") else (-1 if ow == "-1" else 0)
            if oneway == 0 and ow != "no" and (w.tags.get("junction") in ("roundabout", "circular") or hw == "motorway"):
                oneway = 1
            self.ways.append({"hw": hw, "oneway": oneway, "name": w.tags.get("name") or w.tags.get("name:en") or "", "pts": pts})

    h = Ways()
    h.apply_file(str(pbf), locations=True, idx="flex_mem")
    return h.ways


def build(ways):
    use: dict[int, int] = {}
    for w in ways:
        for k, (ref, _, _) in enumerate(w["pts"]):
            use[ref] = use.get(ref, 0) + (2 if k in (0, len(w["pts"]) - 1) else 1)
    junction = {ref for ref, c in use.items() if c >= 2}
    node_index: dict[int, int] = {}
    node_xy, eu, ev, eo, coords, offsets, hws, names = [], [], [], [], [], [0], [], []
    hw_code = {hw: k for k, hw in enumerate(DRIVABLE)}
    name_index = {"": 0}

    def nid(ref, x, y):
        if ref not in node_index:
            node_index[ref] = len(node_xy)
            node_xy.append((x, y))
        return node_index[ref]

    for w in ways:
        pts, start = w["pts"], 0
        for k in range(1, len(pts)):
            if pts[k][0] in junction or k == len(pts) - 1:
                seg = pts[start:k + 1]
                xs, ys = np.array([p[1] for p in seg]), np.array([p[2] for p in seg])
                if len(seg) >= 2 and (seg[0][0] != seg[-1][0] or len(seg) > 2) and haversine(xs[:-1], ys[:-1], xs[1:], ys[1:]).sum() > 0.5:
                    eu.append(nid(seg[0][0], xs[0], ys[0]))
                    ev.append(nid(seg[-1][0], xs[-1], ys[-1]))
                    eo.append(w["oneway"])
                    coords.extend(zip(xs, ys))
                    offsets.append(len(coords))
                    hws.append(hw_code[w["hw"]])
                    names.append(name_index.setdefault(w["name"], len(name_index)))
                start = k
    return {"node_xy": np.array(node_xy), "u": np.array(eu, dtype=np.int32), "v": np.array(ev, dtype=np.int32), "oneway": np.array(eo, dtype=np.int8),
            "coords": np.array(coords), "offsets": np.array(offsets, dtype=np.int64), "hw": np.array(hws, dtype=np.int8),
            "name": np.array(names, dtype=np.int32), "names": np.array(list(name_index), dtype=object), "classes": np.array(DRIVABLE, dtype=object)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Drivable OpenStreetMap road graph for Bengaluru, used to locate moving cameras")
    ap.add_argument("--pbf", type=Path, default=ROOT / "data" / "osm" / "southern-zone-latest.osm.pbf")
    ap.add_argument("--boundary", type=Path, default=ROOT / "assets" / "bengaluru" / "boundary.geojson")
    ap.add_argument("--margin-deg", type=float, default=0.02)
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "bengaluru" / "road_graph.npz")
    a = ap.parse_args()
    import osmium
    from shapely.geometry import shape
    from shapely.prepared import prep
    if not a.pbf.exists():
        a.pbf.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(EXTRACT, headers={"User-Agent": "VahanaDrishti/0.4 (road graph build)"})
        with urllib.request.urlopen(req, timeout=600) as r, open(a.pbf, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
    reader = osmium.io.Reader(str(a.pbf), osmium.osm.osm_entity_bits.NOTHING)
    osm_date = reader.header().get("osmosis_replication_timestamp") or ""
    reader.close()
    g = json.loads(a.boundary.read_text())
    geom = shape(g["features"][0]["geometry"] if "features" in g else g["geometry"]).buffer(a.margin_deg)
    graph = build(read_ways(a.pbf, prep(geom), geom.bounds))
    np.savez_compressed(a.out, **graph, osm_source=np.array(f"OpenStreetMap contributors, ODbL, via {EXTRACT}"), osm_date=np.array(osm_date))
    print(json.dumps({"edges": int(len(graph["u"])), "nodes": int(len(graph["node_xy"])), "oneway_edges": int((graph["oneway"] != 0).sum()),
                      "osm_date": osm_date, "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
