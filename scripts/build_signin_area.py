import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = "vahana-drishti-build/1.0 (+https://github.com/kushal-script/CV_Hakcathon_P5)"


def state(name: str, country: str, threshold: float) -> dict:
    q = urllib.parse.urlencode({"state": name, "country": country, "format": "geojson", "polygon_geojson": 1,
                                "polygon_threshold": threshold, "limit": 1})
    req = urllib.request.Request(f"https://nominatim.openstreetmap.org/search?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        found = json.load(r)["features"]
    if not found:
        raise SystemExit(f"{name}: not found in OpenStreetMap")
    f = found[0]
    p = f["properties"]
    if p.get("osm_type") != "relation" or p.get("addresstype") != "state" or f["geometry"]["type"] not in ("Polygon", "MultiPolygon"):
        raise SystemExit(f"{name}: expected a state boundary, got {p.get('display_name')}")
    return {"type": "Feature", "properties": {"name": name, "osm_relation": p["osm_id"]}, "geometry": f["geometry"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the area a device must be in to sign in, from OpenStreetMap state boundaries")
    ap.add_argument("states", nargs="+")
    ap.add_argument("--country", default="India")
    ap.add_argument("--threshold", type=float, default=0.005, help="simplification in degrees, kept well below the gateway's 2 km margin")
    ap.add_argument("--out", type=Path, default=Path("assets/bengaluru/signin_area.geojson"))
    a = ap.parse_args()
    features = []
    for i, s in enumerate(a.states):
        if i:
            time.sleep(1.1)
        features.append(state(s, a.country, a.threshold))
    name = a.states[0] if len(a.states) == 1 else ", ".join(a.states[:-1]) + " and " + a.states[-1]
    out = {"type": "FeatureCollection", "name": name, "source": "OpenStreetMap contributors (ODbL) via Nominatim",
           "retrieved": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "simplified_degrees": a.threshold, "features": features}
    a.out.write_text(json.dumps(out, separators=(",", ":")))
    print(a.out, a.out.stat().st_size // 1024, "KB:", ", ".join(f"{f['properties']['name']} (relation {f['properties']['osm_relation']})" for f in features))


if __name__ == "__main__":
    main()
