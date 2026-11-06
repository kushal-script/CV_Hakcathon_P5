from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
import urllib.request
from functools import lru_cache

BENGALURU_BBOX = (77.37, 12.83, 77.88, 13.15)
BENGALURU_CENTRE = (12.9716, 77.5946)
PHOTON_URL = os.environ.get("P5_PHOTON_URL", "https://photon.komoot.io/api/")
USER_AGENT = os.environ.get("P5_USER_AGENT", "p5-vehicle-recognition/0.1 (hackathon prototype)")
_last_call = [0.0]


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


@lru_cache(maxsize=2048)
def photon(query: str, limit: int = 5) -> tuple[dict, ...]:
    wait = 0.35 - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    params = {"q": query, "limit": limit, "bbox": ",".join(map(str, BENGALURU_BBOX)),
              "lat": BENGALURU_CENTRE[0], "lon": BENGALURU_CENTRE[1]}
    req = urllib.request.Request(f"{PHOTON_URL}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    _last_call[0] = time.time()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return ()
    out = []
    for rank, f in enumerate(data.get("features", [])):
        p, (lon, lat) = f.get("properties", {}), f["geometry"]["coordinates"][:2]
        out.append({"lat": lat, "lon": lon, "name": p.get("name"), "kind": f"{p.get('osm_key')}={p.get('osm_value')}",
                    "area": p.get("district") or p.get("locality") or p.get("suburb") or p.get("city"),
                    "street": p.get("street"), "osm": f"{p.get('osm_type')}/{p.get('osm_id')}", "rank": rank})
    return tuple(out)
