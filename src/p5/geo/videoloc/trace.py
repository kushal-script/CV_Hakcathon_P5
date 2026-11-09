from __future__ import annotations

from p5.geo.geocode import haversine_m
from p5.geo.route import map_match, mp4_location, position_at
from p5.geo.videoloc.metadata import embedded_gps
from p5.geo.videoloc.signs import load_places, read_signs
from p5.investigate.routes import osrm_routes
from p5.region import inside

CHAIN = ("amenity=fuel", "amenity=bank", "amenity=atm", "amenity=fast_food", "amenity=pharmacy", "amenity=cafe", "amenity=restaurant",
         "highway=bus_stop", "public_transport=", "shop=")
MAX_SAME_NAME = 3
MAX_SPEED_MS = 25.0
SLACK_M = 300.0


def sign_anchor(s: dict, places: dict, prev: dict | None) -> dict | None:
    idx = places["where"].get(s["norm"])
    if not idx or len(idx) > MAX_SAME_NAME:
        return None
    rows = [places["rows"][i] for i in idx if not str(places["rows"][i].get("c", "")).startswith(CHAIN)]
    if not rows:
        return None
    if prev is not None:
        best = min(rows, key=lambda r: haversine_m((prev["lat"], prev["lon"]), (r["y"], r["x"])))
        if haversine_m((prev["lat"], prev["lon"]), (best["y"], best["x"])) > MAX_SPEED_MS * max(s["t"] - prev["t"], 1.0) + SLACK_M:
            return None
    elif len(rows) == 1:
        best = rows[0]
    else:
        return None
    n = len(idx)
    return {"t": s["t"], "lat": best["y"], "lon": best["x"], "kind": "landmark", "source": f"Signboard “{s['text']}”",
            "detail": f"matches {best['label']} ({str(best.get('c', '')).replace('=', ' ')}), {n} place{'s' if n != 1 else ''} in Bengaluru with this name"}


def trace_video(path: str, ocr=None, places_path=None, progress=None) -> dict:
    gps = embedded_gps(path)
    pts = (gps or {}).get("points") or []
    if len(pts) >= 2 and inside(pts[0]["lat"], pts[0]["lon"]) is not False:
        m = map_match(pts)
        line = m["geometry"] if m.get("matched") else [[p["lon"], p["lat"]] for p in pts]
        ends = [{"t": pts[0].get("t") or 0.0, "lat": pts[0]["lat"], "lon": pts[0]["lon"], "kind": "start", "source": gps["source"], "detail": "start of the GPS track"},
                {"t": pts[-1].get("t") or 0.0, "lat": pts[-1]["lat"], "lon": pts[-1]["lon"], "kind": "end", "source": gps["source"], "detail": "end of the GPS track"}]
        return {"status": "traced", "route": line, "roads": m.get("roads", []), "points": ends, "texts": [], "gps": pts,
                "summary": f"Route from the GPS recorded in the video file ({gps['source']}), snapped to the road network."}
    anchors = []
    loc = mp4_location(path) or (pts[0] if pts else None)
    if loc and inside(loc["lat"], loc["lon"]) is not False:
        anchors.append({"t": 0.0, "lat": loc["lat"], "lon": loc["lon"], "kind": "start", "source": "Location stored in the video file",
                        "detail": "phones and some cameras save where recording started"})
    if progress:
        progress(0.1)
    signs = read_signs(path, ocr) if ocr is not None else []
    if progress:
        progress(0.8)
    places = load_places(places_path) if places_path else None
    if places:
        for s in signs:
            a = sign_anchor(s, places, anchors[-1] if anchors else None)
            if a and not any(haversine_m((a["lat"], a["lon"]), (b["lat"], b["lon"])) < 30 for b in anchors):
                anchors.append(a)
    anchors.sort(key=lambda a: a["t"])
    texts = sorted({s["text"] for s in signs})[:30]
    if len(anchors) >= 2:
        line, roads = [], []
        for a, b in zip(anchors, anchors[1:]):
            r = osrm_routes([(a["lat"], a["lon"]), (b["lat"], b["lon"])], alternatives=False)
            line += r[0]["geometry"] if r else [[a["lon"], a["lat"]], [b["lon"], b["lat"]]]
            roads += [x for x in (r[0]["roads"] if r else []) if x not in roads]
        return {"status": "partly_traced", "route": line, "roads": roads, "points": anchors, "texts": texts,
                "summary": f"Traced through {len(anchors)} places the video itself shows or stores. Between them the line is the shortest road route, "
                           "so the exact roads taken in between are not confirmed."}
    if len(anchors) == 1:
        a = anchors[0]
        return {"status": "one_point", "route": None, "roads": [], "points": anchors, "texts": texts,
                "summary": f"One place is known ({a['source'].lower()}). The video has no GPS track and no second readable place name, "
                           "so the path cannot be traced from the background."}
    return {"status": "not_traced", "route": None, "roads": [], "points": [], "texts": texts,
            "summary": "The video has no GPS track, no stored location and no readable place name that is unique in Bengaluru, "
                       "so the path cannot be traced from the background."}


def place_tracks(tracks: list[dict], trace: dict) -> None:
    gps = trace.get("gps") or []
    if not gps:
        return
    for tr in tracks:
        pos = position_at(gps, tr["first_t"])
        if pos:
            tr["lat"], tr["lon"] = pos
