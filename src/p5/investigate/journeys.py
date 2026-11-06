from __future__ import annotations

from p5.geo.geocode import haversine_m
from p5.investigate.routes import osrm_routes
from p5.region import inside


def fastest(a: tuple[float, float], b: tuple[float, float]) -> dict | None:
    routes = osrm_routes([a, b], alternatives=False)
    return min(routes, key=lambda r: r["duration_s"]) if routes else None


def from_sightings(case_id: str, ordered: list[dict], segments: list[dict]) -> list[dict]:
    rows = []
    for (a, b), seg in zip(zip(ordered, ordered[1:]), segments):
        if a.get("t_abs") is None or b.get("t_abs") is None or a["camera_id"] == b["camera_id"]:
            continue
        gap = b["t_abs"] - a["t_abs"]
        routes = seg.get("routes") or []
        if gap <= 0 or not routes:
            continue
        best = min(routes, key=lambda r: r.get("freeflow_s", r["duration_s"]))
        rows.append({"ref": f"{case_id}:{a['camera_id']}:{b['camera_id']}", "from_lat": a["lat"], "from_lon": a["lon"], "to_lat": b["lat"], "to_lon": b["lon"],
                     "depart_ts": a["t_abs"], "observed_s": gap, "freeflow_s": best.get("freeflow_s", best["duration_s"]), "distance_m": best["distance_m"],
                     "note": f"{a.get('camera_name')} to {b.get('camera_name')}"})
    return rows


def from_track(ref: str, pts: list[dict], min_m: float = 1000.0, target_m: float = 3000.0, max_gap_s: float = 900.0, limit: int = 15) -> list[dict]:
    timed = [p for p in pts if p.get("epoch") is not None and p.get("lat") is not None]
    rows, i = [], 0
    while i < len(timed) - 1 and len(rows) < limit:
        j = i + 1
        while j < len(timed) - 1 and haversine_m((timed[i]["lat"], timed[i]["lon"]), (timed[j]["lat"], timed[j]["lon"])) < target_m \
                and timed[j + 1]["epoch"] - timed[i]["epoch"] <= max_gap_s:
            j += 1
        a, b = timed[i], timed[j]
        dt = b["epoch"] - a["epoch"]
        if haversine_m((a["lat"], a["lon"]), (b["lat"], b["lon"])) >= min_m and 0 < dt <= max_gap_s \
                and inside(a["lat"], a["lon"]) is not False and inside(b["lat"], b["lon"]) is not False:
            r = fastest((a["lat"], a["lon"]), (b["lat"], b["lon"]))
            if r:
                rows.append({"ref": f"{ref}:{len(rows)}", "from_lat": a["lat"], "from_lon": a["lon"], "to_lat": b["lat"], "to_lon": b["lon"],
                             "depart_ts": a["epoch"], "observed_s": dt, "freeflow_s": r["duration_s"], "distance_m": r["distance_m"], "note": "dashcam GPS"})
        i = j
    return rows


def parse_import(text: str, max_rows: int = 2000) -> tuple[list[dict], list[str]]:
    import csv
    import io
    from datetime import datetime

    from p5.store import IST
    rows, errors = [], []
    reader = csv.DictReader(io.StringIO(text))
    need = {"from_lat", "from_lon", "to_lat", "to_lon", "depart", "seconds"}
    if not reader.fieldnames or not need <= {f.strip() for f in reader.fieldnames}:
        return [], [f"The CSV needs the columns {', '.join(sorted(need))}."]
    for k, r in enumerate(reader, start=2):
        if len(rows) >= max_rows:
            errors.append(f"Only the first {max_rows} rows were read.")
            break
        try:
            r = {key.strip(): (v or "").strip() for key, v in r.items() if key}
            fl, fo, tl, to, sec = (float(r[x]) for x in ("from_lat", "from_lon", "to_lat", "to_lon", "seconds"))
            when = datetime.fromisoformat(r["depart"])
            if when.tzinfo is None:
                when = when.replace(tzinfo=IST)
        except (ValueError, KeyError):
            errors.append(f"Row {k}: unreadable values.")
            continue
        if sec <= 0 or sec > 6 * 3600:
            errors.append(f"Row {k}: travel time must be between 1 second and 6 hours.")
            continue
        if inside(fl, fo) is False or inside(tl, to) is False:
            errors.append(f"Row {k}: both ends must be inside Bengaluru.")
            continue
        rows.append({"from_lat": fl, "from_lon": fo, "to_lat": tl, "to_lon": to, "depart_ts": when.timestamp(), "observed_s": sec, "note": r.get("note") or None})
    return rows, errors
