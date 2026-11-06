from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone


def _time(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)(Z|[+-]\d{2}:\d{2})?", str(v))
    if not m:
        return None
    y, mo, d, h, mi, s, tz = m.groups()
    dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(float(s)), int((float(s) % 1) * 1e6), tzinfo=timezone.utc)
    if tz and tz != "Z":
        sign = 1 if tz[0] == "+" else -1
        dt = dt.replace(tzinfo=None).timestamp() - sign * (int(tz[1:3]) * 3600 + int(tz[4:6]) * 60)
        return float(dt)
    return dt.timestamp()


def embedded_gps(path: str, timeout: float = 120.0) -> dict | None:
    exe = shutil.which("exiftool")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-ee3", "-n", "-json", "-G3", "-api", "LargeFileSupport=1", path], capture_output=True, timeout=timeout)
        meta = json.loads(out.stdout.decode("utf-8", "ignore") or "[]")
    except (subprocess.SubprocessError, ValueError, OSError):
        return None
    if not meta:
        return None
    tags = meta[0]
    docs: dict[str, dict] = {}
    for key, val in tags.items():
        if ":" not in key:
            continue
        grp, name = key.split(":", 1)
        docs.setdefault(grp, {})[name] = val
    pts = []
    for grp, d in docs.items():
        if not grp.startswith("Doc"):
            continue
        lat, lon = d.get("GPSLatitude"), d.get("GPSLongitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and (lat or lon):
            pts.append({"lat": float(lat), "lon": float(lon), "epoch": _time(d.get("GPSDateTime") or d.get("DateTimeOriginal")),
                        "t": float(d["SampleTime"]) if isinstance(d.get("SampleTime"), (int, float)) else None,
                        "speed_kmh": float(d["GPSSpeed"]) if isinstance(d.get("GPSSpeed"), (int, float)) else None})
    if pts:
        pts.sort(key=lambda p: (p["t"] if p["t"] is not None else 0.0, p["epoch"] or 0.0))
        return {"source": "track embedded in the video file", "points": pts}
    main = docs.get("Main") or docs.get("Track1") or {}
    lat, lon = main.get("GPSLatitude"), main.get("GPSLongitude")
    if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
        c = main.get("GPSCoordinates")
        if isinstance(c, str):
            parts = c.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    lat, lon = float(parts[0]), float(parts[1])
                except ValueError:
                    lat = lon = None
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and (lat or lon):
        return {"source": "location stored in the video file", "points": [{"lat": float(lat), "lon": float(lon), "epoch": None, "t": None, "speed_kmh": None}]}
    return None
