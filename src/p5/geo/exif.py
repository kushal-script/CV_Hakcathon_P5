from __future__ import annotations

import io
from datetime import datetime

from PIL import Image

GPS_IFD = 0x8825
EXIF_IFD = 0x8769


def _deg(value, ref) -> float | None:
    try:
        d, m, s = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    out = d + m / 60 + s / 3600
    return -out if ref in ("S", "W") else out


def read_exif(data: bytes) -> dict:
    try:
        im = Image.open(io.BytesIO(data))
        exif = im.getexif()
    except Exception:
        return {}
    out: dict = {}
    gps = exif.get_ifd(GPS_IFD) if exif else {}
    if gps:
        lat, lon = _deg(gps.get(2), gps.get(1)), _deg(gps.get(4), gps.get(3))
        if lat is not None and lon is not None and not (lat == 0 and lon == 0):
            out["lat"], out["lon"] = lat, lon
        if gps.get(17) is not None:
            try:
                out["heading_deg"] = float(gps.get(17))
            except (TypeError, ValueError):
                pass
        if gps.get(6) is not None:
            try:
                out["altitude_m"] = float(gps.get(6))
            except (TypeError, ValueError):
                pass
        if gps.get(31) is not None:
            try:
                out["accuracy_m"] = float(gps.get(31))
            except (TypeError, ValueError):
                pass
    sub = exif.get_ifd(EXIF_IFD) if exif else {}
    stamp = sub.get(36867) or exif.get(306) if exif else None
    if stamp:
        offset = str(sub.get(36881) or "").strip("\x00 ")
        try:
            out["taken_at"] = datetime.strptime(str(stamp).strip("\x00 "), "%Y:%m:%d %H:%M:%S").isoformat() + offset
        except ValueError:
            out["taken_at"] = str(stamp)
        out["time_zone_recorded"] = bool(offset)
    for tag, name in ((271, "camera_make"), (272, "camera_model")):
        if exif and exif.get(tag):
            out[name] = str(exif.get(tag)).strip("\x00 ")
    return out
