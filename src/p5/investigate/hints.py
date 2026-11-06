from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

from p5.config import section
from p5.geo.geocode import BENGALURU_BBOX, PHOTON_URL, USER_AGENT, photon
from p5.geo.locate import similarity

ROLE_WORDS = [
    (r"(?:coming|came|starting|started|left|leaving)\s+from|from", "origin"),
    (r"heading\s+(?:to|towards)|going\s+(?:to|towards)|drove\s+(?:to|towards)|towards|toward|reached|reach|to", "destination"),
    (r"avoid(?:ing|ed)?", "avoid"),
    (r"via|through|past|passing|near|around|at|in|by|on", "via"),
]
ACTIVITIES = {
    r"petrol|fuel|diesel|cng|gas station|bunk": ("amenity:fuel", "fuel station"),
    r"hospital|clinic|injur": ("amenity:hospital", "hospital"),
    r"\batm\b|cash|bank": ("amenity:atm", "ATM"),
    r"airport|flight": ("aeroway:aerodrome", "airport"),
    r"railway|train|station": ("railway:station", "railway station"),
    r"bus stand|bus station|majestic": ("amenity:bus_station", "bus station"),
    r"\bmall\b|shopping": ("shop:mall", "mall"),
    r"hotel|lodge|stay": ("tourism:hotel", "hotel"),
    r"\bbar\b|\bpub\b|drink": ("amenity:bar", "bar"),
    r"parking|parked": ("amenity:parking", "parking"),
    r"toll": ("barrier:toll_booth", "toll booth"),
    r"temple|church|mosque": ("amenity:place_of_worship", "place of worship"),
    r"restaurant|food|eat|dinner|lunch|breakfast|hotel": ("amenity:restaurant", "restaurant"),
}
DIRECTIONS = {"northeast": 45, "north-east": 45, "southeast": 135, "south-east": 135, "southwest": 225, "south-west": 225,
              "northwest": 315, "north-west": 315, "north": 0, "east": 90, "south": 180, "west": 270}
TIME_RE = re.compile(r"(?:(after|before|around|about|by|at|since|till|until)\s+)?(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", re.I)
TIME24_RE = re.compile(r"(?:(after|before|around|about|by|at|since|till|until)\s+)?\b([01]?\d|2[0-3])[:.]([0-5]\d)\b(?!\s*(?:am|pm))", re.I)
STOP = {"the", "a", "an", "his", "her", "their", "some", "any", "night", "evening", "morning", "afternoon", "time"}


def parse_times(text: str) -> list[dict]:
    out = []
    for m in TIME_RE.finditer(text):
        h, mm, ap = int(m.group(2)), int(m.group(3) or 0), m.group(4).lower().replace(".", "")
        if h > 12:
            continue
        h = (h % 12) + (12 if ap == "pm" else 0)
        out.append({"text": m.group(0).strip(), "minutes": h * 60 + mm, "qualifier": (m.group(1) or "at").lower()})
    for m in TIME24_RE.finditer(text):
        if any(m.start() >= o_start and m.start() < o_end for o_start, o_end in [(x.start(), x.end()) for x in TIME_RE.finditer(text)]):
            continue
        out.append({"text": m.group(0).strip(), "minutes": int(m.group(2)) * 60 + int(m.group(3)), "qualifier": (m.group(1) or "at").lower()})
    return out


CLAUSE_RE = re.compile(r"(?<=[A-Za-z0-9])[.!?;,]\s+|[!?;]\s+")


def parse_places(text: str) -> list[dict]:
    found, seen = [], set()
    clauses, offset = [], 0
    for part in CLAUSE_RE.split(text):
        start = text.find(part, offset)
        clauses.append((part, start))
        offset = start + len(part)
    for pattern, role in ROLE_WORDS:
        rx = re.compile(rf"\b(?i:{pattern})\s+(?i:the\s+)?((?:[A-Z0-9][\w'&-]*)(?:\s+(?:[A-Z0-9][\w'&-]*|of|and|&))*)")
        matches = [(m, base) for clause, base in clauses for m in rx.finditer(clause)]
        for m, base in matches:
            phrase = m.group(1).strip(" .,'")
            words = [w for w in phrase.split() if w.lower() not in STOP]
            if not words or len(phrase) < 3 or phrase.lower() in seen or TIME_RE.fullmatch(phrase):
                continue
            seen.add(phrase.lower())
            found.append({"text": phrase, "role": role, "span": [base + m.start(1), base + m.end(1)]})
    found.sort(key=lambda f: f["span"][0])
    return found


def geocode_places(places: list[dict]) -> list[dict]:
    out = []
    for p in places:
        hits = [h for h in photon(p["text"], 5) if similarity(p["text"], h["name"]) >= section("geocoding")["min_similarity"]]
        if hits:
            h = hits[0]
            out.append({**p, "lat": h["lat"], "lon": h["lon"], "label": h["name"] + (f", {h['area']}" if h.get("area") else ""),
                        "kind": h["kind"]})
        else:
            out.append({**p, "lat": None, "lon": None, "label": None, "kind": None})
    return out


def nearby_pois(tag: str, lat: float, lon: float, limit: int = 3) -> list[dict]:
    params = {"q": tag.split(":")[1].replace("_", " "), "osm_tag": tag, "lat": lat, "lon": lon, "limit": limit,
              "bbox": ",".join(map(str, BENGALURU_BBOX))}
    req = urllib.request.Request(f"{PHOTON_URL}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            feats = json.loads(r.read().decode()).get("features", [])
    except Exception:
        return []
    return [{"lat": f["geometry"]["coordinates"][1], "lon": f["geometry"]["coordinates"][0],
             "label": f["properties"].get("name") or tag.split(":")[1]} for f in feats]


def parse(text: str) -> dict:
    raw = parse_places(text)
    places = geocode_places(raw)
    masked = text
    for p in sorted(raw, key=lambda p: -p["span"][0]):
        masked = masked[:p["span"][0]] + " " * (p["span"][1] - p["span"][0]) + masked[p["span"][1]:]
    acts = []
    low = masked.lower()
    for rx, (tag, name) in ACTIVITIES.items():
        if re.search(rx, low) and tag not in [a["osm_tag"] for a in acts]:
            acts.append({"keyword": re.search(rx, low).group(0), "osm_tag": tag, "name": name})
    direction = next(({"word": w, "bearing": b} for w, b in DIRECTIONS.items() if re.search(rf"\b{w}(?:wards?|bound)?\b", low)), None)
    return {"text": text, "places": places, "times": parse_times(text), "activities": acts, "direction": direction}
