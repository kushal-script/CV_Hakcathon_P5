#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import ssl
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "police_raw"
OUT = ROOT / "assets" / "bengaluru"
UA = "VahanaDrishti/0.3 (https://github.com/kushal-script/CV_Hakcathon_P5; police station dataset)"
BCP = "https://bcp.karnataka.gov.in"
PAGES = {
    "bcp": (f"{BCP}/24/law-&-order-/en", "Bengaluru City Police: Law & Order",
            "Authority for law-and-order, CEN and women police stations, their division and phone numbers."),
    "bcp_traffic": (f"{BCP}/26/traffic-/en", "Bengaluru City Police: Traffic",
                    "Traffic police stations by traffic division with phone numbers, cross-checked against btp."),
    "bcp_cyber": (f"{BCP}/10/cyber-crime-police-station-/en", "Bengaluru City Police: Cyber Crime Police Station",
                  "City-wide Cyber Crime Police Station in the Commissioner's office premises; also states that 8 CEN police stations exist."),
    "bcp_overview": (f"{BCP}/6/overview/en", "Bengaluru City Police: Overview",
                     "States the current law-and-order divisions (zones), each headed by a DCP."),
    "bcp_officers": (f"{BCP}/41/-senior-officers-list-/en", "Bengaluru City Police: Senior Officers list",
                     "Names a DCP for each law-and-order division (West and East zones) and for the four traffic divisions."),
    "btp": ("https://btp.karnataka.gov.in/btp_map_locations/en", "Bengaluru Traffic Police: Traffic Police Station Locator",
            "Authority for traffic police stations, their traffic division (DCP) and phone numbers. Its coordinates break ties between OSM "
            "matches and, when they pass the checks in coordinate_rule, place stations that OSM and the geocoder do not (official_btp)."),
    "ksp": ("https://ksp.karnataka.gov.in/home-fetch?select=district_id&value=5&dependent=district_row_id",
            "Karnataka State Police: Police Station Locator, Bengaluru City (linked from the BCP site)",
            "Used to link alternate official names and to corroborate OSM matches. When they pass the checks in coordinate_rule its "
            "coordinates place stations that OSM, the geocoder and the BTP locator do not (official_ksp); many of its values are invalid "
            "and are dropped. Addresses are not copied: they are cut at 45 characters at the source."),
}
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
NOMINATIM = "https://nominatim.openstreetmap.org/search"
GCC_CA = ("http://secure.globalsign.com/cacert/gsgccr46ovtlsca2025.crt",
          "d160e2de4e56cb10b66c1cb0adcb79cf93c78dcd9db30c2018220262c04063f9")
OSM_BOX = (12.70, 77.25, 13.35, 77.95)
GEO_BOX = (77.37, 12.83, 77.88, 13.15)
FIVE = ("central", "north", "south", "east", "west")
PREFIX = {"law_and_order": "ps", "traffic": "tps", "cen": "cen", "women": "wps", "other": "ops"}
AUTHORITY = ("bcp", "bcp_cyber", "bcp_traffic", "btp")
STOP = {"pi", "ps", "police", "station", "stn", "traffic", "trafic", "tr", "trf", "women", "womem", "womens", "mahila", "cen",
        "crime", "the", "of", "and", "bengaluru", "bangalore", "city", "zone"}
FOLD = (("electronics", "electronic"), ("grounds", "ground"), ("layout", ""), ("puram", "pura"), ("hally", "halli"), ("th", "t"),
        ("dh", "d"), ("bh", "b"), ("sh", "s"), ("gh", "g"), ("kh", "k"), ("ph", "p"), ("w", "v"))
ABBREVIATIONS = {"V V Puram": "Vishveshwarapuram", "K G Nagar": "Kempegowda Nagar", "KS Layout": "Kumaraswamy Layout",
                 "J.P.Nagar": "Jayaprakash Nagar", "C K Acchukattu": "Channammanakere Achukattu", "International Airport": "BIAL",
                 "Devanahalli Airport": "BIAL", "Govindrajpura": "Govindapura"}
DISPLAY = {"White Field": "Whitefield", "B.Gudi": "Basavanagudi"}
SUFFIX = {"halli", "pura", "puram", "palya", "pete"}
NOT_STATION = re.compile(r"outpost|chowk|check ?post|checkpoint|cabin|booth|kiosk|\bbeat\b|commis+ion|headquarter|\bcid\b|director|"
                         r"superintend|prison|reserve|armed|\bdcp\b|\bacp\b|academy|training|quarters|museum", re.I)

ARGS = argparse.Namespace(max_vertices=2000)
INDEX: dict[str, dict] = {}
LAST: dict[str, float] = {}
CTX: list[ssl.SSLContext] = []
OFFLINE = [False]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def cache_key(url: str, data: bytes | None) -> str:
    return hashlib.sha1(url.encode() + b"\n" + (data or b"")).hexdigest()[:20]


def tls() -> ssl.SSLContext:
    if not CTX:
        der = get(GCC_CA[0], accept="application/pkix-cert")
        if hashlib.sha256(der).hexdigest() != GCC_CA[1]:
            raise RuntimeError("GlobalSign intermediate certificate does not match the pinned hash")
        ctx = ssl.create_default_context()
        ctx.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(der))
        CTX.append(ctx)
    return CTX[0]


def get(url: str, params: dict | None = None, data: bytes | None = None, accept: str = "*/*", gap: float = 1.0, tries: int = 6) -> bytes:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    name = cache_key(url, data)
    path = RAW / name
    if path.exists() and name in INDEX:
        return path.read_bytes()
    if OFFLINE[0]:
        raise RuntimeError(f"not cached: {url}")
    host = urllib.parse.urlsplit(url).netloc
    for k in range(tries):
        time.sleep(max(0.0, LAST.get(host, 0.0) + gap - time.time()))
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Accept": accept})
            with urllib.request.urlopen(req, timeout=180, context=tls() if url.startswith("https") else None) as r:
                body = r.read()
            break
        except urllib.error.HTTPError as e:
            if k == tries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            retry = e.headers.get("Retry-After")
            time.sleep(float(retry) if retry and retry.isdigit() else 15 * (k + 1))
        except urllib.error.URLError as e:
            if k == tries - 1 or isinstance(e.reason, ssl.SSLCertVerificationError):
                raise
            time.sleep(2 ** k + 1)
        except (TimeoutError, ConnectionError):
            if k == tries - 1:
                raise
            time.sleep(2 ** k + 1)
        finally:
            LAST[host] = time.time()
    path.write_bytes(body)
    INDEX[name] = {"url": url, "post": data.decode() if data else None, "retrieved": now(), "bytes": len(body)}
    (RAW / "index.json").write_text(json.dumps(INDEX, indent=1))
    return body


def retrieved(url: str, data: bytes | None = None) -> str | None:
    return INDEX.get(cache_key(url, data), {}).get("retrieved")


def overpass(query: str) -> tuple[dict, str, str | None]:
    data = urllib.parse.urlencode({"data": query}).encode()
    order = sorted(OVERPASS, key=lambda u: cache_key(u, data) not in INDEX)
    for k, url in enumerate(order):
        try:
            return json.loads(get(url, data=data, accept="application/json", gap=5.0, tries=4)), url, retrieved(url, data)
        except Exception as e:
            if k == len(order) - 1:
                raise
            print(f"overpass {url} failed ({type(e).__name__}), trying the next mirror", flush=True)


def clean(fragment: str) -> str:
    s = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", fragment)
    s = re.sub(r"(?i)<br\s*/?>|</(p|h\d|div|li)>", "\n", s)
    s = html.unescape(re.sub(r"<[^>]+>", " ", s)).replace("\xa0", " ")
    return "\n".join(t for t in (re.sub(r"\s+", " ", line).strip() for line in s.split("\n")) if t)


def panes(page: str) -> list[tuple[str, str]]:
    names = dict(re.findall(r'(?is)<button[^>]*?data-bs-target="#(content-\d+)"[^>]*>(.*?)</button>', page))
    starts = [(m.start(), m.group(1)) for m in re.finditer(r'<div[^>]+id="(content-\d+)"', page)]
    end = page.find('id="accordionExample"')
    out = []
    for k, (pos, cid) in enumerate(starts):
        stop = starts[k + 1][0] if k + 1 < len(starts) else (end if end > pos else len(page))
        out.append((clean(names.get(cid, cid)), page[pos:stop]))
    return out


def table_rows(fragment: str) -> list[list[str]]:
    return [[clean(c) for c in re.findall(r"(?is)<t[dh]\b[^>]*>(.*?)</t[dh]>", tr)] for tr in re.findall(r"(?is)<tr\b.*?</tr>", fragment)]


def phones(s: str) -> list[str]:
    out = []
    for chunk in re.split(r"[,/;\n|]|\s{2,}", s or ""):
        d = re.sub(r"\D", "", chunk)
        while len(d) >= 8:
            if d.startswith("91") and len(d) >= 12:
                d = d[2:]
                continue
            if d.startswith("080") and len(d) >= 11:
                num, d = "080-" + d[3:11], d[11:]
            elif d.startswith("802") and len(d) == 10:
                num, d = "080-" + d[2:], ""
            elif d[0] in "6789" and len(d) >= 10:
                num, d = d[:10], d[10:]
            elif d[0] == "2":
                num, d = "080-" + d[:8], d[8:]
            else:
                break
            if len(set(num.replace("080-", ""))) > 1 and num not in out:
                out.append(num)
    return out


def fold(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()


def key(name: str) -> str:
    s = re.sub(r"\b\d+\s*(st|nd|rd|th)?\s*(block|stage|phase|sector)\b", " ", fold(name))
    s = re.sub(r"'s\b|\bp\.\s?s\b\.?", " ", s)
    s = re.sub(r"\blay\s*out\b", "layout", s)
    out, run = [], []

    def flush():
        if run:
            w = "".join(run)
            for a, b in FOLD:
                w = w.replace(a, b)
            w = re.sub(r"(.)\1+", r"\1", w)
            out.append(re.sub(r"(.)\1+", r"\1", w[:1] + re.sub(r"[aeiouy]", "", w[1:])))
            run.clear()
    for t in re.sub(r"[^a-z]+", " ", s).split():
        if t in STOP:
            continue
        if len(t) <= 3:
            flush()
            out.append(t)
        else:
            run.append(t)
    flush()
    return "".join(out)


def exact(name: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", fold(name)).split())


def near(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1 or min(len(a), len(b)) < 4 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def kind_of(name: str, kn: str = "") -> str:
    low = fold(name)
    if re.search(r"traf+ic|\btr\b|\btr\.", low) or "ಸಂಚಾರ" in kn:
        return "traffic"
    if re.search(r"\bcen\b", low):
        return "cen"
    if re.search(r"\bwom[ae][nm]|mahila", low) or "ಮಹಿಳಾ" in kn:
        return "women"
    if "dcre" in low or low.startswith("ccb"):
        return "excluded"
    if "cyber" in low:
        return "other"
    return "law_and_order"


def tidy(place: str, upper: bool = False) -> str:
    place = " ".join(place.replace("\u200b", " ").split()).strip(" -.,")
    if upper and place.isupper():
        place = " ".join(w if len(w) <= 2 or not re.search(r"[AEIOU]", w) else w.title() for w in place.split())
    words = []
    for w in place.split():
        parts = re.split(r"(?<=[a-z])(?=[A-Z])", w)
        words.append(parts[0])
        for part in parts[1:]:
            if part.lower() in SUFFIX:
                words[-1] += part.lower()
            else:
                words.append(part)
    place = " ".join(w[:1].upper() + w[1:] if w.islower() else w for w in words)
    return DISPLAY.get(place, place)


def place_of(raw: str, caps_style: bool = False) -> str:
    s = re.sub(r"\(.*?\)", " ", raw)
    s = re.sub(r"(?i)\btr\.?\s*ps\b|\btr\.|\bp\.s\.?", " ", s)
    s = re.sub(r"(?i)\b(pi|ps|police station|police|station|traffic|tr|trf|women|womem|cen|crime)\b", " ", s)
    return tidy(s, caps_style and raw.isupper())


def division_label(label: str) -> str:
    return " ".join(w.capitalize() for w in re.sub(r"(?i)\bdivision\b|\bdcp\b|-", " ", label).split())


def coord(lat, lon) -> tuple[float, float] | None:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    return (lat, lon) if OSM_BOX[0] <= lat <= OSM_BOX[2] and OSM_BOX[1] <= lon <= OSM_BOX[3] else None


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742.0 * math.asin(math.sqrt(h))


def record(src: str, kind: str, raw: str, place: str, division: str | None = None, numbers: list[str] | None = None,
           at: tuple[float, float] | None = None, extra: list[str] | None = None) -> dict:
    names = [place] + (extra or [])
    keys = {k for k in map(key, names) if k}
    keys |= {key(v) for a, v in ABBREVIATIONS.items() if key(a) in keys}
    return {"src": src, "kind": kind, "raw": raw, "place": place, "extra": extra or [], "division": division, "numbers": numbers or [],
            "at": at, "keys": keys}


def parse_bcp(page: str) -> list[dict]:
    out = []
    for label, frag in panes(page):
        division = division_label(label)
        for cells in table_rows(frag):
            if len(cells) < 2:
                continue
            raw = cells[1]
            low = raw.lower()
            if not raw or re.match(r"(sl|police station|unit)\b", low) or re.match(r"(dcp|acp)\b", low):
                continue
            kind = kind_of(raw)
            place = division if kind == "cen" else place_of(raw, True)
            if not place or key(place) == key(division):
                place = division
            out.append(record("bcp", kind, raw, place, division, phones("\n".join(cells[2:]))))
    return out


def parse_bcp_traffic(page: str) -> list[dict]:
    out = []
    for label, frag in panes(page):
        division = division_label(label)
        for cells in table_rows(frag):
            if len(cells) < 2:
                continue
            raw = cells[1]
            low = raw.lower()
            if not raw or re.match(r"(sl|police station|unit)\b", low) or re.match(r"(dcp|acp)\b", low):
                continue
            out.append(record("bcp_traffic", "traffic", raw, place_of(raw, True), division, phones("\n".join(cells[2:]))))
    return out


def parse_btp(page: str) -> list[dict]:
    out, dcp = [], None
    block = page[page.find("dept_docssam"):]
    anchors = list(re.finditer(r"(?is)<a\b[^>]*?data-lat='([^']*)'\s+data-lng='([^']*)'[^>]*>(.*?)</a>", block))
    for k, m in enumerate(anchors):
        label = clean(m.group(3))
        if re.match(r"(?i)dcp\b", label):
            dcp = division_label(label)
            continue
        if re.match(r"(?i)acp\b", label):
            continue
        tail = block[m.end():anchors[k + 1].start() if k + 1 < len(anchors) else len(block)]
        tail = re.split(r"(?i)<li\b|</li>", tail)[0]
        info = clean(tail)
        out.append(record("btp", "traffic", label, place_of(label), dcp, phones(re.sub(r"(?i)email:\S*", " ", info)),
                          coord(m.group(1), m.group(2))))
    return out


def parse_ksp(page: str) -> list[dict]:
    out = []
    for value, label in re.findall(r'(?is)<option value="(.*?)">(.*?)</option>', page):
        parts = html.unescape(value).split("!@#$%")
        if len(parts) < 6:
            continue
        label = clean(label)
        names = [label]
        m = re.match(r"(.*?)\((.*?)\)(.*)", label)
        if m:
            names = [m.group(1) + m.group(3), m.group(2) + m.group(3)]
        places = [place_of(n) for n in names]
        out.append(record("ksp", kind_of(label), label, places[0], None, phones(parts[2]), coord(parts[4], parts[5]), places[1:]))
    return out


def invalidate_duplicates(recs: list[dict]) -> None:
    shared = defaultdict(list)
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            if a["at"] and b["at"] and a["src"] == b["src"] and a["kind"] == b["kind"] and km(a["at"], b["at"]) < 0.05:
                shared[id(a)].append((a, b))
                shared[id(b)].append((b, a))
    for pairs in shared.values():
        r, other = pairs[0]
        r["dropped"] = (r["at"], f"within {km(r['at'], other['at']) * 1000:.0f} m of the {r['src'].upper()} position of '{other['raw']}'")
    for pairs in shared.values():
        pairs[0][0]["at"] = None


def link(recs: list[dict], log: list[str]) -> list[list[int]]:
    parent = list(range(len(recs)))
    srcs = [{r["src"]} for r in recs]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j, why):
        a, b = find(i), find(j)
        if a == b:
            return
        if srcs[a] & srcs[b]:
            log.append(f"not linked ({why}): '{recs[i]['raw']}' [{recs[i]['src']}] and '{recs[j]['raw']}' [{recs[j]['src']}] would merge two rows of one source")
            return
        parent[b] = a
        srcs[a] |= srcs[b]

    by_key = defaultdict(list)
    for i, r in enumerate(recs):
        for k in r["keys"]:
            by_key[(r["kind"], k)].append(i)
    for (_, k), idx in sorted(by_key.items()):
        per_src = Counter(recs[i]["src"] for i in idx)
        idx = [i for i in idx if per_src[recs[i]["src"]] == 1]
        for i in idx[1:]:
            union(idx[0], i, f"name {k}")
    by_num = defaultdict(list)
    for i, r in enumerate(recs):
        for n in r["numbers"]:
            by_num[(r["src"], n)].append(i)
    unique = defaultdict(list)
    for (src, n), idx in by_num.items():
        if len(idx) == 1:
            unique[n].append(idx[0])
    for n, idx in sorted(unique.items()):
        for kind in sorted({recs[i]["kind"] for i in idx}):
            same = [i for i in idx if recs[i]["kind"] == kind]
            for i in same[1:]:
                if find(i) != find(same[0]):
                    union(same[0], i, f"phone {n}")
                    if find(i) == find(same[0]):
                        log.append(f"linked by phone {n}: '{recs[same[0]]['raw']}' [{recs[same[0]]['src']}] = '{recs[i]['raw']}' [{recs[i]['src']}]")
    authority_keys = {(r["kind"], k) for r in recs if r["src"] in AUTHORITY for k in r["keys"]}
    for i, r in enumerate(recs):
        root = find(i)
        if r["src"] not in AUTHORITY or "ksp" in srcs[root]:
            continue
        cands = [j for j, q in enumerate(recs) if q["src"] == "ksp" and q["kind"] == r["kind"] and not (srcs[find(j)] & set(AUTHORITY))
                 and not any((q["kind"], k) in authority_keys for k in q["keys"])
                 and any(near(a, b) for a in r["keys"] for b in q["keys"])]
        if len(cands) == 1:
            log.append(f"linked by near spelling: '{r['raw']}' [{r['src']}] = '{recs[cands[0]]['raw']}' [ksp]")
            union(i, cands[0], "near spelling")
    groups = defaultdict(list)
    for i in range(len(recs)):
        groups[find(i)].append(i)
    return list(groups.values())


def osm_features(data: dict) -> list[dict]:
    out = []
    for e in data["elements"]:
        t = e.get("tags", {})
        lat, lon = e.get("lat", e.get("center", {}).get("lat")), e.get("lon", e.get("center", {}).get("lon"))
        names = [(k, re.split(r"\s+[-\u2013]|\(", v)[0].strip()) for k in ("name", "name:en", "official_name", "alt_name", "old_name",
                 "short_name", "loc_name") for v in t.get(k, "").split(";") if v.strip()]
        label = " ".join([t.get("name", ""), t.get("operator", "")])
        if lat is None or (NOT_STATION.search(label) and "police station" not in label.lower()):
            continue
        variants = [(k, v, kind_of(v, t.get("name:kn", "") if k == "name" else "")) for k, v in names]
        primary_kind = variants[0][2] if variants else kind_of("", t.get("name:kn", ""))
        says_station = bool(re.search(r"police station|ಠಾಣೆ", " ".join([t.get("name", ""), t.get("name:kn", "")]), re.I))
        out.append({"osm": f"{e['type']}/{e['id']}", "at": (lat, lon), "names": variants, "kind": primary_kind,
                    "generic": says_station and not any(key(v) for _, v, _ in variants),
                    "phones": phones(";".join(t.get(k, "") for k in ("phone", "contact:phone"))), "label": t.get("name", "")})
    return out


def pick(st: dict, cands: list[tuple[int, dict]], log: list[str]) -> tuple[dict | None, str]:
    if not cands:
        return None, ""
    top = max(s for s, _ in cands)
    best = [f for s, f in cands if s == top]
    official = st["official_at"]
    if max(km(a["at"], b["at"]) for a in best for b in best) <= 0.3:
        best.sort(key=lambda f: (km(f["at"], official) if official else 0, not f["osm"].startswith("way"), f["osm"]))
        return best[0], ""
    if official:
        close = [f for f in best if km(f["at"], official) <= 0.5]
        if len(close) == 1 or (close and max(km(a["at"], b["at"]) for a in close for b in close) <= 0.3):
            close.sort(key=lambda f: km(f["at"], official))
            log.append(f"{st['name']}: {len(best)} OSM name matches far apart, kept {close[0]['osm']} within "
                       f"{km(close[0]['at'], official) * 1000:.0f} m of the {st['official_src']} coordinate")
            return close[0], f" (chosen among {len(best)} OSM features with this name by the {st['official_src']} coordinate)"
    log.append(f"{st['name']}: ambiguous OSM name matches {[f['osm'] for f in best]}, left for geocoding")
    return None, ""


def match_osm(stations: list[dict], feats: list[dict], log: list[str]) -> None:
    claimed, tried = {}, set()
    for primary in (True, False):
        chosen = {}
        for st in stations:
            if st["osm_feat"] or st["id"] in tried:
                continue
            cands = []
            for f in feats:
                if f["osm"] in claimed:
                    continue
                score = 0
                for k, (tag, name, kind) in enumerate(f["names"]):
                    if kind != st["kind"] or (primary and tag != "name"):
                        continue
                    if exact(name) in st["exact"]:
                        score = max(score, 4 if tag == "name" else 2)
                    elif key(name) and key(name) in st["keys"]:
                        score = max(score, 3 if tag == "name" else 1)
                if score:
                    cands.append((score, f))
            if cands:
                tried.add(st["id"])
            f, why = pick(st, cands, log)
            if f:
                chosen.setdefault(f["osm"], []).append((st, f, why))
        for osm, items in chosen.items():
            if len(items) > 1:
                log.append(f"OSM {osm} matched by several stations {[s['name'] for s, _, _ in items]}, none kept")
                continue
            st, f, why = items[0]
            st["osm_feat"], claimed[osm] = f, st["id"]
            st["coord_note"] = f"OSM {'name' if primary else 'alternative name'} match '{f['label'] or f['names'][0][1]}'{why}"
    numbers = Counter(n for st in stations for n in st["numbers"])
    feat_numbers = Counter(n for f in feats for n in f["phones"])
    for st in stations:
        if st["osm_feat"]:
            continue
        hits = [f for f in feats if f["osm"] not in claimed and f["kind"] == st["kind"]
                and any(n in st["numbers"] and numbers[n] == 1 and feat_numbers[n] == 1 for n in f["phones"])]
        if len(hits) == 1:
            st["osm_feat"], claimed[hits[0]["osm"]] = hits[0], st["id"]
            st["coord_note"] = f"OSM phone match '{hits[0]['label'] or 'unnamed'}' (same number as the official listing)"
    for st in stations:
        if st["osm_feat"] or not st["official_at"]:
            continue
        hits = []
        for f in feats:
            if f["osm"] in claimed or f["kind"] != st["kind"]:
                continue
            d = km(f["at"], st["official_at"])
            fuzzy = any(near(key(n), k) for _, n, _ in f["names"] for k in st["keys"])
            if (f["generic"] and d <= 0.15) or (fuzzy and d <= 0.5):
                hits.append((d, f, fuzzy))
        others = [o for o in stations if o is not st and o["kind"] == st["kind"] and o["official_at"]]
        if len(hits) == 1 and not any(km(hits[0][1]["at"], o["official_at"]) <= 0.15 for o in others):
            d, f, fuzzy = hits[0]
            st["osm_feat"], claimed[f["osm"]] = f, st["id"]
            st["coord_note"] = (f"OSM {'near spelling' if fuzzy else 'unnamed or generic'} feature '{f['label'] or 'unnamed'}' "
                                f"{d * 1000:.0f} m from the {st['official_src']} coordinate")
            log.append(f"{st['name']}: proximity match {f['osm']} '{f['label']}' at {d * 1000:.0f} m")
    for st in stations:
        f = st["osm_feat"]
        if f and st["official_at"] and km(f["at"], st["official_at"]) > 3:
            log.append(f"check: {st['name']} OSM {f['osm']} is {km(f['at'], st['official_at']):.1f} km from the {st['official_src']} coordinate")


def geocode(st: dict, taken: set[str], log: list[str]) -> None:
    word = {"traffic": "traffic ", "women": "women ", "cen": "CEN "}.get(st["kind"], "")
    places = []
    for p in st["query_places"]:
        if key(p) and key(p) not in {key(q) for q in places}:
            places.append(p)
    accepted, queries = [], []
    for p in places[:3]:
        q = f"{p} {word}police station"
        queries.append(q)
        res = json.loads(get(NOMINATIM, {"q": q, "format": "jsonv2", "limit": 10, "namedetails": 1, "countrycodes": "in",
                                         "viewbox": f"{GEO_BOX[0]},{GEO_BOX[3]},{GEO_BOX[2]},{GEO_BOX[1]}", "bounded": 1},
                             accept="application/json", gap=1.1))
        for r in res:
            names = [v for v in (r.get("namedetails") or {}).values() if v] or [r.get("name") or ""]
            cat, typ = r.get("category", ""), r.get("type", "")
            if f"{r['osm_type']}/{r['osm_id']}" in taken or (cat == "highway" and typ != "bus_stop") or cat in ("boundary", "place", "landuse"):
                continue
            ok = [n for n in names if re.search(r"police|ಠಾಣೆ", n, re.I) and kind_of(n) == st["kind"] and key(n) in st["keys"]]
            if ok:
                accepted.append({"at": (float(r["lat"]), float(r["lon"])), "name": ok[0], "osm": f"{r['osm_type']}/{r['osm_id']}",
                                 "what": f"{cat}={typ}", "query": q})
        if accepted:
            break
    if not accepted:
        st["coord_note"] = f"no OSM match; Nominatim found nothing named for this station (queries: {'; '.join(queries)})"
        return
    spread = max(km(a["at"], b["at"]) for a in accepted for b in accepted)
    if spread > 0.5:
        st["coord_note"] = f"no OSM match; Nominatim results named for this station are {spread:.1f} km apart, left empty"
        log.append(f"{st['name']}: ambiguous geocode {[a['osm'] for a in accepted]}")
        return
    a = accepted[0]
    st["geo"] = a
    st["coord_note"] = f"Nominatim result {a['what']} '{a['name']}' ({a['osm']}) for query '{a['query']}'"


def mean(pts: list[tuple[float, float]]) -> tuple[float, float]:
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def assign_divisions(stations: list[dict], log: list[str]) -> tuple[dict, dict, dict]:
    direct = lambda d: re.sub(r"(?i)^traffic\s+", "", d or "").lower()
    groups = defaultdict(list)
    for st in stations:
        if st["official_division"] and st["lat"] is not None:
            groups[st["official_division"]].append((st["lat"], st["lon"]))
    centroid = {d: mean(pts) for d, pts in groups.items()}
    base = {f: centroid[f.capitalize()] for f in FIVE if f.capitalize() in centroid}
    mapping = {}
    for d in dict.fromkeys(st["official_division"] for st in stations if st["official_division"]):
        if direct(d) in FIVE:
            mapping[d] = {"division": direct(d), "method": "same name"}
        elif d in centroid:
            dist = {f: round(km(centroid[d], c), 2) for f, c in base.items()}
            mapping[d] = {"division": min(dist, key=dist.get), "method": "nearest centroid", "distances_km": dist,
                          "centroid": [round(centroid[d][0], 5), round(centroid[d][1], 5)], "stations_with_coordinates": len(groups[d])}
        else:
            log.append(f"division {d} has no station with coordinates, cannot be mapped")
    for st in stations:
        st["division_inferred"] = False
        st["division"] = mapping.get(st["official_division"], {}).get("division")
        st.pop("division_note", None)
    for st in stations:
        if st["division"] is None:
            ref = [o for o in stations if o["division"] and not o["division_inferred"] and o["official_division"] and o["lat"] is not None]
            if st["lat"] is not None and ref:
                o = min(ref, key=lambda o: km((st["lat"], st["lon"]), (o["lat"], o["lon"])))
                st["division"], st["division_inferred"] = o["division"], True
                st["division_note"] = f"nearest station with an official division: {o['id']} ({km((st['lat'], st['lon']), (o['lat'], o['lon'])):.2f} km)"
                log.append(f"{st['name']}: no official division, inferred {o['division']} from {o['name']}")
            else:
                log.append(f"{st['name']}: no official division and no coordinates, division left empty")
    return mapping, base, groups


def km_outside(pt: tuple[float, float], polys: list) -> float:
    if inside(pt, polys):
        return 0.0
    kx, ky = 111.32 * math.cos(math.radians(pt[0])), 110.6
    px, py, best = pt[1] * kx, pt[0] * ky, float("inf")
    for ring in (r for p in polys for r in p):
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            ax, ay, dx, dy = x1 * kx, y1 * ky, (x2 - x1) * kx, (y2 - y1) * ky
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy))) if dx or dy else 0.0
            best = min(best, math.hypot(ax + t * dx - px, ay + t * dy - py))
    return best


def fill_official(stations: list[dict], polys: list, log: list[str]) -> None:
    by_div, by_five = defaultdict(list), defaultdict(list)
    for st in stations:
        if st["coord_source"] == "osm":
            by_div[st["official_division"]].append((st["lat"], st["lon"]))
            by_five[st["division"]].append((st["lat"], st["lon"]))
    outside_osm = [st for st in stations if st["coord_source"] == "osm" and not inside((st["lat"], st["lon"]), polys)]
    for st in stations:
        if st["lat"] is not None:
            continue
        if not st["official"]:
            st["coord_note"] += "; no official locator position"
            log.append(f"{st['name']}: no position (no official locator position)")
            continue
        div = st["official_division"]
        if len(by_div.get(div, [])) >= 3:
            pts, basis = by_div[div], f"{len(by_div[div])} OSM-located stations in {div}"
        else:
            pts = by_five.get(st["division"], [])
            basis = f"{len(pts)} OSM-located stations in the {st['division']} division, as {div} has {len(by_div.get(div, []))}"
        ref = mean(pts) if pts else None
        same = [o for o in st["same_place"] if o["coord_source"] == "osm"]
        notes = []
        for src, at, dropped in st["official"]:
            if dropped:
                notes.append(f"{src.upper()} locator position {at[0]:.6f},{at[1]:.6f} not used: {dropped}")
                continue
            out = km_outside(at, polys)
            if out > 0:
                anchor = min(((km(at, (o["lat"], o["lon"])), o) for o in outside_osm), key=lambda x: x[0], default=None)
                spread_ok = anchor is not None and anchor[0] <= 10
                spread = (f"cluster check {anchor[0]:.2f} km from {anchor[1]['name']} (OSM, outside the district)" if anchor
                          else "cluster check: no OSM-located station outside the district")
                spread_fail = "more than 10 km from an OSM-located station outside the district"
            else:
                d = km(at, ref) if ref else None
                spread_ok = d is not None and d <= 8
                spread = f"{d:.2f} km from the centroid of {basis}" if d is not None else f"no centroid for {basis}"
                spread_fail = "more than 8 km from the centroid"
            near_lo = min(((km(at, (o["lat"], o["lon"])), o) for o in same), key=lambda x: x[0], default=None)
            waived = out > 3 and spread_ok
            failed = [why for why, bad in (("more than 3 km outside the district boundary", out > 3 and not waived), (spread_fail, not spread_ok),
                                           (f"more than 3 km from {near_lo[1]['name'] if near_lo else ''} (OSM)", near_lo and near_lo[0] > 3)) if bad]
            where = ("inside the district boundary" if out == 0 else f"{out:.2f} km outside the district boundary"
                     + (" (boundary check waived by the cluster check)" if waived else ""))
            place_check = f", {near_lo[0]:.2f} km from {near_lo[1]['name']} (OSM)" if near_lo else ""
            notes.append(f"{src.upper()} locator position {at[0]:.6f},{at[1]:.6f}: {where}, {spread}{place_check}: "
                         + ("accepted" if not failed else "rejected, " + " and ".join(failed)))
            if not failed:
                st["lat"], st["lon"], st["coord_source"] = round(at[0], 6), round(at[1], 6), f"official_{src}"
                break
        if st["lat"] is not None:
            st["coord_note"] = "no OSM match or geocode; " + "; ".join(notes)
        else:
            st["coord_note"] += "; " + "; ".join(notes)
            log.append(f"{st['name']}: no position ({'; '.join(notes)})")


def circle(center: tuple[float, float], radius_km: float = 3.0, n: int = 64) -> list:
    lat1, lon1, d = math.radians(center[0]), math.radians(center[1]), radius_km / 6371.0
    pts = []
    for i in range(n):
        b = -2 * math.pi * i / n
        lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(b))
        lon2 = lon1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(lat1), math.cos(d) - math.sin(lat1) * math.sin(lat2))
        pts.append((round(math.degrees(lon2), 6), round(math.degrees(lat2), 6)))
    return oriented(pts + pts[:1], True)


def slug(text: str) -> str:
    out = []
    for t in re.findall(r"[a-z0-9]+", fold(text)):
        if len(t) == 1 and out and out[-1][1]:
            out[-1] = (out[-1][0] + t, True)
        else:
            out.append((t, len(t) == 1))
    return "-".join(t for t, _ in out)


def rings_from(members: list[dict]) -> tuple[list[list], list[list]]:
    outer, inner = [], []
    for m in members:
        if m.get("type") == "way" and m.get("geometry"):
            pts = [(round(p["lon"], 7), round(p["lat"], 7)) for p in m["geometry"]]
            (inner if m.get("role") == "inner" else outer).append(pts)

    def join(ways):
        ways, out = [w[:] for w in ways], []
        while ways:
            ring = ways.pop(0)
            while ring[0] != ring[-1]:
                for k, w in enumerate(ways):
                    if w[0] == ring[-1]:
                        ring += w[1:]
                    elif w[-1] == ring[-1]:
                        ring += w[::-1][1:]
                    elif w[-1] == ring[0]:
                        ring = w + ring[1:]
                    elif w[0] == ring[0]:
                        ring = w[::-1] + ring[1:]
                    else:
                        continue
                    ways.pop(k)
                    break
                else:
                    raise RuntimeError("boundary ways do not close into a ring")
            out.append(ring)
        return out
    return join(outer), join(inner)


def simplify_line(pts: list, eps: float, kx: float) -> list:
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        ax, ay = pts[a][0] * kx, pts[a][1]
        bx, by = pts[b][0] * kx, pts[b][1]
        dx, dy = bx - ax, by - ay
        n = math.hypot(dx, dy)
        best, idx = 0.0, None
        for i in range(a + 1, b):
            px, py = pts[i][0] * kx, pts[i][1]
            d = abs(dy * (px - ax) - dx * (py - ay)) / n if n else math.hypot(px - ax, py - ay)
            if d > best:
                best, idx = d, i
        if idx is not None and best > eps:
            keep[idx] = True
            stack += [(a, idx), (idx, b)]
    return [p for p, k in zip(pts, keep) if k]


def simplify_ring(ring: list, eps: float, kx: float) -> list:
    far = max(range(len(ring)), key=lambda i: (ring[i][0] - ring[0][0]) ** 2 + (ring[i][1] - ring[0][1]) ** 2)
    out = simplify_line(ring[:far + 1], eps, kx)[:-1] + simplify_line(ring[far:], eps, kx)
    return out if len(out) >= 4 else ring


def simplify(polys: list, limit: int) -> tuple[list, float]:
    kx = math.cos(math.radians(13.0))
    count = lambda ps: sum(len(r) for p in ps for r in p)
    if count(polys) <= limit:
        return polys, 0.0
    lo, hi = 0.0, 0.02
    for _ in range(40):
        mid = (lo + hi) / 2
        if count([[simplify_ring(r, mid, kx) for r in p] for p in polys]) > limit:
            lo = mid
        else:
            hi = mid
    return [[simplify_ring(r, hi, kx) for r in p] for p in polys], hi


def oriented(ring: list, ccw: bool) -> list:
    area = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:]))
    return ring if (area > 0) == ccw else ring[::-1]


def inside_ring(pt: tuple[float, float], ring: list) -> bool:
    x, y, hit = pt[1], pt[0], False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def inside(pt: tuple[float, float], polys: list) -> bool:
    return any(inside_ring(pt, p[0]) and not any(inside_ring(pt, h) for h in p[1:]) for p in polys)


def boundary(log: list[str]) -> tuple[list, dict]:
    s, w, n, e = OSM_BOX
    found, url, when = overpass(f'[out:json][timeout:120];relation["boundary"="police"]({s},{w},{n},{e});out tags;')
    police = [r for r in found["elements"] if re.search(r"beng|bang", json.dumps(r.get("tags", {})), re.I)]
    if police:
        rel, why = police[0], "OSM boundary=police relation for the Bengaluru City Police"
    else:
        found, url, when = overpass(f'[out:json][timeout:120];relation["boundary"="administrative"]["admin_level"="5"]'
                                    f'["name"="Bengaluru Urban"]({s},{w},{n},{e});out tags;')
        rel = found["elements"][0]
        why = (f"No OSM boundary=police relation for the Bengaluru City Police was found (Overpass search of boundary=police relations in "
               f"{s}-{n} N, {w}-{e} E), so the Bengaluru Urban district boundary is used")
    log.append(f"boundary: relation {rel['id']} {rel['tags'].get('name')}: {why}")
    query = f"[out:json][timeout:180];relation({rel['id']});out geom;"
    geo, url, when = overpass(query)
    members = geo["elements"][0]["members"]
    outer, inner = rings_from(members)
    polys = [[oriented(o, True)] + [oriented(h, False) for h in inner if inside_ring((h[0][1], h[0][0]), o)] for o in outer]
    before = sum(len(r) for p in polys for r in p)
    polys_s, eps = simplify(polys, ARGS.max_vertices)
    after = sum(len(r) for p in polys_s for r in p)
    props = {"name": rel["tags"].get("name"), "admin_level": rel["tags"].get("admin_level"), "osm_relation": rel["id"],
             "wikidata": rel["tags"].get("wikidata"), "selection": why,
             "source_url": f"{url}?data={urllib.parse.quote(query)}",
             "osm_url": f"https://www.openstreetmap.org/relation/{rel['id']}", "retrieved": when,
             "simplification": {"method": "Douglas-Peucker per ring", "tolerance_deg": round(eps, 7),
                                "tolerance_m_approx": round(eps * 111320, 1), "vertices_before": before, "vertices_after": after},
             "license": "ODbL 1.0, (c) OpenStreetMap contributors"}
    return polys_s, props


def main() -> int:
    global ARGS
    p = argparse.ArgumentParser(description="Build the Bengaluru City Police station dataset from official listings and OpenStreetMap")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--no-geocode", action="store_true")
    p.add_argument("--max-vertices", type=int, default=2000)
    ARGS = p.parse_args()
    OFFLINE[0] = ARGS.offline
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (RAW / "index.json").exists():
        INDEX.update(json.loads((RAW / "index.json").read_text()))
    log: list[str] = []

    pages = {k: get(v[0], accept="text/html").decode("utf-8", "replace") for k, v in PAGES.items()}
    overview = clean(pages["bcp_overview"])
    m = re.search(r"divided into (\d+) zones\s*-\s*(.*?)\.", overview)
    zones = [z.strip() for z in re.split(r",|\band\b", m.group(2)) if z.strip()] if m else []
    recs = parse_bcp(pages["bcp"]) + parse_bcp_traffic(pages["bcp_traffic"]) + parse_btp(pages["btp"]) + parse_ksp(pages["ksp"])
    lo_divisions = list(dict.fromkeys(r["division"] for r in recs if r["src"] == "bcp"))
    if zones and {key(z) for z in zones} != {key(d) for d in lo_divisions}:
        log.append(f"warning: overview lists zones {zones} but the Law & Order page has divisions {lo_divisions}")
    if "Cyber Crime Police Station" in clean(pages["bcp_cyber"]):
        recs.append(record("bcp_cyber", "other", "Cyber Crime Police Station", "Cyber Crime"))
    invalidate_duplicates(recs)
    print("official rows: " + ", ".join(f"{s} {n}" for s, n in Counter(r["src"] for r in recs).items()), flush=True)

    stations, ksp_only = [], []
    for group in link(recs, log):
        rs = [recs[i] for i in group]
        srcs = {r["src"] for r in rs}
        if not srcs & set(AUTHORITY):
            ksp_only.append(rs[0])
            continue
        kind = rs[0]["kind"]
        first = lambda *names: next((r for n in names for r in rs if r["src"] == n), None)
        main_rec = first("bcp", "bcp_cyber", "btp", "bcp_traffic")
        place = main_rec["place"]
        if kind == "law_and_order":
            name = f"{place} Police Station"
        elif kind == "traffic":
            name = f"{place} Traffic Police Station"
        elif kind == "women":
            name = f"{place} Women Police Station"
        elif kind == "cen":
            name = f"{place} CEN Police Station"
        else:
            name = "Cyber Crime Police Station"
        divs = {r["src"]: r["division"] for r in rs if r["division"]}
        if "btp" in divs and "bcp_traffic" in divs and divs["btp"] != divs["bcp_traffic"]:
            log.append(f"{name}: btp says {divs['btp']}, bcp_traffic says {divs['bcp_traffic']}; btp kept")
        division = divs.get("bcp") or divs.get("btp") or divs.get("bcp_traffic")
        suffix = {"law_and_order": "Police Station", "traffic": "Traffic Police Station", "women": "Women Police Station",
                  "cen": "CEN Police Station", "other": "Police Station"}[kind]
        alias_places = []
        for r in rs:
            for pl in [r["place"]] + r["extra"]:
                if pl and fold(pl) != fold(place) and fold(pl) not in map(fold, alias_places):
                    alias_places.append(pl)
        aliases = [f"{pl} {suffix}" for pl in alias_places if kind not in ("cen", "other")]
        numbers = list(dict.fromkeys(n for src in ("bcp", "bcp_traffic", "btp") for r in rs if r["src"] == src for n in r["numbers"]))
        official = [(src, r["at"], None) if r["at"] else (src, r["dropped"][0], r["dropped"][1])
                    for src in ("btp", "ksp") for r in rs if r["src"] == src and (r["at"] or r.get("dropped"))]
        usable = [(src, at) for src, at, why in official if not why]
        stations.append({"id": f"{PREFIX[kind]}-{slug(place)}", "name": name, "kind": kind, "place": place, "official_division": division,
                         "aliases": aliases, "alias_places": alias_places, "numbers": numbers, "srcs": srcs,
                         "keys": set().union(*(r["keys"] for r in rs)), "exact": {exact(name)} | {exact(a) for a in aliases},
                         "official": official, "official_at": usable[0][1] if usable else None,
                         "official_src": usable[0][0] if usable else None, "osm_feat": None, "geo": None, "coord_note": None,
                         "same_place": [], "rows": [f"{r['src']}: {r['raw']}" for r in rs]})
    same_place = defaultdict(list)
    for st in stations:
        if st["kind"] == "law_and_order":
            for k in st["keys"]:
                same_place[k].append(st)
    for st in stations:
        st["query_places"] = [st["place"]] + st["alias_places"]
        if st["kind"] in ("traffic", "women"):
            for lo in {id(o): o for k in list(st["keys"]) for o in same_place.get(k, [])}.values():
                st["same_place"].append(lo)
                st["keys"] = st["keys"] | lo["keys"]
                st["query_places"] += [p for p in [lo["place"]] + lo["alias_places"] if fold(p) not in map(fold, st["query_places"])]
    ids = Counter(s["id"] for s in stations)
    for s in stations:
        if ids[s["id"]] > 1:
            log.append(f"duplicate id {s['id']} for {s['name']}")
    print(f"stations: {len(stations)} ({dict(Counter(s['kind'] for s in stations))}); ksp-only rows: {len(ksp_only)}", flush=True)

    s_, w_, n_, e_ = OSM_BOX
    osm_data, osm_url, osm_when = overpass(f'[out:json][timeout:180];nwr["amenity"="police"]({s_},{w_},{n_},{e_});out center tags;')
    feats = osm_features(osm_data)
    match_osm(stations, feats, log)
    taken = {s["osm_feat"]["osm"] for s in stations if s["osm_feat"]}
    if not ARGS.no_geocode:
        for st in stations:
            if not st["osm_feat"]:
                geocode(st, taken, log)
                if st["geo"]:
                    taken.add(st["geo"]["osm"])

    for st in stations:
        f, g = st["osm_feat"], st["geo"]
        st["lat"], st["lon"], st["coord_source"], st["osm"] = None, None, None, None
        if f:
            st["lat"], st["lon"], st["coord_source"], st["osm"] = round(f["at"][0], 6), round(f["at"][1], 6), "osm", f["osm"]
        elif g:
            st["lat"], st["lon"], st["coord_source"] = round(g["at"][0], 6), round(g["at"][1], 6), "geocoded"
        elif not st["coord_note"]:
            st["coord_note"] = "no OSM match and geocoding skipped"

    polys, bprops = boundary(log)
    before, _, _ = assign_divisions(stations, [])
    fill_official(stations, polys, log)
    mapping, base, groups = assign_divisions(stations, log)
    for d, m in mapping.items():
        if before.get(d, {}).get("division") != m["division"]:
            log.append(f"division {d}: {before.get(d, {}).get('division')} before official positions, {m['division']} after")
    areas = []
    for st in stations:
        if st["lat"] is not None and not inside((st["lat"], st["lon"]), polys):
            areas.append((st, circle((st["lat"], st["lon"]))))
    for st in stations:
        at = None if st["lat"] is None else (st["lat"], st["lon"])
        st["inside_boundary"] = None if at is None else inside(at, polys) or any(inside_ring(at, ring) for _, ring in areas)

    order = {k: i for i, k in enumerate(PREFIX)}
    stations.sort(key=lambda s: (order[s["kind"]], s["official_division"] or "", s["name"]))
    src_order = ["bcp", "bcp_cyber", "bcp_traffic", "btp", "ksp", "osm", "nominatim"]
    out_stations = []
    for st in stations:
        srcs = set(st["srcs"]) | ({"osm"} if st["coord_source"] == "osm" else set()) | ({"nominatim"} if st["coord_source"] == "geocoded" else set())
        row = {"id": st["id"], "name": st["name"], "kind": st["kind"], "official_division": st["official_division"], "division": st["division"],
               "division_inferred": st["division_inferred"], "lat": st["lat"], "lon": st["lon"], "coord_source": st["coord_source"], "osm": st["osm"],
               "address": None, "phone": ", ".join(st["numbers"]) or None, "sources": [s for s in src_order if s in srcs],
               "aliases": st["aliases"], "coord_note": st["coord_note"], "inside_boundary": st["inside_boundary"]}
        if st.get("division_note"):
            row["division_note"] = st["division_note"]
        out_stations.append(row)

    sources = [{"id": k, "name": v[1], "url": v[0], "retrieved": retrieved(v[0]), "note": v[2]} for k, v in PAGES.items()]
    sources.append({"id": "osm", "name": "OpenStreetMap via the Overpass API (amenity=police)", "url": osm_url, "retrieved": osm_when,
                    "note": f"Coordinates of matched stations; query bbox {OSM_BOX} (S, W, N, E). ODbL 1.0, (c) OpenStreetMap contributors."})
    sources.append({"id": "osm_boundary", "name": f"OpenStreetMap relation {bprops['osm_relation']} ({bprops['name']})",
                    "url": bprops["osm_url"], "retrieved": bprops["retrieved"], "note": bprops["selection"]})
    geo_times = sorted(v["retrieved"] for v in INDEX.values() if v["url"].startswith(NOMINATIM))
    sources.append({"id": "nominatim", "name": "Nominatim geocoder (OpenStreetMap data)", "url": NOMINATIM,
                    "retrieved": geo_times[-1] if geo_times else None,
                    "note": f"Only for official stations without an OSM match, bounded to {GEO_BOX} (lon/lat); a result is kept only if its "
                            "name carries the station's place and kind, and results more than 500 m apart are treated as ambiguous."})
    rule = ("Official divisions named Central, North, South, East or West (including the Traffic East, West, North and South divisions) keep "
            "that name, and every other official division is assigned whole to the one of these five whose centroid, the mean coordinate of "
            "the stations with coordinates in that law-and-order division, is nearest to its own centroid.")
    coord_rule = ("A position comes from the matched OpenStreetMap feature (osm), else from a Nominatim result named for the station "
                  "(geocoded), else from the station's BTP locator position and then its KSP locator position (official_btp, official_ksp), "
                  "each accepted only if it either lies inside the Bengaluru Urban district polygon and within 8 km of the centroid of "
                  "OSM-located stations in the same official division (the five-way division when that official division has fewer than 3), or "
                  "lies outside the district polygon, at any distance, and within 10 km of an OSM-located station that is itself outside the "
                  "district (cluster check); is not within 50 m of another station's position of the same kind in the same locator; and, for a "
                  "traffic or women station, lies within 3 km of the OSM position of the law-and-order station of the same place; otherwise "
                  "it is null.")
    doc = {"generated": now(), "sources": sources,
           "coordinate_rule": coord_rule,
           "divisions": [{"id": f, "name": f.capitalize()} for f in FIVE],
           "division_rule": rule,
           "division_centroids": {f: {"centroid": [round(c[0], 5), round(c[1], 5)], "stations_with_coordinates": len(groups[f.capitalize()])}
                                  for f, c in base.items()},
           "division_mapping": mapping,
           "official_structure": {"law_and_order_divisions": lo_divisions, "overview_zones": zones,
                                  "traffic_divisions": list(dict.fromkeys(s["official_division"] for s in stations if s["kind"] == "traffic"))},
           "stations": out_stations,
           "not_included": [{"name": r["raw"], "source": "ksp", "reason": "listed only by the KSP locator, not on the BCP or BTP station lists"}
                            for r in sorted(ksp_only, key=lambda r: r["raw"])]}
    (OUT / "police_stations.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    geom = ({"type": "Polygon", "coordinates": [[[round(x, 6), round(y, 6)] for x, y in r] for r in polys[0]]} if len(polys) == 1 else
            {"type": "MultiPolygon", "coordinates": [[[[round(x, 6), round(y, 6)] for x, y in r] for r in p] for p in polys]})
    features = [{"type": "Feature", "properties": bprops, "geometry": geom}]
    for st, ring in sorted(areas, key=lambda a: a[0]["id"]):
        features.append({"type": "Feature", "properties": {"name": f"{st['name']} area", "reason": "Bengaluru City Police station outside "
                                                           "Bengaluru Urban district", "station_id": st["id"]},
                         "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in ring]]}})
    (OUT / "boundary.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=1) + "\n")

    print("\n".join(log), flush=True)
    c = Counter(s["coord_source"] for s in out_stations)
    print(f"by kind: {dict(Counter(s['kind'] for s in out_stations))}")
    print(f"by division: {dict(Counter(s['division'] for s in out_stations))}")
    print(f"coordinates: {dict(c)}")
    print(f"outside the extended boundary: {[s['id'] for s in out_stations if s['inside_boundary'] is False]}")
    print(f"own 3 km area (outside the district): {sorted(st['id'] for st, _ in areas)}")
    print(f"no position: {[s['id'] for s in out_stations if s['lat'] is None]}")
    print(f"division mapping: {json.dumps(mapping)}")
    print(f"ksp only: {[r['raw'] for r in ksp_only]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
