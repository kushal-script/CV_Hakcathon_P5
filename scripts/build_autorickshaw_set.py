#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "vahana-drishti-build/1.0 (+https://github.com/kushal-script/CV_Hakcathon_P5)"
FREE = re.compile(r"^(CC0|CC BY(-SA)? [0-9.]+|Public domain|PD|GFDL|FAL|Free Art License)", re.I)


def get(params: dict) -> dict:
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode({**params, "format": "json"}), headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(params)


def crawl(root: str, depth: int, limit: int) -> list[str]:
    files, seen, queue = [], {root}, [(root, 0)]
    while queue and len(files) < limit:
        cat, d = queue.pop(0)
        cont = {}
        while True:
            r = get({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmtype": "file|subcat", "cmlimit": 500, **cont})
            for m in r["query"]["categorymembers"]:
                if m["ns"] == 14 and d < depth and m["title"] not in seen:
                    seen.add(m["title"])
                    queue.append((m["title"], d + 1))
                elif m["ns"] == 6 and re.search(r"\.(jpe?g|png)$", m["title"], re.I):
                    files.append(m["title"])
            if "continue" not in r:
                break
            cont = r["continue"]
    return list(dict.fromkeys(files))[:limit]


def main() -> int:
    p = argparse.ArgumentParser(description="Collect free-licensed auto-rickshaw photos from Wikimedia Commons")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--category", default="Category:Auto-rickshaws in India")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--limit", type=int, default=1500)
    args = p.parse_args()
    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    titles = crawl(args.category, args.depth, args.limit)
    rows = []
    for k in range(0, len(titles), 50):
        r = get({"action": "query", "titles": "|".join(titles[k:k + 50]), "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 500})
        for page in r["query"]["pages"].values():
            ii = (page.get("imageinfo") or [{}])[0]
            ext = ii.get("extmetadata", {})
            lic = ext.get("LicenseShortName", {}).get("value", "")
            if not FREE.match(lic) or not ii.get("thumburl"):
                continue
            artist = re.sub("<[^>]+>", "", html.unescape(ext.get("Artist", {}).get("value", ""))).strip()[:120]
            rows.append({"title": page["title"], "page": ii.get("descriptionurl"), "license": lic, "artist": artist, "url": ii["thumburl"]})
    with (args.out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "title", "page", "license", "artist"])
        w.writeheader()
        for k, r in enumerate(rows):
            name = f"{k:05d}.jpg"
            dst = args.out / "raw" / name
            if not dst.exists():
                try:
                    dst.write_bytes(urllib.request.urlopen(urllib.request.Request(r["url"], headers={"User-Agent": UA}), timeout=60).read())
                except Exception:
                    continue
                time.sleep(0.05)
            w.writerow({"file": name, "title": r["title"], "page": r["page"], "license": r["license"], "artist": r["artist"]})
    print(f"{len(titles)} files listed, {len(rows)} free-licensed photos saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
