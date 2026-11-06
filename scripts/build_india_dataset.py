#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from p5.classification.crop import cache_crop
from p5.taxonomy import Taxonomy

CROP_PAD, CROP_SIDE = 0.16, 448
UA = "VahanaDrishti/0.2 (https://github.com/kushal-script/CV_Hakcathon_P5; dataset builder) python-urllib/3.11"
WP = "https://en.wikipedia.org/w/api.php"
WD_SPARQL = "https://query.wikidata.org/sparql"
COMMONS = "https://commons.wikimedia.org/w/api.php"
SKIP_SUBCAT = re.compile(r"\b(interiors?|engines?|dashboards?|badges?|logos?|emblems?|wheels?|steering|seating|seats|boots?|trunks?|instruments?|"
                         r"cockpits?|headlights?|taillights?|tail lights?|grilles?|details?|model cars?|scale models?|toys?|diecast|"
                         r"advertisements?|brochures?|factory|factories|plants?|production lines?|crash(es)?|accidents?|racing|rally|drawings?)\b", re.I)
SKIP_FILE = re.compile(r"\b(interior|engine|dashboard|cockpit|logo|emblem|diecast|die-cast|brochure|scale model)\b", re.I)
FREE = re.compile(r"^(cc0|cc[ -]by(-sa)?|public domain|pd|attribution|gfdl|fal)", re.I)
INDIA_SOURCES = ["Automotive industry in India", "Maruti Suzuki", "Hyundai Motor India", "Tata Motors", "Tata Motors Passenger Vehicles",
                 "Mahindra & Mahindra", "Toyota Kirloskar Motor", "Honda Cars India", "Kia India", "JSW MG Motor India", "Ford India",
                 "Škoda Auto Volkswagen India", "Renault India", "Nissan Motor India", "Hindustan Motors", "Premier Automobiles Limited",
                 "Force Motors", "General Motors India", "Fiat India Automobiles", "Citroën India", "Stellantis India", "Mercedes-Benz India",
                 "BMW India", "Isuzu Motors India", "Datsun"]
CAR_TYPES = ("Q3231690", "Q59773381")
WIKIDATA_QUERY = """
SELECT DISTINCT ?model WHERE {
  ?model wdt:P31/wdt:P279* wd:Q3231690 .
  { ?model wdt:P1071 ?loc . ?loc wdt:P17 wd:Q668 . } UNION { ?model wdt:P495 wd:Q668 . }
  UNION { ?model wdt:P176 ?m . ?m wdt:P17 wd:Q668 . }
}"""


def get(url: str, params: dict | None = None, data: bytes | None = None, accept: str = "application/json", tries: int = 7):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Accept": accept})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                return json.loads(body) if "json" in accept else body
        except urllib.error.HTTPError as e:
            if k == tries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            retry = e.headers.get("Retry-After")
            time.sleep(float(retry) if retry and retry.isdigit() else 15 * (k + 1))
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k + 1)


def sparql(query: str) -> list[dict]:
    res = get(WD_SPARQL, data=urllib.parse.urlencode({"query": query}).encode(), accept="application/sparql-results+json")
    return res["results"]["bindings"]


def article_links(title: str) -> list[str]:
    wt = get(WP, {"action": "parse", "page": title, "prop": "wikitext", "format": "json", "redirects": 1})["parse"]["wikitext"]["*"]
    return sorted({l.strip() for l in re.findall(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]", wt) if ":" not in l})


def titles_to_qids(titles: list[str]) -> dict[str, str]:
    out = {}
    for k in range(0, len(titles), 40):
        res = get(WP, {"action": "query", "titles": "|".join(titles[k:k + 40]), "prop": "pageprops", "ppprop": "wikibase_item",
                       "redirects": 1, "format": "json"})
        for p in res["query"].get("pages", {}).values():
            q = p.get("pageprops", {}).get("wikibase_item")
            if q:
                out[p["title"]] = q
        time.sleep(0.3)
    return out


def resolve_classes(min_year: int | None) -> list[dict]:
    qids = {b["model"]["value"].rsplit("/", 1)[1] for b in sparql(WIKIDATA_QUERY)}
    for src in INDIA_SOURCES:
        qids |= set(titles_to_qids(article_links(src)).values())
        time.sleep(0.5)
    rows = []
    ordered = sorted(qids)
    for k in range(0, len(ordered), 60):
        values = " ".join(f"wd:{q}" for q in ordered[k:k + 60])
        rows += sparql(f"""
SELECT ?model ?modelLabel ?makerLabel ?type ?commons ?sitelink WHERE {{
  VALUES ?model {{ {values} }}
  ?model wdt:P176 ?maker .
  FILTER EXISTS {{ ?model wdt:P31/wdt:P279* ?car . VALUES ?car {{ {" ".join("wd:" + t for t in CAR_TYPES)} }} }}
  OPTIONAL {{ ?model wdt:P373 ?commons . }}
  OPTIONAL {{ ?sitelink schema:about ?model ; schema:isPartOf <https://commons.wikimedia.org/> . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}""")
        time.sleep(1.0)
    classes: dict[str, dict] = {}
    for r in rows:
        q = r["model"]["value"].rsplit("/", 1)[1]
        cat = r.get("commons", {}).get("value")
        if not cat and r.get("sitelink"):
            link = urllib.parse.unquote(r["sitelink"]["value"].rsplit("/", 1)[1]).replace("_", " ")
            cat = link[len("Category:"):] if link.startswith("Category:") else None
        if not cat:
            continue
        label = r["modelLabel"]["value"]
        if re.fullmatch(r"Q\d+", label):
            continue
        c = classes.setdefault(q, {"qid": q, "name": label, "makers": set(), "commons": cat})
        if r.get("makerLabel"):
            c["makers"].add(r["makerLabel"]["value"])
    out = []
    for c in classes.values():
        c["makers"] = sorted(c["makers"])
        out.append(c)
    return sorted(out, key=lambda c: c["name"])


def category_tree(cat: str, depth: int = 2) -> list[str]:
    out, frontier = [], [(cat, 0)]
    while frontier:
        c, d = frontier.pop(0)
        if c in out:
            continue
        out.append(c)
        if d >= depth:
            continue
        cont = {}
        while True:
            res = get(COMMONS, {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{c}", "cmlimit": 500,
                                "cmtype": "subcat", "format": "json", **cont})
            for m in res.get("query", {}).get("categorymembers", []):
                t = m["title"][len("Category:"):]
                if not SKIP_SUBCAT.search(t):
                    frontier.append((t, d + 1))
            if "continue" not in res:
                break
            cont = res["continue"]
        time.sleep(0.1)
    return out


IMAGE_PARAMS = {"prop": "imageinfo", "iiprop": "url|extmetadata|size", "iiextmetadatafilter": "LicenseShortName|Artist|DateTimeOriginal",
                "format": "json"}


def file_rows(pages: dict, seen: set) -> list[dict]:
    out = []
    for p in pages.values():
        t = p["title"]
        if t in seen or not re.search(r"\.(jpe?g|png|webp)$", t, re.I) or SKIP_FILE.search(t):
            continue
        seen.add(t)
        ii = (p.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "")
        artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
        if ii.get("thumburl") and FREE.search(lic or "") and ii.get("width", 0) >= 320:
            out.append({"title": t, "url": ii["thumburl"], "page": ii.get("descriptionurl"), "license": lic,
                        "artist": artist[:120], "date": meta.get("DateTimeOriginal", {}).get("value", "")[:10]})
    return out


def search_images(cat: str, width: int, limit: int) -> list[dict]:
    out, seen, cont = [], set(), {}
    while len(out) < limit:
        res = get(COMMONS, {"action": "query", "generator": "search", "gsrsearch": f'deepcat:"{cat}" filetype:bitmap', "gsrnamespace": 6,
                            "gsrlimit": 50, "iiurlwidth": width, **IMAGE_PARAMS, **cont})
        out += file_rows(res.get("query", {}).get("pages", {}), seen)
        if "continue" not in res:
            break
        cont = res["continue"]
    return out[:limit]


def tree_images(cat: str, width: int, limit: int) -> list[dict]:
    out, seen = [], set()
    for c in category_tree(cat):
        cont = {}
        while len(out) < limit:
            res = get(COMMONS, {"action": "query", "generator": "categorymembers", "gcmtitle": f"Category:{c}", "gcmtype": "file",
                                "gcmlimit": 50, "iiurlwidth": width, **IMAGE_PARAMS, **cont})
            out += file_rows(res.get("query", {}).get("pages", {}), seen)
            if "continue" not in res:
                break
            cont = res["continue"]
        if len(out) >= limit:
            break
    return out[:limit]


def assign_makes(classes: list[dict], known: set[str]) -> list[str]:
    firsts = {b.split()[0].lower() for b in known}
    multi = sorted((b for b in known if " " in b), key=len, reverse=True)
    makes = []
    for c in classes:
        low = c["name"].lower()
        m = next((b for b in multi if low.startswith(b.lower() + " ")), None)
        if m is None:
            first = c["name"].split()[0]
            own = [b for b in c.get("brands", []) if not re.fullmatch(r"Q\d+", b)]
            m = first if first.lower() in firsts or not own else own[0]
        makes.append(m)
    longer = sorted((m for m in set(makes) if " " in m), key=len, reverse=True)
    return [next((x for x in longer if x.lower().startswith(m.lower() + " ")), m) for m in makes]


def india_taxonomy(classes: list[dict], known: set[str]) -> Taxonomy:
    names = [c["name"] for c in classes]
    make_of = assign_makes(classes, known)
    model_of = [n[len(m):].strip() if n.lower().startswith(m.lower() + " ") else n for n, m in zip(names, make_of)]
    return Taxonomy(names=names, make_of=make_of, model_of=model_of, year_of=[0] * len(names), display_make_of=make_of,
                    family_of=list(range(len(names))), family_names=names)


def dhash(img: Image.Image, size: int = 8) -> int:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def unique_categories(classes: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in classes:
        if c["commons"] not in seen:
            seen.add(c["commons"])
            out.append(c)
    return out


def merge_subsets(collected: dict[str, dict], min_overlap: float) -> dict[str, str]:
    titles = {q: {r["title"] for r in c["items"]} for q, c in collected.items()}
    parent = {}
    for q in collected:
        best, share = None, 0.0
        for a in collected:
            if a == q or len(titles[a]) < len(titles[q]) or not titles[q]:
                continue
            o = len(titles[q] & titles[a]) / len(titles[q])
            if o >= min_overlap and o > share:
                best, share = a, o
        if best:
            parent[q] = best
    owner = {}
    for q in collected:
        path, cur = [q], q
        while cur in parent and parent[cur] not in path:
            cur = parent[cur]
            path.append(cur)
        if cur in parent:
            cur = min(path[path.index(parent[cur]):], key=lambda a: collected[a]["name"])
        owner[q] = cur
    return owner


def raw_path(out: Path, qid: str, title: str) -> Path:
    return out / "raw" / qid / (hashlib.sha1(title.encode()).hexdigest()[:16] + ".jpg")


def list_class(c: dict, args) -> tuple[list[dict], str]:
    listed = args.out / "listed" / f"{c['qid']}.json"
    if listed.exists():
        d = json.loads(listed.read_text())
        return d["infos"], d["source"]
    if args.offline:
        raise RuntimeError("not listed yet")
    infos, source = search_images(c["commons"], args.width, args.per_class), "deepcat search"
    if not infos:
        infos, source = tree_images(c["commons"], args.width, args.per_class), "category tree"
    listed.parent.mkdir(parents=True, exist_ok=True)
    listed.write_text(json.dumps({"qid": c["qid"], "source": source, "infos": infos}))
    return infos, source


fetch_slots = threading.BoundedSemaphore(6)


def collect_class(c: dict, args, det, lock) -> dict:
    cache = args.out / "collected" / f"{c['qid']}.json"
    if cache.exists():
        done = json.loads(cache.read_text())
        if len(done["items"]) >= min(args.take, done["listed"]) or done["listed"] <= len(done["items"]):
            return done
    infos, source = list_class(c, args)
    infos = infos[:args.take]

    def one(f):
        dest = raw_path(args.out, c["qid"], f["title"])
        try:
            if not dest.exists():
                url = f["url"].replace(f"/{args.width}px-", f"/{args.fetch_width}px-") if args.fetch_width else f["url"]
                with fetch_slots:
                    data = get(url, accept="image/*", tries=6)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img.thumbnail((args.store_side, args.store_side), Image.BILINEAR)
                dest.parent.mkdir(parents=True, exist_ok=True)
                img.save(dest, quality=90)
            rgb = np.asarray(Image.open(dest).convert("RGB"))
        except Exception:
            return None
        with lock:
            dets = det(rgb, conf=min(0.25, args.min_det), classes={2, 7})
        boxes = [{"box": [round(float(v), 1) for v in d.box], "score": round(float(d.score), 3), "cls": int(d.cls)} for d in dets]
        return {**f, "w": rgb.shape[1], "h": rgb.shape[0], "dets": boxes}

    with ThreadPoolExecutor(args.downloads) as ex:
        items = [r for r in ex.map(one, infos) if r is not None]
    res = {"qid": c["qid"], "name": c["name"], "commons": c["commons"], "source": source, "listed": len(infos), "items": items}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res))
    return res


def main() -> int:
    p = argparse.ArgumentParser(description="Build an Indian-market car model dataset from Wikidata, Wikipedia and Wikimedia Commons")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "india")
    p.add_argument("--assets", type=Path, default=ROOT / "assets" / "india")
    p.add_argument("--stage", choices=["classes", "list", "collect", "clean", "split", "all"], default="all")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--fetch-concurrency", type=int, default=6)
    p.add_argument("--fetch-width", type=int, default=500)
    p.add_argument("--take", type=int, default=150)
    p.add_argument("--clean-side", type=int, default=500)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--store-side", type=int, default=800)
    p.add_argument("--per-class", type=int, default=300)
    p.add_argument("--min-images", type=int, default=40)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--downloads", type=int, default=4)
    p.add_argument("--min-box-frac", type=float, default=0.18)
    p.add_argument("--min-det", type=float, default=0.5)
    p.add_argument("--min-car-share", type=float, default=0.5)
    p.add_argument("--holdout", type=float, default=0.15)
    p.add_argument("--min-holdout", type=int, default=6)
    p.add_argument("--merge-overlap", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    global fetch_slots
    fetch_slots = threading.BoundedSemaphore(args.fetch_concurrency)
    args.out.mkdir(parents=True, exist_ok=True)
    args.assets.mkdir(parents=True, exist_ok=True)
    stages = ["classes", "list", "collect", "clean", "split"] if args.stage == "all" else [args.stage]

    if "classes" in stages:
        classes = resolve_classes(None)
        (args.assets / "candidate_classes.json").write_text(json.dumps(classes, indent=1))
        print(f"candidate classes: {len(classes)}", flush=True)

    if "list" in stages:
        classes = unique_categories(json.loads((args.assets / "candidate_classes.json").read_text()))
        done = 0
        with ThreadPoolExecutor(args.workers) as ex:
            futures = {ex.submit(list_class, c, args): c for c in classes}
            from concurrent.futures import as_completed
            for f in as_completed(futures):
                done += 1
                try:
                    infos, source = f.result()
                    print(f"[{done}/{len(classes)}] {futures[f]['name']}: listed {len(infos)} via {source}", flush=True)
                except Exception as e:
                    print(f"[{done}/{len(classes)}] {futures[f]['name']}: failed {type(e).__name__}: {e}", flush=True)

    if "collect" in stages:
        from p5.detection.detector import Detector
        det = Detector(ROOT / "models" / "detector_coco.onnx", threads=1, iou=0.5)
        classes = unique_categories(json.loads((args.assets / "candidate_classes.json").read_text()))
        classes.sort(key=lambda c: (c.get("india_priority", 1), c["name"]))
        lock = threading.Semaphore(os.cpu_count() or 4)
        done = 0
        with ThreadPoolExecutor(args.workers) as ex:
            futures = {ex.submit(collect_class, c, args, det, lock): c for c in classes}
            from concurrent.futures import as_completed
            for f in as_completed(futures):
                c = futures[f]
                done += 1
                try:
                    r = f.result()
                    print(f"[{done}/{len(classes)}] {c['name']}: listed {r['listed']} kept {len(r['items'])}", flush=True)
                except Exception as e:
                    print(f"[{done}/{len(classes)}] {c['name']}: failed {type(e).__name__}: {e}", flush=True)

    if "clean" in stages:
        classes = unique_categories(json.loads((args.assets / "candidate_classes.json").read_text()))
        collected = {}
        for c in classes:
            cache = args.out / "collected" / f"{c['qid']}.json"
            if cache.exists():
                collected[c["qid"]] = json.loads(cache.read_text())
        owner = merge_subsets(collected, args.merge_overlap)
        (args.assets / "merged_classes.json").write_text(json.dumps(
            [{"qid": q, "name": collected[q]["name"], "merged_into": o, "into_name": collected[o]["name"]} for q, o in owner.items() if q != o], indent=1))
        crops = args.out / "crops"
        kept, hashes, stats = [], {}, Counter()
        stats["class_not_collected"] = len(classes) - len(collected)
        title_owners = defaultdict(set)
        for col in collected.values():
            col["items"] = col["items"][:args.take]
        for q, col in collected.items():
            for r in col["items"]:
                title_owners[r["title"]].add(owner[q])
        seen_titles = set()
        for q, col in collected.items():
            c = collected[owner[q]]
            for r in col["items"]:
                if len(title_owners[r["title"]]) > 1:
                    stats["listed_under_two_classes"] += 1
                    continue
                if r["title"] in seen_titles:
                    continue
                seen_titles.add(r["title"])
                src = raw_path(args.out, q, r["title"])
                stats["images"] += 1
                dets = [d for d in r["dets"] if d["score"] >= args.min_det]
                if not dets:
                    stats["no_vehicle"] += 1
                    continue
                area = lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
                big = max(dets, key=area)
                frac = area(big) / (r["w"] * r["h"])
                if frac < args.min_box_frac:
                    stats["vehicle_too_small"] += 1
                    continue
                if any(d is not big and area(d) > 0.5 * area(big) for d in dets):
                    stats["several_large_vehicles"] += 1
                    continue
                try:
                    img = Image.open(src).convert("RGB")
                    k = min(1.0, args.clean_side / max(img.size))
                    if k < 1.0:
                        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
                    k = img.width / r["w"]
                    crop, inner = cache_crop(img, tuple(v * k for v in big["box"]), CROP_PAD, CROP_SIDE)
                except Exception:
                    stats["unreadable"] += 1
                    continue
                hsh = dhash(crop.crop(tuple(int(round(v)) for v in inner)))
                if hsh in hashes:
                    hashes[hsh].append(c["qid"])
                    stats["duplicate"] += 1
                    continue
                hashes[hsh] = [c["qid"]]
                dest = crops / c["qid"] / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                crop.save(dest, quality=92)
                kept.append({"qid": c["qid"], "class": c["name"], "title": r["title"], "page": r["page"], "license": r["license"],
                             "artist": r["artist"], "date": r["date"], "file": f"{c['qid']}/{src.name}",
                             "det_class": "car" if big["cls"] == 2 else "truck", "det_score": big["score"], "box_frac": round(frac, 3),
                             "dhash": f"{hsh:016x}", **{k: round(float(v), 2) for k, v in zip(("bx1", "by1", "bx2", "by2"), inner)}})
        conflicted = {h for h, q in hashes.items() if len(set(q)) > 1}
        kept = [r for r in kept if int(r["dhash"], 16) not in conflicted]
        stats["cross_class_duplicates"] = len(conflicted)
        stats["kept"] = len(kept)
        with (args.out / "manifest_clean.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(kept[0].keys()))
            w.writeheader()
            w.writerows(kept)
        (args.assets / "clean_stats.json").write_text(json.dumps(dict(stats), indent=1))
        print(f"clean images: {dict(stats)}", flush=True)
    if "split" in stages:
        rows = list(csv.DictReader((args.out / "manifest_clean.csv").open()))
        by_class = defaultdict(list)
        for r in rows:
            by_class[r["qid"]].append(r)
        cand = {c["qid"]: c for c in json.loads((args.assets / "candidate_classes.json").read_text())}
        rng = np.random.default_rng(args.seed)
        classes, split_rows, dropped = [], [], []
        for q, items in sorted(by_class.items(), key=lambda kv: cand[kv[0]]["name"]):
            car_share = sum(r["det_class"] == "car" for r in items) / len(items)
            if len(items) < args.min_images or car_share < args.min_car_share:
                dropped.append({"qid": q, "name": cand[q]["name"], "n": len(items), "car_share": round(car_share, 3),
                                "reason": "too few images" if len(items) < args.min_images else "mostly detected as non-car"})
                continue
            groups = defaultdict(list)
            for r in items:
                groups[re.sub(r"<[^>]+>", "", r["artist"]).strip().lower() or r["title"]].append(r)
            keys = list(groups)
            rng.shuffle(keys)
            keys.sort(key=lambda g: len(groups[g]))
            n = len(items)
            parts = {"test": [], "val": [], "train": []}
            for g in keys:
                target = "test" if len(parts["test"]) < args.holdout * n else "val" if len(parts["val"]) < args.holdout * n else "train"
                parts[target].extend(groups[g])
            if len(parts["train"]) < 0.5 * n or min(len(parts["val"]), len(parts["test"])) < args.min_holdout:
                dropped.append({"qid": q, "name": cand[q]["name"], "n": n, "photographers": len(groups), "reason": "too few independent photographers"})
                continue
            idx = len(classes)
            classes.append({"id": idx, "qid": q, "name": cand[q]["name"], "makers": cand[q]["makers"], "brands": cand[q].get("brands", []),
                            "commons": cand[q]["commons"],
                            "n": n, "train": len(parts["train"]), "val": len(parts["val"]), "test": len(parts["test"])})
            for s, rs in parts.items():
                for r in rs:
                    split_rows.append({"file": r["file"], "class_id": idx, "split": s, "bx1": r["bx1"], "by1": r["by1"], "bx2": r["bx2"],
                                       "by2": r["by2"], "license": r["license"], "artist": r["artist"], "page": r["page"]})
        (args.assets / "classes.json").write_text(json.dumps(classes, indent=1))
        (args.assets / "dropped_classes.json").write_text(json.dumps(dropped, indent=1))
        known = {b for c in cand.values() for b in c.get("brands", []) if not re.fullmatch(r"Q\d+", b)}
        india_taxonomy(classes, known).save(args.assets / "taxonomy")
        with (args.assets / "split.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(split_rows[0].keys()))
            w.writeheader()
            w.writerows(split_rows)
        c = Counter(r["split"] for r in split_rows)
        print(f"classes kept: {len(classes)}; images: {dict(c)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
