import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = "vahana-drishti-build/1.0 (+https://github.com/kushal-script/CV_Hakcathon_P5)"
YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


def get(url: str, params: dict) -> dict:
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode({**params, 'format': 'json'})}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def claim_year(claims: dict, prop: str) -> int | None:
    for c in claims.get(prop, []):
        t = (c.get("mainsnak", {}).get("datavalue", {}).get("value") or {}).get("time", "")
        m = YEAR.search(t)
        if m:
            return int(m.group(1))
    return None


def production_field(wikitext: str) -> str | None:
    m = re.search(r"\{\{\s*Infobox automobile", wikitext, re.I)
    if not m:
        return None
    body = wikitext[m.start():]
    f = re.search(r"^\s*\|\s*production\s*=\s*(.+)$", body, re.I | re.M)
    if not f:
        return None
    value = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", f.group(1), flags=re.S)
    return value.strip() or None


def span(value: str) -> tuple[int, int | None] | None:
    years = [int(y) for y in YEAR.findall(value)]
    if not years:
        return None
    start = min(years)
    end = None if re.search(r"present|current|ongoing", value, re.I) else max(years)
    return start, end


def main() -> None:
    ap = argparse.ArgumentParser(description="Production years of the Indian catalogue's models from Wikipedia, with Wikidata as fallback")
    ap.add_argument("--classes", type=Path, default=Path("assets/india/classes.json"))
    ap.add_argument("--out", type=Path, nargs="+", default=[Path("assets/india/production_years.json"), Path("models/india/production_years.json")])
    a = ap.parse_args()
    classes = json.loads(a.classes.read_text())
    qids = [c["qid"] for c in classes]
    enwiki, wd = {}, {}
    for k in range(0, len(qids), 50):
        ents = get("https://www.wikidata.org/w/api.php", {"action": "wbgetentities", "ids": "|".join(qids[k:k + 50]), "props": "sitelinks|claims", "sitefilter": "enwiki"})["entities"]
        for q, e in ents.items():
            title = e.get("sitelinks", {}).get("enwiki", {}).get("title")
            if title:
                enwiki[q] = title
            wd[q] = (claim_year(e.get("claims", {}), "P571"), claim_year(e.get("claims", {}), "P2669"))
        time.sleep(0.5)
    text = {}
    titles = sorted(set(enwiki.values()))
    for k in range(0, len(titles), 20):
        res = get("https://en.wikipedia.org/w/api.php", {"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main", "redirects": 1,
                                                          "formatversion": 2, "titles": "|".join(titles[k:k + 20])})["query"]
        alias = {r["from"]: r["to"] for r in res.get("redirects", []) + res.get("normalized", [])}
        for p in res.get("pages", []):
            rev = (p.get("revisions") or [{}])[0]
            text[p["title"]] = rev.get("slots", {}).get("main", {}).get("content", "")
        for src, dst in alias.items():
            if dst in text:
                text[src] = text[dst]
        time.sleep(0.5)
    models, counts = {}, {"wikipedia": 0, "wikidata": 0, "none": 0}
    for c in classes:
        q, title = c["qid"], enwiki.get(c["qid"])
        field = production_field(text.get(title, "")) if title else None
        got = span(field) if field else None
        if got and (got[1] is None or got[1] >= got[0]):
            models[c["name"]] = {"from": got[0], "to": got[1], "source": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")), "field": field[:120]}
            counts["wikipedia"] += 1
            continue
        start, end = wd.get(q, (None, None))
        if start:
            models[c["name"]] = {"from": start, "to": end, "source": f"https://www.wikidata.org/wiki/{q}"}
            counts["wikidata"] += 1
        else:
            counts["none"] += 1
    out = {"retrieved": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "source": "English Wikipedia infobox 'production' field, else Wikidata P571 and P2669",
           "models": models}
    for path in a.out:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{len(classes)} models: {counts['wikipedia']} from Wikipedia, {counts['wikidata']} from Wikidata, {counts['none']} without production years")


if __name__ == "__main__":
    main()
