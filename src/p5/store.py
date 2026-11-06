from __future__ import annotations

import base64
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from p5.identity import Identity
from p5.region import registry

IST = timezone(timedelta(hours=5, minutes=30))

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,
  kind TEXT NOT NULL,
  user_id INTEGER,
  username TEXT,
  station_id TEXT,
  division_id TEXT NOT NULL,
  case_id TEXT,
  lat REAL,
  lon REAL,
  summary TEXT NOT NULL,
  data TEXT,
  thumb BLOB,
  note TEXT,
  reviewed INTEGER NOT NULL DEFAULT 0,
  deleted_at REAL,
  deleted_by TEXT
);
CREATE INDEX IF NOT EXISTS events_division_ts ON events(division_id, ts);
CREATE INDEX IF NOT EXISTS events_station_ts ON events(station_id, ts);
CREATE INDEX IF NOT EXISTS events_case ON events(case_id);
CREATE TABLE IF NOT EXISTS detections (
  id INTEGER PRIMARY KEY,
  event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
  ts REAL NOT NULL,
  station_id TEXT,
  division_id TEXT NOT NULL,
  source TEXT NOT NULL,
  label TEXT,
  make TEXT,
  level TEXT,
  confidence REAL,
  catalogue TEXT,
  vehicle_class TEXT,
  colour TEXT,
  wrong_way INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS detections_division_ts ON detections(division_id, ts);
CREATE INDEX IF NOT EXISTS detections_event ON detections(event_id);
CREATE TABLE IF NOT EXISTS cases (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  station_id TEXT,
  division_id TEXT NOT NULL,
  created_by TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',
  target_label TEXT,
  cameras INTEGER NOT NULL DEFAULT 0,
  sightings INTEGER NOT NULL DEFAULT 0,
  deleted_at REAL
);
CREATE INDEX IF NOT EXISTS cases_division ON cases(division_id, updated_at);
CREATE INDEX IF NOT EXISTS cases_station ON cases(station_id, updated_at);
CREATE TABLE IF NOT EXISTS journeys (
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,
  source TEXT NOT NULL,
  ref TEXT,
  division_id TEXT NOT NULL,
  station_id TEXT,
  username TEXT,
  case_id TEXT,
  from_lat REAL NOT NULL,
  from_lon REAL NOT NULL,
  to_lat REAL NOT NULL,
  to_lon REAL NOT NULL,
  depart_ts REAL NOT NULL,
  observed_s REAL NOT NULL,
  freeflow_s REAL NOT NULL,
  distance_m REAL NOT NULL,
  note TEXT,
  deleted_at REAL,
  UNIQUE (source, ref)
);
CREATE INDEX IF NOT EXISTS journeys_division ON journeys(division_id, depart_ts);
"""

KIND_GROUP = {"identify": "identify", "locate": "locate", "video": "video", "camera_analysed": "video"}


def day_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, IST).strftime("%Y-%m-%d")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.local = threading.local()
        self.write_lock = threading.Lock()
        with self.write_lock:
            self.conn().executescript(SCHEMA)

    def conn(self) -> sqlite3.Connection:
        c = getattr(self.local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=10, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode = WAL")
            c.execute("PRAGMA foreign_keys = ON")
            c.execute("PRAGMA synchronous = NORMAL")
            self.local.conn = c
        return c

    @staticmethod
    def scope(ident: Identity, alias: str = "") -> tuple[str, list]:
        p = f"{alias}." if alias else ""
        if ident.is_admin:
            return f"{p}division_id = ?", [ident.division_id]
        return f"{p}station_id = ?", [ident.station_id]

    def record(self, ident: Identity, kind: str, summary: str, data: dict | None = None, case_id: str | None = None,
               lat: float | None = None, lon: float | None = None, thumb: bytes | None = None,
               detections: list[dict] | None = None, station_id: str | None = None) -> int:
        ts = time.time()
        station = station_id or ident.station_id
        with self.write_lock:
            c = self.conn()
            c.execute("BEGIN IMMEDIATE")
            try:
                cur = c.execute(
                    "INSERT INTO events (ts, kind, user_id, username, station_id, division_id, case_id, lat, lon, summary, data, thumb) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, kind, ident.user_id, ident.username, station, ident.division_id, case_id, lat, lon, summary[:300],
                     json.dumps(data, default=float) if data is not None else None, thumb))
                eid = cur.lastrowid
                for d in detections or []:
                    c.execute(
                        "INSERT INTO detections (event_id, ts, station_id, division_id, source, label, make, level, confidence, catalogue, "
                        "vehicle_class, colour, wrong_way) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (eid, ts, station, ident.division_id, d.get("source", "photo"), d.get("label"), d.get("make"), d.get("level"),
                         d.get("confidence"), d.get("catalogue"), d.get("vehicle_class"), d.get("colour"), int(bool(d.get("wrong_way")))))
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
        return eid

    def events(self, ident: Identity, kind: str | None = None, station: str | None = None, q: str | None = None,
               since: float | None = None, until: float | None = None, limit: int = 50, offset: int = 0) -> dict:
        where, args = self.scope(ident)
        clauses = [where, "deleted_at IS NULL"]
        if kind:
            clauses.append("kind = ?")
            args.append(kind)
        if station and ident.is_admin:
            clauses.append("station_id = ?")
            args.append(station)
        if q:
            clauses.append("(summary LIKE ? OR username LIKE ? OR IFNULL(note, '') LIKE ?)")
            args += [f"%{q}%"] * 3
        if since is not None:
            clauses.append("ts >= ?")
            args.append(since)
        if until is not None:
            clauses.append("ts < ?")
            args.append(until)
        sql = " AND ".join(clauses)
        c = self.conn()
        total = c.execute(f"SELECT COUNT(*) FROM events WHERE {sql}", args).fetchone()[0]
        rows = c.execute(
            f"SELECT id, ts, kind, username, station_id, case_id, lat, lon, summary, note, reviewed, thumb IS NOT NULL AS has_thumb "
            f"FROM events WHERE {sql} ORDER BY ts DESC LIMIT ? OFFSET ?", args + [max(1, min(limit, 200)), max(0, offset)]).fetchall()
        stations = registry()["stations"]
        out = []
        for r in rows:
            d = dict(r)
            d["station_name"] = (stations.get(r["station_id"]) or {}).get("name", r["station_id"])
            d["has_thumb"] = bool(d["has_thumb"])
            out.append(d)
        return {"total": total, "events": out}

    def event(self, ident: Identity, eid: int) -> dict | None:
        where, args = self.scope(ident)
        c = self.conn()
        r = c.execute(f"SELECT * FROM events WHERE id = ? AND deleted_at IS NULL AND {where}", [eid] + args).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["data"] = json.loads(d["data"]) if d["data"] else None
        d["thumb"] = base64.b64encode(d["thumb"]).decode() if d["thumb"] else None
        d["detections"] = [dict(x) for x in c.execute("SELECT * FROM detections WHERE event_id = ?", (eid,)).fetchall()]
        d["station_name"] = (registry()["stations"].get(d["station_id"]) or {}).get("name", d["station_id"])
        return d

    def update_event(self, ident: Identity, eid: int, note: str | None = None, reviewed: bool | None = None) -> dict | None:
        where, args = self.scope(ident)
        sets, vals = [], []
        if note is not None:
            sets.append("note = ?")
            vals.append(note[:1000])
        if reviewed is not None:
            sets.append("reviewed = ?")
            vals.append(int(reviewed))
        if sets:
            with self.write_lock:
                self.conn().execute(f"UPDATE events SET {', '.join(sets)} WHERE id = ? AND deleted_at IS NULL AND {where}", vals + [eid] + args)
        return self.event(ident, eid)

    def delete_event(self, ident: Identity, eid: int) -> bool:
        where, args = self.scope(ident)
        with self.write_lock:
            cur = self.conn().execute(f"UPDATE events SET deleted_at = ?, deleted_by = ? WHERE id = ? AND deleted_at IS NULL AND {where}",
                                      [time.time(), ident.username, eid] + args)
        return cur.rowcount > 0

    def index_case(self, case: dict) -> None:
        with self.write_lock:
            self.conn().execute(
                "INSERT INTO cases (id, name, station_id, division_id, created_by, created_at, updated_at, status, target_label, cameras, sightings, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name, station_id = excluded.station_id, "
                "division_id = excluded.division_id, updated_at = excluded.updated_at, status = excluded.status, target_label = excluded.target_label, "
                "cameras = excluded.cameras, sightings = excluded.sightings, deleted_at = excluded.deleted_at",
                (case["id"], case["name"], case.get("station_id"), case["division_id"], case.get("created_by"), case["created"], time.time(),
                 case.get("status", "open"), (case.get("target") or {}).get("label"), len(case.get("cameras", [])),
                 len(case.get("sightings", [])), case.get("deleted_at")))

    def cases(self, ident: Identity, status: str | None = None, station: str | None = None) -> list[dict]:
        where, args = self.scope(ident)
        clauses = [where, "deleted_at IS NULL"]
        if status:
            clauses.append("status = ?")
            args.append(status)
        if station and ident.is_admin:
            clauses.append("station_id = ?")
            args.append(station)
        rows = self.conn().execute(f"SELECT * FROM cases WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT 500", args).fetchall()
        stations = registry()["stations"]
        return [{**dict(r), "station_name": (stations.get(r["station_id"]) or {}).get("name", r["station_id"])} for r in rows]

    def overview(self, ident: Identity, since: float, until: float) -> dict:
        c = self.conn()
        div = ident.division_id
        base = "division_id = ? AND ts >= ? AND ts < ?"
        args = [div, since, until]
        ev = f"{base} AND deleted_at IS NULL"
        det = f"d.division_id = ? AND d.ts >= ? AND d.ts < ? AND e.deleted_at IS NULL"
        join = "detections d JOIN events e ON e.id = d.event_id"

        def rows(sql, a=None):
            return [dict(r) for r in c.execute(sql, a if a is not None else args).fetchall()]

        kinds = {r["kind"]: r["n"] for r in rows(f"SELECT kind, COUNT(*) AS n FROM events WHERE {ev} GROUP BY kind")}
        daily_raw = rows(f"SELECT ts, kind FROM events WHERE {ev}")
        days: dict[str, dict] = {}
        d0 = datetime.fromtimestamp(since, IST).date()
        d1 = datetime.fromtimestamp(until - 1, IST).date()
        span = (d1 - d0).days
        for i in range(span + 1):
            key = (d0 + timedelta(days=i)).strftime("%Y-%m-%d")
            days[key] = {"day": key, "identify": 0, "video": 0, "cases": 0, "other": 0}
        for r in daily_raw:
            k = day_of(r["ts"])
            if k in days:
                grp = KIND_GROUP.get(r["kind"])
                days[k]["identify" if grp == "identify" else "video" if grp == "video" else "cases" if r["kind"].startswith(("case", "hunt", "camera", "route")) else "other"] += 1
        stations = {s["id"]: s for s in registry()["stations"].values() if s.get("division") == div}
        by_station = {sid: {"station_id": sid, "name": s["name"], "kind": s.get("kind"), "lat": s.get("lat"), "lon": s.get("lon"),
                            "events": 0, "identifications": 0, "vehicles": 0, "last": None} for sid, s in stations.items()}
        for r in rows(f"SELECT station_id, COUNT(*) AS n, SUM(kind = 'identify') AS ident, MAX(ts) AS last FROM events WHERE {ev} GROUP BY station_id"):
            sid = r["station_id"]
            if sid is None:
                continue
            row = by_station.setdefault(sid, {"station_id": sid, "name": sid, "kind": None, "lat": None, "lon": None, "events": 0,
                                              "identifications": 0, "vehicles": 0, "last": None})
            row.update(events=r["n"], identifications=r["ident"] or 0, last=r["last"])
        for r in rows(f"SELECT d.station_id, COUNT(*) AS n FROM {join} WHERE {det} GROUP BY d.station_id"):
            if r["station_id"] in by_station:
                by_station[r["station_id"]]["vehicles"] = r["n"]
        cases_open = c.execute("SELECT COUNT(*) FROM cases WHERE division_id = ? AND deleted_at IS NULL AND status = 'open'", (div,)).fetchone()[0]
        cases_total = c.execute("SELECT COUNT(*) FROM cases WHERE division_id = ? AND deleted_at IS NULL", (div,)).fetchone()[0]
        sightings = c.execute("SELECT COALESCE(SUM(sightings), 0) FROM cases WHERE division_id = ? AND deleted_at IS NULL", (div,)).fetchone()[0]
        n_det = c.execute(f"SELECT COUNT(*) FROM {join} WHERE {det}", args).fetchone()[0]
        wrong = c.execute(f"SELECT COUNT(*) FROM {join} WHERE {det} AND d.wrong_way = 1", args).fetchone()[0]
        hist = [0] * 10
        for r in c.execute(f"SELECT d.confidence FROM {join} WHERE {det} AND d.confidence IS NOT NULL", args):
            hist[min(9, max(0, int(r[0] * 10)))] += 1
        return {
            "range": {"from": since, "to": until, "days": span + 1},
            "kpis": {
                "events": sum(kinds.values()),
                "identifications": kinds.get("identify", 0),
                "videos": kinds.get("video", 0) + kinds.get("camera_analysed", 0),
                "vehicles": n_det,
                "wrong_way": wrong,
                "cases_open": cases_open,
                "cases_total": cases_total,
                "sightings": sightings,
                "active_stations": sum(1 for s in by_station.values() if s["events"]),
                "stations": len(stations),
            },
            "kinds": kinds,
            "daily": list(days.values()),
            "top_models": rows(f"SELECT d.label, d.catalogue, COUNT(*) AS n, AVG(d.confidence) AS mean_confidence FROM {join} WHERE {det} "
                               f"AND d.level IN ('class', 'family') GROUP BY d.label, d.catalogue ORDER BY n DESC LIMIT 10"),
            "top_makes": rows(f"SELECT d.make, COUNT(*) AS n FROM {join} WHERE {det} AND d.make IS NOT NULL GROUP BY d.make ORDER BY n DESC LIMIT 10"),
            "levels": rows(f"SELECT d.level, COUNT(*) AS n FROM {join} WHERE {det} AND d.level IS NOT NULL GROUP BY d.level ORDER BY n DESC"),
            "catalogues": rows(f"SELECT d.catalogue, COUNT(*) AS n FROM {join} WHERE {det} AND d.catalogue IS NOT NULL GROUP BY d.catalogue"),
            "vehicle_classes": rows(f"SELECT d.vehicle_class, COUNT(*) AS n FROM {join} WHERE {det} AND d.source = 'video' AND d.vehicle_class IS NOT NULL "
                                    f"GROUP BY d.vehicle_class ORDER BY n DESC"),
            "colours": rows(f"SELECT d.colour, COUNT(*) AS n FROM {join} WHERE {det} AND d.colour IS NOT NULL GROUP BY d.colour ORDER BY n DESC"),
            "confidence_hist": [{"from": i / 10, "to": (i + 1) / 10, "n": n} for i, n in enumerate(hist)],
            "by_station": sorted(by_station.values(), key=lambda s: (-s["events"], s["name"])),
            "points": rows(f"SELECT id, ts, kind, lat, lon, summary, station_id FROM events WHERE {ev} AND lat IS NOT NULL "
                           f"ORDER BY ts DESC LIMIT 500"),
        }

    JOURNEY_FIELDS = ("from_lat", "from_lon", "to_lat", "to_lon", "depart_ts", "observed_s", "freeflow_s", "distance_m")

    def add_journeys(self, ident: Identity, source: str, rows: list[dict], case_id: str | None = None, station_id: str | None = None) -> int:
        added = 0
        with self.write_lock:
            c = self.conn()
            c.execute("BEGIN IMMEDIATE")
            try:
                for r in rows:
                    vals = [float(r[k]) for k in self.JOURNEY_FIELDS]
                    c.execute(
                        "INSERT INTO journeys (ts, source, ref, division_id, station_id, username, case_id, from_lat, from_lon, to_lat, to_lon, "
                        "depart_ts, observed_s, freeflow_s, distance_m, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(source, ref) DO UPDATE SET depart_ts = excluded.depart_ts, observed_s = excluded.observed_s, "
                        "freeflow_s = excluded.freeflow_s, distance_m = excluded.distance_m, deleted_at = NULL",
                        [time.time(), source, r.get("ref"), ident.division_id, station_id or ident.station_id, ident.username, case_id, *vals, r.get("note")])
                    added += 1
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
        return added

    def all_journeys(self) -> list[dict]:
        rows = self.conn().execute(f"SELECT id, source, depart_ts, {', '.join(self.JOURNEY_FIELDS[5:])} FROM journeys WHERE deleted_at IS NULL").fetchall()
        return [dict(r) for r in rows]

    def journeys(self, ident: Identity, limit: int = 100, offset: int = 0) -> dict:
        c = self.conn()
        total = c.execute("SELECT COUNT(*) FROM journeys WHERE division_id = ? AND deleted_at IS NULL", (ident.division_id,)).fetchone()[0]
        rows = c.execute("SELECT * FROM journeys WHERE division_id = ? AND deleted_at IS NULL ORDER BY depart_ts DESC LIMIT ? OFFSET ?",
                         (ident.division_id, max(1, min(limit, 500)), max(0, offset))).fetchall()
        return {"total": total, "journeys": [dict(r) for r in rows]}

    def delete_journey(self, ident: Identity, jid: int) -> bool:
        with self.write_lock:
            cur = self.conn().execute("UPDATE journeys SET deleted_at = ? WHERE id = ? AND division_id = ? AND deleted_at IS NULL",
                                      (time.time(), jid, ident.division_id))
        return cur.rowcount > 0
