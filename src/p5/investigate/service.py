from __future__ import annotations

import base64
import json
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from p5.config import hunt_model, section
from p5.identity import Identity
from p5.region import inside, station
from p5.store import Store
from p5.geo.locate import cluster, ocr_evidence, queries, similarity
from p5.geo.geocode import photon
from p5.geo.route import map_match, mp4_location, parse_track, position_at
from p5.geo.videoloc import file_sha256, locate_video
from p5.geo.videoloc import graph as road_graph
from p5.geo.videoloc.metadata import embedded_gps
from p5.investigate.hints import parse as parse_hints
from p5.investigate.hunt import rank, target_profile
from p5.investigate.journeys import from_sightings, from_track
from p5.investigate.routes import infer
from p5.video.analyse import VideoAnalyser, VideoOptions

PUBLIC_DROP = {"embedding", "probs"}


def _epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def environment(frames: list[np.ndarray], ocr) -> dict:
    ev = [e for f in frames for e in ocr_evidence(f, ocr)]
    points = []
    for q, src in queries(ev, 8):
        for hit in photon(q):
            sim = similarity(q, hit["name"])
            if sim >= section("geocoding")["min_similarity"]:
                points.append({**hit, "score": float(np.mean([e.ocr_conf for e in src])) * sim / (1 + 0.5 * hit["rank"]),
                               "query": q, "evidence_key": q})
    cands = []
    for c in cluster(points)[:3]:
        best = max(c["members"], key=lambda m: m["score"])
        cands.append({"lat": c["lat"], "lon": c["lon"], "label": best["name"], "confidence": round(float(1 - np.exp(-1.5 * c["score"])), 3),
                      "texts": sorted({m["query"] for m in c["members"]})[:5]})
    return {"texts": sorted({e.text for e in ev})[:20], "location_candidates": cands}


def _detections(tracks: list[dict], source: str = "video") -> list[dict]:
    out = []
    for tr in tracks:
        d = tr.get("decision") or {}
        out.append({"source": source, "label": d.get("label"), "make": d.get("make"), "level": d.get("level"),
                    "confidence": d.get("confidence"), "catalogue": d.get("catalogue"), "vehicle_class": tr.get("class"),
                    "colour": (tr.get("colour") or {}).get("name"), "wrong_way": tr.get("wrong_way", False)})
    return out


def plural(n: int, word: str) -> str:
    if n == 1:
        return f"{n} {word}"
    return f"{n} {word}{'es' if word.endswith(('s', 'x', 'z', 'ch', 'sh')) else 's'}"


def video_summary(res: dict) -> str:
    counts = res["counts"]["by_class"]
    parts = ", ".join(plural(n, k) for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    alerts = len(res.get("alerts", []))
    return f"{plural(res['counts']['total'], 'vehicle')} tracked" + (f" ({parts})" if parts else "") + (f", {alerts} wrong-way" if alerts else "")


class CaseStore:
    def __init__(self, root: Path, pipeline, analyser: VideoAnalyser, ocr, store: Store, traffic=None, refit=None,
                 road_graph_path: Path | None = None, geo_workers: int = 2, validation: dict | None = None, places_path: Path | None = None):
        self.root, self.pipeline, self.analyser, self.ocr, self.store = root, pipeline, analyser, ocr, store
        self.traffic, self.refit = traffic or (lambda: None), refit or (lambda: None)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.geo_pool = ThreadPoolExecutor(max_workers=1)
        self.road_graph_path, self.geo_workers, self.validation, self.places_path = road_graph_path, geo_workers, validation, places_path

    def _path(self, cid: str) -> Path:
        if not cid.isalnum() or len(cid) > 32:
            raise KeyError(cid)
        return self.root / cid

    def _load(self, cid: str) -> dict:
        f = self._path(cid) / "case.json"
        if not f.exists():
            raise KeyError(cid)
        return json.loads(f.read_text())

    def _save(self, case: dict) -> None:
        d = self._path(case["id"])
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "case.json.tmp"
        tmp.write_text(json.dumps(case))
        tmp.replace(d / "case.json")
        self.store.index_case(case)

    @staticmethod
    def allowed(case: dict, ident: Identity) -> bool:
        if case.get("deleted_at") or "division_id" not in case:
            return False
        if ident.is_admin:
            return ident.covers(case["division_id"])
        return case.get("station_id") == ident.station_id

    def _owned(self, cid: str, ident: Identity) -> dict:
        case = self._load(cid)
        if not self.allowed(case, ident):
            raise KeyError(cid)
        return case

    def create(self, ident: Identity, name: str | None = None, station_id: str | None = None) -> dict:
        division = ident.division_id
        if ident.is_admin:
            st = station(station_id)
            if not st or not ident.covers(st.get("division")):
                raise ValueError("Choose a police station in your division for this case.")
            division = st["division"]
        else:
            station_id = ident.station_id
        cid = uuid.uuid4().hex[:10]
        case = {"id": cid, "name": (name or f"Case {cid[:4].upper()}").strip()[:120], "created": time.time(), "target": None,
                "cameras": [], "notes": [], "sightings": [], "routes": None, "station_id": station_id,
                "division_id": division, "created_by": ident.username, "status": "open"}
        with self.lock:
            self._save(case)
        self.store.record(ident, "case_created", f"Opened {case['name']}", {"name": case["name"]}, case_id=cid, station_id=station_id)
        return self.public(case)

    def get(self, cid: str, ident: Identity) -> dict:
        return self.public(self._owned(cid, ident))

    def list(self, ident: Identity) -> list[dict]:
        return [{"id": c["id"], "name": c["name"], "created": c["created_at"], "cameras": c["cameras"], "target": c["target_label"],
                 "status": c["status"], "station_id": c["station_id"], "station_name": c["station_name"]}
                for c in self.store.cases(ident)]

    def public(self, case: dict) -> dict:
        c = json.loads(json.dumps(case))
        if c.get("target"):
            for k in PUBLIC_DROP:
                c["target"].pop(k, None)
        for cam in c["cameras"]:
            for tr in (cam.get("analysis") or {}).get("tracks", []):
                for k in PUBLIC_DROP:
                    tr.pop(k, None)
        st = station(c.get("station_id"))
        c["station_name"] = st["name"] if st else c.get("station_id")
        return c

    def update(self, cid: str, ident: Identity, name: str | None = None, status: str | None = None, station_id: str | None = None) -> dict:
        changes = {}
        with self.lock:
            case = self._owned(cid, ident)
            if name and name.strip():
                case["name"] = changes["name"] = name.strip()[:120]
            if status:
                if status not in ("open", "closed"):
                    raise ValueError("Status must be open or closed.")
                case["status"] = changes["status"] = status
            if station_id and station_id != case.get("station_id"):
                if not ident.is_admin:
                    raise PermissionError("Only a division administrator can reassign a case.")
                st = station(station_id)
                if not st or not ident.covers(st.get("division")):
                    raise ValueError("Choose a police station in your division.")
                case["station_id"] = changes["station_id"] = station_id
            self._save(case)
        if changes:
            self.store.record(ident, "case_updated", f"Updated {case['name']}: " + ", ".join(f"{k} = {v}" for k, v in changes.items()),
                              changes, case_id=cid, station_id=case.get("station_id"))
        return self.public(case)

    def delete(self, cid: str, ident: Identity) -> None:
        with self.lock:
            case = self._owned(cid, ident)
            case["deleted_at"], case["deleted_by"] = time.time(), ident.username
            self._save(case)
        self.store.record(ident, "case_deleted", f"Deleted {case['name']}", {"name": case["name"]}, case_id=cid, station_id=case.get("station_id"))

    def set_target(self, cid: str, data: bytes, ident: Identity) -> dict:
        self._owned(cid, ident)
        prof = target_profile(self.pipeline, data)
        with self.lock:
            case = self._owned(cid, ident)
            case["target"] = prof
            case["sightings"] = rank(prof, case["cameras"])
            self._save(case)
        d = prof["decision"]
        self.store.record(ident, "hunt_target", f"Hunt target set: {prof['label']}",
                          {"label": prof["label"], "decision": d, "sightings": len(case["sightings"])}, case_id=cid,
                          thumb=base64.b64decode(prof["thumbnail"]) if prof.get("thumbnail") else None,
                          detections=[{"source": "target", "label": d.get("label"), "make": d.get("make"), "level": d.get("level"),
                                       "confidence": d.get("confidence"), "colour": prof["colour"]["name"], "vehicle_class": "car"}],
                          station_id=case.get("station_id"))
        return self.public(case)

    def add_camera(self, cid: str, video_path: Path, filename: str, name: str | None, lat: float | None, lon: float | None,
                   start: str | None, allowed: tuple[float, float] | None, track: tuple[bytes, str] | None, ident: Identity) -> dict:
        case = self._owned(cid, ident)
        if lat is not None and lon is not None and inside(lat, lon) is False:
            raise ValueError("The camera location must be inside Bengaluru.")
        track_pts = parse_track(*track) if track else None
        cam_id = uuid.uuid4().hex[:8]
        cam_dir = self._path(cid) / "cameras"
        cam_dir.mkdir(exist_ok=True)
        suffix = Path(filename).suffix.lower() if Path(filename).suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v") else ".mp4"
        dest = cam_dir / f"{cam_id}{suffix}"
        shutil.move(str(video_path), dest)
        meta_loc = mp4_location(str(dest))
        embedded = None if track else embedded_gps(str(dest))
        cam = {"id": cam_id, "name": (name or filename)[:120], "file": dest.name, "lat": lat, "lon": lon,
               "location_source": "investigator" if lat is not None else None, "start_time": start, "start_epoch": _epoch(start),
               "allowed_direction": list(allowed) if allowed else None, "status": "queued", "analysis": None,
               "environment": None, "gps": None, "route": None, "sha256": file_sha256(dest), "geolocation": None}
        if cam["lat"] is None and meta_loc and inside(meta_loc["lat"], meta_loc["lon"]) is not False:
            cam["lat"], cam["lon"], cam["location_source"] = meta_loc["lat"], meta_loc["lon"], "video metadata"
        if track:
            pts = track_pts
            cam["gps"] = pts
            if pts and cam["lat"] is None:
                cam["lat"], cam["lon"], cam["location_source"] = pts[0]["lat"], pts[0]["lon"], "GPS track"
        elif embedded and inside(embedded["points"][0]["lat"], embedded["points"][0]["lon"]) is not False:
            pts = [{"lat": p["lat"], "lon": p["lon"], "epoch": p["epoch"], "t": p["t"]} for p in embedded["points"]]
            if any(p["t"] is None for p in pts) and all(p["epoch"] is not None for p in pts):
                for p in pts:
                    p["t"] = p["epoch"] - pts[0]["epoch"]
            if len(pts) >= 2:
                cam["gps"], cam["gps_source"] = pts, embedded["source"]
            if cam["lat"] is None:
                cam["lat"], cam["lon"], cam["location_source"] = pts[0]["lat"], pts[0]["lon"], embedded["source"]
        with self.lock:
            case = self._owned(cid, ident)
            case["cameras"].append(cam)
            self._save(case)
        self.store.record(ident, "camera_added", f"Footage added: {cam['name']}", {"camera": cam["name"], "location_source": cam["location_source"]},
                          case_id=cid, lat=cam["lat"], lon=cam["lon"], station_id=case.get("station_id"))
        job = uuid.uuid4().hex[:10]
        self.jobs[job] = {"id": job, "case_id": cid, "camera_id": cam_id, "status": "queued", "progress": 0.0, "error": None,
                          "station_id": case.get("station_id"), "division_id": case["division_id"]}
        self.pool.submit(self._run_camera, job, cid, cam_id, dest, allowed, ident)
        return {"job_id": job, "camera_id": cam_id}

    def job(self, job_id: str, ident: Identity) -> dict:
        j = self.jobs.get(job_id)
        if not j or (ident.is_admin and not ident.covers(j["division_id"])) or (not ident.is_admin and j["station_id"] != ident.station_id):
            raise KeyError(job_id)
        return {k: v for k, v in j.items() if k not in ("station_id", "division_id")}

    def _run_camera(self, job: str, cid: str, cam_id: str, path: Path, allowed, ident: Identity) -> None:
        state = self.jobs[job]
        state["status"] = "running"
        try:
            res = self.analyser.run(str(path), VideoOptions(allowed_direction=tuple(allowed) if allowed else None),
                                    progress=lambda p: state.__setitem__("progress", round(p * 0.9, 3)))
            frames = res.pop("_keyframes", [])
            env = environment(frames, self.ocr) if self.ocr is not None and frames else None
            with self.lock:
                case = self._load(cid)
                cam = next(c for c in case["cameras"] if c["id"] == cam_id)
                cam["analysis"], cam["environment"], cam["status"] = res, env, "done"
                cam["duration_s"] = res["meta"].get("analysed_s")
                if cam["lat"] is None and env and env["location_candidates"]:
                    best = env["location_candidates"][0]
                    if inside(best["lat"], best["lon"]) is not False:
                        cam["suggested_location"] = best
                track_rows = []
                if cam.get("gps"):
                    cam["route"] = map_match(cam["gps"])
                    track_rows = from_track(f"{cid}:{cam_id}", cam["gps"])
                    for tr in res["tracks"]:
                        pos = position_at(cam["gps"], tr["first_t"])
                        if pos:
                            tr["lat"], tr["lon"] = pos
                before = len(case.get("sightings", []))
                if case.get("target"):
                    case["sightings"] = rank(case["target"], case["cameras"])
                self._save(case)
            self.store.record(ident, "camera_analysed", f"{cam['name']}: {video_summary(res)}",
                              {"counts": res["counts"], "alerts": res["alerts"], "meta": res["meta"]}, case_id=cid,
                              lat=cam["lat"], lon=cam["lon"], detections=_detections(res["tracks"]), station_id=case.get("station_id"))
            if track_rows:
                self.store.add_journeys(ident, "dashcam", track_rows, case_id=cid, station_id=case.get("station_id"))
                self.refit()
            found = len(case.get("sightings", [])) - before
            if case.get("target") and found > 0:
                best = max(case["sightings"], key=lambda s: s["score"])
                self.store.record(ident, "hunt_sighting", f"{found} possible sighting{'s' if found != 1 else ''} of {case['target']['label']} "
                                  f"(best match {best['score']:.0%} at {best['camera_name']})",
                                  {"new": found, "best": {k: best.get(k) for k in ("camera_name", "score", "t_video", "label")}},
                                  case_id=cid, lat=best.get("lat"), lon=best.get("lon"), station_id=case.get("station_id"))
            state["status"], state["progress"] = "done", 1.0
        except Exception as e:
            state["status"], state["error"] = "error", str(e)
            with self.lock:
                case = self._load(cid)
                for c in case["cameras"]:
                    if c["id"] == cam_id:
                        c["status"], c["error"] = "error", str(e)
                self._save(case)

    def update_camera(self, cid: str, cam_id: str, lat: float | None, lon: float | None, start: str | None,
                      name: str | None, ident: Identity) -> dict:
        if lat is not None and lon is not None and inside(lat, lon) is False:
            raise ValueError("The camera location must be inside Bengaluru.")
        with self.lock:
            case = self._owned(cid, ident)
            cam = next(c for c in case["cameras"] if c["id"] == cam_id)
            if lat is not None and lon is not None:
                cam["lat"], cam["lon"], cam["location_source"] = lat, lon, "investigator"
            if start:
                cam["start_time"], cam["start_epoch"] = start, _epoch(start)
            if name:
                cam["name"] = name[:120]
            if case.get("target"):
                case["sightings"] = rank(case["target"], case["cameras"])
            self._save(case)
        return self.public(case)

    def geolocate(self, cid: str, cam_id: str, lat: float, lon: float, radius_m: float, hfov_deg: float, ident: Identity) -> dict:
        if inside(lat, lon) is False:
            raise ValueError("The search area must be inside Bengaluru.")
        if not 300 <= radius_m <= 3000:
            raise ValueError("Choose a search radius between 300 m and 3 km.")
        if not 20 <= hfov_deg <= 170:
            raise ValueError("Enter the camera's horizontal field of view in degrees, between 20 and 170.")
        if not self.road_graph_path or not Path(self.road_graph_path).exists():
            raise ValueError("The road network used to locate footage is not installed on this server.")
        case = self._owned(cid, ident)
        cam = next(c for c in case["cameras"] if c["id"] == cam_id)
        if cam.get("status") != "done":
            raise ValueError("Wait for the footage analysis to finish first.")
        if any(j.get("kind") == "geolocate" and j["status"] in ("queued", "running") and j["camera_id"] == cam_id for j in self.jobs.values()):
            raise ValueError("This footage is already being located.")
        job = uuid.uuid4().hex[:10]
        self.jobs[job] = {"id": job, "kind": "geolocate", "case_id": cid, "camera_id": cam_id, "status": "queued", "progress": 0.0, "error": None,
                          "station_id": case.get("station_id"), "division_id": case["division_id"]}
        self.geo_pool.submit(self._run_geolocate, job, cid, cam_id, lat, lon, radius_m, hfov_deg, ident)
        return {"job_id": job, "camera_id": cam_id}

    def _run_geolocate(self, job: str, cid: str, cam_id: str, lat: float, lon: float, radius_m: float, hfov_deg: float, ident: Identity) -> None:
        state = self.jobs[job]
        state["status"] = "running"
        try:
            case = self._load(cid)
            cam = next(c for c in case["cameras"] if c["id"] == cam_id)
            path = self._path(cid) / "cameras" / cam["file"]
            g = road_graph(self.road_graph_path)
            res = locate_video(str(path), g, (lon, lat), radius_m, hfov_deg, workers=self.geo_workers,
                               progress=lambda p: state.__setitem__("progress", round(min(p, 0.99), 3)), ocr=self.ocr,
                               places_path=self.places_path if self.places_path and Path(self.places_path).exists() else None)
            res.update(requested_by=ident.username, requested_at=time.time(), video_sha256=cam.get("sha256") or file_sha256(path),
                       map_data=g.source, validation=self.validation)
            with self.lock:
                case = self._load(cid)
                cam = next(c for c in case["cameras"] if c["id"] == cam_id)
                cam["geolocation"] = res
                self._save(case)
            loc = res.get("location") or {}
            label = {"candidate": "candidate route", "ambiguous": "two places fit", "not_located": "not located", "not_applicable": "cannot be matched"}.get(res["status"], res["status"])
            conf = f" ({res['confidence']:.0%})" if res.get("confidence") is not None else ""
            self.store.record(ident, "camera_geolocated", f"{cam['name']}: {label}{conf}",
                              {"status": res["status"], "confidence": res.get("confidence"), "log10_likelihood_ratio": res.get("log10_likelihood_ratio"),
                               "search_area": res.get("search_area"), "streets": res.get("streets", [])[:6]},
                              case_id=cid, lat=loc.get("lat"), lon=loc.get("lon"), station_id=case.get("station_id"))
            state["status"], state["progress"] = "done", 1.0
        except Exception as e:
            state["status"], state["error"] = "error", str(e)

    def accept_geolocation(self, cid: str, cam_id: str, ident: Identity) -> dict:
        with self.lock:
            case = self._owned(cid, ident)
            cam = next(c for c in case["cameras"] if c["id"] == cam_id)
            geo = cam.get("geolocation")
            if not geo or geo.get("status") != "candidate" or not geo.get("route"):
                raise ValueError("Only a candidate route can be used for the camera.")
            pts = [{"lat": p["lat"], "lon": p["lon"], "epoch": None, "t": p["t"]} for p in geo["route"]]
            cam["gps"], cam["gps_source"] = pts, "estimated from camera motion"
            cam["route"] = {"matched": True, "geometry": [[p["lon"], p["lat"]] for p in pts], "roads": geo.get("streets", []), "estimated": True}
            cam["lat"], cam["lon"], cam["location_source"] = pts[0]["lat"], pts[0]["lon"], "estimated from camera motion"
            geo["accepted_by"], geo["accepted_at"] = ident.username, time.time()
            for tr in (cam.get("analysis") or {}).get("tracks", []):
                pos = position_at(pts, tr["first_t"])
                if pos:
                    tr["lat"], tr["lon"] = pos
            if case.get("target"):
                case["sightings"] = rank(case["target"], case["cameras"])
            self._save(case)
        self.store.record(ident, "camera_location_confirmed", f"{cam['name']}: estimated route confirmed",
                          {"streets": geo.get("streets", [])[:6], "confidence": geo.get("confidence")},
                          case_id=cid, lat=pts[0]["lat"], lon=pts[0]["lon"], station_id=case.get("station_id"))
        return self.public(case)

    def video_path(self, cid: str, cam_id: str, ident: Identity) -> Path:
        case = self._owned(cid, ident)
        cam = next(c for c in case["cameras"] if c["id"] == cam_id)
        return self._path(cid) / "cameras" / cam["file"]

    def reason(self, cid: str, text: str, ident: Identity, min_score: float | None = None) -> dict:
        min_score = hunt_model()["threshold"] if min_score is None else min_score
        hints = parse_hints(text) if text.strip() else {"places": [], "times": [], "activities": [], "direction": None, "text": ""}
        with self.lock:
            case = self._owned(cid, ident)
        if text.strip():
            case["notes"].append({"text": text[:2000], "at": time.time(), "by": ident.username})
        sightings = [s for s in case["sightings"] if s["score"] >= min_score]
        best: dict[str, dict] = {}
        for s in sightings:
            if s["camera_id"] not in best or s["score"] > best[s["camera_id"]]["score"]:
                best[s["camera_id"]] = s
        result = infer(list(best.values()), hints, case["cameras"], self.traffic())
        ordered = sorted([x for x in best.values() if x.get("lat") is not None], key=lambda x: (x.get("t_abs") is None, x.get("t_abs") or 0))
        observed = from_sightings(cid, ordered, result.get("segments", []))
        if observed:
            self.store.add_journeys(ident, "sighting", observed, case_id=cid, station_id=case.get("station_id"))
            self.refit()
        with self.lock:
            fresh = self._owned(cid, ident)
            fresh["notes"], fresh["routes"] = case["notes"], result
            self._save(fresh)
        top = (result.get("onward") or [None])[0]
        self.store.record(ident, "route_inference",
                          f"Route reasoning: {len(result.get('segments', []))} segment(s), {len(result.get('onward', []))} onward hypotheses"
                          + (f", most likely towards {top.get('destination')} ({top.get('probability', 0):.0%})" if top else ""),
                          {"note": text[:500], "places": [p.get("text") for p in hints.get("places", [])]}, case_id=cid,
                          station_id=case.get("station_id"))
        return result
