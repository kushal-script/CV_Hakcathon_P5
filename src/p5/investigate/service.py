from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from p5.geo.locate import cluster, ocr_evidence, queries, similarity
from p5.geo.geocode import photon
from p5.geo.route import map_match, mp4_location, parse_track, position_at
from p5.investigate.hints import parse as parse_hints
from p5.investigate.hunt import rank, target_profile
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
            if sim >= 0.55:
                points.append({**hit, "score": float(np.mean([e.ocr_conf for e in src])) * sim / (1 + 0.5 * hit["rank"]),
                               "query": q, "evidence_key": q})
    cands = []
    for c in cluster(points)[:3]:
        best = max(c["members"], key=lambda m: m["score"])
        cands.append({"lat": c["lat"], "lon": c["lon"], "label": best["name"], "confidence": round(float(1 - np.exp(-1.5 * c["score"])), 3),
                      "texts": sorted({m["query"] for m in c["members"]})[:5]})
    return {"texts": sorted({e.text for e in ev})[:20], "location_candidates": cands}


class CaseStore:
    def __init__(self, root: Path, pipeline, analyser: VideoAnalyser, ocr):
        self.root, self.pipeline, self.analyser, self.ocr = root, pipeline, analyser, ocr
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.pool = ThreadPoolExecutor(max_workers=1)

    def _path(self, cid: str) -> Path:
        if not cid.isalnum():
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

    def create(self, name: str | None = None) -> dict:
        cid = uuid.uuid4().hex[:10]
        case = {"id": cid, "name": name or f"Case {cid[:4].upper()}", "created": time.time(), "target": None,
                "cameras": [], "notes": [], "sightings": [], "routes": None}
        with self.lock:
            self._save(case)
        return self.public(case)

    def get(self, cid: str) -> dict:
        return self.public(self._load(cid))

    def list(self) -> list[dict]:
        out = []
        for f in sorted(self.root.glob("*/case.json"), key=lambda f: -f.stat().st_mtime):
            c = json.loads(f.read_text())
            out.append({"id": c["id"], "name": c["name"], "created": c["created"], "cameras": len(c["cameras"]),
                        "target": (c.get("target") or {}).get("label")})
        return out

    def public(self, case: dict) -> dict:
        c = json.loads(json.dumps(case))
        if c.get("target"):
            for k in PUBLIC_DROP:
                c["target"].pop(k, None)
        for cam in c["cameras"]:
            for tr in (cam.get("analysis") or {}).get("tracks", []):
                for k in PUBLIC_DROP:
                    tr.pop(k, None)
        return c

    def set_target(self, cid: str, data: bytes) -> dict:
        prof = target_profile(self.pipeline, data)
        with self.lock:
            case = self._load(cid)
            case["target"] = prof
            case["sightings"] = rank(prof, case["cameras"])
            self._save(case)
        return self.public(case)

    def add_camera(self, cid: str, video_path: Path, filename: str, name: str | None, lat: float | None, lon: float | None,
                   start: str | None, allowed: tuple[float, float] | None, track: tuple[bytes, str] | None) -> dict:
        case = self._load(cid)
        cam_id = uuid.uuid4().hex[:8]
        cam_dir = self._path(cid) / "cameras"
        cam_dir.mkdir(exist_ok=True)
        suffix = Path(filename).suffix.lower() or ".mp4"
        dest = cam_dir / f"{cam_id}{suffix}"
        shutil.move(str(video_path), dest)
        meta_loc = mp4_location(str(dest))
        cam = {"id": cam_id, "name": name or filename, "file": dest.name, "lat": lat, "lon": lon,
               "location_source": "investigator" if lat is not None else None, "start_time": start, "start_epoch": _epoch(start),
               "allowed_direction": list(allowed) if allowed else None, "status": "queued", "analysis": None,
               "environment": None, "gps": None, "route": None}
        if cam["lat"] is None and meta_loc:
            cam["lat"], cam["lon"], cam["location_source"] = meta_loc["lat"], meta_loc["lon"], "video metadata"
        if track:
            pts = parse_track(*track)
            cam["gps"] = pts
            if pts and cam["lat"] is None:
                cam["lat"], cam["lon"], cam["location_source"] = pts[0]["lat"], pts[0]["lon"], "GPS track"
        with self.lock:
            case = self._load(cid)
            case["cameras"].append(cam)
            self._save(case)
        job = uuid.uuid4().hex[:10]
        self.jobs[job] = {"id": job, "case_id": cid, "camera_id": cam_id, "status": "queued", "progress": 0.0, "error": None}
        self.pool.submit(self._run_camera, job, cid, cam_id, dest, allowed)
        return {"job_id": job, "camera_id": cam_id}

    def _run_camera(self, job: str, cid: str, cam_id: str, path: Path, allowed) -> None:
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
                    cam["suggested_location"] = best
                if cam.get("gps"):
                    cam["route"] = map_match(cam["gps"])
                    for tr in res["tracks"]:
                        pos = position_at(cam["gps"], tr["first_t"])
                        if pos:
                            tr["lat"], tr["lon"] = pos
                if case.get("target"):
                    case["sightings"] = rank(case["target"], case["cameras"])
                self._save(case)
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
                      name: str | None) -> dict:
        with self.lock:
            case = self._load(cid)
            cam = next(c for c in case["cameras"] if c["id"] == cam_id)
            if lat is not None and lon is not None:
                cam["lat"], cam["lon"], cam["location_source"] = lat, lon, "investigator"
            if start:
                cam["start_time"], cam["start_epoch"] = start, _epoch(start)
            if name:
                cam["name"] = name
            if case.get("target"):
                case["sightings"] = rank(case["target"], case["cameras"])
            self._save(case)
        return self.public(case)

    def video_path(self, cid: str, cam_id: str) -> Path:
        case = self._load(cid)
        cam = next(c for c in case["cameras"] if c["id"] == cam_id)
        return self._path(cid) / "cameras" / cam["file"]

    def reason(self, cid: str, text: str, min_score: float = 0.5) -> dict:
        hints = parse_hints(text) if text.strip() else {"places": [], "times": [], "activities": [], "direction": None, "text": ""}
        with self.lock:
            case = self._load(cid)
        if text.strip():
            case["notes"].append({"text": text, "at": time.time()})
        sightings = [s for s in case["sightings"] if s["score"] >= min_score]
        best: dict[str, dict] = {}
        for s in sightings:
            if s["camera_id"] not in best or s["score"] > best[s["camera_id"]]["score"]:
                best[s["camera_id"]] = s
        result = infer(list(best.values()), hints, case["cameras"])
        with self.lock:
            fresh = self._load(cid)
            fresh["notes"], fresh["routes"] = case["notes"], result
            self._save(fresh)
        return result
