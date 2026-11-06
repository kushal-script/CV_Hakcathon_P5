from __future__ import annotations

import base64
import csv
import io
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel, Field

from p5.classification.classifier import Classifier
from p5.config import section
from p5.detection.detector import Detector
from p5.geo.exif import read_exif
from p5.geo.locate import locate
from p5.geo.traffic import TrafficModel
from p5.identity import Identity, secret, verify
from p5.investigate.journeys import fastest, parse_import
from p5.investigate.service import CaseStore, _detections, video_summary
from p5.io import InputError, load_image
from p5.pipeline import Pipeline
from p5.region import boundary, division_stations, inside, registry, station
from p5.store import IST, Store
from p5.video.analyse import VideoAnalyser, VideoOptions

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path(os.environ.get("P5_CONFIG", ROOT / "configs" / "app.yaml"))
DATA = Path(os.environ.get("P5_DATA_DIR", ROOT / "runs" / "cases"))
MAX_IMAGE = int(os.environ.get("P5_MAX_UPLOAD_MB", "15")) * 2**20
MAX_VIDEO = int(os.environ.get("P5_MAX_VIDEO_MB", "300")) * 2**20

log = logging.getLogger("p5")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Vahana Drishti analysis service", version="0.3.0", docs_url=None, redoc_url=None, openapi_url=None)
state: dict = {}


@app.on_event("startup")
def load() -> None:
    pipe = Pipeline.from_config(CONFIG)
    models = pipe.cfg.models
    coco = models / "detector_coco.onnx"
    analyser = VideoAnalyser(Detector(coco if coco.exists() else models / "detector.onnx", threads=pipe.cfg.threads,
                                      iou=section("video")["nms_iou"]),
                             Classifier(models / "classifier.onnx", models / "classifier_meta.json", pipe.cfg.threads),
                             pipe.tax, pipe.th, pipe.extra, pipe.cfg.prefer, pipe.base.title)
    try:
        from rapidocr import RapidOCR
        ocr = RapidOCR()
    except Exception:
        log.exception("OCR unavailable")
        ocr = None
    store = Store(DATA / "vahana.db")
    state.update(pipeline=pipe, analyser=analyser, ocr=ocr, store=store, secret=secret())
    refit_traffic()
    state["cases"] = CaseStore(DATA / "cases", pipe, analyser, ocr, store, traffic=lambda: state.get("traffic"), refit=refit_traffic)
    if state["secret"] is None:
        log.warning("P5_INTERNAL_SECRET is not set: every request except /api/health will be rejected")
    log.info("services loaded from %s", CONFIG)


TRAFFIC_PRIOR = Path(os.environ.get("P5_REGION_DIR", ROOT / "assets" / "bengaluru")) / "traffic_prior.json"


def refit_traffic() -> None:
    state["traffic"] = TrafficModel.load(TRAFFIC_PRIOR, state["store"].all_journeys())


def identity(request: Request) -> Identity:
    ident = verify(request.headers.get("x-p5-identity"), request.headers.get("x-p5-identity-sig"), state.get("secret"))
    if ident is None:
        raise HTTPException(401, "Sign in to continue.")
    return ident


def admin(ident: Identity = Depends(identity)) -> Identity:
    if not ident.is_admin:
        raise HTTPException(403, "Division administrators only.")
    return ident


def store() -> Store:
    return state["store"]


def cases() -> CaseStore:
    return state["cases"]


async def read_upload(file: UploadFile, limit: int) -> bytes:
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {limit // 2**20} MB.")
    if not data:
        raise HTTPException(400, "Empty upload.")
    return data


async def save_upload(file: UploadFile, limit: int) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=".upload")[1])
    size = 0
    with tmp.open("wb") as fh:
        while chunk := await file.read(2**20):
            size += len(chunk)
            if size > limit:
                fh.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"Video is larger than {limit // 2**20} MB.")
            fh.write(chunk)
    if size == 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "Empty upload.")
    return tmp


def is_video(path: Path) -> bool:
    with path.open("rb") as fh:
        head = fh.read(16)
    return head[4:8] == b"ftyp" or (head[:4] == b"RIFF" and head[8:12] == b"AVI ") or head[:4] == b"\x1a\x45\xdf\xa3"


async def save_video(file: UploadFile, limit: int) -> Path:
    tmp = await save_upload(file, limit)
    if not is_video(tmp):
        tmp.unlink(missing_ok=True)
        raise HTTPException(415, "This file is not a supported video. Please upload an MP4, MOV, AVI, MKV or WebM file.")
    return tmp


def parse_direction(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    try:
        dx, dy = (float(v) for v in value.split(","))
        return (dx, dy) if (dx or dy) else None
    except ValueError:
        raise HTTPException(422, "allowed_direction must be 'dx,dy'.")


def thumbnail(jpeg_b64: str | None, side: int = 360) -> bytes | None:
    if not jpeg_b64:
        return None
    try:
        im = Image.open(io.BytesIO(base64.b64decode(jpeg_b64))).convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=72)
        return buf.getvalue()
    except Exception:
        return None


def photo_location(data: bytes) -> tuple[float | None, float | None]:
    gps = read_exif(data)
    lat, lon = gps.get("lat"), gps.get("lon")
    return (lat, lon) if lat is not None and inside(lat, lon) else (None, None)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok" if state else "loading"}


@app.get("/api/meta")
def meta(ident: Identity = Depends(identity)) -> dict:
    pipe = state["pipeline"]
    report = pipe.cfg.models / "report.json"
    return {"classes": len(pipe.tax.names), "makes": pipe.tax.display_makes, "ocr": bool(state.get("ocr")),
            "catalogues": [{"key": c.key, "title": c.title, "classes": len(c.tax.names)} for c in pipe.catalogues],
            "report": json.loads(report.read_text()) if report.exists() else None}


@app.get("/api/region")
def region(ident: Identity = Depends(identity)) -> dict:
    b = boundary()
    stations = division_stations(ident.division_id) if ident.is_admin else [s for s in [station(ident.station_id)] if s]
    path = Path(os.environ.get("P5_REGION_DIR", ROOT / "assets" / "bengaluru")) / "boundary.geojson"
    return {"bbox": b.bbox if b else None, "boundary": json.loads(path.read_text()) if path.exists() else None,
            "division": registry()["divisions"].get(ident.division_id),
            "stations": [{k: s.get(k) for k in ("id", "name", "kind", "lat", "lon", "official_division")} for s in stations]}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), classify_anyway: bool = Form(False), ident: Identity = Depends(identity)) -> dict:
    data = await read_upload(file, MAX_IMAGE)
    try:
        result = await run_in_threadpool(state["pipeline"].run, data, classify_anyway)
    except Exception:
        log.exception("prediction failed")
        raise HTTPException(500, "Something went wrong while analysing this image. Please try another photo.")
    if result.error:
        raise HTTPException(422, result.error)
    out = result.to_dict()
    lat, lon = photo_location(data)
    dets = [{"source": "photo", "label": v["decision"]["label"], "make": v["decision"].get("make"), "level": v["decision"]["level"],
             "confidence": v["decision"]["confidence"], "catalogue": v["decision"].get("catalogue"), "vehicle_class": "car"}
            for v in out["vehicles"]]
    top = out["vehicles"][0]["decision"] if out["vehicles"] else None
    summary = f"{top['label']} ({top['confidence']:.0%})" if top else "No vehicle found"
    out["event_id"] = store().record(
        ident, "identify", summary,
        {"vehicles": [{"box": v["box"], "decision": {k: v["decision"].get(k) for k in ("level", "label", "confidence", "make", "catalogue", "top5")}}
                      for v in out["vehicles"]],
         "quality": (out.get("quality") or {}).get("flags"), "enhancement": out.get("enhancement"), "image_size": out.get("image_size"),
         "ms": out["timings_ms"].get("total")},
        lat=lat, lon=lon, thumb=thumbnail((out.get("panels") or {}).get("annotated")), detections=dets)
    return out


@app.post("/api/locate")
async def locate_photo(file: UploadFile = File(...), ident: Identity = Depends(identity)) -> dict:
    data = await read_upload(file, MAX_IMAGE)
    try:
        img = load_image(data, 1600)
    except InputError as e:
        raise HTTPException(422, str(e))
    res = await run_in_threadpool(locate, data, img.rgb, state.get("ocr"))
    for c in res.get("candidates", []):
        c["inside_bengaluru"] = inside(c.get("lat"), c.get("lon"))
    best = next((c for c in res.get("candidates", []) if c.get("inside_bengaluru")), None)
    store().record(ident, "locate", f"Located near {best.get('label') or 'a candidate'} ({best.get('confidence', 0):.0%})" if best else "Location not found",
                   {"methods": res.get("methods"), "candidates": len(res.get("candidates", []))},
                   lat=best.get("lat") if best else None, lon=best.get("lon") if best else None)
    return res


@app.post("/api/video")
async def analyse_video(file: UploadFile = File(...), allowed_direction: str | None = Form(None), ident: Identity = Depends(identity)) -> dict:
    tmp = await save_video(file, MAX_VIDEO)
    try:
        res = await run_in_threadpool(state["analyser"].run, str(tmp), VideoOptions(allowed_direction=parse_direction(allowed_direction)))
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        tmp.unlink(missing_ok=True)
    res.pop("_keyframes", None)
    store().record(ident, "video", f"{file.filename or 'Video'}: {video_summary(res)}",
                   {"counts": res["counts"], "alerts": res["alerts"], "meta": res["meta"]}, detections=_detections(res["tracks"]))
    for tr in res["tracks"]:
        tr.pop("embedding", None)
        tr.pop("probs", None)
    return res


@app.get("/api/activity")
def activity(limit: int = 20, ident: Identity = Depends(identity)) -> dict:
    return store().events(ident, limit=min(max(limit, 1), 100))


@app.get("/api/cases")
def list_cases(ident: Identity = Depends(identity)) -> list[dict]:
    return cases().list(ident)


@app.post("/api/cases")
def create_case(name: str | None = Form(None), station_id: str | None = Form(None), ident: Identity = Depends(identity)) -> dict:
    try:
        return cases().create(ident, name, station_id)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/cases/{cid}")
def get_case(cid: str, ident: Identity = Depends(identity)) -> dict:
    try:
        return cases().get(cid, ident)
    except KeyError:
        raise HTTPException(404, "Case not found.")


class CasePatch(BaseModel):
    name: str | None = Field(None, max_length=120)
    status: str | None = None
    station_id: str | None = Field(None, max_length=64)


@app.patch("/api/cases/{cid}")
def patch_case(cid: str, body: CasePatch, ident: Identity = Depends(identity)) -> dict:
    try:
        return cases().update(cid, ident, body.name, body.status, body.station_id)
    except KeyError:
        raise HTTPException(404, "Case not found.")
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/api/cases/{cid}/target")
async def set_target(cid: str, file: UploadFile = File(...), ident: Identity = Depends(identity)) -> dict:
    data = await read_upload(file, MAX_IMAGE)
    try:
        return await run_in_threadpool(cases().set_target, cid, data, ident)
    except KeyError:
        raise HTTPException(404, "Case not found.")
    except InputError as e:
        raise HTTPException(422, str(e))


@app.post("/api/cases/{cid}/cameras")
async def add_camera(cid: str, file: UploadFile = File(...), name: str | None = Form(None), lat: float | None = Form(None),
                     lon: float | None = Form(None), start_time: str | None = Form(None),
                     allowed_direction: str | None = Form(None), gps: UploadFile | None = File(None), ident: Identity = Depends(identity)) -> dict:
    tmp = await save_video(file, MAX_VIDEO)
    track = None
    if gps is not None:
        raw = await gps.read(5 * 2**20 + 1)
        if len(raw) > 5 * 2**20:
            tmp.unlink(missing_ok=True)
            raise HTTPException(413, "GPS track is larger than 5 MB.")
        track = (raw, gps.filename or "track.gpx")
    try:
        return cases().add_camera(cid, tmp, file.filename or "video.mp4", name, lat, lon, start_time, parse_direction(allowed_direction), track, ident)
    except KeyError:
        tmp.unlink(missing_ok=True)
        raise HTTPException(404, "Case not found.")
    except ValueError as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(422, str(e))


@app.patch("/api/cases/{cid}/cameras/{cam_id}")
def update_camera(cid: str, cam_id: str, lat: float | None = Form(None), lon: float | None = Form(None),
                  start_time: str | None = Form(None), name: str | None = Form(None), ident: Identity = Depends(identity)) -> dict:
    try:
        return cases().update_camera(cid, cam_id, lat, lon, start_time, name, ident)
    except (KeyError, StopIteration):
        raise HTTPException(404, "Camera not found.")
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/cases/{cid}/cameras/{cam_id}/video")
def camera_video(cid: str, cam_id: str, ident: Identity = Depends(identity)):
    try:
        return FileResponse(cases().video_path(cid, cam_id, ident))
    except (KeyError, StopIteration):
        raise HTTPException(404, "Video not found.")


@app.get("/api/jobs/{job_id}")
def job(job_id: str, ident: Identity = Depends(identity)) -> dict:
    try:
        return cases().job(job_id, ident)
    except KeyError:
        raise HTTPException(404, "Job not found.")


@app.post("/api/cases/{cid}/reason")
async def reason(cid: str, text: str = Form("", max_length=4000), min_score: float | None = Form(None), ident: Identity = Depends(identity)) -> dict:
    try:
        return await run_in_threadpool(cases().reason, cid, text, ident, min_score)
    except KeyError:
        raise HTTPException(404, "Case not found.")


def window(days: int | None, start: str | None, end: str | None) -> tuple[float, float]:
    try:
        if start and end:
            s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=IST)
            e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=IST) + timedelta(days=1)
            if e <= s or (e - s).days > 366:
                raise ValueError
            return s.timestamp(), e.timestamp()
    except ValueError:
        raise HTTPException(422, "Use from and to dates as YYYY-MM-DD, at most a year apart.")
    days = min(max(days or 30, 1), 366)
    today = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return (today - timedelta(days=days)).timestamp(), today.timestamp()


@app.get("/api/admin/overview")
def admin_overview(days: int = 30, start: str | None = None, end: str | None = None, ident: Identity = Depends(admin)) -> dict:
    since, until = window(days, start, end)
    out = store().overview(ident, since, until)
    out["division"] = registry()["divisions"].get(ident.division_id)
    return out


@app.get("/api/admin/events")
def admin_events(kind: str | None = None, station: str | None = None, q: str | None = None, days: int | None = None,
                 start: str | None = None, end: str | None = None, limit: int = 50, offset: int = 0, ident: Identity = Depends(admin)) -> dict:
    since, until = window(days, start, end) if (days or (start and end)) else (None, None)
    return store().events(ident, kind or None, station or None, (q or "")[:100] or None, since, until, limit, offset)


@app.get("/api/admin/events/{eid}")
def admin_event(eid: int, ident: Identity = Depends(admin)) -> dict:
    ev = store().event(ident, eid)
    if ev is None:
        raise HTTPException(404, "Event not found.")
    return ev


class EventPatch(BaseModel):
    note: str | None = Field(None, max_length=1000)
    reviewed: bool | None = None


@app.patch("/api/admin/events/{eid}")
def admin_event_patch(eid: int, body: EventPatch, ident: Identity = Depends(admin)) -> dict:
    ev = store().update_event(ident, eid, body.note, body.reviewed)
    if ev is None:
        raise HTTPException(404, "Event not found.")
    return ev


@app.delete("/api/admin/events/{eid}")
def admin_event_delete(eid: int, ident: Identity = Depends(admin)) -> dict:
    if not store().delete_event(ident, eid):
        raise HTTPException(404, "Event not found.")
    store().record(ident, "admin_event_deleted", f"Deleted event {eid}", {"event_id": eid}, station_id=None)
    return {"ok": True}


@app.get("/api/admin/cases")
def admin_cases(status: str | None = None, station: str | None = None, ident: Identity = Depends(admin)) -> list[dict]:
    return store().cases(ident, status or None, station or None)


class CaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    station_id: str = Field(..., max_length=64)


@app.post("/api/admin/cases")
def admin_case_create(body: CaseCreate, ident: Identity = Depends(admin)) -> dict:
    try:
        return cases().create(ident, body.name, body.station_id)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.delete("/api/admin/cases/{cid}")
def admin_case_delete(cid: str, ident: Identity = Depends(admin)) -> dict:
    try:
        cases().delete(cid, ident)
    except KeyError:
        raise HTTPException(404, "Case not found.")
    return {"ok": True}


@app.get("/api/admin/stations")
def admin_stations(days: int = 30, ident: Identity = Depends(admin)) -> dict:
    since, until = window(days, None, None)
    return {"division": registry()["divisions"].get(ident.division_id), "stations": store().overview(ident, since, until)["by_station"]}


@app.get("/api/admin/export.csv")
def admin_export(days: int = 30, start: str | None = None, end: str | None = None, ident: Identity = Depends(admin)) -> Response:
    since, until = window(days, start, end)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "time_ist", "kind", "station", "user", "summary", "lat", "lon", "case_id", "reviewed", "note"])
    offset = 0
    while True:
        page = store().events(ident, since=since, until=until, limit=200, offset=offset)
        for e in page["events"]:
            w.writerow([e["id"], datetime.fromtimestamp(e["ts"], IST).strftime("%Y-%m-%d %H:%M:%S"), e["kind"], e["station_name"], e["username"],
                        e["summary"], e["lat"], e["lon"], e["case_id"], e["reviewed"], e["note"]])
        offset += 200
        if offset >= page["total"] or not page["events"]:
            break
    name = f"vahana-{ident.division_id}-{datetime.now(IST).strftime('%Y%m%d-%H%M')}.csv"
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/admin/traffic")
def admin_traffic(limit: int = 100, offset: int = 0, ident: Identity = Depends(admin)) -> dict:
    model = state["traffic"]
    return {**model.summary(), **store().journeys(ident, limit, offset)}


@app.post("/api/admin/traffic/refit")
def admin_traffic_refit(ident: Identity = Depends(admin)) -> dict:
    refit_traffic()
    return state["traffic"].summary()


@app.delete("/api/admin/traffic/journeys/{jid}")
def admin_traffic_delete(jid: int, ident: Identity = Depends(admin)) -> dict:
    if not store().delete_journey(ident, jid):
        raise HTTPException(404, "Journey not found.")
    refit_traffic()
    store().record(ident, "journey_deleted", f"Deleted travel-time journey {jid}", {"journey_id": jid})
    return {"ok": True}


def _import_journeys(job: dict, rows: list[dict], ident: Identity) -> None:
    job["status"] = "running"
    kept = []
    try:
        for k, r in enumerate(rows):
            route = fastest((r["from_lat"], r["from_lon"]), (r["to_lat"], r["to_lon"]))
            if route:
                kept.append({**r, "ref": f"import:{job['id']}:{k}", "freeflow_s": route["duration_s"], "distance_m": route["distance_m"]})
            job["progress"] = round((k + 1) / len(rows), 3)
            time.sleep(1.0)
        added = store().add_journeys(ident, "import", kept) if kept else 0
        refit_traffic()
        store().record(ident, "journeys_imported", f"Imported {added} travel-time journeys", {"rows": len(rows), "added": added})
        job.update(status="done", progress=1.0, added=added, unrouted=len(rows) - len(kept))
    except Exception as e:
        job.update(status="error", error=str(e))


@app.post("/api/admin/traffic/import")
async def admin_traffic_import(file: UploadFile = File(...), ident: Identity = Depends(admin)) -> dict:
    data = await read_upload(file, 2 * 2**20)
    rows, errors = parse_import(data.decode("utf-8", "ignore"))
    if not rows:
        raise HTTPException(422, " ".join(errors) or "No usable rows.")
    job = {"id": uuid.uuid4().hex[:10], "status": "queued", "progress": 0.0, "error": None, "station_id": None, "division_id": ident.division_id,
           "rows": len(rows), "skipped": errors[:20]}
    cases().jobs[job["id"]] = job
    threading.Thread(target=_import_journeys, args=(job, rows, ident), daemon=True).start()
    return {"job_id": job["id"], "rows": len(rows), "skipped": errors[:20]}
