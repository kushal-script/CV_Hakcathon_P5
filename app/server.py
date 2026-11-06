from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from p5.classification.classifier import Classifier
from p5.config import section
from p5.detection.detector import Detector
from p5.geo.locate import locate
from p5.io import InputError, load_image
from p5.investigate.service import CaseStore
from p5.pipeline import Pipeline
from p5.video.analyse import VideoAnalyser, VideoOptions

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path(os.environ.get("P5_CONFIG", ROOT / "configs" / "app.yaml"))
DATA = Path(os.environ.get("P5_DATA_DIR", ROOT / "runs" / "cases"))
WEB = Path(os.environ.get("P5_WEB_DIR", ROOT / "web"))
MAX_IMAGE = int(os.environ.get("P5_MAX_UPLOAD_MB", "15")) * 2**20
MAX_VIDEO = int(os.environ.get("P5_MAX_VIDEO_MB", "300")) * 2**20
ORIGINS = [o.strip() for o in os.environ.get("P5_ALLOWED_ORIGINS", "*").split(",") if o.strip()]

log = logging.getLogger("p5")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="P5 Vehicle Recognition API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST", "PATCH"], allow_headers=["*"])
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
    state.update(pipeline=pipe, analyser=analyser, ocr=ocr, cases=CaseStore(DATA, pipe, analyser, ocr))
    log.info("services loaded from %s", CONFIG)


async def read_upload(file: UploadFile, limit: int) -> bytes:
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {limit // 2**20} MB.")
    if not data:
        raise HTTPException(400, "Empty upload.")
    return data


async def save_upload(file: UploadFile, limit: int) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=Path(file.filename or "video.mp4").suffix)[1])
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


def cases() -> CaseStore:
    return state["cases"]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok" if state else "loading", "ocr": bool(state.get("ocr"))}


@app.get("/api/meta")
def meta() -> dict:
    pipe = state["pipeline"]
    report = pipe.cfg.models / "report.json"
    return {"classes": len(pipe.tax.names), "makes": pipe.tax.display_makes,
            "catalogues": [{"key": c.key, "title": c.title, "classes": len(c.tax.names)} for c in pipe.catalogues],
            "report": json.loads(report.read_text()) if report.exists() else None}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), classify_anyway: bool = Form(False)) -> dict:
    data = await read_upload(file, MAX_IMAGE)
    try:
        result = await run_in_threadpool(state["pipeline"].run, data, classify_anyway)
    except Exception:
        log.exception("prediction failed")
        raise HTTPException(500, "Something went wrong while analysing this image. Please try another photo.")
    if result.error:
        raise HTTPException(422, result.error)
    return result.to_dict()


@app.post("/api/locate")
async def locate_photo(file: UploadFile = File(...)) -> dict:
    data = await read_upload(file, MAX_IMAGE)
    try:
        img = load_image(data, 1600)
    except InputError as e:
        raise HTTPException(422, str(e))
    return await run_in_threadpool(locate, data, img.rgb, state.get("ocr"))


@app.post("/api/video")
async def analyse_video(file: UploadFile = File(...), allowed_direction: str | None = Form(None)) -> dict:
    tmp = await save_video(file, MAX_VIDEO)
    try:
        res = await run_in_threadpool(state["analyser"].run, str(tmp), VideoOptions(allowed_direction=parse_direction(allowed_direction)))
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        tmp.unlink(missing_ok=True)
    res.pop("_keyframes", None)
    for tr in res["tracks"]:
        tr.pop("embedding", None)
        tr.pop("probs", None)
    return res


@app.get("/api/cases")
def list_cases() -> list[dict]:
    return cases().list()


@app.post("/api/cases")
def create_case(name: str | None = Form(None)) -> dict:
    return cases().create(name)


@app.get("/api/cases/{cid}")
def get_case(cid: str) -> dict:
    try:
        return cases().get(cid)
    except KeyError:
        raise HTTPException(404, "Case not found.")


@app.post("/api/cases/{cid}/target")
async def set_target(cid: str, file: UploadFile = File(...)) -> dict:
    data = await read_upload(file, MAX_IMAGE)
    try:
        return await run_in_threadpool(cases().set_target, cid, data)
    except KeyError:
        raise HTTPException(404, "Case not found.")
    except InputError as e:
        raise HTTPException(422, str(e))


@app.post("/api/cases/{cid}/cameras")
async def add_camera(cid: str, file: UploadFile = File(...), name: str | None = Form(None), lat: float | None = Form(None),
                     lon: float | None = Form(None), start_time: str | None = Form(None),
                     allowed_direction: str | None = Form(None), gps: UploadFile | None = File(None)) -> dict:
    tmp = await save_video(file, MAX_VIDEO)
    track = None
    if gps is not None:
        raw = await gps.read(5 * 2**20 + 1)
        if len(raw) > 5 * 2**20:
            tmp.unlink(missing_ok=True)
            raise HTTPException(413, "GPS track is larger than 5 MB.")
        track = (raw, gps.filename or "track.gpx")
    try:
        return cases().add_camera(cid, tmp, file.filename or "video.mp4", name, lat, lon, start_time,
                                  parse_direction(allowed_direction), track)
    except KeyError:
        tmp.unlink(missing_ok=True)
        raise HTTPException(404, "Case not found.")


@app.patch("/api/cases/{cid}/cameras/{cam_id}")
def update_camera(cid: str, cam_id: str, lat: float | None = Form(None), lon: float | None = Form(None),
                  start_time: str | None = Form(None), name: str | None = Form(None)) -> dict:
    try:
        return cases().update_camera(cid, cam_id, lat, lon, start_time, name)
    except (KeyError, StopIteration):
        raise HTTPException(404, "Camera not found.")


@app.get("/api/cases/{cid}/cameras/{cam_id}/video")
def camera_video(cid: str, cam_id: str):
    try:
        return FileResponse(cases().video_path(cid, cam_id))
    except (KeyError, StopIteration):
        raise HTTPException(404, "Video not found.")


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict:
    j = cases().jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found.")
    return j


@app.post("/api/cases/{cid}/reason")
async def reason(cid: str, text: str = Form(""), min_score: float | None = Form(None)) -> dict:
    try:
        return await run_in_threadpool(cases().reason, cid, text, min_score)
    except KeyError:
        raise HTTPException(404, "Case not found.")


if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
