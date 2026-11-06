from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from p5.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path(os.environ.get("P5_CONFIG", ROOT / "configs" / "app.yaml"))
MAX_BYTES = int(os.environ.get("P5_MAX_UPLOAD_MB", "15")) * 2**20
ORIGINS = [o.strip() for o in os.environ.get("P5_ALLOWED_ORIGINS", "*").split(",") if o.strip()]

log = logging.getLogger("p5")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="P5 Vehicle Recognition API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])
pipeline: Pipeline | None = None


@app.on_event("startup")
def load() -> None:
    global pipeline
    pipeline = Pipeline.from_config(CONFIG)
    log.info("pipeline loaded from %s", CONFIG)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok" if pipeline else "loading"}


@app.get("/api/meta")
def meta() -> dict:
    models = pipeline.cfg.models
    report = models / "report.json"
    return {"classes": len(pipeline.tax.names), "makes": pipeline.tax.display_makes,
            "report": json.loads(report.read_text()) if report.exists() else None}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), classify_anyway: bool = Form(False)) -> dict:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Image is larger than {MAX_BYTES // 2**20} MB.")
    if not data:
        raise HTTPException(400, "Empty upload.")
    try:
        result = await run_in_threadpool(pipeline.run, data, classify_anyway)
    except Exception:
        log.exception("prediction failed")
        raise HTTPException(500, "Something went wrong while analysing this image. Please try another photo.")
    if result.error:
        raise HTTPException(422, result.error)
    log.info("request %s status=%s total_ms=%s", result.request_id, result.status, result.timings_ms.get("total"))
    return result.to_dict()
