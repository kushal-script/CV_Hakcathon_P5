from __future__ import annotations

import base64

import numpy as np

from p5 import render
from p5.classical.colour import vehicle_colour
from p5.confidence.policy import decide
from p5.confidence.temperature import softmax_t
from p5.detection.select import select_boxes
from p5.io import load_image
from p5.video.analyse import embedding

COLOUR_GROUPS = [{"white", "silver"}, {"silver", "grey"}, {"grey", "black"}, {"red", "maroon"}, {"orange", "brown"},
                 {"orange", "yellow"}, {"blue", "purple"}]
WEIGHTS = {"appearance": 0.55, "model": 0.30, "colour": 0.15}


def colour_match(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.5
    if a == b:
        return 1.0
    return 0.5 if any(a in g and b in g for g in COLOUR_GROUPS) else 0.0


def target_profile(pipeline, data: bytes) -> dict:
    img = load_image(data, pipeline.cfg.max_side)
    dets = pipeline.detector(img.rgb, conf=0.05)
    sel = select_boxes(dets, pipeline.th.tau_lo, pipeline.th.tau_det)
    box = sel.primary.box if sel.primary else (0, 0, img.rgb.shape[1], img.rgb.shape[0])
    crop = pipeline.classifier.prepare(img.image, box)
    logits, feats = pipeline.classifier.run([crop])
    d = decide(logits[0], pipeline.tax, pipeline.th)
    colour = vehicle_colour(crop)
    return {
        "embedding": embedding(feats[0]).astype(np.float32).tolist(),
        "probs": softmax_t(logits[0], pipeline.th.t_d).astype(np.float32).tolist(),
        "colour": colour,
        "decision": {"level": d.level, "label": d.label, "confidence": d.confidence, "make": d.make, "top5": d.top5[:5]},
        "box": [round(v, 1) for v in box],
        "detected": sel.primary is not None,
        "thumbnail": base64.b64encode(render.jpeg(crop, 200, 85)).decode(),
        "label": f"{colour['name']} {d.label}",
    }


def score(target: dict, track: dict) -> dict | None:
    if not track.get("embedding") or not track.get("probs"):
        return None
    app = float(np.dot(target["embedding"], track["embedding"]))
    app = float(np.clip((app - 0.35) / 0.45, 0.0, 1.0))
    model = float(np.sqrt(np.asarray(target["probs"]) * np.asarray(track["probs"])).sum())
    col = colour_match(target["colour"]["name"], (track.get("colour") or {}).get("name"))
    total = WEIGHTS["appearance"] * app + WEIGHTS["model"] * model + WEIGHTS["colour"] * col
    return {"score": round(total, 4), "appearance": round(app, 4), "model_overlap": round(model, 4), "colour": col}


def rank(target: dict, cameras: list[dict], min_score: float = 0.35) -> list[dict]:
    sightings = []
    for cam in cameras:
        for tr in (cam.get("analysis") or {}).get("tracks", []):
            s = score(target, tr)
            if s is None or s["score"] < min_score:
                continue
            sightings.append({"camera_id": cam["id"], "camera_name": cam.get("name"),
                              "lat": tr.get("lat", cam.get("lat")), "lon": tr.get("lon", cam.get("lon")),
                              "t_video": tr["first_t"], "t_abs": (cam.get("start_epoch") or 0) + tr["first_t"]
                              if cam.get("start_epoch") is not None else None,
                              "track_id": tr["id"], "label": tr["label"], "direction": tr["direction"]["compass"],
                              "thumbnail": tr.get("thumbnail"), **s})
    sightings.sort(key=lambda s: -s["score"])
    return sightings
