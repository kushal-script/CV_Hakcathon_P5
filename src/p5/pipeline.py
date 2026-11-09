from __future__ import annotations

import base64
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from p5 import render
from p5.catalogue import Catalogue, primary, tagged
from p5.classical.features import extract, hsv_hist
from p5.classical.preprocess import enhance
from p5.classical.quality import analyse
from p5.classical.saliency import classical_roi
from p5.classical.svm import ClassicalModel
from p5.classification.cam import class_activation_map, overlay
from p5.classification.classifier import Classifier
from p5.classification.vehicle_type import AUTO_NOTE, load as load_vehicle_type
from p5.confidence.policy import Decision, Thresholds, decide
from p5.detection.detector import Detector
from p5.detection.select import select_boxes
from p5.hybrid.fusion import fuse
from p5.io import InputError, load_image
from p5.taxonomy import Taxonomy
from p5.video.analyse import embedding


@dataclass
class AppConfig:
    models: Path
    serve_size: int | None = None
    flip_tta: bool = False
    fusion: bool = False
    max_side: int = 1600
    max_extra_vehicles: int = 2
    threads: int = 2
    panels: bool = True
    enhance: bool = True
    comparison: bool = True
    keep_logits: bool = False
    india: bool = False
    prefer: str = "india"
    quality: dict | None = None
    enhancement: dict | None = None

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        data = yaml.safe_load(path.read_text())
        data["models"] = (path.parent / data["models"]).resolve()
        return cls(**data)


@dataclass
class VehicleResult:
    box: list[float]
    detection_score: float
    decision: dict
    classical: dict | None = None
    geometry: dict | None = None
    markets: dict | None = None


@dataclass
class PipelineResult:
    request_id: str
    status: str
    vehicles: list[VehicleResult] = field(default_factory=list)
    quality: dict | None = None
    enhancement: list[str] = field(default_factory=list)
    comparison: dict | None = None
    panels: dict[str, str] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    image_size: list[int] | None = None
    primary_logits: list[float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


US_TITLE = "US market, Stanford Cars ({n} models)"
INDIA_TITLE = "Indian market ({n} models)"


def top1(scores: np.ndarray, tax: Taxonomy) -> dict:
    p = np.exp(scores - scores.max())
    p = p / p.sum()
    i = int(p.argmax())
    return {"label": tax.names[i], "class_id": i, "confidence": float(p[i])}


class Pipeline:
    def __init__(self, cfg: AppConfig):
        m = cfg.models
        self.cfg = cfg
        self.tax = Taxonomy.load(m / "taxonomy")
        self.th = Thresholds.load(m / "thresholds.yaml")
        self.detector = Detector(m / "detector.onnx", threads=cfg.threads)
        self.classifier = Classifier(m / "classifier.onnx", m / "classifier_meta.json", cfg.threads, cfg.serve_size)
        self.fc_weight = np.load(m / "head.npz")["fc_weight"]
        self.classical = ClassicalModel.load(m / "classical_global.npz") if (m / "classical_global.npz").exists() else None
        self.base = Catalogue("us", US_TITLE.format(n=len(self.tax.names)), self.tax, self.th, self.classifier, self.fc_weight)
        self.extra = [Catalogue.load("india", INDIA_TITLE, m / "india", cfg.threads, cfg.serve_size)] \
            if cfg.india and (m / "india" / "classifier.onnx").exists() else []
        self.vehicle_type = load_vehicle_type(m)

    @property
    def catalogues(self) -> list[Catalogue]:
        return [self.base, *self.extra]

    def market_decisions(self, crops, logits, work_img, boxes) -> list[list]:
        out = [[(self.base, logits[k], None)] for k in range(len(boxes))]
        for cat in self.extra:
            cc = crops if cat.same_input(self.base) else [cat.classifier.prepare(work_img, b) for b in boxes]
            z, f = cat.classifier.run(cc, self.cfg.flip_tta)
            for k in range(len(boxes)):
                out[k].append((cat, z[k], f[k]))
        return out

    @classmethod
    def from_config(cls, path: Path) -> "Pipeline":
        return cls(AppConfig.load(path))

    def run(self, data: bytes, classify_anyway: bool = False) -> PipelineResult:
        rid = uuid.uuid4().hex[:12]
        t: dict[str, float] = {}
        start = clock = time.perf_counter()

        def tick(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            t[name] = round((now - clock) * 1000, 1)
            clock = now

        try:
            img = load_image(data, self.cfg.max_side)
        except InputError as e:
            return PipelineResult(rid, "error", error=str(e))
        tick("io")
        rgb = img.rgb
        q = analyse(rgb, self.cfg.quality)
        tick("quality")
        if q.flags and max(rgb.shape[:2]) > 1024:
            s = 1024 / max(rgb.shape[:2])
            img.image = img.image.resize((round(img.image.width * s), round(img.image.height * s)), Image.LANCZOS)
            rgb = np.asarray(img.image)
        knobs = self.cfg.enhancement or {}
        enh = enhance(rgb, q, **knobs) if self.cfg.enhance else enhance(rgb, type(q)(**{**asdict(q), "flags": []}), **knobs)
        tick("enhance")
        conf = min(self.th.tau_lo, 0.05)
        dets = self.detector(enh.rgb, conf=conf)
        work, applied = enh.rgb, enh.applied
        if enh.applied:
            raw_dets = self.detector(rgb, conf=conf)
            if (raw_dets[0].score if raw_dets else 0.0) >= (dets[0].score if dets else 0.0):
                dets, work, applied = raw_dets, rgb, [a + " (rejected: detector preferred original)" for a in enh.applied]
        work_img = Image.fromarray(work) if work is not rgb else img.image
        sel = select_boxes(dets, self.th.tau_lo, self.th.tau_det, 1 + self.cfg.max_extra_vehicles)
        tick("detect")
        result = PipelineResult(rid, sel.status, quality={**asdict(q)}, enhancement=applied,
                                image_size=[img.image.width, img.image.height])

        boxes, scores = [], []
        if sel.primary is not None:
            boxes, scores = [sel.primary.box] + [d.box for d in sel.others], [sel.primary.score] + [d.score for d in sel.others]
        sroi = None
        if not boxes and classify_anyway:
            sroi = classical_roi(work)
            boxes, scores = [sroi.box], [0.0]
            tick("classical_roi")
        if not boxes:
            result.vehicles = []
            result.panels = self.panels(rgb, enh.rgb, work, enh.applied, [], None, None, None) if self.cfg.panels else {}
            t["total"] = round((time.perf_counter() - start) * 1000, 1)
            result.timings_ms = t
            return result

        crops = [self.classifier.prepare(work_img, b) for b in boxes]
        logits, feats = self.classifier.run(crops, self.cfg.flip_tta)
        p_auto = [self.vehicle_type.autorickshaw(embedding(f)) for f in feats] if self.vehicle_type is not None else [0.0] * len(boxes)
        tick("classify")

        per_market = self.market_decisions(crops, logits, work_img, boxes) if self.extra else None
        if self.extra:
            tick("classify_markets")
        primary_roi = None
        vehicles = []
        cls_out = None
        cam_source = None
        named_locally = [any(c.key == self.cfg.prefer and decide(z, c.tax, c.th, detected=True).level == "class" for c, z, _ in per_market[k][1:])
                         if per_market else False for k in range(len(boxes))]
        for k, (box, score) in enumerate(zip(boxes, scores)):
            if self.vehicle_type is not None and p_auto[k] >= self.vehicle_type.threshold and not named_locally[k]:
                auto = {**asdict(Decision("vehicle_type", "Auto-rickshaw", p_auto[k], None, note=AUTO_NOTE)), "catalogue": None, "catalogue_title": None}
                vehicles.append(VehicleResult([round(v, 1) for v in box], round(score, 4), auto, None, None, None))
                continue
            z_fused = None
            classical, geo_d = None, None
            if k == 0 and self.classical is not None and (self.cfg.comparison or self.cfg.panels or self.cfg.fusion):
                vec, primary_roi, geo = extract(work, box)
                cls_scores = self.classical.scores(vec[None])[0]
                cls_out = cls_scores
                classical = top1(cls_scores / self.classical.temperature, self.tax)
                geo_d = asdict(geo)
                if self.cfg.fusion and self.th.beta > 0:
                    z_fused = fuse(logits[k], self.classical.proba(vec[None])[0], self.th.beta)
            d: Decision = decide(logits[k], self.tax, self.th, z_fused, detected=True)
            warn = []
            if sel.status == "weak" and k == 0:
                warn.append("Possible vehicle: low detector confidence.")
            if sroi is not None and k == 0:
                warn.append("Forced classification: no vehicle was detected, so this answer is unreliable.")
            d.note = " ".join([d.note, *warn]).strip()
            if per_market:
                ds = [d] + [decide(z, cat.tax, cat.th, detected=True) for cat, z, _ in per_market[k][1:]]
                for x in ds[1:]:
                    x.note = " ".join([x.note, *warn]).strip()
                cats = [c for c, _, _ in per_market[k]]
                pick = primary(ds, next((i for i, c in enumerate(cats) if c.key == self.cfg.prefer), 0))
                markets = {c.key: tagged(x, c) for c, x in zip(cats, ds)}
                chosen = markets[cats[pick].key]
                if k == 0:
                    cam_source = per_market[k][pick]
                vehicles.append(VehicleResult([round(v, 1) for v in box], round(score, 4), chosen, classical, geo_d, markets))
                continue
            vehicles.append(VehicleResult([round(v, 1) for v in box], round(score, 4), tagged(d, self.base), classical, geo_d))
        tick("refine_classical")
        result.vehicles = vehicles

        if self.cfg.keep_logits:
            result.primary_logits = [round(float(v), 5) for v in logits[0]]
        lead = vehicles[0]
        hybrid = lead.markets[self.base.key] if lead.markets else lead.decision
        comparison = {"deep": top1(logits[0] / self.th.t_d, self.tax), "hybrid": hybrid,
                      "final_catalogue": lead.decision.get("catalogue_title") if lead.decision.get("catalogue") != self.base.key else None}
        if self.classical is not None and self.cfg.comparison:
            comparison["classical_on_detector_crop"] = vehicles[0].classical
            sroi = sroi or classical_roi(work)
            vec_c, _, _ = extract(work, sroi.box, sroi)
            comparison["classical_only"] = {**top1(self.classical.scores(vec_c[None])[0] / self.classical.temperature, self.tax),
                                            "roi": [round(v, 1) for v in sroi.box]}
        result.comparison = comparison
        tick("comparison")
        if self.cfg.panels:
            if cam_source is not None and cam_source[2] is not None:
                cat, z, f = cam_source
                cam = class_activation_map(f, cat.fc_weight, int(z.argmax()), crops[0].shape[1::-1])
            else:
                cam = class_activation_map(feats[0], self.fc_weight, int(logits[0].argmax()), crops[0].shape[1::-1])
            result.panels = self.panels(rgb, enh.rgb, work, enh.applied, vehicles, primary_roi, sroi, overlay(crops[0], cam))
            tick("panels")
        t["total"] = round((time.perf_counter() - start) * 1000, 1)
        result.timings_ms = t
        return result

    def panels(self, rgb, enhanced, work, applied, vehicles, roi, sroi, cam_img) -> dict[str, str]:
        out = {}
        annotated = work.copy()
        for k in list(range(1, len(vehicles))) + ([0] if vehicles else []):
            d = vehicles[k].decision
            colour = render.COLOURS["secondary"] if k else render.COLOURS.get(d["level"], render.COLOURS["class"])
            short = d["label"] if d["level"] in ("class", "family") else (d.get("make") or "vehicle")
            text = f"{d['label']} {d['confidence']:.0%}" if k == 0 else f"{k + 1}. {short}"
            render.draw_box(annotated, vehicles[k].box, text, colour, below=k > 0)
        out["annotated"] = render.jpeg(annotated, 1024)
        if applied:
            out["original"] = render.jpeg(rgb)
            out["enhanced"] = render.jpeg(enhanced)
        if roi is not None:
            out["grabcut"] = render.jpeg(render.mask_overlay(work, roi.mask, roi.contour))
            out["edges"] = render.jpeg(render.edges_panel(roi.edges))
            out["colour_histogram"] = render.jpeg(render.hsv_hist_panel(hsv_hist(work, roi.mask)))
        if sroi is not None:
            sal = work.copy()
            render.draw_box(sal, sroi.box, "classical ROI", render.COLOURS["abstain"])
            out["saliency_roi"] = render.jpeg(sal)
        if cam_img is not None:
            out["cam"] = render.jpeg(cam_img, 368)
        return {k: base64.b64encode(v).decode() for k, v in out.items()}
