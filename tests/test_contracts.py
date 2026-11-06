import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from p5.classical.preprocess import enhance
from p5.classical.quality import analyse
from p5.classical.refine import refine
from p5.confidence.policy import Thresholds, decide
from p5.detection.detector import Detection, letterbox
from p5.detection.select import select_boxes
from p5.eval.detection import detection_report, iou
from p5.io import InputError, load_image
from p5.taxonomy import Taxonomy

TAX = Taxonomy.load(Path(__file__).resolve().parent.parent / "assets" / "taxonomy")


def png(w, h, mode="RGB", colour=(120, 120, 120)):
    buf = io.BytesIO()
    Image.new(mode, (w, h), colour if mode != "L" else 120).save(buf, "PNG")
    return buf.getvalue()


def test_io_rejects_non_image():
    with pytest.raises(InputError):
        load_image(b"not an image")


def test_io_rejects_tiny():
    with pytest.raises(InputError):
        load_image(png(20, 20))
    assert load_image(png(80, 40)).rgb.shape == (40, 80, 3)


def test_io_grey_and_rgba_become_rgb():
    assert load_image(png(200, 100, "L")).rgb.shape == (100, 200, 3)
    assert load_image(png(200, 100, "RGBA", (0, 0, 0, 0))).rgb.shape == (100, 200, 3)


def test_io_downscales_large():
    assert max(load_image(png(4000, 3000)).rgb.shape[:2]) == 1600


def test_policy_levels():
    th = Thresholds(tau_hi=0.5, tau_fam=0.5, tau_make=0.5, tau_ood=5.0, t_d=1.0)
    z = np.full(196, -10.0)
    z[10] = 20.0
    assert decide(z, TAX, th).level == "class"
    a, b = TAX.year_twins[0]
    z2 = np.full(196, -10.0)
    z2[a] = z2[b] = 20.0
    assert decide(z2, TAX, th).level == "family"
    assert decide(np.zeros(196), TAX, th).level in ("ood", "ood_make")
    assert decide(z, TAX, th, detected=False).level == "no_vehicle"


def test_select_boxes_status():
    assert select_boxes([], 0.1, 0.25).status == "none"
    assert select_boxes([Detection((0, 0, 10, 10), 0.15)], 0.1, 0.25).status == "weak"
    sel = select_boxes([Detection((0, 0, 10, 10), 0.9), Detection((0, 0, 100, 100), 0.8)], 0.1, 0.25)
    assert sel.status == "strong" and sel.primary.box == (0, 0, 100, 100)


def test_letterbox_shape():
    canvas, r, (dx, dy) = letterbox(np.zeros((300, 600, 3), np.uint8), 640)
    assert canvas.shape == (640, 640, 3) and r == pytest.approx(640 / 600)


def test_iou_and_report():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    rep = detection_report({1: [([0, 0, 10, 10], 0.9)]}, {1: [0, 0, 10, 10]})
    assert rep["mean_iou"] == pytest.approx(1.0) and rep["map_50"] == pytest.approx(1.0)


def test_quality_flags_dark_and_enhance():
    dark = np.full((200, 300, 3), 20, np.uint8)
    q = analyse(dark)
    assert "dark" in q.flags
    assert "gamma" in enhance(dark, q).applied


def test_refine_returns_box_inside_image():
    rgb = np.full((200, 300, 3), 200, np.uint8)
    rgb[60:140, 80:220] = (30, 60, 200)
    roi = refine(rgb, (70, 50, 230, 150))
    x1, y1, x2, y2 = roi.box
    assert 0 <= x1 < x2 <= 300 and 0 <= y1 < y2 <= 200
