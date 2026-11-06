from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from p5.detection.detector import Detection


@dataclass
class Selection:
    status: str
    primary: Detection | None
    others: list[Detection] = field(default_factory=list)


def select_boxes(dets: list[Detection], tau_lo: float, tau_det: float, max_boxes: int = 5) -> Selection:
    cand = [d for d in dets if d.score >= tau_lo]
    if not cand:
        return Selection("none", None)

    def rank(d: Detection) -> float:
        x1, y1, x2, y2 = d.box
        return d.score * float(np.sqrt(max((x2 - x1) * (y2 - y1), 1.0)))

    cand.sort(key=rank, reverse=True)
    primary = cand[0]
    status = "strong" if primary.score >= tau_det else "weak"
    others = [d for d in cand[1:] if d.score >= tau_det][: max_boxes - 1]
    return Selection(status, primary, others)
