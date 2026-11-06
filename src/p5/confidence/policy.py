from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from p5.confidence.temperature import softmax_t
from p5.taxonomy import Taxonomy

LEVELS = ("class", "family", "make", "abstain", "ood_make", "ood", "no_vehicle")


@dataclass(frozen=True)
class Thresholds:
    tau_lo: float = 0.10
    tau_det: float = 0.25
    tau_hi: float = 0.55
    tau_fam: float = 0.60
    tau_make: float = 0.60
    tau_ood: float = 0.0
    t_d: float = 1.0
    t_f: float | None = None
    beta: float = 0.0

    @classmethod
    def load(cls, path: Path) -> "Thresholds":
        data = yaml.safe_load(path.read_text()) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path, extra: dict | None = None) -> None:
        path.write_text(yaml.safe_dump({**asdict(self), **(extra or {})}, sort_keys=False))


@dataclass
class Decision:
    level: str
    label: str
    confidence: float
    class_id: int | None
    top5: list[tuple[str, float]] = field(default_factory=list)
    make: str | None = None
    ood_score: float = 0.0
    note: str = ""


def mls(z: np.ndarray) -> float:
    return float(np.max(z))


def group_mass(p: np.ndarray, group_of: list, g) -> float:
    return float(sum(p[i] for i, x in enumerate(group_of) if x == g))


def decide(z_deep: np.ndarray, tax: Taxonomy, th: Thresholds, z_fused: np.ndarray | None = None,
           detected: bool = True) -> Decision:
    if not detected:
        return Decision("no_vehicle", "No vehicle detected", 0.0, None)
    z, t = (z_fused, th.t_f or th.t_d) if z_fused is not None else (z_deep, th.t_d)
    p = softmax_t(z, t)
    order = np.argsort(-p)
    top = int(order[0])
    top5 = [(tax.names[i], float(p[i])) for i in order[:5]]
    score = mls(z_deep)
    dmake = tax.display_make_of[top]
    make_p = group_mass(p, tax.display_make_of, dmake)
    in_catalogue = score >= th.tau_ood
    if in_catalogue and p[top] >= th.tau_hi:
        return Decision("class", tax.names[top], float(p[top]), top, top5, dmake, score)
    fam = tax.family_of[top]
    fam_p = group_mass(p, tax.family_of, fam)
    if in_catalogue and fam_p >= th.tau_fam:
        years = sorted({tax.year_of[i] for i, f in enumerate(tax.family_of) if f == fam})
        label = f"{tax.family_names[fam]} ({' or '.join(map(str, years))})"
        return Decision("family", label, fam_p, top, top5, dmake, score, "The model cannot separate these model years.")
    if in_catalogue and make_p >= th.tau_make:
        return Decision("make", f"{dmake} (model uncertain)", make_p, top, top5, dmake, score)
    if in_catalogue:
        return Decision("abstain", "Not sure which model this is", float(p[top]), top, top5, dmake, score)
    if make_p >= th.tau_make:
        return Decision("ood_make", f"Possibly a {dmake}, but not one of the 196 known models", make_p, top, top5,
                        dmake, score)
    return Decision("ood", "Not one of the 196 known models", float(p[top]), top, top5, None, score)
