from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Box = tuple[float, float, float, float]


@dataclass
class Quality:
    luma_mean: float
    luma_p5: float
    luma_p95: float
    rms_contrast: float
    cast: float
    noise_sigma: float
    sharpness: float
    flags: list[str] = field(default_factory=list)


@dataclass
class Enhanced:
    rgb: np.ndarray
    applied: list[str]


@dataclass
class Roi:
    box: Box
    mask: np.ndarray
    contour: np.ndarray | None
    edges: np.ndarray
    edge_density: float
    source: str


@dataclass
class Geometry:
    aspect: float
    extent: float
    solidity: float
    symmetry: float
    hu: list[float]
    viewpoint: str

    def vector(self) -> np.ndarray:
        return np.array([self.aspect, self.extent, self.solidity, self.symmetry, *self.hu], dtype=np.float32)
