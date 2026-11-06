from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT = next((d / "configs" / "investigation.yaml" for d in (ROOT, Path.cwd()) if (d / "configs" / "investigation.yaml").exists()),
               ROOT / "configs" / "investigation.yaml")


@lru_cache(maxsize=1)
def settings() -> dict:
    path = Path(os.environ.get("P5_INVESTIGATION_CONFIG", DEFAULT))
    data = yaml.safe_load(path.read_text())
    svc = data.setdefault("services", {})
    for key, env in (("photon_url", "P5_PHOTON_URL"), ("osrm_url", "P5_OSRM_URL"), ("valhalla_url", "P5_VALHALLA_URL"),
                     ("user_agent", "P5_USER_AGENT")):
        if os.environ.get(env):
            svc[key] = os.environ[env]
    return data


def section(name: str) -> dict:
    return settings()[name]


@lru_cache(maxsize=1)
def hunt_model() -> dict:
    import json
    cfg = Path(os.environ.get("P5_INVESTIGATION_CONFIG", DEFAULT))
    return json.loads((cfg.parent / section("hunt")["model"]).read_text())
