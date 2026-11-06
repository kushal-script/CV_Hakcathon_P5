from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import asdict, dataclass

ROLES = ("station", "division_admin")


@dataclass(frozen=True)
class Identity:
    user_id: int
    username: str
    role: str
    station_id: str | None
    division_id: str
    display_name: str

    @property
    def is_admin(self) -> bool:
        return self.role == "division_admin"

    def to_dict(self) -> dict:
        return asdict(self)


def secret() -> bytes | None:
    value = os.environ.get("P5_INTERNAL_SECRET", "")
    return value.encode() if len(value) >= 32 else None


def sign(payload: dict, key: bytes) -> tuple[str, str]:
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=").decode()
    return encoded, hmac.new(key, encoded.encode(), hashlib.sha256).hexdigest()


def verify(encoded: str | None, signature: str | None, key: bytes | None, max_skew_s: int = 120) -> Identity | None:
    if not key or not encoded or not signature or len(encoded) > 4096:
        return None
    expected = hmac.new(key, encoded.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature.lower()):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    except ValueError:
        return None
    if abs(time.time() - float(data.get("ts", 0))) > max_skew_s:
        return None
    role, station, division = data.get("r"), data.get("s"), data.get("d")
    if role not in ROLES or not division or (role == "station") != bool(station):
        return None
    return Identity(int(data["uid"]), str(data["u"]), role, station, str(division), str(data.get("n") or data["u"]))
