"""Savings gate — decide whether to emit compact or JSON.

Cheap pre-check runs before full encoding. Full check compares actual
byte sizes. Threshold is configurable via env/config; default 15%.
"""

from __future__ import annotations

import json
import os

DEFAULT_THRESHOLD = 0.15


def threshold() -> float:
    raw = os.environ.get("JCODEMUNCH_ENCODING_THRESHOLD")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return DEFAULT_THRESHOLD


def json_size(response: dict) -> int:
    return len(json.dumps(response, separators=(",", ":")))


def savings_ratio(json_bytes: int, compact_bytes: int) -> float:
    if json_bytes <= 0:
        return 0.0
    return max(0.0, (json_bytes - compact_bytes) / json_bytes)


def passes(json_bytes: int, compact_bytes: int) -> bool:
    return savings_ratio(json_bytes, compact_bytes) >= threshold()
