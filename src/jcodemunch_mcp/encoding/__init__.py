"""Compact response encoding — the MUNCH format.

Dispatcher entry point. Given a tool name and a response dict, returns
either the original dict (JSON passthrough) or a MUNCH payload string
together with the encoding id and a savings measurement.

Usage from server.py:

    from .encoding import encode_response

    payload, meta = encode_response(tool_name, result, requested_format)
    if meta["encoding"] == "json":
        text = json.dumps(payload, separators=(",", ":"))
    else:
        text = payload

`requested_format` is one of: "auto" (default), "compact", "json".
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from . import gate, generic
from .schemas import registry

logger = logging.getLogger(__name__)

_FORMATS = ("auto", "compact", "json")


def default_format() -> str:
    raw = os.environ.get("JCODEMUNCH_DEFAULT_FORMAT", "auto").lower()
    return raw if raw in _FORMATS else "auto"


def encode_response(
    tool_name: str,
    response: Any,
    requested_format: str | None = None,
) -> tuple[Any, dict]:
    """Return (payload, meta).

    payload is either the original dict (for json path) or a MUNCH string.
    meta is a dict with keys: encoding, json_bytes, encoded_bytes,
    encoding_tokens_saved.
    """
    fmt = (requested_format or default_format()).lower()
    if fmt not in _FORMATS:
        fmt = "auto"

    if fmt == "json" or not isinstance(response, dict):
        return response, {"encoding": "json"}

    json_bytes = gate.json_size(response)

    try:
        encoder = registry.for_tool(tool_name)
        if encoder is not None:
            payload, enc_id = encoder.encode(tool_name, response)
        else:
            payload, enc_id = generic.encode(tool_name, response)
    except Exception:
        logger.debug("Encoder failed for %s; falling back to JSON", tool_name, exc_info=True)
        return response, {"encoding": "json", "json_bytes": json_bytes}

    encoded_bytes = len(payload)
    if fmt == "auto" and not gate.passes(json_bytes, encoded_bytes):
        return response, {
            "encoding": "json",
            "json_bytes": json_bytes,
            "encoded_bytes": encoded_bytes,
        }

    return payload, {
        "encoding": enc_id,
        "json_bytes": json_bytes,
        "encoded_bytes": encoded_bytes,
        "encoding_tokens_saved": max(0, (json_bytes - encoded_bytes) // 4),
    }


__all__ = ["encode_response", "default_format"]
