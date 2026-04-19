"""Compact encoder for get_dependency_cycles."""

from .. import schema_driven as sd

TOOLS = ("get_dependency_cycles",)
ENCODING_ID = "dc1"

_TABLES = [
    sd.TableSpec(
        key="cycles",
        tag="y",
        cols=["length", "files"],
        types={"length": "int"},
    ),
]
_SCALARS = ("repo", "cycle_count")
_META = ("timing_ms",)


def encode(tool: str, response: dict) -> tuple[str, str]:
    return sd.encode(tool, response, ENCODING_ID, _TABLES, _SCALARS, meta_keys=_META)


def decode(payload: str) -> dict:
    return sd.decode(payload, _TABLES, _SCALARS, meta_keys=_META)
