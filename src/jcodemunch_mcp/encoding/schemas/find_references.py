"""Compact encoder for find_references."""

from .. import schema_driven as sd

TOOLS = ("find_references",)
ENCODING_ID = "fr1"

_TABLES = [
    sd.TableSpec(
        key="references",
        tag="r",
        cols=["file", "line", "column", "specifier", "kind"],
        intern=["file", "specifier"],
        types={"line": "int", "column": "int"},
    ),
    sd.TableSpec(
        key="results",
        tag="b",
        cols=["identifier", "reference_count"],
        types={"reference_count": "int"},
    ),
]
_SCALARS = ("repo", "identifier", "reference_count", "note")
_META = ("timing_ms", "truncated", "tokens_saved", "total_tokens_saved")


def encode(tool: str, response: dict) -> tuple[str, str]:
    return sd.encode(tool, response, ENCODING_ID, _TABLES, _SCALARS, meta_keys=_META)


def decode(payload: str) -> dict:
    return sd.decode(payload, _TABLES, _SCALARS, meta_keys=_META)
