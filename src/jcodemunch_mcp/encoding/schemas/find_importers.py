"""Compact encoder for find_importers."""

from .. import schema_driven as sd

TOOLS = ("find_importers",)
ENCODING_ID = "fi1"

_TABLES = [
    sd.TableSpec(
        key="importers",
        tag="i",
        cols=["file", "specifier", "line", "column"],
        intern=["file", "specifier"],
        types={"line": "int", "column": "int"},
    ),
    sd.TableSpec(
        key="results",
        tag="b",
        cols=["file", "importer_count"],
        intern=["file"],
        types={"importer_count": "int"},
    ),
]
_SCALARS = ("repo", "file", "importer_count", "note")
_META = ("timing_ms", "truncated", "tokens_saved", "total_tokens_saved")


def encode(tool: str, response: dict) -> tuple[str, str]:
    return sd.encode(tool, response, ENCODING_ID, _TABLES, _SCALARS, meta_keys=_META)


def decode(payload: str) -> dict:
    return sd.decode(payload, _TABLES, _SCALARS, meta_keys=_META)
