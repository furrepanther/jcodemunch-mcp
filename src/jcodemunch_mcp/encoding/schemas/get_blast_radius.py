"""Compact encoder for get_blast_radius."""

from .. import schema_driven as sd

TOOLS = ("get_blast_radius",)
ENCODING_ID = "br1"

_TABLES = [
    sd.TableSpec(
        key="affected_symbols",
        tag="a",
        cols=["id", "name", "kind", "file", "line", "depth"],
        intern=["file", "id"],
        types={"line": "int", "depth": "int"},
    ),
    sd.TableSpec(
        key="importer_files",
        tag="f",
        cols=["file", "depth"],
        intern=["file"],
        types={"depth": "int"},
    ),
]
_SCALARS = (
    "repo", "symbol", "direction", "depth",
    "importer_file_count", "affected_symbol_count",
)
_NESTED = {"symbol_info": ["id", "name", "kind", "file", "line"]}
_META = ("timing_ms", "truncated", "cross_repo", "methodology")
_JSON = ("files_by_depth",)


def encode(tool: str, response: dict) -> tuple[str, str]:
    return sd.encode(
        tool, response, ENCODING_ID, _TABLES, _SCALARS,
        nested_dicts=_NESTED, meta_keys=_META, json_blobs=_JSON,
    )


def decode(payload: str) -> dict:
    return sd.decode(
        payload, _TABLES, _SCALARS,
        nested_dicts=_NESTED, meta_keys=_META, json_blobs=_JSON,
    )
