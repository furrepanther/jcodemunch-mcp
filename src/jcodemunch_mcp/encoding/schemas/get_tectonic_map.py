"""Compact encoder for get_tectonic_map."""

from .. import schema_driven as sd

TOOLS = ("get_tectonic_map",)
ENCODING_ID = "tm1"

_TABLES = [
    sd.TableSpec(
        key="plates",
        tag="p",
        cols=["label", "file_count", "representative"],
        intern=["representative"],
        types={"file_count": "int"},
    ),
    sd.TableSpec(
        key="drifter_summary",
        tag="z",
        cols=["file", "score", "nearest_plate"],
        intern=["file"],
        types={"score": "float"},
    ),
    sd.TableSpec(
        key="isolated_files",
        tag="q",
        cols=["file"],
        intern=["file"],
    ),
]
_SCALARS = ("repo", "plate_count", "file_count")
_META = ("timing_ms", "methodology")
_JSON = ("signals_used", "_meta")


def encode(tool: str, response: dict) -> tuple[str, str]:
    return sd.encode(
        tool, response, ENCODING_ID, _TABLES, _SCALARS,
        meta_keys=_META, json_blobs=_JSON,
    )


def decode(payload: str) -> dict:
    return sd.decode(
        payload, _TABLES, _SCALARS, meta_keys=_META, json_blobs=_JSON,
    )
