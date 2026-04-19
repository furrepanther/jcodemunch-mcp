"""Compact encoder for get_signal_chains."""

from .. import schema_driven as sd

TOOLS = ("get_signal_chains",)
ENCODING_ID = "sc1"

_TABLES = [
    sd.TableSpec(
        key="chains",
        tag="c",
        cols=["gateway", "gateway_kind", "leaves", "depth", "symbol_path"],
        intern=["gateway"],
        types={"depth": "int"},
    ),
]
_SCALARS = (
    "repo", "gateway_count", "chain_count", "orphan_symbols", "orphan_symbol_pct",
)
_META = (
    "timing_ms", "max_depth", "include_tests",
    "symbols_on_chains", "total_functions_methods",
)
_JSON = ("kind_summary",)


def encode(tool: str, response: dict) -> tuple[str, str]:
    return sd.encode(
        tool, response, ENCODING_ID, _TABLES, _SCALARS,
        meta_keys=_META, json_blobs=_JSON,
    )


def decode(payload: str) -> dict:
    return sd.decode(
        payload, _TABLES, _SCALARS, meta_keys=_META, json_blobs=_JSON,
    )
