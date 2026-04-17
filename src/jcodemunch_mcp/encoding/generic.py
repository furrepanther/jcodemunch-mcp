"""Generic fallback encoder — shape-sniffer for tools without a custom schema.

Strategy:
    * Walk the response dict once; separate scalar top-level fields from
      array-of-homogeneous-dicts tables.
    * For each table, intern repeated string values (paths, symbol names)
      that benefit from legend substitution.
    * Emit MUNCH with a tag per table: 't' + letter per table index.

Round-trip is best-effort. Types are preserved only for columns declared
at encode time in an embedded type line; columns not declared decode as
strings. Use custom per-tool encoders when strict round-trip matters.
"""

from __future__ import annotations

from typing import Any

from .format import (
    Legends,
    assemble,
    encode_scalar,
    parse_header,
    parse_scalars,
    read_table,
    split_sections,
    write_header,
    write_scalars,
    write_table,
)

ENCODING_ID = "gen1"

_TABLE_TAGS = "tuvwxyz"
_MIN_TABLE_ROWS = 2


def _is_homogeneous_dict_array(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < _MIN_TABLE_ROWS:
        return False
    if not all(isinstance(x, dict) for x in value):
        return False
    first_keys = tuple(value[0].keys())
    return all(tuple(x.keys()) == first_keys for x in value)


def _collect_prefixes(samples: list[str], min_len: int = 6) -> list[str]:
    """Find common path-like prefixes from a list of strings."""
    buckets: dict[str, int] = {}
    for s in samples:
        if not isinstance(s, str) or len(s) < min_len:
            continue
        # progressive prefixes split on / or \
        pos = 0
        while True:
            nxt = min(
                (i for i in (s.find("/", pos + 1), s.find("\\", pos + 1)) if i > 0),
                default=-1,
            )
            if nxt < 0 or nxt >= len(s) - 1:
                break
            prefix = s[: nxt + 1]
            if len(prefix) >= min_len:
                buckets[prefix] = buckets.get(prefix, 0) + 1
            pos = nxt
    return sorted(buckets, key=lambda p: -buckets[p] * len(p))


def encode(tool_name: str, response: dict) -> tuple[str, str]:
    scalars: dict[str, Any] = {}
    tables: list[tuple[str, list[str], list[dict]]] = []
    tag_iter = iter(_TABLE_TAGS)

    for key, value in response.items():
        if _is_homogeneous_dict_array(value):
            try:
                tag = next(tag_iter)
            except StopIteration:
                scalars[key] = "[omitted: table-tag exhaustion]"
                continue
            cols = list(value[0].keys())
            tables.append((tag, cols, value))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            scalars[key] = value
        else:
            # nested dicts or mixed arrays fall back to JSON literal
            import json as _json
            scalars[key] = _json.dumps(value, separators=(",", ":"))

    # Build path legend from string-valued table columns
    path_legend = Legends(prefix="@")
    for _, _, rows in tables:
        for row in rows:
            for v in row.values():
                if isinstance(v, str):
                    path_legend.observe(v)
    for prefix in _collect_prefixes(list(path_legend._counts.keys())):
        path_legend._counts.setdefault(prefix, 2)
        path_legend._order.insert(0, prefix)
    # De-duplicate while preserving first occurrence
    seen: set[str] = set()
    path_legend._order = [x for x in path_legend._order if not (x in seen or seen.add(x))]
    path_legend.finalize()

    # Scalar section also encodes the table column schema so decode can reverse
    scalars["__tables"] = ",".join(f"{tag}:{'|'.join(cols)}" for tag, cols, _ in tables)

    sections: list[str] = []
    legend_text = path_legend.write()
    if legend_text:
        sections.append(legend_text)
    sections.append(write_scalars(scalars))
    for tag, cols, rows in tables:
        data_rows = [
            [path_legend.encode_prefix(v) if isinstance(v, str) else v for v in (r[c] for c in cols)]
            for r in rows
        ]
        sections.append(write_table(tag, data_rows))

    header = write_header(tool_name, ENCODING_ID)
    payload = assemble(header, *sections)
    return payload, ENCODING_ID


def decode(payload: str) -> dict:
    head, blocks = split_sections(payload)
    parse_header(head)  # validate
    legend = Legends(prefix="@")
    scalar_text = ""
    table_blocks: list[str] = []
    for b in blocks:
        if b.startswith("@"):
            legend = Legends.read(b, prefix="@")
        elif "=" in b.splitlines()[0] and not b.splitlines()[0].startswith(tuple(_TABLE_TAGS) + ("#",)):
            scalar_text = b
        else:
            table_blocks.append(b)

    scalars = parse_scalars(scalar_text) if scalar_text else {}
    table_spec = scalars.pop("__tables", "")
    schemas: dict[str, list[str]] = {}
    for part in [p for p in table_spec.split(",") if p]:
        tag, _, cols = part.partition(":")
        schemas[tag] = cols.split("|")

    result: dict[str, Any] = dict(scalars)
    for block in table_blocks:
        for tag, cols in schemas.items():
            rows = read_table(block, tag)
            if not rows:
                continue
            # find the outer key this table belonged to — not recoverable
            # from the tag alone. Generic decode emits under "table_<tag>".
            out_key = f"table_{tag}"
            result[out_key] = [
                {c: legend.decode_prefix(v) for c, v in zip(cols, r)}
                for r in rows
            ]
    return result
