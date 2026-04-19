"""Unit tests for MUNCH format primitives."""

from jcodemunch_mcp.encoding.format import (
    Legends,
    assemble,
    parse_header,
    parse_scalars,
    read_table,
    split_sections,
    write_header,
    write_scalars,
    write_table,
)


def test_header_round_trip():
    h = write_header("get_call_hierarchy", "ch1")
    meta = parse_header(h)
    assert meta == {"tool": "get_call_hierarchy", "enc": "ch1"}


def test_scalars_quoting():
    pairs = {"repo": "foo", "note": "hello world", "n": 42, "flag": True, "none": None}
    line = write_scalars(pairs)
    parsed = parse_scalars(line)
    assert parsed["repo"] == "foo"
    assert parsed["note"] == "hello world"
    assert parsed["n"] == "42"
    assert parsed["flag"] == "T"
    assert parsed["none"] == ""


def test_scalars_embedded_quotes():
    pairs = {"msg": 'she said "hi"'}
    line = write_scalars(pairs)
    parsed = parse_scalars(line)
    assert parsed["msg"] == 'she said "hi"'


def test_table_round_trip():
    rows = [["a.py", 12, "call"], ["b.py", 44, "ref"]]
    text = write_table("c", rows)
    parsed = read_table(text, "c")
    assert parsed == [["a.py", "12", "call"], ["b.py", "44", "ref"]]


def test_legends_dedup_and_encode():
    leg = Legends(prefix="@")
    for v in ["src/foo/", "src/foo/", "src/foo/", "src/bar/", "x"]:
        leg.observe(v)
    leg.finalize(min_uses=2, min_chars_saved=1)
    assert leg.encode_prefix("src/foo/thing.py").startswith("@")
    out = leg.write()
    leg2 = Legends.read(out, prefix="@")
    encoded = leg.encode_prefix("src/foo/a.py")
    decoded = leg2.decode_prefix(encoded)
    assert decoded == "src/foo/a.py"


def test_assemble_and_split():
    header = write_header("demo", "gen1")
    payload = assemble(header, "@1=x/", "k=v", "c,row1")
    head, blocks = split_sections(payload)
    assert head == header
    assert blocks == ["@1=x/", "k=v", "c,row1"]
