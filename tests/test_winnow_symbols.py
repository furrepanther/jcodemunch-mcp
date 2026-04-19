"""Tests for winnow_symbols — multi-axis constraint-chain query."""

from __future__ import annotations

import pytest

from jcodemunch_mcp.tools.winnow_symbols import winnow_symbols
from jcodemunch_mcp.tools.index_folder import index_folder


def _build_repo(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    store = tmp_path / "store"
    store.mkdir()
    (src / "a.py").write_text(
        "def simple():\n"
        "    return 1\n"
        "\n"
        "def validate_input(x):\n"
        "    if x > 0:\n"
        "        if x < 10:\n"
        "            return True\n"
        "        else:\n"
        "            return False\n"
        "    elif x < 0:\n"
        "        return False\n"
        "    return None\n"
        "\n"
        "def run_query(sql):\n"
        "    db.Exec(sql)\n"
        "    return None\n"
    )
    (src / "b.py").write_text(
        "from functools import wraps\n"
        "\n"
        "def deprecated(fn):\n"
        "    @wraps(fn)\n"
        "    def _w(*a, **k):\n"
        "        return fn(*a, **k)\n"
        "    return _w\n"
        "\n"
        "@deprecated\n"
        "def old_api():\n"
        "    return 42\n"
    )
    r = index_folder(str(src), use_ai_summaries=False, storage_path=str(store))
    assert r["success"] is True
    return r["repo"], str(store)


class TestWinnowValidation:
    def test_rejects_bad_axis(self, tmp_path):
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(
            repo=repo,
            criteria=[{"axis": "bogus", "op": "eq", "value": 1}],
            storage_path=store,
        )
        assert "error" in out
        assert "bogus" in out["error"]

    def test_rejects_bad_op(self, tmp_path):
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(
            repo=repo,
            criteria=[{"axis": "kind", "op": "!=", "value": "function"}],
            storage_path=store,
        )
        assert "error" in out

    def test_rejects_missing_value(self, tmp_path):
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(
            repo=repo,
            criteria=[{"axis": "kind", "op": "eq"}],
            storage_path=store,
        )
        assert "error" in out

    def test_rejects_bad_rank_by(self, tmp_path):
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(repo=repo, criteria=[], rank_by="zzz", storage_path=store)
        assert "error" in out

    def test_unknown_repo_errors(self):
        out = winnow_symbols(repo="nope/nope", criteria=[])
        assert "error" in out


class TestWinnowFiltering:
    def test_kind_filter_functions_only(self, tmp_path):
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(
            repo=repo,
            criteria=[{"axis": "kind", "op": "eq", "value": "function"}],
            storage_path=store,
        )
        assert "results" in out
        assert out["matched"] >= 1
        assert all(r["kind"] == "function" for r in out["results"])

    def test_name_regex(self, tmp_path):
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(
            repo=repo,
            criteria=[{"axis": "name", "op": "matches", "value": "^validate_"}],
            storage_path=store,
        )
        names = [r["name"] for r in out["results"]]
        assert "validate_input" in names
        assert "simple" not in names

    def test_complexity_threshold(self, tmp_path):
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(
            repo=repo,
            criteria=[
                {"axis": "kind", "op": "eq", "value": "function"},
                {"axis": "complexity", "op": ">", "value": 2},
            ],
            storage_path=store,
        )
        for r in out["results"]:
            assert r["cyclomatic"] > 2

    def test_file_glob(self, tmp_path):
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(
            repo=repo,
            criteria=[{"axis": "file", "op": "matches", "value": "*b.py"}],
            storage_path=store,
        )
        for r in out["results"]:
            assert r["file"].replace("\\", "/").endswith("b.py")

    def test_calls_contains(self, tmp_path):
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(
            repo=repo,
            criteria=[{"axis": "calls", "op": "contains", "value": "Exec"}],
            storage_path=store,
        )
        names = [r["name"] for r in out["results"]]
        assert "run_query" in names

    def test_and_semantics(self, tmp_path):
        """kind=function AND name matches ^validate → only validate_input."""
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(
            repo=repo,
            criteria=[
                {"axis": "kind", "op": "eq", "value": "function"},
                {"axis": "name", "op": "matches", "value": "^validate"},
            ],
            storage_path=store,
        )
        names = [r["name"] for r in out["results"]]
        assert names == ["validate_input"]


class TestWinnowRanking:
    def test_rank_by_name_asc(self, tmp_path):
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(
            repo=repo,
            criteria=[{"axis": "kind", "op": "eq", "value": "function"}],
            rank_by="name",
            order="asc",
            storage_path=store,
        )
        names = [r["name"] for r in out["results"]]
        assert names == sorted(names)

    def test_rank_by_complexity_desc(self, tmp_path):
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(
            repo=repo,
            criteria=[{"axis": "kind", "op": "eq", "value": "function"}],
            rank_by="complexity",
            order="desc",
            storage_path=store,
        )
        scores = [r["cyclomatic"] for r in out["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_max_results_respected(self, tmp_path):
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(
            repo=repo,
            criteria=[{"axis": "kind", "op": "eq", "value": "function"}],
            max_results=1,
            storage_path=store,
        )
        assert len(out["results"]) <= 1


class TestWinnowMeta:
    def test_meta_reports_scan_count_and_axes(self, tmp_path):
        repo, store = _build_repo(tmp_path)
        out = winnow_symbols(
            repo=repo,
            criteria=[{"axis": "kind", "op": "eq", "value": "function"}],
            storage_path=store,
        )
        assert out["total_scanned"] >= out["matched"]
        assert "supported_axes" in out["_meta"]
        assert "kind" in out["_meta"]["supported_axes"]
