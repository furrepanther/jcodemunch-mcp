"""Tests for check_references composite tool."""
import pytest
from pathlib import Path
from jcodemunch_mcp.tools.index_folder import index_folder


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestCheckReferences:
    def test_finds_import_reference(self, tmp_path):
        """Detects a symbol referenced via import."""
        src = tmp_path / "src"
        src.mkdir()
        _write(src / "utils.js", "export function helper() {}")
        _write(src / "app.js", "import { helper } from './utils';")

        idx = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=str(tmp_path / "idx"))
        repo = idx["repo"]

        from jcodemunch_mcp.tools.check_references import check_references
        result = check_references(repo=repo, identifier="helper", storage_path=str(tmp_path / "idx"))
        assert result["is_referenced"] is True
        assert result["import_count"] >= 1
        assert "import_references" in result

    def test_finds_content_reference(self, tmp_path):
        """Detects a symbol used in file content (not just imports)."""
        src = tmp_path / "src"
        src.mkdir()
        _write(src / "utils.py", "def helper(): pass")
        _write(src / "app.py", "from .utils import helper\nresult = helper()")

        idx = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=str(tmp_path / "idx"))
        repo = idx["repo"]

        from jcodemunch_mcp.tools.check_references import check_references
        result = check_references(repo=repo, identifier="helper", storage_path=str(tmp_path / "idx"))
        assert result["is_referenced"] is True
        assert result["content_count"] >= 1

    def test_no_references(self, tmp_path):
        """Symbol with no references anywhere."""
        src = tmp_path / "src"
        src.mkdir()
        _write(src / "utils.py", "def helper(): pass")
        _write(src / "app.py", "print('hello')")

        idx = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=str(tmp_path / "idx"))
        repo = idx["repo"]

        from jcodemunch_mcp.tools.check_references import check_references
        result = check_references(repo=repo, identifier="helper", storage_path=str(tmp_path / "idx"))
        assert result["is_referenced"] is False
        assert result["import_count"] == 0
        assert result["content_count"] == 0

    def test_search_content_false(self, tmp_path):
        """search_content=False skips content search, only checks imports."""
        src = tmp_path / "src"
        src.mkdir()
        _write(src / "utils.py", "def helper(): pass")
        _write(src / "app.py", "# helper is used here in a comment\nprint('hello')")

        idx = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=str(tmp_path / "idx"))
        repo = idx["repo"]

        from jcodemunch_mcp.tools.check_references import check_references
        result = check_references(
            repo=repo, identifier="helper",
            search_content=False, storage_path=str(tmp_path / "idx"),
        )
        # Import check finds nothing, content search skipped
        assert result["is_referenced"] is False
        assert "content_references" not in result

    def test_batch_identifiers(self, tmp_path):
        """check_references with identifiers list returns grouped results."""
        src = tmp_path / "src"
        src.mkdir()
        _write(src / "utils.js", "export function helper() {}\nexport function unused() {}")
        _write(src / "app.js", "import { helper } from './utils';")

        idx = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=str(tmp_path / "idx"))
        repo = idx["repo"]

        from jcodemunch_mcp.tools.check_references import check_references
        result = check_references(
            repo=repo, identifiers=["helper", "unused"],
            storage_path=str(tmp_path / "idx"),
        )
        assert "results" in result
        assert len(result["results"]) == 2
        by_id = {r["identifier"]: r for r in result["results"]}
        assert by_id["helper"]["is_referenced"] is True
        assert by_id["unused"]["is_referenced"] is False

    def test_not_indexed(self, tmp_path):
        """Unindexed repo returns error."""
        from jcodemunch_mcp.tools.check_references import check_references
        result = check_references(repo="nonexistent/repo", identifier="foo", storage_path=str(tmp_path / "idx"))
        assert "error" in result
