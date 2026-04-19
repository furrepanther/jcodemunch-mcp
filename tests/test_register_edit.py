"""Tests for register_edit (Feature 4)."""

import pytest
from pathlib import Path


class TestRegisterEdit:
    """Tests for register_edit function."""

    def test_clears_bm25_cache(self, tmp_path: Path):
        """After register_edit, BM25 cache is cleared."""
        from jcodemunch_mcp.tools.search_symbols import search_symbols, _result_cache_get, _result_cache_put, _result_cache_lock, _result_cache
        from tests.conftest_helpers import create_mini_index, get_index
        repo, storage_path = create_mini_index(tmp_path)

        # Do a search to populate BM25 cache
        result = search_symbols(repo=repo, query="my_func", storage_path=storage_path)
        assert result["result_count"] >= 1

        # Get the index and verify BM25 cache exists
        index = get_index(repo, storage_path)
        assert len(index._bm25_cache) > 0

        # Register edit
        from jcodemunch_mcp.tools.register_edit import register_edit
        edit_result = register_edit(
            repo=repo,
            file_paths=["test_module.py"],
            storage_path=storage_path,
        )

        # BM25 cache should be cleared
        assert edit_result["bm25_cache_cleared"] is True

    def test_records_in_journal(self, tmp_path: Path):
        """Register edit records in session journal."""
        from jcodemunch_mcp.tools.session_journal import SessionJournal
        from jcodemunch_mcp.tools.register_edit import register_edit
        from tests.conftest_helpers import create_mini_index

        # Create fresh journal for test
        journal = SessionJournal()
        journal._edits.clear()

        repo, storage_path = create_mini_index(tmp_path)

        # Register edit with journal
        register_edit(
            repo=repo,
            file_paths=["test_module.py"],
            storage_path=storage_path,
            _journal=journal,  # inject for testing
        )

        ctx = journal.get_context()
        assert len(ctx["files_edited"]) == 1
        assert ctx["files_edited"][0]["file"] == "test_module.py"

    def test_invalidates_search_result_cache(self, tmp_path: Path):
        """After register_edit, search result cache is invalidated for repo."""
        from jcodemunch_mcp.tools.search_symbols import search_symbols, _result_cache, _result_cache_lock
        from jcodemunch_mcp.tools.register_edit import register_edit
        from tests.conftest_helpers import create_mini_index

        repo, storage_path = create_mini_index(tmp_path)

        # Do a search to populate cache
        search_symbols(repo=repo, query="my_func", storage_path=storage_path)

        # Verify cache has entries
        with _result_cache_lock:
            cache_count_before = len(_result_cache)

        # Register edit
        register_edit(
            repo=repo,
            file_paths=["test_module.py"],
            storage_path=storage_path,
        )

        # Cache should be invalidated for this repo
        with _result_cache_lock:
            # Check that entries for this repo are gone
            remaining_for_repo = sum(1 for k in _result_cache if k[0] == repo)
        assert remaining_for_repo == 0

    def test_nonexistent_repo_returns_error(self):
        """Nonexistent repo returns error dict."""
        from jcodemunch_mcp.tools.register_edit import register_edit
        result = register_edit(
            repo="nonexistent/repo",
            file_paths=["test.py"],
            storage_path=None,
        )
        assert "error" in result

    def test_nonexistent_file_no_crash(self, tmp_path: Path):
        """File not in index doesn't crash, invalidated_symbols == 0."""
        from jcodemunch_mcp.tools.register_edit import register_edit
        from tests.conftest_helpers import create_mini_index

        repo, storage_path = create_mini_index(tmp_path)

        result = register_edit(
            repo=repo,
            file_paths=["nonexistent_file.py"],
            storage_path=storage_path,
        )

        # Should succeed but with 0 invalidated symbols
        assert "error" not in result
        assert result["invalidated_symbols"] == 0

    def test_returns_meta_with_timing(self, tmp_path: Path):
        """Result includes _meta with timing_ms."""
        from jcodemunch_mcp.tools.register_edit import register_edit
        from tests.conftest_helpers import create_mini_index

        repo, storage_path = create_mini_index(tmp_path)

        result = register_edit(
            repo=repo,
            file_paths=["test_module.py"],
            storage_path=storage_path,
        )

        assert "_meta" in result
        assert "timing_ms" in result["_meta"]