"""Tests for the check_freshness tool."""

from unittest.mock import patch

import pytest

from jcodemunch_mcp.parser import Symbol
from jcodemunch_mcp.storage import IndexStore
from jcodemunch_mcp.tools.check_freshness import check_freshness


_SHA_A = "abc123def456abc123def456abc123def456abc1"
_SHA_B = "999000111222333444555666777888999000111a"


def _seed_local_repo(tmp_path, git_head=_SHA_A, source_root=None):
    """Seed a minimal local repo index."""
    store = IndexStore(base_path=str(tmp_path))
    root = source_root or str(tmp_path / "myrepo")
    symbol = Symbol(
        id="src-main-py::run#function",
        file="src/main.py",
        name="run",
        qualified_name="run",
        kind="function",
        language="python",
        signature="def run():",
        byte_offset=0,
        byte_length=45,
    )
    store.save_index(
        owner="local",
        name="myrepo",
        source_files=["src/main.py"],
        symbols=[symbol],
        raw_files={"src/main.py": "def run(): pass\n"},
        languages={"python": 1},
        file_languages={"src/main.py": "python"},
        file_summaries={"src/main.py": "Entry point."},
        git_head=git_head,
        source_root=root,
    )


def _seed_github_repo(tmp_path):
    """Seed a GitHub-indexed repo (no source_root)."""
    store = IndexStore(base_path=str(tmp_path))
    symbol = Symbol(
        id="src-main-py::run#function",
        file="src/main.py",
        name="run",
        qualified_name="run",
        kind="function",
        language="python",
        signature="def run():",
        byte_offset=0,
        byte_length=45,
    )
    store.save_index(
        owner="github",
        name="upstream",
        source_files=["src/main.py"],
        symbols=[symbol],
        raw_files={"src/main.py": "def run(): pass\n"},
        languages={"python": 1},
        file_languages={"src/main.py": "python"},
        file_summaries={"src/main.py": "Entry point."},
        git_head=_SHA_A,
        source_root="",
    )


def test_fresh_repo(tmp_path):
    """Same SHA at index time and current HEAD → fresh=True."""
    _seed_local_repo(tmp_path, git_head=_SHA_A)
    with patch("jcodemunch_mcp.tools.check_freshness._get_git_head", return_value=_SHA_A):
        result = check_freshness("local/myrepo", storage_path=str(tmp_path))
    assert result["fresh"] is True
    assert result["is_local"] is True
    assert result["indexed_sha"] == _SHA_A
    assert result["current_sha"] == _SHA_A
    assert result["commits_behind"] is None


def test_stale_repo(tmp_path):
    """Different SHA → fresh=False with commits_behind count."""
    _seed_local_repo(tmp_path, git_head=_SHA_A)
    with (
        patch("jcodemunch_mcp.tools.check_freshness._get_git_head", return_value=_SHA_B),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "3\n"
        result = check_freshness("local/myrepo", storage_path=str(tmp_path))
    assert result["fresh"] is False
    assert result["commits_behind"] == 3
    assert result["indexed_sha"] == _SHA_A
    assert result["current_sha"] == _SHA_B


def test_github_repo(tmp_path):
    """GitHub-indexed repo (no source_root) → is_local=False, fresh=None."""
    _seed_github_repo(tmp_path)
    result = check_freshness("github/upstream", storage_path=str(tmp_path))
    assert result["is_local"] is False
    assert result["fresh"] is None
    assert "message" in result


def test_no_stored_sha(tmp_path):
    """Index with no stored SHA → fresh=None with re-run message."""
    _seed_local_repo(tmp_path, git_head="")
    result = check_freshness("local/myrepo", storage_path=str(tmp_path))
    assert result["fresh"] is None
    assert result["is_local"] is True
    assert "Re-run index_folder" in result["message"]


def test_git_unavailable(tmp_path):
    """_get_git_head returns None → fresh=None with helpful message."""
    _seed_local_repo(tmp_path, git_head=_SHA_A)
    with patch("jcodemunch_mcp.tools.check_freshness._get_git_head", return_value=None):
        result = check_freshness("local/myrepo", storage_path=str(tmp_path))
    assert result["fresh"] is None
    assert result["is_local"] is True
    assert "git" in result["message"].lower()


def test_commits_behind_unavailable(tmp_path):
    """Stale but subprocess raises → fresh=False, commits_behind=None."""
    _seed_local_repo(tmp_path, git_head=_SHA_A)
    with (
        patch("jcodemunch_mcp.tools.check_freshness._get_git_head", return_value=_SHA_B),
        patch("subprocess.run", side_effect=OSError("git not found")),
    ):
        result = check_freshness("local/myrepo", storage_path=str(tmp_path))
    assert result["fresh"] is False
    assert result["commits_behind"] is None


def test_repo_not_indexed(tmp_path):
    """Unknown repo name → error key in response."""
    result = check_freshness("nonexistent/repo", storage_path=str(tmp_path))
    assert "error" in result
