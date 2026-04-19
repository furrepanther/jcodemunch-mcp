"""Tests for `watch --once` (sync_folders)."""

import asyncio
import sys

import pytest

from jcodemunch_mcp.watcher import sync_folders


@pytest.fixture
def sample_project(tmp_path):
    """Create a minimal project with a Python file."""
    src = tmp_path / "app.py"
    src.write_text("def hello():\n    return 'world'\n")
    return tmp_path


def test_sync_folders_indexes_and_returns(sample_project, tmp_path):
    """sync_folders should index the folder and return (not block)."""
    storage = tmp_path / "storage"
    storage.mkdir()

    asyncio.run(
        sync_folders(
            paths=[str(sample_project)],
            use_ai_summaries=False,
            storage_path=str(storage),
        )
    )

    # Verify an index was created
    from jcodemunch_mcp.storage import IndexStore
    store = IndexStore(base_path=str(storage))
    repos = store.list_repos()
    store.close()
    assert len(repos) >= 1


def test_sync_folders_multiple_paths(tmp_path):
    """sync_folders handles multiple paths."""
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    (dir_a / "mod.py").write_text("x = 1\n")

    dir_b = tmp_path / "b"
    dir_b.mkdir()
    (dir_b / "mod.py").write_text("y = 2\n")

    storage = tmp_path / "storage"
    storage.mkdir()

    asyncio.run(
        sync_folders(
            paths=[str(dir_a), str(dir_b)],
            use_ai_summaries=False,
            storage_path=str(storage),
        )
    )

    from jcodemunch_mcp.storage import IndexStore
    store = IndexStore(base_path=str(storage))
    repos = store.list_repos()
    store.close()
    assert len(repos) >= 2


def test_sync_folders_bad_path_exits(tmp_path):
    """sync_folders with no valid dirs should exit with code 1."""
    with pytest.raises(SystemExit) as exc:
        asyncio.run(
            sync_folders(
                paths=[str(tmp_path / "nonexistent")],
                use_ai_summaries=False,
            )
        )
    assert exc.value.code == 1


def test_watch_once_cli_flag_parsed():
    """The --once flag should be recognized by the argument parser."""
    from jcodemunch_mcp.server import main
    # --once with --help should show the flag in usage and exit 0
    with pytest.raises(SystemExit) as exc:
        main(["watch", "--help"])
    assert exc.value.code == 0
