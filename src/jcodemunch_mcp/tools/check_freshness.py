"""Check whether a local index is still fresh against the current git HEAD."""

import subprocess
import time
from pathlib import Path
from typing import Optional

from ..storage import IndexStore
from ..storage.index_store import _get_git_head
from ._utils import resolve_repo


def check_freshness(
    repo: str,
    storage_path: Optional[str] = None,
) -> dict:
    """Compare the git HEAD SHA at index time against the current HEAD.

    Works only for repos indexed with index_folder (local). For repos indexed
    with index_repo (GitHub), returns is_local=False — use index_repo's
    incremental mode for GitHub freshness checks.

    Returns:
        {
            "fresh": bool,
            "indexed_sha": str,       # SHA stored at index time
            "current_sha": str,       # current HEAD SHA
            "commits_behind": int,    # commits since index (None if unavailable)
            "is_local": bool,         # False for GitHub-indexed repos
            "source_root": str,       # path checked
        }
    """
    start = time.perf_counter()

    try:
        owner, name = resolve_repo(repo, storage_path)
    except ValueError as e:
        return {"error": str(e)}

    store = IndexStore(base_path=storage_path)
    index = store.load_index(owner, name)
    if not index:
        return {"error": f"Repository not indexed: {owner}/{name}"}

    # GitHub-indexed repos store a tree SHA, not a local commit SHA
    if not index.source_root:
        return {
            "fresh": None,
            "is_local": False,
            "message": (
                "Freshness check requires a locally indexed repo (index_folder). "
                "For GitHub repos, call index_repo — it compares tree SHAs automatically."
            ),
            "indexed_sha": index.git_head or "",
        }

    source_root = Path(index.source_root)
    indexed_sha = index.git_head or ""

    current_sha = _get_git_head(source_root) or ""

    if not indexed_sha:
        return {
            "fresh": None,
            "is_local": True,
            "source_root": str(source_root),
            "message": (
                "No SHA stored at index time — repo may not be a git repo, "
                "or was indexed before git tracking was added. Re-run index_folder."
            ),
        }

    if not current_sha:
        return {
            "fresh": None,
            "is_local": True,
            "source_root": str(source_root),
            "indexed_sha": indexed_sha,
            "message": "Could not read current git HEAD. Is git installed and is this a git repo?",
        }

    fresh = indexed_sha == current_sha

    commits_behind = None
    if not fresh:
        try:
            result = subprocess.run(
                ["git", "rev-list", "--count", f"{indexed_sha}..{current_sha}"],
                cwd=str(source_root),
                capture_output=True, text=True, timeout=5,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                commits_behind = int(result.stdout.strip())
        except Exception:
            pass  # commits_behind stays None — not fatal

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "fresh": fresh,
        "is_local": True,
        "source_root": str(source_root),
        "indexed_sha": indexed_sha,
        "current_sha": current_sha,
        "commits_behind": commits_behind,
        "_meta": {"timing_ms": round(elapsed, 1)},
    }
