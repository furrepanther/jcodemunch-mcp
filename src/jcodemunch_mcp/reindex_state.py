"""Per-repo reindex state container and backpressure primitives.

Provides:
- Per-repo state with __slots__ for memory efficiency.
- Reindex lifecycle: start → done / failed.
- Query: is_any_reindex_in_progress(), get_reindex_status().
- Freshness mode: relaxed / strict (for waiting on watcher reindex to complete).
- wait_for_fresh_result() — wait for a repo's reindex to finish, return fresh result.
"""

from __future__ import annotations

import threading
import time
from typing import Optional
from collections import OrderedDict
from dataclasses import dataclass


# ── NamedTuple for watcher changes ────────────────────────────────────────────

class WatcherChange(tuple):
    """A watcher change with (change_type, path, old_hash).

    change_type: str  — "added" | "modified" | "deleted"
    path: str        — absolute file path
    old_hash: str    — content hash BEFORE the change (empty for "added")
    """
    __slots__ = ()

    def __new__(cls, change_type: str, path: str, old_hash: str = ""):
        return super().__new__(cls, (change_type, path, old_hash))

    @property
    def change_type(self) -> str:
        return self[0]

    @property
    def path(self) -> str:
        return self[1]

    @property
    def old_hash(self) -> str:
        return self[2]


# ── Per-repo state ────────────────────────────────────────────────────────────

@dataclass(slots=True)
class _RepoState:
    """Lightweight per-repo reindex state with __slots__ for memory efficiency."""
    reindexing: bool = False
    reindex_finished: bool = False
    reindex_error: Optional[str] = None
    last_reindex_start: float = 0.0
    last_reindex_done: float = 0.0
    last_result: Optional[dict] = None


# ── Module-level state ───────────────────────────────────────────────────────

_states_lock = threading.RLock()
_repo_states: dict[str, _RepoState] = OrderedDict()

# Freshness mode: "relaxed" (default) or "strict"
# strict = await_freshness_if_strict() blocks callers until reindex is done
_freshness_mode: dict[str, str] = OrderedDict()
_DEFAULT_FRESHNESS = "relaxed"


# ── Core state access ─────────────────────────────────────────────────────────

def _get_state(repo: str) -> _RepoState:
    """Get or create the per-repo state container."""
    with _states_lock:
        if repo not in _repo_states:
            _repo_states[repo] = _RepoState()
        return _repo_states[repo]


# ── Reindex lifecycle ─────────────────────────────────────────────────────────

def mark_reindex_start(repo: str) -> None:
    """Mark a repo as actively reindexing."""
    with _states_lock:
        state = _get_state(repo)
        state.reindexing = True
        state.reindex_finished = False
        state.reindex_error = None
        state.last_reindex_start = time.monotonic()


def mark_reindex_done(repo: str, result: Optional[dict] = None) -> None:
    """Mark a repo's reindex as successfully completed."""
    with _states_lock:
        state = _get_state(repo)
        state.reindexing = False
        state.reindex_finished = True
        state.reindex_error = None
        state.last_reindex_done = time.monotonic()
        if result is not None:
            state.last_result = result


def mark_reindex_failed(repo: str, error: str) -> None:
    """Mark a repo's reindex as failed."""
    with _states_lock:
        state = _get_state(repo)
        state.reindexing = False
        state.reindex_finished = True
        state.reindex_error = error
        state.last_reindex_done = time.monotonic()


# ── Query functions ──────────────────────────────────────────────────────────

def get_reindex_status(repo: str) -> dict:
    """Return the reindex status dict for a repo."""
    with _states_lock:
        state = _get_state(repo)
        return {
            "reindexing": state.reindexing,
            "reindex_finished": state.reindex_finished,
            "reindex_error": state.reindex_error,
            "last_reindex_start": state.last_reindex_start,
            "last_reindex_done": state.last_reindex_done,
        }


def is_any_reindex_in_progress() -> bool:
    """Return True if any repo is currently being reindexed."""
    with _states_lock:
        return any(s.reindexing for s in _repo_states.values())


# ── Freshness mode ────────────────────────────────────────────────────────────

def set_freshness_mode(mode: str) -> None:
    """Set freshness mode for a repo: 'relaxed' (default) or 'strict'."""
    with _states_lock:
        _freshness_mode["_global"] = mode


def get_freshness_mode() -> str:
    """Get the current global freshness mode."""
    with _states_lock:
        return _freshness_mode.get("_global", _DEFAULT_FRESHNESS)


def await_freshness_if_strict(repo: str, timeout_ms: int = 500) -> bool:
    """Block the caller until the repo's reindex finishes (strict mode only).

    In relaxed mode, this returns immediately.
    In strict mode, waits up to timeout_ms for reindexing to complete.
    Returns True if the repo is fresh (not reindexing or finished), False on timeout.
    """
    if get_freshness_mode() != "strict":
        return True

    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        with _states_lock:
            state = _get_state(repo)
            if not state.reindexing:
                return True
        time.sleep(0.05)
    return False


# ── wait_for_fresh ────────────────────────────────────────────────────────────

def wait_for_fresh_result(
    repo: str,
    timeout_ms: int = 500,
) -> dict:
    """Wait for a repo's in-progress reindex to finish, then return its result.

    Args:
        repo: Repository identifier.
        timeout_ms: Maximum time to wait (default 500ms).

    Returns:
        The last result dict for the repo (from mark_reindex_done), or a
        "stale" dict if the repo is still reindexing after timeout.
    """
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        with _states_lock:
            state = _repo_states.get(repo)
            if state is None:
                # Never seen this repo — return empty stale dict
                return {
                    "status": "stale",
                    "repo": repo,
                    "reindexing": False,
                    "error": None,
                }
            if state.reindexing:
                pass  # keep waiting
            elif state.reindex_error:
                return {
                    "status": "error",
                    "repo": repo,
                    "reindexing": False,
                    "error": state.reindex_error,
                }
            else:
                # Finished successfully
                return {
                    "status": "fresh",
                    "repo": repo,
                    "reindexing": False,
                    "reindex_finished": True,
                    "result": state.last_result,
                }
        time.sleep(0.05)

    # Timeout — return stale status
    with _states_lock:
        state = _repo_states.get(repo)
        if state:
            reindexing = state.reindexing
            error = state.reindex_error
        else:
            reindexing = False
            error = None
    return {
        "status": "stale",
        "repo": repo,
        "reindexing": reindexing,
        "error": error,
    }
