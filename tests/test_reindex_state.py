"""Tests for reindex_state module."""

import pytest

from jcodemunch_mcp.reindex_state import _get_state, _freshness_mode, _repo_states


@pytest.fixture(autouse=True)
def reset_state():
    """Reset module-level state before each test."""
    _repo_states.clear()
    _freshness_mode.clear()
    yield
    _repo_states.clear()
    _freshness_mode.clear()


class TestRepoStateCreation:
    def test_get_state_creates_new_state(self):
        state = _get_state("test/repo")
        assert state is not None

    def test_get_state_returns_same_instance(self):
        state1 = _get_state("test/repo")
        state2 = _get_state("test/repo")
        assert state1 is state2

    def test_get_state_different_repos_are_independent(self):
        state1 = _get_state("repo/a")
        state2 = _get_state("repo/b")
        assert state1 is not state2
