"""Tests for the _pulse.json per-call activity signal."""

import json
import os
from pathlib import Path

import pytest


def test_write_pulse_creates_file(tmp_path, monkeypatch):
    """write_pulse writes _pulse.json when JCODEMUNCH_EVENT_LOG=1."""
    monkeypatch.setenv("JCODEMUNCH_EVENT_LOG", "1")
    from jcodemunch_mcp.storage.token_tracker import write_pulse

    write_pulse("search_symbols", tokens_saved=1234, base_path=str(tmp_path))

    pulse = tmp_path / "_pulse.json"
    assert pulse.exists()
    data = json.loads(pulse.read_text())
    assert data["tool"] == "search_symbols"
    assert data["tokens_saved"] == 1234
    assert "last_call_at" in data
    assert "calls_since_boot" in data
    assert "session_tokens_saved" in data


def test_write_pulse_noop_without_env(tmp_path, monkeypatch):
    """write_pulse is a no-op when JCODEMUNCH_EVENT_LOG is not set."""
    monkeypatch.delenv("JCODEMUNCH_EVENT_LOG", raising=False)
    from jcodemunch_mcp.storage.token_tracker import write_pulse

    write_pulse("search_symbols", tokens_saved=100, base_path=str(tmp_path))

    pulse = tmp_path / "_pulse.json"
    assert not pulse.exists()


def test_write_pulse_atomic_overwrite(tmp_path, monkeypatch):
    """Successive calls overwrite the pulse file atomically."""
    monkeypatch.setenv("JCODEMUNCH_EVENT_LOG", "1")
    from jcodemunch_mcp.storage.token_tracker import write_pulse

    write_pulse("search_symbols", tokens_saved=100, base_path=str(tmp_path))
    write_pulse("get_file_outline", tokens_saved=200, base_path=str(tmp_path))

    pulse = tmp_path / "_pulse.json"
    data = json.loads(pulse.read_text())
    assert data["tool"] == "get_file_outline"
    assert data["tokens_saved"] == 200
