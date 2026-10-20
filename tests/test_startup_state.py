from __future__ import annotations

import asyncio
import json
import sys
import types
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "tree_sitter_language_pack" not in sys.modules:
    stub = types.ModuleType("tree_sitter_language_pack")

    def _get_parser(*args, **kwargs):  # pragma: no cover - import shim only
        raise RuntimeError("tree_sitter_language_pack stub should not be used in startup-state tests")

    stub.get_parser = _get_parser  # type: ignore[attr-defined]
    sys.modules["tree_sitter_language_pack"] = stub

from jcodemunch_mcp import server as server_module
from jcodemunch_mcp.startup_state import clear_startup_state, read_startup_state, write_startup_state


def test_run_stdio_server_reports_startup_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ANTIGRAVITY_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("JCODEMUNCH_STARTUP_STATE_RECORD", "jcodemunch_startup_state")

    writes: list[tuple[str, dict[str, object]]] = []

    def record_startup_state(state: str, **kwargs: object) -> dict[str, object]:
        payload = {"state": state, **kwargs}
        writes.append((state, dict(kwargs)))
        return payload

    class DummyStdioServer:
        async def __aenter__(self):
            return object(), object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_server_run(*args, **kwargs):
        await asyncio.sleep(0)

    async def fake_startup_maintenance(storage_path):
        record_startup_state(
            "ready",
            phase="startup_warmup_complete",
            detail="Background warmup complete.",
            metadata={"storage_path": storage_path or ""},
        )

    class DummyIndexStore:
        def __init__(self, base_path=None):
            self.base_path = base_path

        def close(self):
            return None

    monkeypatch.setattr(server_module, "_write_startup_state", record_startup_state)
    monkeypatch.setattr(server_module, "_run_startup_maintenance", fake_startup_maintenance)
    monkeypatch.setattr(server_module.server, "run", fake_server_run)
    monkeypatch.setattr("mcp.server.stdio.stdio_server", lambda: DummyStdioServer())
    monkeypatch.setattr("jcodemunch_mcp.storage.IndexStore", DummyIndexStore)

    asyncio.run(server_module.run_stdio_server())

    assert writes, "run_stdio_server() did not emit any startup states"
    assert any(state == "initializing" and kwargs.get("phase") == "server_boot" for state, kwargs in writes)
    assert any(state == "ready" and kwargs.get("phase") == "startup_warmup_complete" for state, kwargs in writes)


def test_startup_state_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTIGRAVITY_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("JCODEMUNCH_STARTUP_STATE_RECORD", "jcodemunch_startup_state")

    first = write_startup_state(
        "initializing",
        phase="server_boot",
        detail="Booting jcodemunch",
        metadata={"component": "stdio"},
    )
    second = write_startup_state(
        "ready",
        phase="accepting_connections",
        detail="Transport ready",
        metadata={"component": "stdio"},
    )

    stored = read_startup_state()
    db_path = tmp_path / "runtime_state.sqlite3"

    assert db_path.is_file()
    assert not (tmp_path / "jcodemunch_startup_state.json").exists()
    assert first["state"] == "initializing"
    assert second["state"] == "ready"
    assert first["started_at_utc"] == second["started_at_utc"]
    assert stored is not None
    assert stored["detail"] == "Transport ready"
    assert stored["metadata"] == {"component": "stdio"}

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM json_records WHERE name = ?",
            ("jcodemunch_startup_state",),
        ).fetchone()
    assert row is not None
    payload = json.loads(row[0])
    assert payload["state"] == "ready"

    assert clear_startup_state() is True
    assert read_startup_state() is None
