"""Atomic startup state tracking for jcodemunch-mcp."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STARTUP_STATE_RECORD_ENV_VAR = "JCODEMUNCH_STARTUP_STATE_RECORD"
LEGACY_STARTUP_STATE_PATH_ENV_VAR = "JCODEMUNCH_STARTUP_STATE_PATH"
RUNTIME_STATE_DIR_ENV_VAR = "ANTIGRAVITY_RUNTIME_STATE_DIR"
STARTUP_STATE_FILENAME = "jcodemunch_startup_state.json"
STARTUP_STATE_RECORD_NAME = "jcodemunch_startup_state"
DEFAULT_RUNTIME_STATE_DIR = Path("/mnt/f/Production/antigravity/runtime")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _runtime_state_dir() -> Path:
    raw = os.environ.get(RUNTIME_STATE_DIR_ENV_VAR, "").strip()
    return Path(raw) if raw else DEFAULT_RUNTIME_STATE_DIR


def _startup_state_record_name() -> str:
    raw = os.environ.get(STARTUP_STATE_RECORD_ENV_VAR, "").strip()
    return raw or STARTUP_STATE_RECORD_NAME


def _legacy_startup_state_path() -> Path:
    raw = os.environ.get(LEGACY_STARTUP_STATE_PATH_ENV_VAR, "").strip()
    if raw:
        return Path(raw)
    storage_root = os.environ.get("CODE_INDEX_PATH", str(Path.home() / ".code-index"))
    return Path(storage_root) / STARTUP_STATE_FILENAME


def _startup_state_db_path() -> Path:
    return _runtime_state_dir() / "runtime_state.sqlite3"


def _connect(*, ensure_schema: bool = False) -> sqlite3.Connection:
    db_path = _startup_state_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    if ensure_schema:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS json_records (
                name TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            )
            """
        )
    return conn


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_db_record(name: str) -> dict[str, Any] | None:
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM json_records WHERE name = ?",
                (str(name).strip(),),
            ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return None
        raise
    if row is None:
        return None
    try:
        payload = json.loads(str(row["payload_json"]))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def read_startup_state() -> dict[str, Any] | None:
    payload = _read_db_record(_startup_state_record_name())
    if payload is not None:
        return payload
    return _read_json_file(_legacy_startup_state_path())


def write_startup_state(
    state: str,
    *,
    phase: str = "",
    detail: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_state = str(state or "").strip().lower() or "initializing"
    normalized_phase = str(phase or "").strip()
    normalized_detail = str(detail or "").strip()
    current = read_startup_state() or {}
    started_at_utc = str(current.get("started_at_utc", "")).strip() or _utc_now()
    payload: dict[str, Any] = {
        "state": normalized_state,
        "phase": normalized_phase,
        "detail": normalized_detail,
        "pid": os.getpid(),
        "started_at_utc": started_at_utc,
        "updated_at_utc": _utc_now(),
        "metadata": dict(metadata or {}),
    }
    payload_json = json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True)
    with _connect(ensure_schema=True) as conn:
        conn.execute(
            """
            INSERT INTO json_records (name, payload_json, updated_at_utc)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                payload_json = excluded.payload_json,
                updated_at_utc = excluded.updated_at_utc
            """,
            (_startup_state_record_name(), payload_json, payload["updated_at_utc"]),
        )
        conn.commit()
    return payload


def clear_startup_state() -> bool:
    removed = False
    try:
        with _connect(ensure_schema=True) as conn:
            cursor = conn.execute("DELETE FROM json_records WHERE name = ?", (_startup_state_record_name(),))
            conn.commit()
            removed = cursor.rowcount > 0
    except sqlite3.OperationalError:
        removed = False
    legacy_path = _legacy_startup_state_path()
    try:
        legacy_path.unlink()
        removed = True
    except FileNotFoundError:
        pass
    return removed
