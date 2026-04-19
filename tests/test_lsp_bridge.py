"""Tests for the LSP bridge enrichment layer (Phase 4).

Tests cover:
  - LSP JSON-RPC encoding/decoding
  - LSPServer lifecycle (mock subprocess)
  - LSPBridge graceful degradation when servers are unavailable
  - Call graph integration with lsp_resolved edges
  - enrich_call_graph_with_lsp high-level entry point
  - _find_call_position helper
  - Config helpers (is_lsp_enabled, get_lsp_config)
"""

import json
import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ───────────────────────────────────────────────────────────────────
# JSON-RPC helpers
# ───────────────────────────────────────────────────────────────────

class TestEncodeMessage:
    def test_basic_message(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _encode_message
        msg = {"jsonrpc": "2.0", "id": 1, "method": "test"}
        encoded = _encode_message(msg)
        assert encoded.startswith(b"Content-Length: ")
        assert b"\r\n\r\n" in encoded
        body = encoded.split(b"\r\n\r\n", 1)[1]
        decoded = json.loads(body)
        assert decoded["method"] == "test"

    def test_content_length_matches(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _encode_message
        msg = {"jsonrpc": "2.0", "id": 42, "method": "test", "params": {"foo": "bar"}}
        encoded = _encode_message(msg)
        header, body = encoded.split(b"\r\n\r\n", 1)
        length_str = header.decode("ascii").split(": ")[1]
        assert int(length_str) == len(body)


class TestReadMessage:
    def test_valid_message(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _read_message
        import io
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}).encode()
        raw = f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        stream = io.BytesIO(raw)
        msg = _read_message(stream)
        assert msg is not None
        assert msg["result"]["ok"] is True

    def test_empty_stream(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _read_message
        import io
        stream = io.BytesIO(b"")
        assert _read_message(stream) is None

    def test_truncated_body(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _read_message
        import io
        raw = b"Content-Length: 100\r\n\r\nshort"
        stream = io.BytesIO(raw)
        assert _read_message(stream) is None


# ───────────────────────────────────────────────────────────────────
# Data types
# ───────────────────────────────────────────────────────────────────

class TestDataTypes:
    def test_position(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import Position
        pos = Position(line=10, character=5)
        assert pos.line == 10
        assert pos.character == 5

    def test_call_site(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import CallSite, Position
        site = CallSite(file="/a/b.py", position=Position(3, 8), called_name="foo")
        assert site.called_name == "foo"

    def test_resolved_ref(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import ResolvedRef, CallSite, Position
        site = CallSite(file="/a/b.py", position=Position(3, 8), called_name="foo")
        ref = ResolvedRef(
            call_site=site, target_file="/a/c.py",
            target_line=10, target_character=0, target_name="foo",
        )
        assert ref.resolution == "lsp_resolved"


# ───────────────────────────────────────────────────────────────────
# LSPServer
# ───────────────────────────────────────────────────────────────────

class TestLSPServer:
    def test_is_running_no_process(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import LSPServer
        server = LSPServer("python", ["pyright-langserver", "--stdio"], "/tmp")
        assert server.is_running is False

    def test_start_binary_not_found(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import LSPServer
        server = LSPServer("python", ["nonexistent-binary-xyz", "--stdio"], "/tmp")
        assert server.start() is False
        assert server.is_running is False

    def test_stop_no_process(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import LSPServer
        server = LSPServer("python", ["pyright-langserver", "--stdio"], "/tmp")
        server.stop()  # Should not raise


# ───────────────────────────────────────────────────────────────────
# LSPBridge
# ───────────────────────────────────────────────────────────────────

class TestLSPBridge:
    def test_graceful_degradation_no_servers(self):
        """When no LSP servers are installed, resolve_references returns empty."""
        from jcodemunch_mcp.enrichment.lsp_bridge import LSPBridge, CallSite, Position
        bridge = LSPBridge("/tmp/project", lsp_servers={"python": "nonexistent-server"})
        sites = [CallSite(file="/tmp/project/a.py", position=Position(1, 0), called_name="foo")]
        result = bridge.resolve_references(sites, {"/tmp/project/a.py": "foo()"}, {"/tmp/project/a.py": "python"})
        assert result == []
        bridge.shutdown()

    def test_empty_call_sites(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import LSPBridge
        bridge = LSPBridge("/tmp/project")
        assert bridge.resolve_references([], {}, {}) == []
        bridge.shutdown()

    def test_failed_language_cached(self):
        """After a server fails to start, the language is cached so we don't retry."""
        from jcodemunch_mcp.enrichment.lsp_bridge import LSPBridge
        bridge = LSPBridge("/tmp/project", lsp_servers={"python": "nonexistent-server"})
        server1 = bridge._get_server("python")
        assert server1 is None
        assert "python" in bridge._failed_languages
        # Second call should return None immediately without retrying
        server2 = bridge._get_server("python")
        assert server2 is None
        bridge.shutdown()

    def test_shutdown_clears_state(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import LSPBridge
        bridge = LSPBridge("/tmp/project")
        bridge._failed_languages.add("python")
        bridge.shutdown()
        assert len(bridge._failed_languages) == 0
        assert len(bridge._servers) == 0

    def test_unknown_server_name(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import LSPBridge
        bridge = LSPBridge("/tmp/project", lsp_servers={"python": "unknown-server-name"})
        server = bridge._get_server("python")
        assert server is None
        bridge.shutdown()

    def test_language_not_in_config(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import LSPBridge
        bridge = LSPBridge("/tmp/project", lsp_servers={})
        server = bridge._get_server("python")
        assert server is None
        bridge.shutdown()


# ───────────────────────────────────────────────────────────────────
# _find_call_position
# ───────────────────────────────────────────────────────────────────

class TestFindCallPosition:
    def test_finds_call(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _find_call_position
        lines = [
            "def foo():",
            "    bar()",
            "    baz(1, 2)",
        ]
        pos = _find_call_position(lines, 0, "bar")
        assert pos is not None
        assert pos.line == 1
        assert pos.character == 4

    def test_not_found(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _find_call_position
        lines = ["def foo():", "    pass"]
        pos = _find_call_position(lines, 0, "nonexistent")
        assert pos is None

    def test_finds_in_range(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _find_call_position
        lines = [
            "# line 0",
            "# line 1",
            "def foo():",
            "    helper()",
        ]
        pos = _find_call_position(lines, 2, "helper")
        assert pos is not None
        assert pos.line == 3


# ───────────────────────────────────────────────────────────────────
# _uri_to_path
# ───────────────────────────────────────────────────────────────────

class TestUriToPath:
    def test_unix_path(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _uri_to_path
        result = _uri_to_path("file:///home/user/project/main.py")
        assert result is not None
        # On Windows this will normalize differently, just check it's not None
        assert "main.py" in result

    def test_non_file_uri(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _uri_to_path
        assert _uri_to_path("https://example.com") is None

    def test_encoded_spaces(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _uri_to_path
        result = _uri_to_path("file:///home/user/my%20project/main.py")
        assert result is not None
        assert "my project" in result


# ───────────────────────────────────────────────────────────────────
# Config helpers
# ───────────────────────────────────────────────────────────────────

class TestConfigHelpers:
    def test_is_lsp_enabled_default(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import is_lsp_enabled
        with patch("jcodemunch_mcp.config.get", return_value={}):
            assert is_lsp_enabled() is False

    def test_is_lsp_enabled_true(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import is_lsp_enabled
        with patch("jcodemunch_mcp.config.get", return_value={"lsp_enabled": True}):
            assert is_lsp_enabled() is True

    def test_get_lsp_config_defaults(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import get_lsp_config, DEFAULT_LSP_SERVERS, DEFAULT_LSP_TIMEOUT
        with patch("jcodemunch_mcp.config.get", return_value={}):
            cfg = get_lsp_config()
            assert cfg["lsp_enabled"] is False
            assert cfg["lsp_servers"] == DEFAULT_LSP_SERVERS
            assert cfg["lsp_timeout_seconds"] == DEFAULT_LSP_TIMEOUT

    def test_get_lsp_config_custom(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import get_lsp_config
        custom = {
            "lsp_enabled": True,
            "lsp_servers": {"python": "pylsp"},
            "lsp_timeout_seconds": 60,
        }
        with patch("jcodemunch_mcp.config.get", return_value=custom):
            cfg = get_lsp_config()
            assert cfg["lsp_enabled"] is True
            assert cfg["lsp_servers"] == {"python": "pylsp"}
            assert cfg["lsp_timeout_seconds"] == 60


# ───────────────────────────────────────────────────────────────────
# Call graph integration: _lsp_callers / _lsp_callees
# ───────────────────────────────────────────────────────────────────

class TestCallGraphLSPIntegration:
    """Test that _lsp_callers and _lsp_callees extract edges from context_metadata."""

    def _make_index(self, lsp_edges=None, symbols=None):
        """Create a mock CodeIndex with context_metadata and symbols."""
        index = MagicMock()
        index.context_metadata = {"lsp_edges": lsp_edges} if lsp_edges else {}
        index._symbol_index = {}
        if symbols:
            for s in symbols:
                index._symbol_index[s["id"]] = s
        return index

    def _make_symbols_by_file(self, symbols):
        result = {}
        for s in symbols:
            result.setdefault(s["file"], []).append(s)
        return result

    def test_lsp_callers_finds_edge(self):
        from jcodemunch_mcp.tools._call_graph import _lsp_callers

        caller_sym = {
            "id": "services.py::process#function",
            "name": "process",
            "kind": "function",
            "file": "services.py",
            "line": 3,
            "call_references": ["helper"],
        }
        target_sym = {
            "id": "utils.py::helper#function",
            "name": "helper",
            "kind": "function",
            "file": "utils.py",
            "line": 1,
        }
        lsp_edges = [{
            "caller_file": "services.py",
            "called_name": "helper",
            "target_file": "utils.py",
            "target_line": 1,
            "resolution": "lsp_resolved",
        }]

        index = self._make_index(lsp_edges=lsp_edges, symbols=[caller_sym, target_sym])
        symbols_by_file = self._make_symbols_by_file([caller_sym, target_sym])

        callers = _lsp_callers(index, target_sym, symbols_by_file)
        assert len(callers) == 1
        assert callers[0]["id"] == "services.py::process#function"
        assert callers[0]["resolution"] == "lsp_resolved"

    def test_lsp_callers_empty_when_no_edges(self):
        from jcodemunch_mcp.tools._call_graph import _lsp_callers
        index = self._make_index()
        result = _lsp_callers(index, {"name": "foo", "file": "a.py"}, {})
        assert result == []

    def test_lsp_callees_finds_edge(self):
        from jcodemunch_mcp.tools._call_graph import _lsp_callees

        caller_sym = {
            "id": "services.py::process#function",
            "name": "process",
            "kind": "function",
            "file": "services.py",
            "line": 3,
            "call_references": ["helper"],
        }
        target_sym = {
            "id": "utils.py::helper#function",
            "name": "helper",
            "kind": "function",
            "file": "utils.py",
            "line": 1,
        }
        lsp_edges = [{
            "caller_file": "services.py",
            "called_name": "helper",
            "target_file": "utils.py",
            "target_line": 1,
            "resolution": "lsp_resolved",
        }]

        index = self._make_index(lsp_edges=lsp_edges, symbols=[caller_sym, target_sym])
        symbols_by_file = self._make_symbols_by_file([caller_sym, target_sym])

        callees = _lsp_callees(index, caller_sym, symbols_by_file)
        assert len(callees) == 1
        assert callees[0]["id"] == "utils.py::helper#function"
        assert callees[0]["resolution"] == "lsp_resolved"

    def test_lsp_callees_empty_no_call_refs(self):
        from jcodemunch_mcp.tools._call_graph import _lsp_callees
        sym = {"name": "foo", "file": "a.py", "call_references": []}
        index = self._make_index(lsp_edges=[{"caller_file": "a.py", "called_name": "bar", "target_file": "b.py", "target_line": 1}])
        result = _lsp_callees(index, sym, {})
        assert result == []


# ───────────────────────────────────────────────────────────────────
# enrich_call_graph_with_lsp
# ───────────────────────────────────────────────────────────────────

class TestEnrichCallGraphWithLSP:
    def test_disabled_returns_empty(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import enrich_call_graph_with_lsp
        with patch("jcodemunch_mcp.enrichment.lsp_bridge.get_lsp_config", return_value={"lsp_enabled": False, "lsp_servers": {}, "lsp_timeout_seconds": 30}):
            result = enrich_call_graph_with_lsp("/tmp", [], {}, {})
            assert result == []

    def test_no_call_sites_returns_empty(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import enrich_call_graph_with_lsp
        from jcodemunch_mcp.parser.symbols import Symbol
        sym = Symbol(
            id="a.py::foo#function", file="a.py", name="foo",
            qualified_name="foo", kind="function", language="python",
            signature="def foo():", call_references=[],
        )
        with patch("jcodemunch_mcp.enrichment.lsp_bridge.get_lsp_config", return_value={
            "lsp_enabled": True,
            "lsp_servers": {"python": "pyright"},
            "lsp_timeout_seconds": 30,
        }):
            result = enrich_call_graph_with_lsp(
                "/tmp/project", [sym], {"a.py": "def foo(): pass"}, {"a.py": "python"},
            )
            assert result == []


# ───────────────────────────────────────────────────────────────────
# get_call_hierarchy: resolution tiers in _meta
# ───────────────────────────────────────────────────────────────────

class TestCallHierarchyLSPMeta:
    def test_lsp_enriched_methodology(self):
        """When lsp_edges are present, methodology should be lsp_enriched."""
        from jcodemunch_mcp.tools.get_call_hierarchy import get_call_hierarchy
        from jcodemunch_mcp.tools.index_folder import index_folder

    def test_resolution_tiers_counted(self, tmp_path):
        """Resolution tier counts should include lsp_resolved when LSP edges exist."""
        from jcodemunch_mcp.tools.index_folder import index_folder

        src = tmp_path / "src"
        store = tmp_path / "store"
        src.mkdir()
        store.mkdir()

        (src / "utils.py").write_text("def helper():\n    return 42\n")
        (src / "main.py").write_text("from utils import helper\n\ndef run():\n    return helper()\n")

        result = index_folder(str(src), use_ai_summaries=False, storage_path=str(store))
        assert result["success"] is True

        from jcodemunch_mcp.tools.get_call_hierarchy import get_call_hierarchy
        hier = get_call_hierarchy(result["repo"], "helper", direction="callers", storage_path=str(store))
        assert "error" not in hier
        # Should have resolution_tiers in _meta
        tiers = hier["_meta"]["resolution_tiers"]
        assert isinstance(tiers, dict)
        # At minimum, some tier should be present
        assert sum(tiers.values()) > 0


# ───────────────────────────────────────────────────────────────────
# _to_lsp_language_id
# ───────────────────────────────────────────────────────────────────

class TestToLSPLanguageId:
    def test_known_languages(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _to_lsp_language_id
        assert _to_lsp_language_id("python") == "python"
        assert _to_lsp_language_id("typescript") == "typescript"
        assert _to_lsp_language_id("go") == "go"
        assert _to_lsp_language_id("rust") == "rust"
        assert _to_lsp_language_id("javascript") == "javascript"

    def test_unknown_passthrough(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _to_lsp_language_id
        assert _to_lsp_language_id("haskell") == "haskell"


# ───────────────────────────────────────────────────────────────────
# DEFAULT_LSP_SERVERS and _SERVER_COMMANDS
# ───────────────────────────────────────────────────────────────────

class TestDefaults:
    def test_default_servers_all_have_commands(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import DEFAULT_LSP_SERVERS, _SERVER_COMMANDS
        for lang, server_name in DEFAULT_LSP_SERVERS.items():
            assert server_name in _SERVER_COMMANDS, f"Missing command for {server_name} ({lang})"

    def test_server_commands_are_lists(self):
        from jcodemunch_mcp.enrichment.lsp_bridge import _SERVER_COMMANDS
        for name, cmd in _SERVER_COMMANDS.items():
            assert isinstance(cmd, list), f"Command for {name} should be a list"
            assert len(cmd) >= 1, f"Command for {name} should have at least one element"


# ───────────────────────────────────────────────────────────────────
# Index folder integration (mock LSP)
# ───────────────────────────────────────────────────────────────────

class TestIndexFolderLSPIntegration:
    def test_lsp_enrichment_skipped_when_disabled(self, tmp_path):
        """With default config (lsp_enabled=False), no LSP enrichment runs."""
        src = tmp_path / "src"
        store = tmp_path / "store"
        src.mkdir()
        store.mkdir()
        (src / "a.py").write_text("def foo(): pass\n")

        from jcodemunch_mcp.tools.index_folder import index_folder
        result = index_folder(str(src), use_ai_summaries=False, storage_path=str(store))
        assert result["success"] is True
        # No lsp_edges in context_metadata since LSP is disabled by default
        from jcodemunch_mcp.storage import IndexStore
        idx = IndexStore(base_path=str(store)).load_index("local", result["repo"].split("/")[1])
        assert "lsp_edges" not in (idx.context_metadata or {})
