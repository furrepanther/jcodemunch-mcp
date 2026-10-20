"""Startup-import regressions for jcodemunch-mcp."""

from __future__ import annotations

import builtins
import importlib
import sys


def test_server_import_does_not_eagerly_load_tree_sitter_language_pack(monkeypatch):
    """Server startup should not import the tree-sitter extractor package eagerly."""
    for name in list(sys.modules):
        if name == "jcodemunch_mcp" or name.startswith("jcodemunch_mcp."):
            sys.modules.pop(name, None)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tree_sitter_language_pack" or name.startswith("tree_sitter_language_pack."):
            raise AssertionError("tree_sitter_language_pack was imported during server startup")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("jcodemunch_mcp.server")
    assert hasattr(module, "server")
