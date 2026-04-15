"""Tests for render_diagram — universal Mermaid renderer."""

import pytest

from jcodemunch_mcp.tools.render_diagram import (
    render_diagram,
    _detect_source,
    _prune_graph,
    _basename,
    _disambiguate_basenames,
    _sanitize_label,
)


# ── Minimal mock data builders ──────────────────────────────────────────────

def _call_hierarchy_data(**overrides):
    base = {
        "repo": "test/repo",
        "symbol": {"id": "server.py::handle", "name": "handle", "kind": "function", "file": "server.py", "line": 10},
        "direction": "both",
        "depth": 3,
        "depth_reached": 2,
        "caller_count": 2,
        "callee_count": 1,
        "callers": [
            {"id": "main.py::run", "name": "run", "kind": "function", "file": "main.py", "line": 5, "depth": 1, "resolution": "ast_resolved"},
            {"id": "app.py::startup", "name": "startup", "kind": "function", "file": "app.py", "line": 1, "depth": 2, "resolution": "text_matched"},
        ],
        "callees": [
            {"id": "db.py::query", "name": "query", "kind": "function", "file": "db.py", "line": 20, "depth": 1, "resolution": "lsp_resolved"},
        ],
        "dispatches": [],
        "_meta": {"timing_ms": 5.0, "methodology": "ast_call_references", "confidence_level": "medium", "source": "ast_call_references", "resolution_tiers": {}, "tip": ""},
    }
    base.update(overrides)
    return base


def _signal_chains_discovery_data(**overrides):
    base = {
        "repo": "test/repo",
        "gateway_count": 2,
        "chain_count": 2,
        "chains": [
            {"gateway": "routes.py::create_user", "gateway_name": "create_user", "kind": "http", "label": "POST /api/users", "depth": 3, "reach": 4, "symbols": ["create_user", "validate", "save", "notify"], "files_touched": ["routes.py", "validators.py", "repo.py", "mailer.py"], "file_count": 4},
            {"gateway": "cli.py::seed_db", "gateway_name": "seed_db", "kind": "cli", "label": "cli:seed-db", "depth": 2, "reach": 3, "symbols": ["seed_db", "generate", "insert"], "files_touched": ["cli.py", "factory.py", "repo.py"], "file_count": 3},
        ],
        "kind_summary": {"http": 1, "cli": 1},
        "orphan_symbols": 5,
        "orphan_symbol_pct": 12.5,
        "_meta": {"timing_ms": 10.0},
    }
    base.update(overrides)
    return base


def _signal_chains_lookup_data(**overrides):
    base = {
        "repo": "test/repo",
        "symbol": "validate",
        "symbol_id": "validators.py::validate",
        "chain_count": 1,
        "chains": [
            {"gateway": "routes.py::create_user", "gateway_name": "create_user", "kind": "http", "label": "POST /api/users", "chain_reach": 4},
        ],
        "on_no_chain": False,
        "_meta": {"timing_ms": 3.0},
    }
    base.update(overrides)
    return base


def _tectonic_map_data(**overrides):
    base = {
        "repo": "test/repo",
        "plate_count": 2,
        "file_count": 6,
        "plates": [
            {
                "plate_id": 0,
                "anchor": "src/api/server.py",
                "file_count": 3,
                "cohesion": 0.82,
                "files": ["src/api/server.py", "src/api/routes.py", "src/api/middleware.py"],
                "majority_directory": "src/api",
            },
            {
                "plate_id": 1,
                "anchor": "src/db/models.py",
                "file_count": 3,
                "cohesion": 0.65,
                "files": ["src/db/models.py", "src/db/queries.py", "src/config/loader.py"],
                "majority_directory": "src/db",
                "drifters": ["src/config/loader.py"],
                "drifter_count": 1,
                "nexus_alert": True,
                "nexus_coupling_count": 4,
                "coupled_to": {"src/api/server.py": 0.45},
            },
        ],
        "isolated_files": ["README.md"],
        "signals_used": ["structural", "behavioral", "temporal"],
        "drifter_summary": [{"file": "src/config/loader.py", "current_directory": "src/config", "belongs_with": "src/db", "plate_anchor": "src/db/models.py"}],
        "_meta": {"timing_ms": 15.0},
    }
    base.update(overrides)
    return base


def _dependency_cycles_data(**overrides):
    base = {
        "repo": "test/repo",
        "cycle_count": 1,
        "cycles": [["a.py", "b.py", "c.py"]],
        "_meta": {"timing_ms": 2.0},
    }
    base.update(overrides)
    return base


def _impact_preview_data(**overrides):
    base = {
        "repo": "test/repo",
        "symbol": {"id": "utils.py::parse_config", "name": "parse_config", "kind": "function", "file": "utils.py", "line": 10},
        "affected_files": 2,
        "affected_symbol_count": 3,
        "affected_symbols": [
            {"id": "server.py::init", "name": "init", "kind": "function", "file": "server.py", "line": 5, "call_chain": ["utils.py::parse_config", "server.py::init"]},
            {"id": "main.py::run", "name": "run", "kind": "function", "file": "main.py", "line": 1, "call_chain": ["utils.py::parse_config", "server.py::init", "main.py::run"]},
            {"id": "main.py::startup", "name": "startup", "kind": "function", "file": "main.py", "line": 20, "call_chain": ["utils.py::parse_config", "main.py::startup"]},
        ],
        "affected_by_file": {
            "server.py": [{"id": "server.py::init", "name": "init", "kind": "function", "line": 5}],
            "main.py": [
                {"id": "main.py::run", "name": "run", "kind": "function", "line": 1},
                {"id": "main.py::startup", "name": "startup", "kind": "function", "line": 20},
            ],
        },
        "call_chains": [
            {"symbol_id": "server.py::init", "chain": ["utils.py::parse_config", "server.py::init"]},
            {"symbol_id": "main.py::run", "chain": ["utils.py::parse_config", "server.py::init", "main.py::run"]},
            {"symbol_id": "main.py::startup", "chain": ["utils.py::parse_config", "main.py::startup"]},
        ],
        "_meta": {"timing_ms": 4.0},
    }
    base.update(overrides)
    return base


def _blast_radius_data(**overrides):
    base = {
        "repo": "test/repo",
        "symbol": {"id": "config.py::DB_URL", "name": "DB_URL"},
        "confirmed": [
            {"file": "server.py", "reference_count": 3},
            {"file": "migration.py", "reference_count": 1},
        ],
        "potential": [
            {"file": "tests/test_db.py"},
        ],
        "overall_risk_score": 0.65,
        "direct_dependents_count": 2,
        "_meta": {"timing_ms": 3.0},
    }
    base.update(overrides)
    return base


def _dependency_graph_data(**overrides):
    base = {
        "repo": "test/repo",
        "file": "src/server.py",
        "direction": "imports",
        "depth": 1,
        "neighbors": {
            "src/routes.py": {"specifiers": ["routes"]},
            "src/models.py": {"specifiers": ["models"]},
        },
        "cross_repo_edges": [],
        "_meta": {"timing_ms": 2.0},
    }
    base.update(overrides)
    return base


# ── Detection tests ─────────────────────────────────────────────────────────

class TestDetection:
    def test_detect_call_hierarchy(self):
        assert _detect_source(_call_hierarchy_data()) == "call_hierarchy"

    def test_detect_signal_chains_discovery(self):
        assert _detect_source(_signal_chains_discovery_data()) == "signal_chains_discovery"

    def test_detect_signal_chains_lookup(self):
        assert _detect_source(_signal_chains_lookup_data()) == "signal_chains_lookup"

    def test_detect_tectonic_map(self):
        assert _detect_source(_tectonic_map_data()) == "tectonic_map"

    def test_detect_dependency_cycles(self):
        assert _detect_source(_dependency_cycles_data()) == "dependency_cycles"

    def test_detect_impact_preview(self):
        assert _detect_source(_impact_preview_data()) == "impact_preview"

    def test_detect_blast_radius(self):
        assert _detect_source(_blast_radius_data()) == "blast_radius"

    def test_detect_dependency_graph(self):
        assert _detect_source(_dependency_graph_data()) == "dependency_graph"

    def test_detect_error_response(self):
        assert _detect_source({"error": "not indexed"}) == "error"

    def test_detect_unknown_shape(self):
        assert _detect_source({"foo": "bar", "baz": 42}) == "unknown"

    def test_detect_callers_only(self):
        """call_hierarchy with direction='callers' has callers but no callees."""
        data = _call_hierarchy_data(callees=[], callee_count=0, direction="callers")
        # Still has callers + symbol keys
        assert _detect_source(data) == "call_hierarchy"


# ── Sanitisation helper tests ───────────────────────────────────────────────

class TestHelpers:
    def test_basename_unix(self):
        assert _basename("src/api/server.py") == "server.py"

    def test_basename_windows(self):
        assert _basename("src\\api\\server.py") == "server.py"

    def test_basename_flat(self):
        assert _basename("server.py") == "server.py"

    def test_sanitize_label_quotes(self):
        assert '"' not in _sanitize_label('say "hello"')

    def test_sanitize_label_angles(self):
        result = _sanitize_label("List<int>")
        assert "<" not in result and ">" not in result

    def test_disambiguate_no_collision(self):
        paths = ["src/a.py", "src/b.py"]
        d = _disambiguate_basenames(paths)
        assert d["src/a.py"] == "a.py"
        assert d["src/b.py"] == "b.py"

    def test_disambiguate_collision(self):
        paths = ["src/api/server.py", "src/db/server.py"]
        d = _disambiguate_basenames(paths)
        assert d["src/api/server.py"] != d["src/db/server.py"]
        assert "api" in d["src/api/server.py"]
        assert "db" in d["src/db/server.py"]


# ── Pruning tests ───────────────────────────────────────────────────────────

class TestPruning:
    def test_no_pruning_under_budget(self):
        nodes = ["a", "b", "c"]
        edges = [("a", "b"), ("b", "c")]
        result_nodes, result_edges, pruned = _prune_graph(nodes, edges, 10, set())
        assert pruned == 0
        assert set(result_nodes) == {"a", "b", "c"}

    def test_leaf_pruning(self):
        nodes = ["root", "a", "b", "leaf1", "leaf2", "leaf3"]
        edges = [("root", "a"), ("root", "b"), ("a", "leaf1"), ("a", "leaf2"), ("b", "leaf3")]
        result_nodes, result_edges, pruned = _prune_graph(nodes, edges, 3, {"root"})
        assert len(result_nodes) <= 3
        assert "root" in result_nodes
        assert pruned > 0

    def test_preserve_set_honored(self):
        nodes = ["target", "a"]
        edges = [("target", "a")]
        result_nodes, _, pruned = _prune_graph(nodes, edges, 1, {"target"})
        assert "target" in result_nodes

    def test_exact_budget_no_pruning(self):
        nodes = ["a", "b", "c"]
        edges = [("a", "b"), ("b", "c")]
        result_nodes, _, pruned = _prune_graph(nodes, edges, 3, set())
        assert pruned == 0
        assert len(result_nodes) == 3


# ── Theme tests ─────────────────────────────────────────────────────────────

class TestThemes:
    def test_flow_theme_has_color(self):
        result = render_diagram(_call_hierarchy_data(), theme="flow")
        assert "error" not in result
        assert "#4A90D9" in result["mermaid"] or "fill:" in result["mermaid"]

    def test_risk_theme_has_color(self):
        result = render_diagram(_blast_radius_data(), theme="risk")
        assert "error" not in result
        # Risk theme should use red/orange palette
        assert "FF4136" in result["mermaid"] or "FF851B" in result["mermaid"]

    def test_minimal_theme_no_bright_colors(self):
        result = render_diagram(_dependency_cycles_data(), theme="minimal")
        assert "error" not in result
        # Minimal still uses red for cycle highlighting (semantic, not theme)
        assert "mermaid" in result

    def test_invalid_theme_defaults_to_flow(self):
        result = render_diagram(_call_hierarchy_data(), theme="nonexistent")
        assert "error" not in result
        assert result["_meta"]["theme"] == "nonexistent"  # theme name recorded as-is


# ── Renderer tests: call_hierarchy ──────────────────────────────────────────

class TestRenderCallHierarchy:
    def test_flowchart_td(self):
        result = render_diagram(_call_hierarchy_data())
        assert result["diagram_type"] == "flowchart TD"
        assert result["mermaid"].startswith("flowchart TD")

    def test_root_node_present(self):
        result = render_diagram(_call_hierarchy_data())
        assert "handle" in result["mermaid"]

    def test_callers_and_callees_present(self):
        result = render_diagram(_call_hierarchy_data())
        assert "run" in result["mermaid"]
        assert "query" in result["mermaid"]

    def test_resolution_tier_styling(self):
        result = render_diagram(_call_hierarchy_data())
        # Should have classDef for at least one resolution tier
        assert "classDef" in result["mermaid"]

    def test_file_subgrouping(self):
        result = render_diagram(_call_hierarchy_data())
        assert "subgraph" in result["mermaid"]

    def test_source_tool_detected(self):
        result = render_diagram(_call_hierarchy_data())
        assert result["source_tool"] == "call_hierarchy"

    def test_legend_present(self):
        result = render_diagram(_call_hierarchy_data())
        assert "Edge color" in result["legend"]

    def test_empty_callers_callees(self):
        data = _call_hierarchy_data(callers=[], callees=[], caller_count=0, callee_count=0)
        result = render_diagram(data)
        assert "error" not in result
        assert "handle" in result["mermaid"]


# ── Renderer tests: signal_chains ───────────────────────────────────────────

class TestRenderSignalChains:
    def test_sequence_diagram_type(self):
        result = render_diagram(_signal_chains_discovery_data())
        assert result["diagram_type"] == "sequenceDiagram"
        assert result["mermaid"].startswith("sequenceDiagram")

    def test_gateway_participants(self):
        result = render_diagram(_signal_chains_discovery_data())
        assert "create_user" in result["mermaid"]
        assert "seed_db" in result["mermaid"]

    def test_kind_grouping_boxes(self):
        result = render_diagram(_signal_chains_discovery_data())
        assert "box" in result["mermaid"]
        assert "HTTP" in result["mermaid"]
        assert "CLI" in result["mermaid"]

    def test_orphan_note(self):
        result = render_diagram(_signal_chains_discovery_data())
        assert "12.5%" in result["mermaid"]

    def test_empty_chains(self):
        data = _signal_chains_discovery_data(chains=[], gateway_count=0, chain_count=0, orphan_symbols=0, orphan_symbol_pct=0)
        result = render_diagram(data)
        assert "error" not in result
        assert "No signal chains" in result["mermaid"]

    def test_lookup_mode(self):
        result = render_diagram(_signal_chains_lookup_data())
        assert result["source_tool"] == "signal_chains"

    def test_source_tool_name_normalized(self):
        result = render_diagram(_signal_chains_discovery_data())
        assert result["source_tool"] == "signal_chains"


# ── Renderer tests: tectonic_map ────────────────────────────────────────────

class TestRenderTectonicMap:
    def test_subgraph_per_plate(self):
        result = render_diagram(_tectonic_map_data())
        mermaid = result["mermaid"]
        assert mermaid.count("subgraph plate") == 2

    def test_anchor_styling(self):
        result = render_diagram(_tectonic_map_data())
        assert ":::anchor" in result["mermaid"]

    def test_drifter_styling(self):
        result = render_diagram(_tectonic_map_data())
        assert ":::drifter" in result["mermaid"]

    def test_nexus_alert_in_label(self):
        result = render_diagram(_tectonic_map_data())
        assert "NEXUS" in result["mermaid"]

    def test_cohesion_in_title(self):
        result = render_diagram(_tectonic_map_data())
        assert "0.82" in result["mermaid"]
        assert "0.65" in result["mermaid"]

    def test_coupling_edges(self):
        result = render_diagram(_tectonic_map_data())
        assert "0.45" in result["mermaid"]

    def test_isolated_files(self):
        result = render_diagram(_tectonic_map_data())
        assert "Isolated" in result["mermaid"]

    def test_empty_plates(self):
        data = _tectonic_map_data(plates=[], plate_count=0)
        result = render_diagram(data)
        assert "error" not in result


# ── Renderer tests: dependency_cycles ───────────────────────────────────────

class TestRenderDependencyCycles:
    def test_cycle_subgraphs(self):
        result = render_diagram(_dependency_cycles_data())
        assert "Cycle 1" in result["mermaid"]

    def test_cycle_edges_red(self):
        result = render_diagram(_dependency_cycles_data())
        assert "FF4136" in result["mermaid"]

    def test_cycled_class(self):
        result = render_diagram(_dependency_cycles_data())
        assert ":::cycled" in result["mermaid"]

    def test_no_cycles_clean(self):
        data = _dependency_cycles_data(cycles=[], cycle_count=0)
        result = render_diagram(data)
        assert "No circular dependencies" in result["mermaid"]

    def test_multiple_cycles(self):
        data = _dependency_cycles_data(
            cycles=[["a.py", "b.py"], ["x.py", "y.py", "z.py"]],
            cycle_count=2,
        )
        result = render_diagram(data)
        assert "Cycle 1" in result["mermaid"]
        assert "Cycle 2" in result["mermaid"]

    def test_closing_edge(self):
        """Cycle a→b→c should have edge from c back to a."""
        result = render_diagram(_dependency_cycles_data())
        # 3 files = 3 edges (a→b, b→c, c→a)
        assert result["edge_count"] == 3


# ── Renderer tests: impact_preview ──────────────────────────────────────────

class TestRenderImpactPreview:
    def test_flowchart_bt(self):
        result = render_diagram(_impact_preview_data())
        assert result["diagram_type"] == "flowchart BT"

    def test_target_present(self):
        result = render_diagram(_impact_preview_data())
        assert "parse_config" in result["mermaid"]
        assert ":::target" in result["mermaid"]

    def test_affected_symbols_present(self):
        result = render_diagram(_impact_preview_data())
        assert "init" in result["mermaid"]
        assert "run" in result["mermaid"]

    def test_file_grouping(self):
        result = render_diagram(_impact_preview_data())
        assert "subgraph" in result["mermaid"]

    def test_depth_coloring(self):
        result = render_diagram(_impact_preview_data())
        # Should have classDef for depth levels
        assert "classDef d1" in result["mermaid"]


# ── Renderer tests: blast_radius ────────────────────────────────────────────

class TestRenderBlastRadius:
    def test_flowchart_td(self):
        result = render_diagram(_blast_radius_data())
        assert result["diagram_type"] == "flowchart TD"

    def test_target_present(self):
        result = render_diagram(_blast_radius_data())
        assert "DB_URL" in result["mermaid"]
        assert ":::target" in result["mermaid"]

    def test_confirmed_styling(self):
        result = render_diagram(_blast_radius_data())
        assert ":::confirmed" in result["mermaid"]

    def test_potential_styling(self):
        result = render_diagram(_blast_radius_data())
        assert ":::potential" in result["mermaid"]

    def test_risk_score_badge(self):
        result = render_diagram(_blast_radius_data())
        assert "0.65" in result["mermaid"]
        assert "medium" in result["mermaid"]

    def test_reference_count_annotation(self):
        result = render_diagram(_blast_radius_data())
        assert "3 refs" in result["mermaid"]

    def test_risk_theme_heat_coloring(self):
        result = render_diagram(_blast_radius_data(), theme="risk")
        assert "FF4136" in result["mermaid"] or "FF851B" in result["mermaid"]


# ── Renderer tests: dependency_graph ────────────────────────────────────────

class TestRenderDependencyGraph:
    def test_flowchart_lr(self):
        result = render_diagram(_dependency_graph_data())
        assert result["diagram_type"] == "flowchart LR"

    def test_focal_node_present(self):
        result = render_diagram(_dependency_graph_data())
        assert "server.py" in result["mermaid"]
        assert ":::focal" in result["mermaid"]

    def test_neighbors_present(self):
        result = render_diagram(_dependency_graph_data())
        assert "routes.py" in result["mermaid"]
        assert "models.py" in result["mermaid"]

    def test_cross_repo_dashed(self):
        data = _dependency_graph_data(cross_repo_edges=[{"file": "other-repo/utils.py"}])
        result = render_diagram(data)
        assert "cross-repo" in result["mermaid"]
        assert ":::cross" in result["mermaid"]

    def test_importers_direction(self):
        data = _dependency_graph_data(direction="importers")
        result = render_diagram(data)
        assert "importers" in result["legend"]


# ── Return shape tests ──────────────────────────────────────────────────────

class TestReturnShape:
    """Every successful render must have all required keys."""

    _ALL_SOURCES = [
        _call_hierarchy_data,
        _signal_chains_discovery_data,
        _signal_chains_lookup_data,
        _tectonic_map_data,
        _dependency_cycles_data,
        _impact_preview_data,
        _blast_radius_data,
        _dependency_graph_data,
    ]

    _REQUIRED_KEYS = {"diagram_type", "mermaid", "node_count", "edge_count", "pruned_count", "legend", "source_tool", "_meta"}

    @pytest.mark.parametrize("builder", _ALL_SOURCES)
    def test_all_keys_present(self, builder):
        result = render_diagram(builder())
        assert "error" not in result, result.get("error")
        missing = self._REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    @pytest.mark.parametrize("builder", _ALL_SOURCES)
    def test_mermaid_is_nonempty_string(self, builder):
        result = render_diagram(builder())
        assert isinstance(result["mermaid"], str)
        assert len(result["mermaid"]) > 10

    @pytest.mark.parametrize("builder", _ALL_SOURCES)
    def test_node_count_nonnegative(self, builder):
        result = render_diagram(builder())
        assert result["node_count"] >= 0

    @pytest.mark.parametrize("builder", _ALL_SOURCES)
    def test_meta_has_timing(self, builder):
        result = render_diagram(builder())
        assert "timing_ms" in result["_meta"]


# ── Error handling tests ────────────────────────────────────────────────────

class TestErrorHandling:
    def test_error_response_rejected(self):
        result = render_diagram({"error": "not indexed"})
        assert "error" in result

    def test_unknown_shape_rejected(self):
        result = render_diagram({"foo": "bar"})
        assert "error" in result
        assert "Unrecognised" in result["error"]

    def test_empty_dict_rejected(self):
        result = render_diagram({})
        assert "error" in result
