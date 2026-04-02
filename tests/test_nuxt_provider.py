"""Tests for the Nuxt.js context provider."""

from pathlib import Path

import pytest

from jcodemunch_mcp.parser.context.nuxt import (
    NuxtContextProvider,
    _nuxt_route_from_path,
    _nuxt_api_route_from_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_nuxt_project(tmp_path: Path) -> None:
    _write(tmp_path / "nuxt.config.ts", "export default defineNuxtConfig({})")
    _write(tmp_path / "package.json", '{"dependencies": {"nuxt": "^3.0.0"}}')


# ---------------------------------------------------------------------------
# Route path conversion
# ---------------------------------------------------------------------------

class TestNuxtRouteFromPath:
    def test_index(self):
        assert _nuxt_route_from_path("pages/index.vue") == "/"

    def test_simple_path(self):
        assert _nuxt_route_from_path("pages/users/index.vue") == "/users"

    def test_non_index_page(self):
        assert _nuxt_route_from_path("pages/about.vue") == "/about"

    def test_dynamic_param(self):
        assert _nuxt_route_from_path("pages/users/[id].vue") == "/users/:id"

    def test_catch_all(self):
        assert _nuxt_route_from_path("pages/posts/[...slug].vue") == "/posts/*"

    def test_nested(self):
        assert _nuxt_route_from_path("pages/admin/users/[id].vue") == "/admin/users/:id"


class TestNuxtApiRouteFromPath:
    def test_simple_get(self):
        result = _nuxt_api_route_from_path("server/api/users.get.ts")
        assert result == {"method": "GET", "endpoint": "/api/users"}

    def test_simple_post(self):
        result = _nuxt_api_route_from_path("server/api/users.post.ts")
        assert result == {"method": "POST", "endpoint": "/api/users"}

    def test_dynamic_param(self):
        result = _nuxt_api_route_from_path("server/api/users/[id].get.ts")
        assert result == {"method": "GET", "endpoint": "/api/users/:id"}

    def test_no_method_suffix(self):
        result = _nuxt_api_route_from_path("server/api/health.ts")
        assert result == {"method": "ALL", "endpoint": "/api/health"}

    def test_nested_path(self):
        result = _nuxt_api_route_from_path("server/api/admin/settings.put.ts")
        assert result == {"method": "PUT", "endpoint": "/api/admin/settings"}


# ---------------------------------------------------------------------------
# NuxtContextProvider
# ---------------------------------------------------------------------------

class TestNuxtContextProvider:
    def test_detect_nuxt_ts(self, tmp_path):
        _make_nuxt_project(tmp_path)
        provider = NuxtContextProvider()
        assert provider.detect(tmp_path) is True

    def test_detect_nuxt_js(self, tmp_path):
        _write(tmp_path / "nuxt.config.js", "module.exports = {}")
        provider = NuxtContextProvider()
        assert provider.detect(tmp_path) is True

    def test_detect_no_config(self, tmp_path):
        provider = NuxtContextProvider()
        assert provider.detect(tmp_path) is False

    def test_page_routing(self, tmp_path):
        _make_nuxt_project(tmp_path)
        _write(tmp_path / "pages" / "index.vue", "<template><div/></template>")
        _write(tmp_path / "pages" / "users" / "[id].vue", "<template><div/></template>")

        provider = NuxtContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)

        ctx = provider.get_file_context("pages/index.vue")
        assert ctx is not None
        assert "nuxt-page" in ctx.tags
        assert "/" in ctx.properties.get("route", "")

        ctx2 = provider.get_file_context("pages/users/[id].vue")
        assert ctx2 is not None
        assert ":id" in ctx2.properties.get("route", "")

    def test_server_api_routes(self, tmp_path):
        _make_nuxt_project(tmp_path)
        _write(tmp_path / "server" / "api" / "users.get.ts",
               "export default defineEventHandler(() => [])")
        _write(tmp_path / "server" / "api" / "users.post.ts",
               "export default defineEventHandler(() => {})")

        provider = NuxtContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)

        ctx = provider.get_file_context("server/api/users.get.ts")
        assert ctx is not None
        assert "nuxt-api" in ctx.tags
        assert ctx.properties["method"] == "GET"
        assert ctx.properties["endpoint"] == "/api/users"

        meta = provider.get_metadata()
        assert "nuxt_api" in meta
        assert "GET /api/users" in meta["nuxt_api"]

    def test_auto_import_composables(self, tmp_path):
        _make_nuxt_project(tmp_path)
        _write(tmp_path / "composables" / "useAuth.ts",
               "export function useAuth() { return {} }")
        _write(tmp_path / "composables" / "useCart.ts",
               "export function useCart() { return {} }")
        _write(tmp_path / "pages" / "index.vue", """
<script setup>
const { user } = useAuth()
const { items } = useCart()
</script>
""")

        provider = NuxtContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)

        extras = provider.get_extra_imports()
        page_imports = extras.get("pages/index.vue", [])
        specifiers = {imp["specifier"] for imp in page_imports}
        assert "composables/useAuth.ts" in specifiers
        assert "composables/useCart.ts" in specifiers

    def test_auto_import_utils(self, tmp_path):
        _make_nuxt_project(tmp_path)
        _write(tmp_path / "utils" / "formatDate.ts",
               "export function formatDate() {}")
        _write(tmp_path / "pages" / "index.vue",
               "<script setup>\nconst d = formatDate(new Date())\n</script>")

        provider = NuxtContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)

        extras = provider.get_extra_imports()
        page_imports = extras.get("pages/index.vue", [])
        specifiers = {imp["specifier"] for imp in page_imports}
        assert "utils/formatDate.ts" in specifiers

    def test_no_auto_import_self(self, tmp_path):
        """A composable should not create an auto-import edge to itself."""
        _make_nuxt_project(tmp_path)
        _write(tmp_path / "composables" / "useAuth.ts",
               "export function useAuth() { return useAuth() }")

        provider = NuxtContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)

        extras = provider.get_extra_imports()
        # composables/useAuth.ts should NOT import itself
        self_imports = extras.get("composables/useAuth.ts", [])
        assert not any(i["specifier"] == "composables/useAuth.ts" for i in self_imports)

    def test_stats(self, tmp_path):
        _make_nuxt_project(tmp_path)
        _write(tmp_path / "pages" / "index.vue", "<template><div/></template>")
        _write(tmp_path / "server" / "api" / "health.ts", "export default () => 'ok'")
        _write(tmp_path / "composables" / "useAuth.ts", "export function useAuth() {}")

        provider = NuxtContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)

        stats = provider.stats()
        assert stats["page_routes"] >= 1
        assert stats["api_routes"] >= 1
        assert stats["auto_import_symbols"] >= 1

    def test_no_pages_is_safe(self, tmp_path):
        _make_nuxt_project(tmp_path)
        provider = NuxtContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)
        assert provider.stats()["page_routes"] == 0

    def test_metadata_has_routes(self, tmp_path):
        _make_nuxt_project(tmp_path)
        _write(tmp_path / "pages" / "users" / "index.vue", "<template/>")

        provider = NuxtContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)

        meta = provider.get_metadata()
        assert "nuxt_routes" in meta
        assert "/users" in meta["nuxt_routes"]
