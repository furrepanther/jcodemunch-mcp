"""Tests for session state persistence (Feature 10: Session-Aware Routing)."""
import json
import time
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest


class TestSessionStateSaveLoad:
    """Tests for SessionState save/load roundtrip."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save journal + cache, load it back, verify identical."""
        from jcodemunch_mcp.tools.session_state import SessionState
        from jcodemunch_mcp.tools.session_journal import SessionJournal
        
        # Create state and journal
        state = SessionState(base_path=str(tmp_path))
        journal = SessionJournal()
        journal.record_read("src/main.py", "get_symbol_source")
        journal.record_search("UserService", 3)
        journal.record_edit("src/auth.py")
        
        # Create a search cache
        cache = OrderedDict()
        cache[("local/test", "2026-04-05T12:00:00Z", "query1")] = {"result_count": 5}
        cache[("local/test", "2026-04-05T12:00:00Z", "query2")] = {"result_count": 0}
        
        # Save
        state.save(journal, cache, max_queries=50)
        
        # Verify file exists
        state_file = tmp_path / "_session_state.json"
        assert state_file.exists()
        
        # Load
        loaded = state.load(max_age_minutes=60)
        assert loaded is not None
        
        # Verify journal data
        assert "journal" in loaded
        assert "src/main.py" in loaded["journal"]["files_accessed"]
        assert "UserService" in loaded["journal"]["queries"]
        assert "src/auth.py" in loaded["journal"]["files_edited"]
        
        # Verify cache data
        assert "search_cache" in loaded
        assert len(loaded["search_cache"]) == 2

    def test_load_returns_none_when_stale(self, tmp_path):
        """Save, advance time past max_age, load → None."""
        from jcodemunch_mcp.tools.session_state import SessionState
        from jcodemunch_mcp.tools.session_journal import SessionJournal
        
        state = SessionState(base_path=str(tmp_path))
        journal = SessionJournal()
        
        # Save
        state.save(journal, OrderedDict())
        
        # Modify the saved_at timestamp to be old
        state_file = tmp_path / "_session_state.json"
        data = json.loads(state_file.read_text(encoding="utf-8"))
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=60)).isoformat()
        data["saved_at"] = old_time
        state_file.write_text(json.dumps(data), encoding="utf-8")
        
        # Load with max_age=30 minutes
        loaded = state.load(max_age_minutes=30)
        assert loaded is None

    def test_load_returns_none_when_missing(self, tmp_path):
        """Load from non-existent path → None."""
        from jcodemunch_mcp.tools.session_state import SessionState
        
        state = SessionState(base_path=str(tmp_path))
        loaded = state.load(max_age_minutes=60)
        assert loaded is None


class TestSessionStateRestoreJournal:
    """Tests for restore_journal method."""

    def test_restore_journal_populates_entries(self, tmp_path):
        """Restore, verify get_context() shows restored files/queries."""
        from jcodemunch_mcp.tools.session_state import SessionState
        from jcodemunch_mcp.tools.session_journal import SessionJournal
        
        # Create original journal and save
        state = SessionState(base_path=str(tmp_path))
        journal1 = SessionJournal()
        journal1.record_read("src/main.py", "get_symbol_source")
        journal1.record_search("UserService", 3)
        state.save(journal1, OrderedDict())
        
        # Load and restore to new journal
        loaded = state.load(max_age_minutes=60)
        journal2 = SessionJournal()
        count = state.restore_journal(journal2, loaded)
        
        # Verify entries restored
        assert count > 0
        ctx = journal2.get_context()
        # files_accessed is a list of dicts with "file" key
        assert any(f.get("file") == "src/main.py" for f in ctx["files_accessed"])
        # recent_searches is a list of dicts with "query" key
        assert any(q.get("query") == "UserService" for q in ctx["recent_searches"])


class TestSessionStateRestoreCache:
    """Tests for restore_search_cache method."""

    def test_restore_skips_stale_cache_entries(self, tmp_path):
        """Save with index_snapshot A, change index to B, restore → cache entries skipped."""
        from jcodemunch_mcp.tools.session_state import SessionState
        from jcodemunch_mcp.tools.session_journal import SessionJournal
        
        state = SessionState(base_path=str(tmp_path))
        journal = SessionJournal()
        
        # Create cache with old indexed_at
        cache = OrderedDict()
        old_indexed_at = "2026-04-05T10:00:00Z"
        cache[("local/test", old_indexed_at, "query1")] = {"result_count": 5}
        
        state.save(journal, cache, max_queries=50)
        
        # Load with current_indexes having different indexed_at
        loaded = state.load(max_age_minutes=60)
        current_indexes = {"local/test": "2026-04-05T12:00:00Z"}
        
        # Restore
        new_cache = OrderedDict()
        count = state.restore_search_cache(new_cache, loaded, current_indexes)
        
        # Should skip the entry because indexed_at changed
        assert count == 0
        assert len(new_cache) == 0


class TestSessionStateMaxQueries:
    """Tests for max_queries cap on saved cache."""

    def test_max_queries_caps_saved_cache(self, tmp_path):
        """Save 100 cache entries with max_queries=10, load → only top 10 by hit_count."""
        from jcodemunch_mcp.tools.session_state import SessionState
        from jcodemunch_mcp.tools.session_journal import SessionJournal
        
        state = SessionState(base_path=str(tmp_path))
        journal = SessionJournal()
        
        # Create 100 cache entries with different hit counts
        cache = OrderedDict()
        for i in range(100):
            key = ("local/test", "2026-04-05T12:00:00Z", f"query{i}")
            # Higher index = higher hit_count (to make it interesting)
            cache[key] = {
                "result_count": i,
                "_hit_count": i,  # internal field used by save() for ranking
            }
        
        # Save with max_queries=10
        state.save(journal, cache, max_queries=10)
        
        # Load
        loaded = state.load(max_age_minutes=60)
        assert len(loaded["search_cache"]) == 10
        
        # Verify top 10 by hit_count were kept (indices 90-99)
        hit_counts = [e["hit_count"] for e in loaded["search_cache"]]
        assert min(hit_counts) == 90
        assert max(hit_counts) == 99


class TestSessionStateClear:
    """Tests for clear method."""

    def test_clear_removes_file(self, tmp_path):
        """Save, clear, verify file deleted."""
        from jcodemunch_mcp.tools.session_state import SessionState
        from jcodemunch_mcp.tools.session_journal import SessionJournal
        
        state = SessionState(base_path=str(tmp_path))
        journal = SessionJournal()
        
        # Save
        state.save(journal, OrderedDict())
        state_file = tmp_path / "_session_state.json"
        assert state_file.exists()
        
        # Clear
        state.clear()
        assert not state_file.exists()


class TestSessionStateConcurrency:
    """Tests for thread safety."""

    def test_concurrent_save_no_crash(self, tmp_path):
        """Two threads calling save simultaneously should not crash."""
        from jcodemunch_mcp.tools.session_state import SessionState
        from jcodemunch_mcp.tools.session_journal import SessionJournal
        import threading
        
        state = SessionState(base_path=str(tmp_path))
        journal = SessionJournal()
        
        errors = []
        
        def save_thread():
            try:
                for _ in range(10):
                    state.save(journal, OrderedDict())
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=save_thread) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert not errors


class TestSessionStateNegativeEvidenceLog:
    """Tests for negative evidence log persistence."""

    def test_negative_evidence_log_persisted(self, tmp_path):
        """Record negative evidence, save, load, verify entries present."""
        from jcodemunch_mcp.tools.session_state import SessionState
        from jcodemunch_mcp.tools.session_journal import SessionJournal
        
        state = SessionState(base_path=str(tmp_path))
        journal = SessionJournal()
        
        # Simulate recording negative evidence
        neg_evidence = {
            "query": "csrf protection",
            "repo": "local/myapp-abc123",
            "verdict": "no_implementation_found",
            "scanned_symbols": 1422,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Save with negative evidence log
        state.save(journal, OrderedDict(), negative_evidence_log=[neg_evidence])
        
        # Load
        loaded = state.load(max_age_minutes=60)
        assert "negative_evidence_log" in loaded
        assert len(loaded["negative_evidence_log"]) == 1
        assert loaded["negative_evidence_log"][0]["query"] == "csrf protection"


class TestSessionStateStorageLocation:
    """Tests for storage location."""

    def test_default_storage_path(self, tmp_path, monkeypatch):
        """Session state should be stored in ~/.code-index/_session_state.json."""
        from jcodemunch_mcp.tools.session_state import SessionState
        from jcodemunch_mcp.tools.session_journal import SessionJournal
        
        monkeypatch.setenv("CODE_INDEX_PATH", str(tmp_path))
        
        # Create state without explicit base_path
        state = SessionState()
        journal = SessionJournal()
        
        state.save(journal, OrderedDict())
        
        # Verify file is in CODE_INDEX_PATH
        expected_path = tmp_path / "_session_state.json"
        assert expected_path.exists()