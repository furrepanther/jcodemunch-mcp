"""Tests for turn budget (Feature 7)."""

import pytest
import threading
import time
from unittest.mock import patch


class TestTurnBudget:
    """Tests for TurnBudget class."""

    def test_new_turn_resets_counters(self):
        """After turn_gap_seconds, new turn starts and counters reset."""
        from jcodemunch_mcp.tools.turn_budget import TurnBudget, get_turn_budget
        # Create fresh instance with short gap
        budget = TurnBudget(turn_gap_seconds=0.1, turn_budget_tokens=10000)
        budget.record_output(5000)
        assert budget._turn_tokens == 5000
        # Wait for new turn
        time.sleep(0.15)
        # Check new turn
        assert budget.is_new_turn()
        info = budget.record_output(100)
        assert info["turn_tokens_used"] == 100  # reset happened

    def test_budget_warning_at_80_percent(self):
        """Record up to 80% of budget, assert budget_warning returned."""
        from jcodemunch_mcp.tools.turn_budget import TurnBudget
        budget = TurnBudget(turn_gap_seconds=30.0, turn_budget_tokens=10000)
        # 80% = 8000 tokens
        info = budget.record_output(8000)
        assert "budget_warning" in info
        assert "80%" in info["budget_warning"] or "exhausted" in info["budget_warning"].lower()

    def test_budget_exhausted_message(self):
        """Record past 100%, assert warning includes 'exhausted'."""
        from jcodemunch_mcp.tools.turn_budget import TurnBudget
        budget = TurnBudget(turn_gap_seconds=30.0, turn_budget_tokens=10000)
        info = budget.record_output(12000)
        assert "budget_warning" in info
        assert "exhausted" in info["budget_warning"].lower()

    def test_should_compact_when_exhausted(self):
        """should_compact() returns True past threshold."""
        from jcodemunch_mcp.tools.turn_budget import TurnBudget
        budget = TurnBudget(turn_gap_seconds=30.0, turn_budget_tokens=10000)
        budget.record_output(9000)  # 90%
        assert budget.should_compact() is True

    def test_disabled_when_zero(self):
        """Set budget=0, verify no tracking happens."""
        from jcodemunch_mcp.tools.turn_budget import TurnBudget
        budget = TurnBudget(turn_gap_seconds=30.0, turn_budget_tokens=0)
        info = budget.record_output(5000)
        # Should return empty dict when disabled
        assert info == {}

    def test_thread_safety(self):
        """Concurrent record_output from 5 threads."""
        from jcodemunch_mcp.tools.turn_budget import TurnBudget
        budget = TurnBudget(turn_gap_seconds=30.0, turn_budget_tokens=100000)
        errors = []

        def writer(i: int):
            try:
                for j in range(100):
                    budget.record_output(100)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        # Total should be 50000 tokens
        assert budget._turn_tokens == 50000

    def test_get_turn_budget_singleton(self):
        """get_turn_budget returns singleton."""
        from jcodemunch_mcp.tools.turn_budget import get_turn_budget
        b1 = get_turn_budget()
        b2 = get_turn_budget()
        assert b1 is b2

    def test_budget_info_has_required_fields(self):
        """Budget info dict has expected fields."""
        from jcodemunch_mcp.tools.turn_budget import TurnBudget
        budget = TurnBudget(turn_gap_seconds=30.0, turn_budget_tokens=10000)
        info = budget.record_output(1000)
        assert "turn_tokens_used" in info
        assert "turn_budget_remaining" in info