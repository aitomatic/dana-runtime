"""D5 MCP Cancellation Tracker — remote cancellation acknowledgement.

AC: Cancellation leaks no owned process.
AC: Exactly one terminal fact per call.
"""

from __future__ import annotations

from dana.core.mcp.cancellation import MCPCancellationTracker


class TestMCPCancellationTracker:
    """MCPCancellationTracker — track, cancel, forget."""

    def test_track_and_forget(self):
        """Track a call, then forget it."""
        tracker = MCPCancellationTracker()
        tracker.track("call-1", 42)
        assert tracker.is_tracked("call-1")
        assert tracker.tracked_count == 1

        tracker.forget("call-1")
        assert not tracker.is_tracked("call-1")
        assert tracker.tracked_count == 0

    def test_request_cancellation(self):
        """Request cancellation marks the call as cancelled."""
        tracker = MCPCancellationTracker()
        tracker.track("call-1", 42)
        tracker.request_cancellation("call-1")

        assert tracker.is_cancelled("call-1")
        assert tracker.cancelled_count == 1

    def test_forget_clears_cancelled(self):
        """Forgetting a cancelled call clears both tracked and cancelled state."""
        tracker = MCPCancellationTracker()
        tracker.track("call-1", 42)
        tracker.request_cancellation("call-1")
        tracker.forget("call-1")

        assert not tracker.is_cancelled("call-1")
        assert not tracker.is_tracked("call-1")
        assert tracker.tracked_count == 0
        assert tracker.cancelled_count == 0

    def test_is_cancelled_without_request(self):
        """A tracked call that was not cancelled returns False."""
        tracker = MCPCancellationTracker()
        tracker.track("call-1", 42)
        assert not tracker.is_cancelled("call-1")

    def test_is_tracked_returns_true_for_cancelled(self):
        """A cancelled call is still tracked until forgotten."""
        tracker = MCPCancellationTracker()
        tracker.track("call-1", 42)
        tracker.request_cancellation("call-1")
        assert tracker.is_tracked("call-1")

    def test_multiple_calls(self):
        """Track multiple calls independently."""
        tracker = MCPCancellationTracker()
        tracker.track("call-1", 42)
        tracker.track("call-2", 43)
        tracker.track("call-3", 44)

        assert tracker.tracked_count == 3

        tracker.request_cancellation("call-2")
        assert tracker.cancelled_count == 1
        assert not tracker.is_cancelled("call-1")
        assert tracker.is_cancelled("call-2")
        assert not tracker.is_cancelled("call-3")

        tracker.forget("call-2")
        assert tracker.tracked_count == 2
        assert tracker.cancelled_count == 0

    def test_forget_unknown_is_noop(self):
        """Forgetting an unknown call is a no-op."""
        tracker = MCPCancellationTracker()
        tracker.forget("unknown-call")  # Should not raise
        assert tracker.tracked_count == 0

    def test_is_cancelled_unknown(self):
        """is_cancelled on an unknown call returns False."""
        tracker = MCPCancellationTracker()
        assert not tracker.is_cancelled("unknown-call")

    def test_is_tracked_unknown(self):
        """is_tracked on an unknown call returns False."""
        tracker = MCPCancellationTracker()
        assert not tracker.is_tracked("unknown-call")

    def test_track_with_string_request_id(self):
        """Track with a string MCP request ID."""
        tracker = MCPCancellationTracker()
        tracker.track("call-1", "req-abc-123")
        assert tracker.is_tracked("call-1")
        tracker.forget("call-1")
        assert not tracker.is_tracked("call-1")
