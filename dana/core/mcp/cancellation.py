"""MCP Cancellation Tracker — remote cancellation acknowledgement.

Per ADR-005 (Cancellation-First Tool Execution Engine):
- Cancellation is terminal only after the remote system acknowledges it.
- Otherwise effect disposition remains unknown.
- Exactly one terminal fact per call.

The tracker maintains the cancellation state for in-flight MCP tool calls.
When cancellation is requested, the tracker records the request. When the
response arrives, the caller checks if cancellation was requested and
reports the appropriate terminal fact.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


class MCPCancellationTracker:
    """Tracks cancellation state for MCP tool calls.

    Maintains two sets of state:
    - ``_tracked``: mapping of tool_call_id -> MCP request ID (for active calls)
    - ``_cancelled``: set of tool_call_ids that have been requested for cancellation

    Usage::

        tracker = MCPCancellationTracker()
        tracker.track("call-1", 42)
        tracker.request_cancellation("call-1")
        assert tracker.is_cancelled("call-1")
        tracker.forget("call-1")
    """

    def __init__(self) -> None:
        self._tracked: dict[str, int | str] = {}
        self._cancelled: set[str] = set()

    def track(self, tool_call_id: str, mcp_request_id: int | str) -> None:
        """Track a new in-flight MCP call.

        Args:
            tool_call_id: The tool_call_id from the request.
            mcp_request_id: The MCP protocol request ID.
        """
        self._tracked[tool_call_id] = mcp_request_id

    def request_cancellation(self, tool_call_id: str) -> None:
        """Record a cancellation request for a tool call.

        The call is marked as cancelled. When the response arrives, the
        caller should check ``is_cancelled()`` and report the terminal fact.
        """
        self._cancelled.add(tool_call_id)
        logger.info("Cancellation requested for '%s'", tool_call_id)

    def is_cancelled(self, tool_call_id: str) -> bool:
        """Check if cancellation was requested for a tool call.

        Returns:
            True if cancellation was requested, False otherwise.
        """
        return tool_call_id in self._cancelled

    def is_tracked(self, tool_call_id: str) -> bool:
        """Check if a tool call is being tracked.

        Returns:
            True if the call is tracked (in-flight or cancelled).
        """
        return tool_call_id in self._tracked or tool_call_id in self._cancelled

    def forget(self, tool_call_id: str) -> None:
        """Stop tracking a tool call (after it completes).

        Removes the call from both tracked and cancelled state.
        """
        self._tracked.pop(tool_call_id, None)
        self._cancelled.discard(tool_call_id)

    @property
    def tracked_count(self) -> int:
        """Number of currently tracked in-flight calls."""
        return len(self._tracked)

    @property
    def cancelled_count(self) -> int:
        """Number of calls that have been requested for cancellation."""
        return len(self._cancelled)
