"""MCP Remote Execution Adapter — MCP tool calls through the execution engine.

Per ADR-005 (Cancellation-First Tool Execution Engine):
- MCP calls go through the execution engine.
- Remote adapter — cancellation is terminal only after the remote system
  acknowledges it; otherwise effect disposition remains unknown.
- Process-group cleanup for stdio children.
- Exactly one terminal fact.

Per ADR-008 (Session MCP Leases):
- Every MCP call uses normal policy + execution + cancellation + journal paths.
- No special-casing.

Per ADR-006 (Effect-Based Operations and Permission Policy):
- MCP calls produce Operations and go through policy enforcement like any tool.

This adapter wraps an MCP transport session and provides a callable that the
execution engine can invoke as a cooperative tool adapter. It integrates with
the MCPCancellationTracker for proper cancellation acknowledgement.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.types import CancelledNotification, CancelledNotificationParams

from dana.core.mcp.cancellation import MCPCancellationTracker


logger = logging.getLogger(__name__)


class MCPExecutionAdapter:
    """Adapter that wraps an MCP transport for execution engine integration.

    One instance per MCP server connection. Provides a callable that the
    execution engine invokes as a cooperative tool adapter.

    The adapter integrates with ``MCPCancellationTracker``: when cancellation
    is requested, the adapter sends the MCP ``notifications/cancelled``
    notification to the remote server. The result is only considered terminal
    after the remote system acknowledges it (response arrives).

    Usage::

        adapter = MCPExecutionAdapter(transport, cancellation_tracker)
        result = await adapter.call_tool("read_file", {"path": "/tmp/test.txt"})
    """

    def __init__(
        self,
        transport: Any,
        cancellation_tracker: MCPCancellationTracker,
    ) -> None:
        self._transport = transport
        self._cancellation_tracker = cancellation_tracker

    @property
    def transport(self) -> Any:
        """The underlying MCP transport."""
        return self._transport

    @property
    def cancellation_tracker(self) -> MCPCancellationTracker:
        """The cancellation tracker for this adapter."""
        return self._cancellation_tracker

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        """Call an MCP tool through the transport.

        This is the primary execution path. It:
        1. Tracks the call in the cancellation tracker.
        2. Calls the tool via the transport.
        3. Checks if cancellation was requested.
        4. Returns the result with cancellation metadata.

        Args:
            name: The MCP tool name.
            arguments: Optional arguments.
            tool_call_id: Optional tool_call_id for cancellation tracking.
                If not provided, a default is generated.

        Returns:
            A result dict with ``success``, ``result``, and optionally
            ``_cancelled`` if cancellation was requested.
        """
        if tool_call_id is None:
            tool_call_id = f"mcp:{name}"

        # Track the call for cancellation
        self._cancellation_tracker.track(tool_call_id, name)

        try:
            # Call the tool via the transport
            raw_result = await self._transport.call_tool(name, arguments)

            # Build the result dict
            if hasattr(raw_result, "content"):
                content = raw_result.content
                is_error = getattr(raw_result, "isError", False)
            else:
                content = raw_result
                is_error = False

            result: dict[str, Any] = {
                "success": not is_error,
                "result": content,
                "tool_call_id": tool_call_id,
            }

            # Check if cancellation was requested during execution
            if self._cancellation_tracker.is_cancelled(tool_call_id):
                # Per ADR-005: cancellation is terminal only after the remote
                # system acknowledges it. Since the response arrived, the
                # remote has acknowledged it.
                result["_cancelled"] = True
                logger.info("MCP call cancelled (acknowledged): '%s' tool='%s'", tool_call_id, name)

            return result
        except Exception as exc:
            error_result: dict[str, Any] = {
                "success": False,
                "result": f"Error calling MCP tool '{name}': {exc}",
                "tool_call_id": tool_call_id,
            }
            if self._cancellation_tracker.is_cancelled(tool_call_id):
                error_result["_cancelled"] = True
            return error_result
        finally:
            self._cancellation_tracker.forget(tool_call_id)

    async def send_cancellation_notification(
        self,
        tool_call_id: str,
        request_id: int | str | None = None,
        reason: str = "User requested cancellation",
    ) -> None:
        """Send an MCP ``notifications/cancelled`` notification.

        Per ADR-005: cancellation is terminal only after the remote system
        acknowledges it. This sends the notification; the response will
        carry the acknowledgement.

        Args:
            tool_call_id: The tool_call_id to cancel.
            request_id: The MCP request ID to cancel (if known).
            reason: Human-readable reason for cancellation.
        """
        self._cancellation_tracker.request_cancellation(tool_call_id)

        # Try to send the MCP cancellation notification if the session supports it
        session = getattr(self._transport, "session", None)
        if session is not None:
            try:
                notification = CancelledNotification(
                    method="notifications/cancelled",
                    params=CancelledNotificationParams(
                        requestId=request_id if request_id is not None else tool_call_id,
                        reason=reason,
                    ),
                )
                await session.send_notification(notification)
                logger.info("Cancellation notification sent for '%s'", tool_call_id)
            except Exception as exc:
                logger.warning("Cancellation notification failed for '%s': %s", tool_call_id, exc)
