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

Wire-id correlation lives in :mod:`dana.core.mcp.correlation`: the transports
wrap their own send stream (a supported constructor argument of ``ClientSession``)
so this adapter never reaches into the SDK session's private attributes.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from mcp.types import CancelledNotification, CancelledNotificationParams

from dana.core.mcp.cancellation import MCPCancellationTracker
from dana.core.mcp.correlation import _active_mcp_call


logger = logging.getLogger(__name__)


class MCPExecutionAdapter:
    """Adapter that wraps an MCP transport for execution engine integration.

    One instance per MCP server connection. Provides a callable that the
    execution engine invokes as a cooperative tool adapter.

    The adapter integrates with ``MCPCancellationTracker``: when cancellation
    is requested, the adapter sends the MCP ``notifications/cancelled``
    notification to the remote server. A normal response still reports its actual
    outcome; requesting cancellation alone is never an acknowledgement.

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
        self._pending_cancellations: dict[str, str] = {}

    @property
    def transport(self) -> Any:
        """The underlying MCP transport."""
        return self._transport

    @property
    def cancellation_tracker(self) -> MCPCancellationTracker:
        """The cancellation tracker for this adapter."""
        return self._cancellation_tracker

    def pending_cancellation_reason(self, tool_call_id: str) -> str:
        """Reason recorded for a cancellation requested before the wire id was known.

        Consumed once (popped): the notification that carries it is sent exactly
        once, so the reason must not linger for a later call that reuses the id.
        """
        return self._pending_cancellations.pop(tool_call_id, "User requested cancellation")

    @property
    def pending_cancellation_count(self) -> int:
        """Cancellations still waiting for their wire id to be observed.

        Observability only: a non-zero count after every call settled means a
        cancellation never reached the wire (no id was ever observed for it).
        """
        return len(self._pending_cancellations)

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
            A result dict with ``success``, ``result``, the harness ``tool_call_id``,
            and the observed ``transport_request_id`` when available.
        """
        if tool_call_id is None:
            tool_call_id = f"mcp:{name}:{uuid4().hex}"

        # Track the call for cancellation BEFORE any await so a cancellation that
        # arrives during the call is observed (the tracker is keyed by the harness
        # tool_call_id, never by the tool name — concurrent same-name calls must
        # stay distinct).
        self._cancellation_tracker.track(tool_call_id, f"pending:{tool_call_id}")
        token = _active_mcp_call.set((self, tool_call_id))

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

            # Record the transport-level request id this call actually used, so the
            # harness id (tool_call_id) and the JSON-RPC id remain distinguishable.
            transport_request_id = self._cancellation_tracker.request_id_for(tool_call_id)
            if transport_request_id is not None:
                result["transport_request_id"] = transport_request_id

            return result
        except Exception as exc:
            error_result: dict[str, Any] = {
                "success": False,
                "result": f"Error calling MCP tool '{name}': {exc}",
                "tool_call_id": tool_call_id,
            }
            transport_request_id = self._cancellation_tracker.request_id_for(tool_call_id)
            if transport_request_id is not None:
                error_result["transport_request_id"] = transport_request_id
            if self._cancellation_tracker.is_cancelled(tool_call_id):
                # The call raised while a cancellation was pending: the remote
                # system may or may not have committed the effect, and no response
                # arrived to acknowledge it. Report the truthful disposition — the
                # terminal state is unknown, never silently "cancelled".
                error_result["_effect_unknown"] = True
            return error_result
        finally:
            _active_mcp_call.reset(token)
            self._pending_cancellations.pop(tool_call_id, None)
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
            tool_call_id: The harness tool_call_id for the call to cancel.
            request_id: The MCP JSON-RPC request id to cancel. When omitted, the
                id recorded from the outgoing request is used. A cancellation
                requested before dispatch waits for that real ID to be observed.
            reason: Human-readable reason for cancellation.
        """
        self._cancellation_tracker.request_cancellation(tool_call_id)

        # Never substitute a harness ID or tool name for a JSON-RPC request ID.
        recorded = self._cancellation_tracker.request_id_for(tool_call_id)
        effective_request_id = request_id if request_id is not None else recorded
        if effective_request_id is None or (isinstance(effective_request_id, str) and effective_request_id.startswith("pending:")):
            self._pending_cancellations[tool_call_id] = reason
            return

        # Try to send the MCP cancellation notification if the session supports it
        session = getattr(self._transport, "session", None)
        if session is not None:
            try:
                notification = CancelledNotification(
                    method="notifications/cancelled",
                    params=CancelledNotificationParams(
                        requestId=effective_request_id,
                        reason=reason,
                    ),
                )
                await session.send_notification(notification)
                logger.info("Cancellation notification sent for '%s'", tool_call_id)
            except Exception as exc:
                logger.warning("Cancellation notification failed for '%s': %s", tool_call_id, exc)

    async def call_tool_from_dict(
        self,
        tool_call: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any]:
        """Call an MCP tool from a tool_call dict (engine integration).

        Extracts the function name and arguments from the tool_call dict
        and delegates to ``call_tool``.

        Args:
            tool_call: The tool call dict with ``function`` and ``arguments``.
            tool_call_id: The tool_call_id for cancellation tracking.

        Returns:
            A result dict from ``call_tool``.
        """
        name = tool_call.get("function", "")
        arguments = tool_call.get("arguments", {})
        return await self.call_tool(name, arguments, tool_call_id)
