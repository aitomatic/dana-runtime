"""MCP wire correlation: observe the SDK's own JSON-RPC request ids.

Per ADR-005 the harness tracks each in-flight call by its *harness*
``tool_call_id``, while cancellation must name the transport's own JSON-RPC
request id. The id is assigned inside the SDK, so the only honest way to learn
it is to observe the outgoing frame — never to guess the SDK's private counter.

The supported seam is the session's constructor: ``ClientSession(read_stream,
write_stream)`` takes the send stream as a parameter, so a transport wraps the
stream it passes in (:func:`correlating_write_stream`) instead of reaching into
the constructed session's private attributes after the fact. The wrapper is a
plain :class:`~anyio.abc.ObjectSendStream` delegate, so it is a drop-in stream
for the SDK and for any other MCP client code.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from anyio.abc import ObjectSendStream
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCRequest


if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from dana.core.mcp.execution import MCPExecutionAdapter


#: The MCP call currently being dispatched in this context: (adapter, harness id).
#: Context-local, so concurrent calls — including several with the same tool name
#: — never observe each other's identity.
_active_mcp_call: ContextVar[tuple[MCPExecutionAdapter, str] | None] = ContextVar("dana_active_mcp_call", default=None)


class CorrelatingSendStream(ObjectSendStream[SessionMessage]):
    """Send-stream delegate that records the SDK-assigned id of each ``tools/call``.

    Observes the frame at its shared send boundary and, when a cancellation was
    requested before the id was known, sends the ``notifications/cancelled``
    notification for the id that is now observable. Every other message (notably
    ``notifications/*``) passes through untouched.
    """

    def __init__(self, stream: ObjectSendStream[SessionMessage]) -> None:
        self._stream = stream

    async def send(self, item: SessionMessage) -> None:
        await self._stream.send(item)
        request = item.message.root
        active = _active_mcp_call.get()
        if active is not None and isinstance(request, JSONRPCRequest) and request.method == "tools/call":
            adapter, tool_call_id = active
            adapter.cancellation_tracker.record_request_id(tool_call_id, request.id)
            if adapter.cancellation_tracker.is_cancelled(tool_call_id):
                reason = adapter.pending_cancellation_reason(tool_call_id)
                await adapter.send_cancellation_notification(tool_call_id, request.id, reason)

    async def aclose(self) -> None:
        await self._stream.aclose()


def correlating_write_stream(stream: Any) -> Any:
    """Wrap ``stream`` for wire-id observation, idempotently.

    Returns ``stream`` unchanged when it is already a wrapper, so a transport that
    passes an already-wrapped stream does not nest delegates.
    """
    if isinstance(stream, CorrelatingSendStream):
        return stream
    return CorrelatingSendStream(stream)
