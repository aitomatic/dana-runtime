"""D5 MCP Execution Adapter — remote execution through the engine.

AC: Exactly one terminal fact per call.
AC: Cancellation leaks no owned process.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anyio
from mcp import ClientSession, types
from mcp.shared.message import SessionMessage
import pytest

from dana.core.mcp.cancellation import MCPCancellationTracker
from dana.core.mcp.correlation import correlating_write_stream
from dana.core.mcp.execution import MCPExecutionAdapter
from dana.core.mcp.protocol import call_tool


pytestmark = pytest.mark.asyncio


class TestMCPExecutionAdapter:
    """MCPExecutionAdapter — call_tool, cancellation, error handling."""

    @pytest.fixture
    def mock_transport(self):
        """Create a mock MCP transport."""
        transport = MagicMock()
        transport.session = AsyncMock()
        transport.call_tool = AsyncMock()
        return transport

    @pytest.fixture
    def cancellation_tracker(self):
        return MCPCancellationTracker()

    @pytest.fixture
    def adapter(self, mock_transport, cancellation_tracker):
        return MCPExecutionAdapter(mock_transport, cancellation_tracker)

    async def test_call_tool_success(self, adapter, mock_transport):
        """AC: Successful call returns result with success=True."""
        mock_result = MagicMock()
        mock_result.content = [{"type": "text", "text": "Hello, world!"}]
        mock_result.isError = False
        mock_transport.call_tool.return_value = mock_result

        result = await adapter.call_tool("greet", {"name": "World"})

        assert result["success"] is True
        assert result["result"] == mock_result.content
        assert "_cancelled" not in result

    async def test_call_tool_error(self, adapter, mock_transport):
        """AC: Error call returns result with success=False."""
        mock_result = MagicMock()
        mock_result.content = [{"type": "text", "text": "Error: something went wrong"}]
        mock_result.isError = True
        mock_transport.call_tool.return_value = mock_result

        result = await adapter.call_tool("failing_tool", {})

        assert result["success"] is False
        assert "_cancelled" not in result

    async def test_call_tool_exception(self, adapter, mock_transport):
        """AC: Exception during call returns error result."""
        mock_transport.call_tool.side_effect = RuntimeError("Connection lost")

        result = await adapter.call_tool("broken_tool", {})

        assert result["success"] is False
        assert "Error calling MCP tool" in result["result"]
        assert "_cancelled" not in result

    async def test_successful_response_is_not_a_cancellation_ack(self, adapter, mock_transport, cancellation_tracker):
        """A cancellation request does not erase the remote's completed effect."""
        mock_result = MagicMock()
        mock_result.content = [{"type": "text", "text": "partial result"}]
        mock_result.isError = False
        mock_transport.call_tool.return_value = mock_result

        # Request cancellation before the call completes
        cancellation_tracker.request_cancellation("call-1")

        result = await adapter.call_tool("slow_tool", {}, tool_call_id="call-1")

        assert result["success"] is True
        assert "_cancelled" not in result
        assert "_effect_unknown" not in result

    async def test_call_tool_cancelled_with_error_is_effect_unknown(self, adapter, mock_transport, cancellation_tracker):
        """AC: A failure raised while cancellation is pending is NOT an acknowledgement.

        The remote raised instead of replying, so there is no confirmation that it
        stopped before committing its effect. The truthful disposition is
        effect-unknown (ADR-011: never auto-retry), not ``_cancelled``.
        """
        mock_transport.call_tool.side_effect = RuntimeError("Cancelled by user")

        cancellation_tracker.request_cancellation("call-2")

        result = await adapter.call_tool("failing_tool", {}, tool_call_id="call-2")

        assert result["success"] is False
        assert result["_effect_unknown"] is True
        assert "_cancelled" not in result

    async def test_call_tool_tracks_and_forgets(self, adapter, mock_transport, cancellation_tracker):
        """AC: Exactly one terminal fact — call is tracked then forgotten."""
        mock_result = MagicMock()
        mock_result.content = []
        mock_result.isError = False
        mock_transport.call_tool.return_value = mock_result

        assert cancellation_tracker.tracked_count == 0

        await adapter.call_tool("some_tool", {}, tool_call_id="call-3")

        # After completion, the call should be forgotten
        assert cancellation_tracker.tracked_count == 0
        assert not cancellation_tracker.is_tracked("call-3")

    async def test_send_cancellation_notification_no_session(self, cancellation_tracker):
        """AC: Cancellation notification without session is still tracked."""
        transport = MagicMock()
        transport.session = None
        adapter = MCPExecutionAdapter(transport, cancellation_tracker)

        await adapter.send_cancellation_notification("call-5")

        assert cancellation_tracker.is_cancelled("call-5")

    async def test_call_tool_default_tool_call_id(self, adapter, mock_transport):
        """AC: Default tool_call_id is generated when not provided."""
        mock_result = MagicMock()
        mock_result.content = []
        mock_result.isError = False
        mock_transport.call_tool.return_value = mock_result

        first, second = await asyncio.gather(adapter.call_tool("my_tool", {}), adapter.call_tool("my_tool", {}))

        assert first["tool_call_id"].startswith("mcp:my_tool:")
        assert second["tool_call_id"].startswith("mcp:my_tool:")
        assert first["tool_call_id"] != second["tool_call_id"]

    async def test_properties(self, adapter, mock_transport, cancellation_tracker):
        """AC: Properties return the correct values."""
        assert adapter.transport is mock_transport
        assert adapter.cancellation_tracker is cancellation_tracker


@asynccontextmanager
async def _memory_mcp():
    incoming, reader = anyio.create_memory_object_stream(10)
    writer, outgoing = anyio.create_memory_object_stream(10)
    tracker = MCPCancellationTracker()
    async with ClientSession(reader, correlating_write_stream(writer)) as session:
        session._tool_output_schemas["same"] = None
        transport = SimpleNamespace(session=session, call_tool=lambda name, args: call_tool(session, name, args))
        yield session, MCPExecutionAdapter(transport, tracker), tracker, incoming, outgoing


class TestMCPWireIdentity:
    async def test_same_name_calls_record_real_ids_and_cancel_independently(self):
        async with _memory_mcp() as (session, adapter, tracker, incoming, outgoing):
            ready = asyncio.Event()
            requests, cancellations = [], []

            async def remote():
                async with incoming, outgoing:
                    async for message in outgoing:
                        frame = message.message.root
                        if isinstance(frame, types.JSONRPCRequest):
                            if frame.method == "ping":
                                await incoming.send(
                                    SessionMessage(
                                        message=types.JSONRPCMessage(
                                            types.JSONRPCResponse(jsonrpc="2.0", id=frame.id, result={}),
                                        )
                                    )
                                )
                                continue
                            assert frame.method == "tools/call"
                            requests.append(frame)
                            if len(requests) == 2:
                                ready.set()
                        elif isinstance(frame, types.JSONRPCNotification):
                            cancellations.append(frame)
                            for request in reversed(requests):
                                await incoming.send(
                                    SessionMessage(
                                        message=types.JSONRPCMessage(
                                            types.JSONRPCResponse(
                                                jsonrpc="2.0",
                                                id=request.id,
                                                result={
                                                    "content": [{"type": "text", "text": request.params["arguments"]["slot"]}],
                                                },
                                            ),
                                        )
                                    )
                                )
                            return

            server = asyncio.create_task(remote())
            calls = []
            try:
                await asyncio.wait_for(session.send_ping(), 1)
                calls = [asyncio.create_task(adapter.call_tool("same", {"slot": slot}, f"harness-{slot}")) for slot in ["first", "second"]]
                await asyncio.wait_for(ready.wait(), 1)
                wire_ids = {frame.params["arguments"]["slot"]: frame.id for frame in requests}
                assert wire_ids == {"first": 1, "second": 2}
                assert tracker.tracked_count == 2
                assert tracker.request_id_for("harness-first") == wire_ids["first"]
                assert tracker.request_id_for("harness-second") == wire_ids["second"]
                await adapter.send_cancellation_notification("harness-first")
                results = await asyncio.wait_for(asyncio.gather(*calls), 1)
                await server
                assert [result["transport_request_id"] for result in results] == [1, 2]
                assert [result["tool_call_id"] for result in results] == ["harness-first", "harness-second"]
                assert [result["result"][0].text for result in results] == ["first", "second"]
                assert len(cancellations) == 1
                assert cancellations[0].method == "notifications/cancelled"
                assert cancellations[0].params["requestId"] == wire_ids["first"]
                assert all(result["success"] and "_cancelled" not in result for result in results)
                assert tracker.tracked_count == tracker.cancelled_count == 0
            finally:
                for task in [server, *calls]:
                    task.cancel()
                await asyncio.gather(server, *calls, return_exceptions=True)

    async def test_cancellation_before_dispatch_waits_for_observed_wire_id(self):
        async with _memory_mcp() as (_, adapter, tracker, incoming, outgoing):
            frames = []

            async def remote():
                async with incoming, outgoing:
                    async for message in outgoing:
                        frame = message.message.root
                        frames.append(frame)
                        if isinstance(frame, types.JSONRPCNotification):
                            request = frames[0]
                            await incoming.send(
                                SessionMessage(
                                    message=types.JSONRPCMessage(
                                        types.JSONRPCResponse(jsonrpc="2.0", id=request.id, result={"content": []}),
                                    )
                                )
                            )
                            return

            await adapter.send_cancellation_notification("early")
            server = asyncio.create_task(remote())
            try:
                result = await asyncio.wait_for(adapter.call_tool("same", {}, "early"), 1)
                await server
                assert len(frames) == 2
                assert isinstance(frames[0], types.JSONRPCRequest)
                assert frames[0].method == "tools/call"
                assert frames[1].params["requestId"] == frames[0].id
                assert result["transport_request_id"] == frames[0].id
                assert "_cancelled" not in result
                assert tracker.tracked_count == tracker.cancelled_count == 0
            finally:
                server.cancel()
                await asyncio.gather(server, return_exceptions=True)
