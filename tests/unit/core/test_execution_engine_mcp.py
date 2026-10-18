"""D5 Execution Engine — MCP remote adapter registration and cancellation.

AC: MCP calls go through the execution engine.
AC: Cancellation leaks no owned process.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
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
from dana.core.tool.catalog import ToolCatalog, ToolCatalogEntry, ToolIdentity
from dana.core.tool.execution_engine import ToolExecutionEngine


class TestExecutionEngineMCP:
    """ToolExecutionEngine — MCP remote adapter integration."""

    @pytest.fixture
    def catalog(self):
        """Create a catalog with an MCP tool entry."""
        entry = ToolCatalogEntry(
            identity=ToolIdentity(name="filesystem:read_file", source="mcp:filesystem"),
            schema={
                "type": "function",
                "function": {
                    "name": "filesystem:read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            },
            adapter=None,
            aliases=frozenset({"read_file"}),
            cancellable=True,
            max_latency_ms=5000,
        )
        return ToolCatalog([entry])

    @pytest.fixture
    def engine(self, catalog):
        return ToolExecutionEngine(catalog)

    @pytest.fixture
    def mock_transport(self):
        transport = MagicMock()
        transport.session = AsyncMock()
        transport.call_tool = AsyncMock()
        return transport

    def test_register_remote_adapter(self, engine, mock_transport):
        """AC: Register an MCP execution adapter."""
        tracker = MCPCancellationTracker()
        adapter = MCPExecutionAdapter(mock_transport, tracker)
        engine.register_remote_adapter("filesystem", adapter)

        adapters = engine.remote_adapters
        assert "filesystem" in adapters
        assert adapters["filesystem"] is adapter

    def test_unregister_remote_adapter(self, engine, mock_transport):
        """AC: Unregister an MCP execution adapter."""
        tracker = MCPCancellationTracker()
        adapter = MCPExecutionAdapter(mock_transport, tracker)
        engine.register_remote_adapter("filesystem", adapter)
        engine.unregister_remote_adapter("filesystem")

        assert "filesystem" not in engine.remote_adapters

    def test_mcp_cancellation_tracker(self, engine):
        """AC: Engine provides an MCP cancellation tracker."""
        tracker = engine.mcp_cancellation_tracker
        assert isinstance(tracker, MCPCancellationTracker)

    @pytest.mark.asyncio
    async def test_cancel_live_mcp_call_reaches_its_wire_request(self, engine):
        incoming, reader = anyio.create_memory_object_stream(10)
        writer, outgoing = anyio.create_memory_object_stream(10)
        ready = asyncio.Event()
        requests, cancellations = [], []

        async def remote():
            async with incoming, outgoing:
                async for message in outgoing:
                    frame = message.message.root
                    if isinstance(frame, types.JSONRPCRequest):
                        requests.append(frame)
                        ready.set()
                    elif isinstance(frame, types.JSONRPCNotification):
                        cancellations.append(frame)
                        await incoming.send(
                            SessionMessage(
                                message=types.JSONRPCMessage(
                                    types.JSONRPCResponse(jsonrpc="2.0", id=requests[0].id, result={"content": []}),
                                )
                            )
                        )
                        return

        async with ClientSession(reader, correlating_write_stream(writer)) as session:
            session._tool_output_schemas["filesystem:read_file"] = None
            transport = SimpleNamespace(session=session, call_tool=lambda name, args: call_tool(session, name, args))
            tracker = MCPCancellationTracker()
            engine.register_remote_adapter("filesystem", MCPExecutionAdapter(transport, tracker))
            server = asyncio.create_task(remote())
            execution = asyncio.create_task(
                engine.execute_async(
                    {
                        "function": "filesystem:read_file",
                        "arguments": {},
                        "tool_call_id": "live-call",
                    }
                )
            )
            try:
                await asyncio.wait_for(ready.wait(), 1)
                assert "live-call" in engine._in_flight
                engine.cancel("live-call")
                result = await asyncio.wait_for(execution, 1)
                await server
                await engine.aclose()
                assert len(requests) == len(cancellations) == 1
                assert cancellations[0].method == "notifications/cancelled"
                assert cancellations[0].params["requestId"] == requests[0].id
                assert result["transport_request_id"] == requests[0].id
                assert result["tool_call_id"] == "live-call"
                assert result["success"] and "_cancelled" not in result
                assert engine._in_flight == {}
                assert not engine._cancel_tasks
                assert tracker.tracked_count == tracker.cancelled_count == 0
            finally:
                server.cancel()
                execution.cancel()
                await asyncio.gather(server, execution, return_exceptions=True)
                await engine.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_sdk", [False, True])
    async def test_cancel_then_immediate_failure_is_unknown_without_stale_tracking(self, engine, tmp_path, use_sdk):
        tracker = MCPCancellationTracker()
        path = tmp_path / "committed"
        wire_ids = []
        observed = []
        server = None

        async with AsyncExitStack() as stack:
            session = None
            if use_sdk:
                incoming, reader = anyio.create_memory_object_stream(10)
                writer, outgoing = anyio.create_memory_object_stream(10)
                session = await stack.enter_async_context(ClientSession(reader, correlating_write_stream(writer)))
                session._tool_output_schemas["filesystem:read_file"] = None

                async def remote():
                    async with incoming, outgoing:
                        async for message in outgoing:
                            frame = message.message.root
                            if isinstance(frame, types.JSONRPCRequest):
                                wire_ids.append(frame.id)
                                path.write_text("committed")
                                await incoming.send(
                                    SessionMessage(
                                        message=types.JSONRPCMessage(
                                            types.JSONRPCResponse(jsonrpc="2.0", id=frame.id, result={"content": []}),
                                        )
                                    )
                                )

                server = asyncio.create_task(remote())

            async def transport_call(name, arguments):
                if session is not None:
                    await call_tool(session, name, arguments)
                else:
                    path.write_text("committed")
                engine.cancel("immediate-failure")
                observed.append(tracker.is_cancelled("immediate-failure"))
                # No await between cancellation and failure: the notification task
                # cannot publish state on behalf of this transport.
                raise RuntimeError("connection lost after committed effect")

            adapter = MCPExecutionAdapter(SimpleNamespace(session=session, call_tool=transport_call), tracker)
            engine.register_remote_adapter("filesystem", adapter)
            try:
                result = await asyncio.wait_for(
                    engine.execute_async(
                        {
                            "function": "filesystem:read_file",
                            "arguments": {},
                            "tool_call_id": "immediate-failure",
                        }
                    ),
                    1,
                )
                await engine.aclose()
                assert observed == [True]
                assert path.read_text() == "committed"
                assert result["success"] is False
                assert result["_effect_unknown"] is True
                assert "_cancelled" not in result
                assert result["tool_call_id"] == "immediate-failure"
                if use_sdk:
                    assert wire_ids == [result["transport_request_id"]]
                assert engine._in_flight == {}
                assert not engine._cancel_tasks
                assert tracker.tracked_count == tracker.cancelled_count == 0
                assert adapter.pending_cancellation_count == 0
            finally:
                if server is not None:
                    server.cancel()
                    await asyncio.gather(server, return_exceptions=True)
                await engine.aclose()

    def test_cancel_mcp_tool_no_adapter(self, engine):
        """AC: Cancelling an MCP tool without a registered adapter logs a warning."""
        from dana.core.tool.execution_engine import _InFlight

        entry = engine._catalog.get("filesystem:read_file")
        in_flight = _InFlight("mcp-call-1", entry)
        engine._in_flight["mcp-call-1"] = in_flight

        # Should not raise — just log a warning
        engine.cancel("mcp-call-1")

        assert in_flight.is_cancelled is True

    def test_close_clears_remote_adapters(self, engine, mock_transport):
        """AC: Close clears remote adapters."""
        tracker = MCPCancellationTracker()
        adapter = MCPExecutionAdapter(mock_transport, tracker)
        engine.register_remote_adapter("filesystem", adapter)

        engine.close()

        assert engine.remote_adapters == {}
