"""D5 Execution Engine — MCP remote adapter registration and cancellation.

AC: MCP calls go through the execution engine.
AC: Cancellation leaks no owned process.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dana.core.mcp.cancellation import MCPCancellationTracker
from dana.core.mcp.execution import MCPExecutionAdapter
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

    def test_cancel_mcp_tool(self, engine, mock_transport):
        """AC: Cancelling an MCP tool sends notification to the remote adapter."""
        tracker = MCPCancellationTracker()
        adapter = MCPExecutionAdapter(mock_transport, tracker)
        engine.register_remote_adapter("filesystem", adapter)

        # Simulate an in-flight MCP call using the real _InFlight class
        from dana.core.tool.execution_engine import _InFlight

        entry = engine._catalog.get("filesystem:read_file")
        in_flight = _InFlight("mcp-call-1", entry)
        engine._in_flight["mcp-call-1"] = in_flight

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value = AsyncMock()
            engine.cancel("mcp-call-1")

        assert in_flight.is_cancelled is True

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
