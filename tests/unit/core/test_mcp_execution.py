"""D5 MCP Execution Adapter — remote execution through the engine.

AC: Exactly one terminal fact per call.
AC: Cancellation leaks no owned process.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dana.core.mcp.cancellation import MCPCancellationTracker
from dana.core.mcp.execution import MCPExecutionAdapter


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

    async def test_call_tool_cancelled_acknowledged(self, adapter, mock_transport, cancellation_tracker):
        """AC: Cancellation requested before response arrives is acknowledged."""
        mock_result = MagicMock()
        mock_result.content = [{"type": "text", "text": "partial result"}]
        mock_result.isError = False
        mock_transport.call_tool.return_value = mock_result

        # Request cancellation before the call completes
        cancellation_tracker.request_cancellation("call-1")

        result = await adapter.call_tool("slow_tool", {}, tool_call_id="call-1")

        assert result["success"] is True
        assert result["_cancelled"] is True

    async def test_call_tool_cancelled_with_error(self, adapter, mock_transport, cancellation_tracker):
        """AC: Cancellation with error response is acknowledged."""
        mock_transport.call_tool.side_effect = RuntimeError("Cancelled by user")

        cancellation_tracker.request_cancellation("call-2")

        result = await adapter.call_tool("failing_tool", {}, tool_call_id="call-2")

        assert result["success"] is False
        assert result["_cancelled"] is True

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

    async def test_send_cancellation_notification(self, adapter, mock_transport, cancellation_tracker):
        """AC: Sending cancellation notification marks the call as cancelled."""
        await adapter.send_cancellation_notification("call-4", request_id=42)

        assert cancellation_tracker.is_cancelled("call-4")

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

        result = await adapter.call_tool("my_tool", {})

        assert result["tool_call_id"] == "mcp:my_tool"

    async def test_properties(self, adapter, mock_transport, cancellation_tracker):
        """AC: Properties return the correct values."""
        assert adapter.transport is mock_transport
        assert adapter.cancellation_tracker is cancellation_tracker
