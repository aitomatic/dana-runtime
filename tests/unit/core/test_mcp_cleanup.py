"""D5 MCP Cleanup — clean close, stdio child reaping, and session teardown.

AC: stdio children reaped (no leaks).
AC: Clean close.
AC: Cancellation leaks no owned process.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dana.core.mcp.cleanup import MCPCleanupHandler


class TestMCPCleanupHandler:
    """MCPCleanupHandler — register, close, reap, assert_no_leaks."""

    @pytest.fixture
    def handler(self):
        return MCPCleanupHandler()

    def test_register_transport(self, handler):
        """AC: Register a transport for cleanup tracking."""
        transport = MagicMock()
        handler.register_transport("filesystem", transport)
        assert handler.transport_count == 1
        assert "filesystem" in handler.transports

    def test_register_child_pid(self, handler):
        """AC: Register a child PID for reaping."""
        handler.register_child_pid("filesystem", 12345)
        assert handler.child_pid_count == 1

    def test_unregister_transport(self, handler):
        """AC: Unregister a transport removes it from tracking."""
        transport = MagicMock()
        handler.register_transport("filesystem", transport)
        handler.register_child_pid("filesystem", 12345)
        handler.unregister_transport("filesystem")
        assert handler.transport_count == 0
        assert handler.child_pid_count == 0

    @pytest.mark.asyncio
    async def test_close_transport(self, handler):
        """AC: Close a single transport."""
        transport = MagicMock()
        transport.close = AsyncMock()
        handler.register_transport("filesystem", transport)
        handler.register_child_pid("filesystem", 12345)

        await handler.close_transport("filesystem")

        transport.close.assert_awaited_once()
        assert handler.transport_count == 0

    @pytest.mark.asyncio
    async def test_close_transport_no_close_method(self, handler):
        """AC: Close a transport without a close method is a no-op."""
        transport = MagicMock(spec=[])  # No close method
        handler.register_transport("filesystem", transport)

        await handler.close_transport("filesystem")  # Should not raise

        assert handler.transport_count == 0

    @pytest.mark.asyncio
    async def test_close_all(self, handler):
        """AC: Close all registered transports."""
        transport_a = MagicMock()
        transport_a.close = AsyncMock()
        transport_b = MagicMock()
        transport_b.close = AsyncMock()

        handler.register_transport("server-a", transport_a)
        handler.register_transport("server-b", transport_b)
        handler.register_child_pid("server-a", 12345)
        handler.register_child_pid("server-b", 67890)

        await handler.close_all()

        transport_a.close.assert_awaited_once()
        transport_b.close.assert_awaited_once()
        assert handler.transport_count == 0
        assert handler.child_pid_count == 0

    @pytest.mark.asyncio
    async def test_close_all_with_error(self, handler):
        """AC: Close all continues even if one transport fails."""
        transport_a = MagicMock()
        transport_a.close = AsyncMock(side_effect=RuntimeError("Close failed"))
        transport_b = MagicMock()
        transport_b.close = AsyncMock()

        handler.register_transport("server-a", transport_a)
        handler.register_transport("server-b", transport_b)

        await handler.close_all()  # Should not raise

        transport_b.close.assert_awaited_once()

    def test_assert_no_leaks_clean(self, handler):
        """AC: No leaks when all child PIDs have exited."""
        with patch("os.kill") as mock_kill:
            mock_kill.side_effect = ProcessLookupError  # Process already exited
            handler.register_child_pid("filesystem", 99999)
            handler.assert_no_leaks()  # Should not raise

    def test_assert_no_leaks_detected(self, handler):
        """AC: Leak detection raises RuntimeError."""
        with patch("os.kill") as mock_kill:
            mock_kill.return_value = None  # Process exists
            handler.register_child_pid("filesystem", 99999)
            with pytest.raises(RuntimeError, match="Subprocess leak detected"):
                handler.assert_no_leaks()

    def test_assert_no_leaks_empty(self, handler):
        """AC: No leaks when no child PIDs registered."""
        handler.assert_no_leaks()  # Should not raise

    def test_reap_child_pids(self, handler):
        """AC: Reap child PIDs sends SIGTERM."""
        with patch("os.kill") as mock_kill:
            handler.register_child_pid("filesystem", 12345)
            handler._reap_child_pids("filesystem")
            mock_kill.assert_called_once_with(12345, 15)  # SIGTERM = 15

    def test_reap_child_pids_already_exited(self, handler):
        """AC: Reap child PIDs handles already-exited processes."""
        with patch("os.kill") as mock_kill:
            mock_kill.side_effect = ProcessLookupError
            handler.register_child_pid("filesystem", 12345)
            handler._reap_child_pids("filesystem")  # Should not raise

    def test_register_multiple_pids(self, handler):
        """AC: Register multiple child PIDs for the same server."""
        handler.register_child_pid("filesystem", 12345)
        handler.register_child_pid("filesystem", 67890)
        assert handler.child_pid_count == 2

    def test_transport_count(self, handler):
        """AC: Transport count reflects registered transports."""
        assert handler.transport_count == 0
        handler.register_transport("a", MagicMock())
        assert handler.transport_count == 1
        handler.register_transport("b", MagicMock())
        assert handler.transport_count == 2
