"""D5 MCP Session Restore — required/optional lease restore on session load.

AC: Optional failure degrades; required failure stops preflight.
"""

from __future__ import annotations

import pytest

from dana.core.mcp.leases import LeaseState, MCPLease, MCPLeaseManager
from dana.core.mcp.restore import MCPRestoreHandler


pytestmark = pytest.mark.asyncio


class TestMCPRestoreHandler:
    """MCPRestoreHandler — restore leases, preflight failures, degraded leases."""

    @pytest.fixture
    def lease_manager(self):
        return MCPLeaseManager()

    @pytest.fixture
    def handler(self, lease_manager):
        return MCPRestoreHandler(lease_manager)

    async def test_restore_leases_no_reconnect(self, handler, lease_manager):
        """AC: Restore leases without reconnection keeps persisted state."""
        leases = [
            MCPLease(server_name="server-a", required=True, state=LeaseState.ACTIVE),
            MCPLease(server_name="server-b", required=False, state=LeaseState.ACTIVE),
        ]

        restored = await handler.restore_leases(leases)

        assert len(restored) == 2
        assert lease_manager.get_lease("server-a") is not None
        assert lease_manager.get_lease("server-b") is not None

    async def test_restore_leases_with_reconnect_success(self, handler, lease_manager):
        """AC: Successful reconnection reactivates the lease."""
        leases = [
            MCPLease(server_name="server-a", required=True, state=LeaseState.FAILED),
        ]

        async def reconnect_fn(lease):
            return True

        restored = await handler.restore_leases(leases, reconnect_fn=reconnect_fn)

        assert len(restored) == 1
        lease = lease_manager.get_lease("server-a")
        assert lease is not None
        assert lease.state == LeaseState.ACTIVE

    async def test_restore_leases_reconnect_failure_required(self, handler, lease_manager):
        """AC: Required lease reconnect failure keeps lease FAILED (preflight stops)."""
        leases = [
            MCPLease(server_name="server-a", required=True, state=LeaseState.FAILED),
        ]

        async def reconnect_fn(lease):
            return False

        restored = await handler.restore_leases(leases, reconnect_fn=reconnect_fn)

        assert len(restored) == 1
        lease = lease_manager.get_lease("server-a")
        assert lease is not None
        assert lease.state == LeaseState.FAILED

        # Preflight should catch this
        failures = handler.get_preflight_failures()
        assert len(failures) == 1
        assert failures[0].server_name == "server-a"

    async def test_restore_leases_reconnect_failure_optional(self, handler, lease_manager):
        """AC: Optional lease reconnect failure degrades (session continues)."""
        leases = [
            MCPLease(server_name="server-b", required=False, state=LeaseState.ACTIVE),
        ]

        async def reconnect_fn(lease):
            return False

        restored = await handler.restore_leases(leases, reconnect_fn=reconnect_fn)

        assert len(restored) == 1
        lease = lease_manager.get_lease("server-b")
        assert lease is not None
        assert lease.state == LeaseState.DEGRADED

        # Preflight should NOT catch this (it's optional)
        failures = handler.get_preflight_failures()
        assert len(failures) == 0

        # But degraded leases should be reported
        degraded = handler.get_degraded_leases()
        assert len(degraded) == 1
        assert degraded[0].server_name == "server-b"

    async def test_restore_leases_reconnect_exception_required(self, handler, lease_manager):
        """AC: Required lease reconnect exception keeps lease FAILED."""
        leases = [
            MCPLease(server_name="server-a", required=True, state=LeaseState.ACTIVE),
        ]

        async def reconnect_fn(lease):
            raise RuntimeError("Connection refused")

        restored = await handler.restore_leases(leases, reconnect_fn=reconnect_fn)

        assert len(restored) == 1
        lease = lease_manager.get_lease("server-a")
        assert lease is not None
        assert lease.state == LeaseState.FAILED

    async def test_restore_leases_reconnect_exception_optional(self, handler, lease_manager):
        """AC: Optional lease reconnect exception degrades."""
        leases = [
            MCPLease(server_name="server-b", required=False, state=LeaseState.ACTIVE),
        ]

        async def reconnect_fn(lease):
            raise RuntimeError("Connection refused")

        restored = await handler.restore_leases(leases, reconnect_fn=reconnect_fn)

        assert len(restored) == 1
        lease = lease_manager.get_lease("server-b")
        assert lease is not None
        assert lease.state == LeaseState.DEGRADED

    async def test_restore_released_lease(self, handler, lease_manager):
        """AC: Released leases stay released after restore."""
        leases = [
            MCPLease(server_name="server-a", required=True, state=LeaseState.RELEASED),
        ]

        restored = await handler.restore_leases(leases)

        assert len(restored) == 1
        lease = lease_manager.get_lease("server-a")
        assert lease is not None
        assert lease.state == LeaseState.RELEASED

    async def test_get_preflight_failures_empty(self, handler, lease_manager):
        """AC: No preflight failures when all required leases are active."""
        lease_manager.create_lease("server-a", required=True).activate([])
        lease_manager.create_lease("server-b", required=False).activate([])

        failures = handler.get_preflight_failures()
        assert len(failures) == 0

    async def test_get_preflight_failures_with_failures(self, handler, lease_manager):
        """AC: Preflight failures include failed required leases."""
        lease_a = lease_manager.create_lease("server-a", required=True)
        lease_a.fail("Connection lost")
        lease_b = lease_manager.create_lease("server-b", required=False)
        lease_b.fail("Optional failure")

        failures = handler.get_preflight_failures()
        assert len(failures) == 1
        assert failures[0].server_name == "server-a"

    async def test_get_degraded_leases(self, handler, lease_manager):
        """AC: Degraded leases include failed/degraded optional leases."""
        lease_a = lease_manager.create_lease("server-a", required=True)
        lease_a.fail("Required failure")
        lease_b = lease_manager.create_lease("server-b", required=False)
        lease_b.degrade("Optional degraded")

        degraded = handler.get_degraded_leases()
        assert len(degraded) == 1
        assert degraded[0].server_name == "server-b"

    async def test_handler_property(self, handler, lease_manager):
        """AC: Handler property returns the lease manager."""
        assert handler.lease_manager is lease_manager
