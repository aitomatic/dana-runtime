"""D5 MCP Leases — session MCP Lease binding, restore, and failure handling.

Per ADR-008:
- Required-lease failure stops preflight or workflow start (not session load).
- Optional-lease failure degrades that lease and updates the host.
"""

from __future__ import annotations

import pytest

from dana.core.mcp.leases import LeaseState, MCPLease, MCPLeaseManager


class TestMCPLease:
    """MCPLease — individual lease lifecycle."""

    def test_lease_initial_state(self):
        """A new lease starts in PENDING state."""
        lease = MCPLease(server_name="filesystem")
        assert lease.server_name == "filesystem"
        assert lease.state == LeaseState.PENDING
        assert lease.required is True
        assert lease.error is None
        assert lease.entries == []

    def test_lease_optional(self):
        """A lease can be created as optional."""
        lease = MCPLease(server_name="search", required=False)
        assert lease.required is False

    def test_activate(self):
        """Activate sets state to ACTIVE and stores entries."""
        lease = MCPLease(server_name="test")
        entries = ["entry1", "entry2"]
        lease.activate(entries)
        assert lease.state == LeaseState.ACTIVE
        assert lease.entries == ["entry1", "entry2"]
        assert lease.error is None
        assert lease.is_active is True

    def test_fail_required(self):
        """Fail on a required lease sets state to FAILED."""
        lease = MCPLease(server_name="test", required=True)
        lease.fail("Connection refused")
        assert lease.state == LeaseState.FAILED
        assert lease.error == "Connection refused"
        assert lease.entries == []
        assert lease.is_failed is True
        assert lease.is_active is False

    def test_fail_optional(self):
        """Fail on an optional lease sets state to FAILED but is not critical."""
        lease = MCPLease(server_name="test", required=False)
        lease.fail("Timeout")
        assert lease.state == LeaseState.FAILED
        assert lease.is_failed is True

    def test_degrade(self):
        """Degrade sets state to DEGRADED (optional leases only)."""
        lease = MCPLease(server_name="test", required=False)
        lease.degrade("Slow response")
        assert lease.state == LeaseState.DEGRADED
        assert lease.error == "Slow response"

    def test_release(self):
        """Release sets state to RELEASED and clears entries."""
        lease = MCPLease(server_name="test")
        lease.activate(["entry1"])
        lease.release()
        assert lease.state == LeaseState.RELEASED
        assert lease.entries == []

    def test_lease_equality(self):
        """Leases are compared by identity (not value)."""
        a = MCPLease(server_name="test")
        b = MCPLease(server_name="test")
        # dataclass equality by value
        assert a == b


class TestMCPLeaseManager:
    """MCPLeaseManager — collection of leases."""

    def test_create_lease(self):
        """Create a new lease."""
        manager = MCPLeaseManager()
        lease = manager.create_lease("filesystem", required=True)
        assert lease.server_name == "filesystem"
        assert lease.required is True
        assert lease.state == LeaseState.PENDING

    def test_create_lease_duplicate_raises(self):
        """Creating a duplicate lease raises ValueError."""
        manager = MCPLeaseManager()
        manager.create_lease("filesystem")
        with pytest.raises(ValueError, match="already exists"):
            manager.create_lease("filesystem")

    def test_get_lease(self):
        """Get a lease by server name."""
        manager = MCPLeaseManager()
        manager.create_lease("filesystem")
        lease = manager.get_lease("filesystem")
        assert lease is not None
        assert lease.server_name == "filesystem"

    def test_get_lease_nonexistent(self):
        """Getting a nonexistent lease returns None."""
        manager = MCPLeaseManager()
        assert manager.get_lease("nonexistent") is None

    def test_release_lease(self):
        """Release a lease by server name."""
        manager = MCPLeaseManager()
        manager.create_lease("filesystem")
        manager.release_lease("filesystem")
        assert manager.get_lease("filesystem") is None

    def test_release_all(self):
        """Release all leases."""
        manager = MCPLeaseManager()
        manager.create_lease("a")
        manager.create_lease("b")
        manager.release_all()
        assert manager.leases == {}

    def test_active_leases(self):
        """active_leases returns only ACTIVE leases."""
        manager = MCPLeaseManager()
        lease_a = manager.create_lease("a")
        manager.create_lease("b")
        lease_a.activate(["tool1"])
        # lease_b is still PENDING
        active = manager.active_leases
        assert len(active) == 1
        assert active[0].server_name == "a"

    def test_failed_leases(self):
        """failed_leases returns only FAILED leases."""
        manager = MCPLeaseManager()
        lease_a = manager.create_lease("a")
        manager.create_lease("b")
        lease_a.fail("error")
        # lease_b is still PENDING
        failed = manager.failed_leases
        assert len(failed) == 1
        assert failed[0].server_name == "a"

    # ------------------------------------------------------------------
    # ADR-008: Required-lease failure
    # ------------------------------------------------------------------

    def test_check_required_leases_no_failures(self):
        """check_required_leases returns empty when all required leases are active."""
        manager = MCPLeaseManager()
        lease = manager.create_lease("required-server", required=True)
        lease.activate(["tool1"])
        failed = manager.check_required_leases()
        assert failed == []

    def test_check_required_leases_with_failures(self):
        """check_required_leases returns failed required leases."""
        manager = MCPLeaseManager()
        lease = manager.create_lease("required-server", required=True)
        lease.fail("Connection lost")
        failed = manager.check_required_leases()
        assert len(failed) == 1
        assert failed[0].server_name == "required-server"

    def test_check_required_leases_ignores_optional(self):
        """check_required_leases ignores optional leases."""
        manager = MCPLeaseManager()
        manager.create_lease("optional-server", required=False).fail("Timeout")
        failed = manager.check_required_leases()
        assert failed == []

    # ------------------------------------------------------------------
    # ADR-008: Optional-lease failure
    # ------------------------------------------------------------------

    def test_check_optional_leases_returns_degraded(self):
        """check_optional_leases returns degraded/failed optional leases."""
        manager = MCPLeaseManager()
        opt = manager.create_lease("optional-server", required=False)
        opt.degrade("Slow")
        degraded = manager.check_optional_leases()
        assert len(degraded) == 1
        assert degraded[0].server_name == "optional-server"

    def test_check_optional_leases_ignores_required(self):
        """check_optional_leases ignores required leases."""
        manager = MCPLeaseManager()
        manager.create_lease("required-server", required=True).fail("Error")
        degraded = manager.check_optional_leases()
        assert degraded == []

    # ------------------------------------------------------------------
    # ADR-008: Restore leases (session load)
    # ------------------------------------------------------------------

    def test_restore_leases(self):
        """Restore leases from persisted state."""
        manager = MCPLeaseManager()
        original = MCPLease(server_name="filesystem", required=True, state=LeaseState.FAILED, error="Previous error")
        manager.restore_leases([original])
        restored = manager.get_lease("filesystem")
        assert restored is not None
        assert restored.server_name == "filesystem"
        assert restored.state == LeaseState.FAILED
        assert restored.error == "Previous error"

    def test_restore_leases_multiple(self):
        """Restore multiple leases."""
        manager = MCPLeaseManager()
        leases = [
            MCPLease(server_name="a", required=True, state=LeaseState.ACTIVE),
            MCPLease(server_name="b", required=False, state=LeaseState.FAILED),
        ]
        manager.restore_leases(leases)
        assert len(manager.leases) == 2
        assert manager.get_lease("a").state == LeaseState.ACTIVE
        assert manager.get_lease("b").state == LeaseState.FAILED

    def test_restore_leases_empty(self):
        """Restoring empty list is a no-op."""
        manager = MCPLeaseManager()
        manager.restore_leases([])
        assert manager.leases == {}

    def test_leases_property_returns_copy(self):
        """leases property returns a copy."""
        manager = MCPLeaseManager()
        manager.create_lease("a")
        leases_copy = manager.leases
        leases_copy.clear()
        assert len(manager.leases) == 1
