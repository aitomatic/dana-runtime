"""Session MCP Leases — lease binding, restore, and failure handling.

Per ADR-008 (Session MCP Leases):
- Discovered tools enter the Tool Catalog with namespaced Tool Identity.
- Required-lease failure stops preflight or workflow start (not session load).
- Optional-lease failure degrades that lease and updates the host.

Leases represent the session's binding to an MCP server. Each lease tracks
whether the server's tools are required for the session to function, or
optional (best-effort).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any


logger = logging.getLogger(__name__)


class LeaseState:
    """Enum-like constants for lease states."""

    PENDING = "pending"
    ACTIVE = "active"
    FAILED = "failed"
    DEGRADED = "degraded"
    RELEASED = "released"


@dataclass
class MCPLease:
    """A session's lease on an MCP server.

    ``server_name``   — the MCP server's implementation name.
    ``required``      — if True, lease failure stops preflight/workflow start.
    ``state``         — current lease state (pending/active/failed/degraded/released).
    ``error``         — error message if the lease failed or degraded.
    ``entries``       — the catalog entries discovered from this server.
    ``transport_info`` — opaque transport info for lease restore.
    """

    server_name: str
    required: bool = True
    state: str = LeaseState.PENDING
    error: str | None = None
    entries: list[Any] = field(default_factory=list)
    transport_info: dict[str, Any] = field(default_factory=dict)

    def activate(self, entries: list[Any]) -> None:
        """Mark the lease as active with the given catalog entries."""
        self.state = LeaseState.ACTIVE
        self.entries = list(entries)
        self.error = None
        logger.info("MCP lease activated for '%s' (%d entries)", self.server_name, len(entries))

    def fail(self, error: str) -> None:
        """Mark the lease as failed.

        If the lease is required, this will stop preflight/workflow start.
        If optional, the session degrades and continues.
        """
        self.state = LeaseState.FAILED
        self.error = error
        self.entries = []
        if self.required:
            logger.error("Required MCP lease failed for '%s': %s", self.server_name, error)
        else:
            logger.warning("Optional MCP lease failed for '%s': %s", self.server_name, error)

    def degrade(self, error: str) -> None:
        """Mark the lease as degraded (optional lease only).

        The server's tools are partially available or the connection is
        degraded. The session continues but the host is updated.
        """
        self.state = LeaseState.DEGRADED
        self.error = error
        logger.info("MCP lease degraded for '%s': %s", self.server_name, error)

    def release(self) -> None:
        """Release the lease (cleanup on session end)."""
        self.state = LeaseState.RELEASED
        self.entries = []
        logger.info("MCP lease released for '%s'", self.server_name)

    @property
    def is_active(self) -> bool:
        return self.state == LeaseState.ACTIVE

    @property
    def is_failed(self) -> bool:
        return self.state == LeaseState.FAILED


class MCPLeaseManager:
    """Manages a collection of MCP leases for a session.

    Handles lease lifecycle: creation, activation, failure, degradation,
    release, and restore from persisted state.
    """

    def __init__(self) -> None:
        self._leases: dict[str, MCPLease] = {}

    @property
    def leases(self) -> dict[str, MCPLease]:
        """All leases keyed by server name (copy)."""
        return dict(self._leases)

    @property
    def active_leases(self) -> list[MCPLease]:
        """Leases in ACTIVE state."""
        return [lease for lease in self._leases.values() if lease.is_active]

    @property
    def failed_leases(self) -> list[MCPLease]:
        """Leases in FAILED state."""
        return [lease for lease in self._leases.values() if lease.is_failed]

    def create_lease(self, server_name: str, required: bool = True) -> MCPLease:
        """Create a new lease for an MCP server.

        Args:
            server_name: The MCP server name.
            required: Whether the server's tools are required.

        Returns:
            The new ``MCPLease``.

        Raises:
            ValueError: If a lease for this server already exists.
        """
        if server_name in self._leases:
            raise ValueError(f"Lease already exists for server '{server_name}'")
        lease = MCPLease(server_name=server_name, required=required)
        self._leases[server_name] = lease
        logger.info("MCP lease created for '%s' (required=%s)", server_name, required)
        return lease

    def get_lease(self, server_name: str) -> MCPLease | None:
        """Get the lease for a server, or None."""
        return self._leases.get(server_name)

    def release_lease(self, server_name: str) -> None:
        """Release a lease by server name."""
        lease = self._leases.get(server_name)
        if lease is not None:
            lease.release()
            del self._leases[server_name]

    def release_all(self) -> None:
        """Release all leases."""
        for lease in list(self._leases.values()):
            lease.release()
        self._leases.clear()

    def check_required_leases(self) -> list[MCPLease]:
        """Check all required leases and return failed ones.

        Per ADR-008: required-lease failure stops preflight or workflow start
        (not session load). Call this during preflight to determine if the
        session can proceed.

        Returns:
            List of failed required leases. Empty if all required leases are
            active or pending.
        """
        failed: list[MCPLease] = []
        for lease in self._leases.values():
            if lease.required and lease.state == LeaseState.FAILED:
                failed.append(lease)
        return failed

    def check_optional_leases(self) -> list[MCPLease]:
        """Check optional leases and return degraded/failed ones.

        Per ADR-008: optional-lease failure degrades that lease and updates
        the host. The session continues but the host is informed.

        Returns:
            List of degraded or failed optional leases.
        """
        degraded: list[MCPLease] = []
        for lease in self._leases.values():
            if not lease.required and lease.state in (LeaseState.FAILED, LeaseState.DEGRADED):
                degraded.append(lease)
        return degraded

    def restore_leases(self, leases: list[MCPLease]) -> None:
        """Restore leases from persisted state (e.g. session load).

        Per ADR-008: lease failure does NOT stop session load. Failed leases
        are restored in their failed state; the session can still load and
        the host is updated.

        Args:
            leases: List of ``MCPLease`` objects to restore.
        """
        for lease in leases:
            self._leases[lease.server_name] = lease
        logger.info("Restored %d MCP leases", len(leases))
