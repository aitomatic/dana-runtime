"""MCP Session Restore — required/optional lease restore on session load.

Per ADR-008 (Session MCP Leases):
- Required-lease failure stops preflight or workflow start (not session load).
- Optional-lease failure degrades that lease and updates the host.
- Clean close on session teardown.

On session load, persisted MCP leases are restored. Required leases that
failed during the previous session are restored in their failed state — the
session can still load. Optional leases that failed are degraded.

The restore process:
1. Load persisted lease state from the session journal.
2. For each lease, attempt to reconnect to the MCP server.
3. If reconnection succeeds, the lease is re-activated.
4. If reconnection fails:
   - Required lease: restored in FAILED state (preflight will catch it).
   - Optional lease: restored in DEGRADED state (session continues).
"""

from __future__ import annotations

import logging
from typing import Any

from dana.core.mcp.leases import LeaseState, MCPLease, MCPLeaseManager


logger = logging.getLogger(__name__)


class MCPRestoreHandler:
    """Handles MCP lease restore on session load.

    Restores leases from persisted state, attempting reconnection for each.
    Required-lease failures are preserved (preflight will catch them).
    Optional-lease failures degrade gracefully.

    Usage::

        handler = MCPRestoreHandler(lease_manager)
        results = await handler.restore_leases(persisted_leases)
    """

    def __init__(self, lease_manager: MCPLeaseManager) -> None:
        self._lease_manager = lease_manager

    @property
    def lease_manager(self) -> MCPLeaseManager:
        """The lease manager being restored."""
        return self._lease_manager

    async def restore_leases(
        self,
        persisted_leases: list[MCPLease],
        reconnect_fn: Any = None,
    ) -> list[MCPLease]:
        """Restore leases from persisted state.

        Args:
            persisted_leases: List of ``MCPLease`` objects from the journal.
            reconnect_fn: Optional async callable ``(MCPLease) -> bool`` that
                attempts to reconnect to the MCP server. If None, all leases
                are restored in their persisted state without reconnection.

        Returns:
            The list of restored leases (in the lease manager).
        """
        # Restore all leases into the manager first
        self._lease_manager.restore_leases(persisted_leases)

        restored: list[MCPLease] = []
        for lease in persisted_leases:
            if lease.state == LeaseState.RELEASED:
                # Released leases stay released
                restored.append(lease)
                continue

            if reconnect_fn is not None:
                try:
                    reconnected = await reconnect_fn(lease)
                    if reconnected:
                        # Reconnection succeeded — reactivate
                        lease.activate(lease.entries)
                        logger.info("Lease reconnected for '%s'", lease.server_name)
                    else:
                        self._handle_failed_reconnect(lease)
                except Exception as exc:
                    logger.warning(
                        "Lease reconnect error for '%s': %s",
                        lease.server_name,
                        exc,
                    )
                    self._handle_failed_reconnect(lease)
            else:
                # No reconnect function — keep persisted state
                logger.info(
                    "Lease restored (persisted) for '%s' (state=%s)",
                    lease.server_name,
                    lease.state,
                )

            restored.append(lease)

        return restored

    def _handle_failed_reconnect(self, lease: MCPLease) -> None:
        """Handle a failed reconnection attempt.

        Per ADR-008:
        - Required lease: stays FAILED (preflight will stop).
        - Optional lease: degrades (session continues).
        """
        if lease.required:
            lease.fail(f"Failed to reconnect to MCP server '{lease.server_name}'")
            logger.error("Required lease reconnect failed for '%s'", lease.server_name)
        else:
            lease.degrade(f"Failed to reconnect to MCP server '{lease.server_name}'")
            logger.warning("Optional lease reconnect failed for '%s'", lease.server_name)

    def get_preflight_failures(self) -> list[MCPLease]:
        """Get required leases that failed, for preflight checks.

        Returns:
            List of failed required leases. Empty if all required leases
            are active or pending.
        """
        return self._lease_manager.check_required_leases()

    def get_degraded_leases(self) -> list[MCPLease]:
        """Get degraded or failed optional leases.

        Returns:
            List of degraded or failed optional leases.
        """
        return self._lease_manager.check_optional_leases()
