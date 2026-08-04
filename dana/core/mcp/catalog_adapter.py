"""MCP Catalog Adapter — MCP → Tool Catalog integration with namespaced identity.

Per ADR-008 (Session MCP Leases): discovered tools enter the Tool Catalog with
namespaced Tool Identity; required-lease failure stops preflight or workflow
start (not session load); optional-lease failure degrades that lease and
updates the host.

Per ADR-004 (Stable Tool Identity and Versioned Catalog): MCP tools follow the
same catalog rules as built-ins — duplicate stable identities or aliases fail
catalog construction; per-turn immutable version; changes only between turns.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.types import Tool as MCPTool

from dana.core.mcp.protocol import MCPHandshakeResult, discover_tools, perform_handshake
from dana.core.mcp.schema_conversion import mcp_tool_to_catalog_entry
from dana.core.tool.catalog import ToolCatalogEntry, ToolIdentity


logger = logging.getLogger(__name__)


class MCPCatalogAdapter:
    """Adapter that discovers MCP tools and registers them in the Tool Catalog.

    One adapter instance per MCP server connection. Manages the lifecycle of
    discovered tool entries: initial discovery, collision-safe registration,
    invalidation, and re-discovery.

    Usage::

        adapter = MCPCatalogAdapter(server_name="filesystem")
        async with transport.connect():
            entries = await adapter.discover_and_register(transport)
            # entries are now ready for ToolCatalog construction
            ...
            # On server restart:
            await adapter.invalidate()
            entries = await adapter.discover_and_register(transport)
    """

    def __init__(self, server_name: str) -> None:
        self._server_name = server_name
        self._entries: list[ToolCatalogEntry] = []
        self._handshake_result: MCPHandshakeResult | None = None
        self._valid = False

    @property
    def server_name(self) -> str:
        """The MCP server name this adapter is bound to."""
        return self._server_name

    @property
    def entries(self) -> list[ToolCatalogEntry]:
        """Currently discovered catalog entries (copy)."""
        return list(self._entries)

    @property
    def handshake_result(self) -> MCPHandshakeResult | None:
        """The handshake result from the last successful discovery."""
        return self._handshake_result

    @property
    def is_valid(self) -> bool:
        """Whether the adapter's entries are currently valid."""
        return self._valid

    async def discover_and_register(
        self,
        transport: Any,
    ) -> list[ToolCatalogEntry]:
        """Perform handshake, discover tools, convert to catalog entries.

        Args:
            transport: An MCP transport instance (must be connected).

        Returns:
            List of ``ToolCatalogEntry`` objects ready for catalog insertion.

        Raises:
            RuntimeError: If the handshake or discovery fails.
            ValueError: If converted entries have duplicate identities or
                aliases (deterministic collision error per ADR-004).
        """
        # Perform handshake
        if transport.session is None:
            raise RuntimeError("Transport not connected. Connect before discovering tools.")

        handshake = await perform_handshake(
            transport.session,
            client_name="dana",
            client_version="0.2.0",
        )
        self._handshake_result = handshake

        # Discover tools
        raw_tools: tuple[MCPTool, ...] = await discover_tools(transport.session)

        # Convert to catalog entries
        entries: list[ToolCatalogEntry] = []
        for tool in raw_tools:
            entry = mcp_tool_to_catalog_entry(tool, self._server_name)
            entries.append(entry)

        # Validate for collisions (deterministic — raises ValueError on duplicate)
        # This catches same-named tools from different MCP servers AND
        # collisions with built-in tools (enforced at catalog construction time).
        self._validate_entries(entries)

        self._entries = entries
        self._valid = True
        logger.info(
            "MCP catalog adapter discovered %d tools from server '%s'",
            len(entries),
            self._server_name,
        )
        return list(self._entries)

    async def invalidate(self) -> None:
        """Invalidate the current entries.

        Called when the MCP server restarts or the connection is lost.
        After invalidation, ``discover_and_register()`` must be called again
        before the entries can be used.

        Per ADR-004: catalog changes only occur between turns. Invalidation
        marks the current entries as stale; the session must rebuild the
        catalog on the next turn.
        """
        self._entries = []
        self._valid = False
        self._handshake_result = None
        logger.info("MCP catalog adapter for '%s' invalidated", self._server_name)

    async def re_discover(self, transport: Any) -> list[ToolCatalogEntry]:
        """Invalidate then re-discover tools from the server.

        Convenience wrapper for ``invalidate()`` + ``discover_and_register()``.

        Args:
            transport: An MCP transport instance (must be connected).

        Returns:
            Fresh list of ``ToolCatalogEntry`` objects.
        """
        await self.invalidate()
        return await self.discover_and_register(transport)

    def _validate_entries(self, entries: list[ToolCatalogEntry]) -> None:
        """Validate entries for collisions.

        Raises ``ValueError`` on the first duplicate identity or alias
        (deterministic collision error per ADR-004).

        This validates within the MCP server's own entries. Cross-server
        collisions (e.g. same tool name from two different MCP servers) are
        caught at ``ToolCatalog`` construction time because the namespaced
        identities will differ (``server_a:tool`` vs ``server_b:tool``), but
        the aliases (original tool name) will collide — which is the
        deterministic collision behavior we want.
        """
        # Build a temporary catalog to validate — this catches duplicates
        # within this server's entries (shouldn't happen, but defensive).
        seen_names: dict[str, ToolIdentity] = {}
        seen_aliases: dict[str, ToolIdentity] = {}

        for entry in entries:
            if entry.identity.name in seen_names:
                prev = seen_names[entry.identity.name]
                raise ValueError(
                    f"Duplicate tool name within server '{self._server_name}': '{entry.identity.name}' (conflicts with {prev})"
                )
            seen_names[entry.identity.name] = entry.identity

            for alias in entry.aliases:
                if alias in seen_aliases:
                    prev = seen_aliases[alias]
                    raise ValueError(f"Duplicate alias within server '{self._server_name}': '{alias}' (conflicts with {prev.name})")
                seen_aliases[alias] = entry.identity
