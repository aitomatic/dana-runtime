"""D5 MCP Catalog Adapter — MCP → Tool Catalog integration.

AC #1: Deterministic collisions — same-named tools from different MCP servers
produce deterministic error.
AC #2: Dynamic catalog invalidation — catalog clears and re-discovers after
server restart.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mcp import types
import pytest

from dana.core.mcp.catalog_adapter import MCPCatalogAdapter
from dana.core.mcp.protocol import MCPHandshakeResult
from dana.core.tool.catalog import ToolCatalog, ToolCatalogEntry, ToolIdentity


pytestmark = pytest.mark.asyncio


class TestMCPCatalogAdapter:
    """MCPCatalogAdapter — discover, register, invalidate, re-discover."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def mock_transport(self):
        """Create a mock MCP transport with a fake session."""
        transport = MagicMock()
        transport.session = AsyncMock()
        return transport

    @pytest.fixture
    def handshake_result(self):
        return MCPHandshakeResult(
            server_name="test-server",
            server_version="1.0.0",
            capabilities=types.ServerCapabilities(tools=types.ToolsCapability(listChanged=True)),
        )

    # ------------------------------------------------------------------
    # discover_and_register
    # ------------------------------------------------------------------

    async def test_discover_and_register_basic(self, mock_transport, handshake_result):
        """AC #2: Discover tools and register them as catalog entries."""
        tools = [
            types.Tool(
                name="read_file",
                description="Read a file",
                inputSchema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            ),
            types.Tool(
                name="write_file",
                description="Write a file",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            ),
        ]

        # We need to mock perform_handshake and discover_tools at the module level
        # Since catalog_adapter imports them directly, we patch the module functions
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("dana.core.mcp.catalog_adapter.perform_handshake", AsyncMock(return_value=handshake_result))
            mp.setattr("dana.core.mcp.catalog_adapter.discover_tools", AsyncMock(return_value=tuple(tools)))

            adapter = MCPCatalogAdapter(server_name="filesystem")
            entries = await adapter.discover_and_register(mock_transport)

        assert len(entries) == 2
        assert adapter.is_valid is True
        assert adapter.server_name == "filesystem"
        assert adapter.handshake_result is not None
        assert adapter.handshake_result.server_name == "test-server"

        # Check namespaced identities
        assert entries[0].identity.name == "filesystem:read_file"
        assert entries[0].identity.source == "mcp:filesystem"
        assert "read_file" in entries[0].aliases

        assert entries[1].identity.name == "filesystem:write_file"
        assert entries[1].identity.source == "mcp:filesystem"
        assert "write_file" in entries[1].aliases

    async def test_discover_and_register_empty(self, mock_transport, handshake_result):
        """Discovering no tools returns empty list."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("dana.core.mcp.catalog_adapter.perform_handshake", AsyncMock(return_value=handshake_result))
            mp.setattr("dana.core.mcp.catalog_adapter.discover_tools", AsyncMock(return_value=()))

            adapter = MCPCatalogAdapter(server_name="empty-server")
            entries = await adapter.discover_and_register(mock_transport)

        assert len(entries) == 0
        assert adapter.is_valid is True
        assert adapter.entries == []

    async def test_discover_and_register_not_connected(self):
        """Raises RuntimeError when transport is not connected."""
        transport = MagicMock()
        transport.session = None

        adapter = MCPCatalogAdapter(server_name="test")
        with pytest.raises(RuntimeError, match="not connected"):
            await adapter.discover_and_register(transport)

    # ------------------------------------------------------------------
    # Deterministic collisions (AC #1)
    # ------------------------------------------------------------------

    async def test_deterministic_collision_same_server(self, mock_transport, handshake_result):
        """AC #1: Duplicate tool names within the same server produce deterministic error."""
        tools = [
            types.Tool(
                name="duplicate_tool",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="duplicate_tool",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("dana.core.mcp.catalog_adapter.perform_handshake", AsyncMock(return_value=handshake_result))
            mp.setattr("dana.core.mcp.catalog_adapter.discover_tools", AsyncMock(return_value=tuple(tools)))

            adapter = MCPCatalogAdapter(server_name="dup-server")
            with pytest.raises(ValueError, match="Duplicate tool name"):
                await adapter.discover_and_register(mock_transport)

    async def test_cross_server_collision_via_catalog(self):
        """AC #1: Same-named tools from different MCP servers produce collision at catalog level.

        Two servers each have a tool named 'search'. The namespaced identities
        differ (server_a:search vs server_b:search), but the aliases collide
        (both have alias 'search'). ToolCatalog construction should catch this.
        """
        entry_a = ToolCatalogEntry(
            identity=ToolIdentity(name="server_a:search", source="mcp:server_a"),
            schema={},
            adapter=None,
            aliases=frozenset({"search"}),
        )
        entry_b = ToolCatalogEntry(
            identity=ToolIdentity(name="server_b:search", source="mcp:server_b"),
            schema={},
            adapter=None,
            aliases=frozenset({"search"}),
        )

        with pytest.raises(ValueError, match="Duplicate alias"):
            ToolCatalog([entry_a, entry_b])

    async def test_cross_server_no_collision_different_aliases(self):
        """Different aliases from different servers do not collide."""
        entry_a = ToolCatalogEntry(
            identity=ToolIdentity(name="server_a:search", source="mcp:server_a"),
            schema={},
            adapter=None,
            aliases=frozenset({"search_a"}),
        )
        entry_b = ToolCatalogEntry(
            identity=ToolIdentity(name="server_b:search", source="mcp:server_b"),
            schema={},
            adapter=None,
            aliases=frozenset({"search_b"}),
        )

        catalog = ToolCatalog([entry_a, entry_b])
        assert catalog.get("server_a:search") is entry_a
        assert catalog.get("server_b:search") is entry_b
        assert catalog.get("search_a") is entry_a
        assert catalog.get("search_b") is entry_b

    async def test_mcp_vs_builtin_collision(self):
        """AC #1: MCP tool colliding with built-in tool name produces deterministic error.

        Edge case: an MCP-discovered tool has the same namespaced name as a
        built-in tool.
        """
        builtin_entry = ToolCatalogEntry(
            identity=ToolIdentity(name="builtin_tool", source="builtin"),
            schema={},
            adapter=lambda args: "ok",
        )
        mcp_entry = ToolCatalogEntry(
            identity=ToolIdentity(name="builtin_tool", source="mcp:server"),
            schema={},
            adapter=None,
            aliases=frozenset({"mcp_alias"}),
        )

        with pytest.raises(ValueError, match="Duplicate tool name"):
            ToolCatalog([builtin_entry, mcp_entry])

    # ------------------------------------------------------------------
    # Invalidation (AC #2)
    # ------------------------------------------------------------------

    async def test_invalidate_clears_entries(self, mock_transport, handshake_result):
        """AC #2: Invalidation clears entries and marks adapter as invalid."""
        tools = [
            types.Tool(
                name="my_tool",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("dana.core.mcp.catalog_adapter.perform_handshake", AsyncMock(return_value=handshake_result))
            mp.setattr("dana.core.mcp.catalog_adapter.discover_tools", AsyncMock(return_value=tuple(tools)))

            adapter = MCPCatalogAdapter(server_name="test")
            await adapter.discover_and_register(mock_transport)
            assert len(adapter.entries) == 1
            assert adapter.is_valid is True

            await adapter.invalidate()

        assert adapter.is_valid is False
        assert adapter.entries == []
        assert adapter.handshake_result is None

    async def test_re_discover_after_invalidation(self, mock_transport, handshake_result):
        """AC #2: Re-discovery after invalidation returns fresh entries."""
        tools_v1 = [
            types.Tool(
                name="tool_v1",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
        tools_v2 = [
            types.Tool(
                name="tool_v2",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("dana.core.mcp.catalog_adapter.perform_handshake", AsyncMock(return_value=handshake_result))
            mp.setattr("dana.core.mcp.catalog_adapter.discover_tools", AsyncMock(side_effect=[tuple(tools_v1), tuple(tools_v2)]))

            adapter = MCPCatalogAdapter(server_name="test")
            entries_v1 = await adapter.discover_and_register(mock_transport)
            assert len(entries_v1) == 1
            assert entries_v1[0].identity.name == "test:tool_v1"

            entries_v2 = await adapter.re_discover(mock_transport)
            assert len(entries_v2) == 1
            assert entries_v2[0].identity.name == "test:tool_v2"
            assert adapter.is_valid is True

    async def test_invalidation_while_in_flight(self):
        """Edge case: invalidation while entries are in use.

        The adapter marks entries as stale. The session must not use stale
        entries for tool execution.
        """
        adapter = MCPCatalogAdapter(server_name="test")
        # Simulate: entries were discovered, then invalidated
        adapter._entries = [
            ToolCatalogEntry(
                identity=ToolIdentity(name="test:tool", source="mcp:test"),
                schema={},
                adapter=None,
            ),
        ]
        adapter._valid = True

        await adapter.invalidate()

        assert adapter.is_valid is False
        # The session should check is_valid before using entries
        # If is_valid is False, the session must re-discover

    # ------------------------------------------------------------------
    # entries property returns copy
    # ------------------------------------------------------------------

    async def test_entries_returns_copy(self, mock_transport, handshake_result):
        """entries property returns a copy, not the internal list."""
        tools = [
            types.Tool(
                name="tool",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("dana.core.mcp.catalog_adapter.perform_handshake", AsyncMock(return_value=handshake_result))
            mp.setattr("dana.core.mcp.catalog_adapter.discover_tools", AsyncMock(return_value=tuple(tools)))

            adapter = MCPCatalogAdapter(server_name="test")
            await adapter.discover_and_register(mock_transport)

        entries_copy = adapter.entries
        entries_copy.clear()
        assert len(adapter.entries) == 1  # internal list unchanged
