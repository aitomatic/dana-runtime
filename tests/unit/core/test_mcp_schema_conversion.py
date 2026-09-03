"""D5 MCP Schema Conversion — MCP Tool → ToolCatalogEntry mapping.

AC #2: Schema conversion test asserts MCP tool definition → Tool Identity
mapping fidelity.
"""

from __future__ import annotations

from mcp import types

from dana.core.mcp.schema_conversion import mcp_tool_to_catalog_entry
from dana.core.tool.catalog import ToolCatalogEntry, ToolIdentity


class TestMCPToolToCatalogEntry:
    """mcp_tool_to_catalog_entry — MCP Tool → ToolCatalogEntry."""

    def test_basic_conversion(self):
        """AC #2: MCP tool definition maps to ToolCatalogEntry with correct identity."""
        tool = types.Tool(
            name="read_file",
            description="Read a file from disk",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        )

        entry = mcp_tool_to_catalog_entry(tool, server_name="filesystem")

        assert isinstance(entry, ToolCatalogEntry)
        assert isinstance(entry.identity, ToolIdentity)
        # Namespaced name
        assert entry.identity.name == "filesystem:read_file"
        assert entry.identity.source == "mcp:filesystem"
        # Original MCP name as alias
        assert "read_file" in entry.aliases
        # Schema is OpenAI-compatible
        assert entry.schema["type"] == "function"
        assert entry.schema["function"]["name"] == "filesystem:read_file"
        assert entry.schema["function"]["description"] == "Read a file from disk"
        assert entry.schema["function"]["parameters"] == tool.inputSchema
        # Adapter is None (set by session)
        assert entry.adapter is None

    def test_conversion_no_description(self):
        """Tool with no description gets empty string."""
        tool = types.Tool(
            name="no_desc",
            inputSchema={"type": "object", "properties": {}},
        )

        entry = mcp_tool_to_catalog_entry(tool, server_name="test")

        assert entry.schema["function"]["description"] == ""

    def test_conversion_different_servers_same_tool_name(self):
        """Same tool name from different servers gets different namespaced names."""
        tool = types.Tool(
            name="search",
            inputSchema={"type": "object", "properties": {}},
        )

        entry_a = mcp_tool_to_catalog_entry(tool, server_name="server_a")
        entry_b = mcp_tool_to_catalog_entry(tool, server_name="server_b")

        assert entry_a.identity.name == "server_a:search"
        assert entry_b.identity.name == "server_b:search"
        assert entry_a.identity != entry_b.identity
        # Both have the original name as alias
        assert "search" in entry_a.aliases
        assert "search" in entry_b.aliases

    def test_conversion_round_trip_via_catalog(self):
        """Converted entries can be added to a ToolCatalog without collision."""
        from dana.core.tool.catalog import ToolCatalog

        tool_a = types.Tool(name="read", inputSchema={"type": "object", "properties": {}})
        tool_b = types.Tool(name="write", inputSchema={"type": "object", "properties": {}})

        entry_a = mcp_tool_to_catalog_entry(tool_a, server_name="srv")
        entry_b = mcp_tool_to_catalog_entry(tool_b, server_name="srv")

        catalog = ToolCatalog([entry_a, entry_b])
        assert catalog.get("srv:read") is entry_a
        assert catalog.get("srv:write") is entry_b
        # Also findable by original MCP name alias
        assert catalog.get("read") is entry_a
        assert catalog.get("write") is entry_b
