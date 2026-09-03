"""MCP tool schema → Tool Catalog schema conversion.

Per ADR-004: discovered MCP tools enter the session Tool Catalog with
namespaced Tool Identity; duplicate detection applies.
"""

from __future__ import annotations

from typing import Any

from mcp.types import Tool as MCPTool

from dana.core.tool.catalog import ToolCatalogEntry, ToolIdentity


def mcp_tool_to_catalog_entry(
    tool: MCPTool,
    server_name: str,
) -> ToolCatalogEntry:
    """Convert an MCP ``Tool`` definition to a ``ToolCatalogEntry``.

    The entry's ``identity.name`` is the MCP tool name prefixed with the
    server name (``{server_name}:{tool_name}``) to avoid collisions across
    servers. The original MCP tool name is stored as an alias so the
    execution engine can look it up by either name.

    The ``inputSchema`` from the MCP tool is mapped to an OpenAI-compatible
    tool schema dict with ``type: "function"``.

    Args:
        tool: The MCP ``Tool`` definition from a tools/list response.
        server_name: The MCP server name (used for namespacing).

    Returns:
        A ``ToolCatalogEntry`` ready for insertion into a ``ToolCatalog``.
    """
    namespaced_name = f"{server_name}:{tool.name}"

    # Build an OpenAI-compatible function schema from the MCP inputSchema.
    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": namespaced_name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }

    identity = ToolIdentity(name=namespaced_name, source=f"mcp:{server_name}")

    return ToolCatalogEntry(
        identity=identity,
        schema=schema,
        adapter=None,  # Set by the session when wiring the adapter
        aliases=frozenset({tool.name}),
    )
