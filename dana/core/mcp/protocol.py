"""MCP protocol handshake, capabilities discovery, and tools/list integration.

Per ADR-008: use official ``mcp>=1.28,<2`` package, not ad hoc JSON-RPC.
Per ADR-004: discovered MCP tools enter the session Tool Catalog with
namespaced Tool Identity; duplicate detection applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, types


@dataclass(frozen=True)
class MCPHandshakeResult:
    """Result of a successful MCP handshake with a server.

    ``server_name``    — the server's implementation name.
    ``server_version`` — the server's implementation version.
    ``capabilities``   — raw ``ServerCapabilities`` from the handshake.
    ``tools``          — list of ``Tool`` definitions from tools/list.
    """

    server_name: str
    server_version: str
    capabilities: types.ServerCapabilities
    tools: tuple[types.Tool, ...] = field(default_factory=tuple)


async def perform_handshake(
    session: ClientSession,
    client_name: str = "dana",
    client_version: str = "0.2.0",
) -> MCPHandshakeResult:
    """Perform the MCP initialize handshake and return server info + capabilities.

    Args:
        session: An already-connected ``ClientSession`` (transport wired).
        client_name: Client implementation name sent during handshake.
        client_version: Client implementation version sent during handshake.

    Returns:
        ``MCPHandshakeResult`` with server identity and capabilities.

    Raises:
        RuntimeError: If the server rejects the handshake or protocol version
            is unsupported.
    """
    result = await session.initialize()

    return MCPHandshakeResult(
        server_name=result.serverInfo.name,
        server_version=result.serverInfo.version,
        capabilities=result.capabilities,
    )


async def discover_tools(
    session: ClientSession,
) -> tuple[types.Tool, ...]:
    """Call tools/list and return the discovered tool definitions.

    Args:
        session: An initialized ``ClientSession``.

    Returns:
        Tuple of ``Tool`` definitions. Empty if the server has no tools or
        does not support the tools capability.
    """
    caps = session.get_server_capabilities()
    if caps is None or caps.tools is None:
        return ()

    result = await session.list_tools()
    return tuple(result.tools)


async def call_tool(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> types.CallToolResult:
    """Call a tool on the MCP server.

    Args:
        session: An initialized ``ClientSession``.
        name: The tool name to call.
        arguments: Optional arguments to pass to the tool.

    Returns:
        ``CallToolResult`` from the server.
    """
    return await session.call_tool(name, arguments)
