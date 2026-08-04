"""MCP Protocol & Transports — official mcp package integration.

Per ADR-008: use official ``mcp>=1.28,<2`` package, not ad hoc JSON-RPC.
"""

from dana.core.mcp.protocol import MCPHandshakeResult, discover_tools, perform_handshake
from dana.core.mcp.schema_conversion import mcp_tool_to_catalog_entry


__all__ = [
    "MCPHandshakeResult",
    "discover_tools",
    "mcp_tool_to_catalog_entry",
    "perform_handshake",
]
