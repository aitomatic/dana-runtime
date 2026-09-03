"""MCP transport adapters — stdio and HTTP.

Per ADR-008: stdio servers default to dedicated managed processes; HTTP
connections may be pooled internally when descriptors + credentials match.
"""

from dana.core.mcp.transports.http import MCPHttpTransport
from dana.core.mcp.transports.stdio import MCPStdioTransport


__all__ = [
    "MCPHttpTransport",
    "MCPStdioTransport",
]
