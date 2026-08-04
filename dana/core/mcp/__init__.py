"""MCP Protocol, Transports, Catalog Adapter, Leases, and Configuration.

Per ADR-008: use official ``mcp>=1.28,<2`` package, not ad hoc JSON-RPC.
Per ADR-004: discovered MCP tools enter the session Tool Catalog with
namespaced Tool Identity; duplicate detection applies.
Per ADR-012: rollback disables MCP configuration loading.
"""

from dana.core.mcp.catalog_adapter import MCPCatalogAdapter
from dana.core.mcp.config import (
    MCPConfig,
    MCPServerConfig,
    filter_env_for_server,
    is_mcp_enabled,
    load_mcp_config,
    load_mcp_config_from_dict,
)
from dana.core.mcp.leases import LeaseState, MCPLease, MCPLeaseManager
from dana.core.mcp.protocol import MCPHandshakeResult, discover_tools, perform_handshake
from dana.core.mcp.schema_conversion import mcp_tool_to_catalog_entry


__all__ = [
    "MCPCatalogAdapter",
    "MCPConfig",
    "MCPHandshakeResult",
    "MCPLease",
    "MCPLeaseManager",
    "MCPServerConfig",
    "LeaseState",
    "discover_tools",
    "filter_env_for_server",
    "is_mcp_enabled",
    "load_mcp_config",
    "load_mcp_config_from_dict",
    "mcp_tool_to_catalog_entry",
    "perform_handshake",
]
