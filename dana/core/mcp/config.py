"""MCP Configuration — server configuration loading and environment allowlist.

Per ADR-012 (Migration Shadow Cutover Retirement):
- Rollback disables MCP configuration loading; other tools remain available.
- Rollback never deletes journal facts.

Configuration is loaded from a JSON/YAML file that defines MCP server
definitions (command, args, env, etc.) and an environment variable allowlist.
The allowlist controls which environment variables are passed to MCP server
subprocesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server.

    ``name``     — unique server name (used for namespacing tool identities).
    ``command``  — the command to start the server (e.g. ``npx``, ``python``).
    ``args``     — command-line arguments.
    ``env``      — environment variables to pass (after allowlist filtering).
    ``cwd``      — working directory for the server process.
    ``transport`` — transport type (``stdio`` or ``http``).
    ``url``      — URL for HTTP transport.
    ``headers``  — HTTP headers for HTTP transport.
    ``timeout``  — connection timeout in seconds.
    """

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    transport: str = "stdio"
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0


@dataclass
class MCPConfig:
    """Complete MCP configuration for a session.

    ``servers``         — list of MCP server configurations.
    ``env_allowlist``   — list of environment variable keys allowed to pass
                          through to MCP server subprocesses. If empty, no
                          env vars are passed (block all).
    ``enabled``         — if False, MCP configuration loading is disabled
                          (rollback mechanism per ADR-012).
    """

    servers: list[MCPServerConfig] = field(default_factory=list)
    env_allowlist: list[str] = field(default_factory=list)
    enabled: bool = True


def load_mcp_config(path: str) -> MCPConfig:
    """Load MCP configuration from a JSON file.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        An ``MCPConfig`` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the config file is malformed JSON.
        ValueError: If the config structure is invalid.
    """
    with open(path) as f:
        raw: dict[str, Any] = json.load(f)

    return _parse_mcp_config(raw)


def load_mcp_config_from_dict(raw: dict[str, Any]) -> MCPConfig:
    """Load MCP configuration from a dictionary (for testing).

    Args:
        raw: Dictionary with the same structure as the JSON config file.

    Returns:
        An ``MCPConfig`` instance.

    Raises:
        ValueError: If the config structure is invalid.
    """
    return _parse_mcp_config(raw)


def load_mcp_config_from_env(env_var: str = "DANA_MCP_SERVERS") -> MCPConfig | None:
    """Load MCP configuration from an environment variable (D7.5 AC #4).

    Reads ``env_var`` as a JSON array of server specs (the ``mcp_servers``
    payload). Each spec has the shape documented in :func:`_parse_mcp_config`
    (``name``, ``command``, ``args``, ``env``, ``cwd``, ``transport``).

    Args:
        env_var: The environment variable name (default ``DANA_MCP_SERVERS``).

    Returns:
        An ``MCPConfig`` if the variable is set and parses, or ``None`` if it
        is unset/empty (MCP disabled by absence of config).

    Raises:
        json.JSONDecodeError: If the value is malformed JSON.
        ValueError: If the config structure is invalid.
    """
    raw = os.environ.get(env_var)
    if not raw or not raw.strip():
        return None
    servers_raw = json.loads(raw)
    if not isinstance(servers_raw, list):
        raise ValueError(f"{env_var} must be a JSON array of server specs")
    return _parse_mcp_config({"mcp_servers": servers_raw, "mcp_enabled": True})


def _parse_mcp_config(raw: dict[str, Any]) -> MCPConfig:
    """Parse a raw dictionary into an MCPConfig.

    Expected structure::

        {
            "mcp_servers": [
                {
                    "name": "filesystem",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
                    "env": {"ALLOWED_KEY": "value"},
                    "cwd": "/workspace",
                    "transport": "stdio"
                }
            ],
            "mcp_env_allowlist": ["API_KEY", "HOME", "PATH"],
            "mcp_enabled": true
        }
    """
    enabled = raw.get("mcp_enabled", True)

    # Parse env allowlist
    env_allowlist: list[str] = raw.get("mcp_env_allowlist", [])

    # Parse servers
    servers_raw: list[dict[str, Any]] = raw.get("mcp_servers", [])
    servers: list[MCPServerConfig] = []

    for srv in servers_raw:
        name = srv.get("name", "")
        if not name:
            raise ValueError("MCP server config missing required 'name' field")

        transport = srv.get("transport", "stdio")
        if transport not in ("stdio", "http"):
            raise ValueError(f"Unsupported transport '{transport}' for server '{name}'")

        server = MCPServerConfig(
            name=name,
            command=srv.get("command"),
            args=srv.get("args", []),
            env=srv.get("env", {}),
            cwd=srv.get("cwd"),
            transport=transport,
            url=srv.get("url"),
            headers=srv.get("headers", {}),
            timeout=srv.get("timeout", 30.0),
        )

        # Validate: stdio requires command; http requires url
        if transport == "stdio" and not server.command:
            raise ValueError(f"Stdio server '{name}' requires 'command'")
        if transport == "http" and not server.url:
            raise ValueError(f"HTTP server '{name}' requires 'url'")

        servers.append(server)

    return MCPConfig(
        servers=servers,
        env_allowlist=env_allowlist,
        enabled=enabled,
    )


def filter_env_for_server(
    server_config: MCPServerConfig,
    env_allowlist: list[str],
) -> dict[str, str]:
    """Filter environment variables for an MCP server subprocess.

    Applies the allowlist to the server's declared env vars. If the allowlist
    is empty, no env vars are passed (block all).

    Args:
        server_config: The server configuration.
        env_allowlist: List of allowed environment variable keys.

    Returns:
        Filtered environment dict.
    """
    if not env_allowlist:
        # Empty allowlist = block all env vars
        return {}

    filtered: dict[str, str] = {}
    for key, value in server_config.env.items():
        if key in env_allowlist:
            filtered[key] = value
        else:
            logger.debug(
                "Env var '%s' blocked by allowlist for server '%s'",
                key,
                server_config.name,
            )

    return filtered


def is_mcp_enabled(config: MCPConfig | None) -> bool:
    """Check if MCP is enabled.

    Per ADR-012: rollback disables MCP configuration loading. When disabled,
    no MCP servers are started and no MCP tools enter the catalog.

    Args:
        config: The MCP configuration, or None.

    Returns:
        True if MCP is enabled and config exists.
    """
    if config is None:
        return False
    return config.enabled
