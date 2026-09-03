"""D5 MCP Configuration — configuration loading and environment allowlist.

AC #3: Allowlisted environment enforced — env var with disallowed key is stripped.
AC #4: Configuration loading succeeds and rollback disables MCP config loading.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from dana.core.mcp.config import (
    MCPConfig,
    MCPServerConfig,
    filter_env_for_server,
    is_mcp_enabled,
    load_mcp_config,
    load_mcp_config_from_dict,
)


class TestMCPServerConfig:
    """MCPServerConfig — individual server configuration."""

    def test_stdio_server_defaults(self):
        """Stdio server defaults."""
        config = MCPServerConfig(name="test", command="npx", args=["-y", "server"])
        assert config.name == "test"
        assert config.command == "npx"
        assert config.args == ["-y", "server"]
        assert config.env == {}
        assert config.cwd is None
        assert config.transport == "stdio"
        assert config.url is None
        assert config.timeout == 30.0

    def test_http_server(self):
        """HTTP server configuration."""
        config = MCPServerConfig(
            name="http-server",
            transport="http",
            url="http://localhost:8080/mcp",
            headers={"Authorization": "Bearer token"},
            timeout=60.0,
        )
        assert config.transport == "http"
        assert config.url == "http://localhost:8080/mcp"
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.timeout == 60.0


class TestLoadMCPConfig:
    """load_mcp_config — JSON file loading."""

    def test_load_basic_config(self):
        """AC #4: Load a basic MCP config from a dict."""
        raw = {
            "mcp_servers": [
                {
                    "name": "filesystem",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    "env": {"ALLOWED_KEY": "value"},
                    "transport": "stdio",
                },
                {
                    "name": "web-search",
                    "transport": "http",
                    "url": "http://localhost:8080/mcp",
                    "headers": {"X-API-Key": "secret"},
                },
            ],
            "mcp_env_allowlist": ["ALLOWED_KEY", "PATH"],
            "mcp_enabled": True,
        }

        config = load_mcp_config_from_dict(raw)

        assert config.enabled is True
        assert config.env_allowlist == ["ALLOWED_KEY", "PATH"]
        assert len(config.servers) == 2

        # Stdio server
        assert config.servers[0].name == "filesystem"
        assert config.servers[0].command == "npx"
        assert config.servers[0].transport == "stdio"

        # HTTP server
        assert config.servers[1].name == "web-search"
        assert config.servers[1].transport == "http"
        assert config.servers[1].url == "http://localhost:8080/mcp"

    def test_load_from_json_file(self):
        """AC #4: Load config from a JSON file."""
        raw = {
            "mcp_servers": [
                {
                    "name": "test-server",
                    "command": "python",
                    "args": ["-m", "server"],
                    "transport": "stdio",
                },
            ],
            "mcp_env_allowlist": ["HOME"],
            "mcp_enabled": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(raw, f)
            f.flush()
            config_path = f.name

        try:
            config = load_mcp_config(config_path)
            assert len(config.servers) == 1
            assert config.servers[0].name == "test-server"
            assert config.servers[0].command == "python"
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_load_file_not_found(self):
        """FileNotFoundError when config file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_mcp_config("/nonexistent/path/config.json")

    def test_load_malformed_json(self):
        """json.JSONDecodeError when config file is malformed."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json}")
            f.flush()
            config_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_mcp_config(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_missing_name_raises(self):
        """Server config without name raises ValueError."""
        raw = {
            "mcp_servers": [
                {
                    "command": "npx",
                    "transport": "stdio",
                },
            ],
        }
        with pytest.raises(ValueError, match="missing required 'name'"):
            load_mcp_config_from_dict(raw)

    def test_stdio_without_command_raises(self):
        """Stdio server without command raises ValueError."""
        raw = {
            "mcp_servers": [
                {
                    "name": "bad-server",
                    "transport": "stdio",
                },
            ],
        }
        with pytest.raises(ValueError, match="requires 'command'"):
            load_mcp_config_from_dict(raw)

    def test_http_without_url_raises(self):
        """HTTP server without url raises ValueError."""
        raw = {
            "mcp_servers": [
                {
                    "name": "bad-http",
                    "transport": "http",
                },
            ],
        }
        with pytest.raises(ValueError, match="requires 'url'"):
            load_mcp_config_from_dict(raw)

    def test_unsupported_transport_raises(self):
        """Unsupported transport raises ValueError."""
        raw = {
            "mcp_servers": [
                {
                    "name": "bad-transport",
                    "command": "npx",
                    "transport": "websocket",
                },
            ],
        }
        with pytest.raises(ValueError, match="Unsupported transport"):
            load_mcp_config_from_dict(raw)

    def test_empty_servers(self):
        """Empty servers list is valid."""
        raw = {"mcp_servers": [], "mcp_env_allowlist": [], "mcp_enabled": True}
        config = load_mcp_config_from_dict(raw)
        assert config.servers == []
        assert config.env_allowlist == []

    def test_default_enabled(self):
        """mcp_enabled defaults to True when not specified."""
        raw = {"mcp_servers": []}
        config = load_mcp_config_from_dict(raw)
        assert config.enabled is True


class TestFilterEnvForServer:
    """filter_env_for_server — environment allowlist enforcement."""

    def test_allowlist_allows_matching_keys(self):
        """AC #3: Allowed keys pass through."""
        server = MCPServerConfig(
            name="test",
            command="npx",
            env={"API_KEY": "secret123", "HOME": "/home/user", "PATH": "/usr/bin"},
        )
        filtered = filter_env_for_server(server, ["API_KEY", "PATH"])
        assert filtered == {"API_KEY": "secret123", "PATH": "/usr/bin"}

    def test_allowlist_blocks_disallowed_keys(self):
        """AC #3: Disallowed keys are stripped."""
        server = MCPServerConfig(
            name="test",
            command="npx",
            env={"API_KEY": "secret123", "DB_PASSWORD": "hunter2", "PATH": "/usr/bin"},
        )
        filtered = filter_env_for_server(server, ["PATH"])
        assert filtered == {"PATH": "/usr/bin"}
        assert "API_KEY" not in filtered
        assert "DB_PASSWORD" not in filtered

    def test_empty_allowlist_blocks_all(self):
        """AC #3: Empty allowlist blocks all env vars."""
        server = MCPServerConfig(
            name="test",
            command="npx",
            env={"API_KEY": "secret123", "PATH": "/usr/bin"},
        )
        filtered = filter_env_for_server(server, [])
        assert filtered == {}

    def test_no_env_vars(self):
        """Server with no env vars returns empty dict."""
        server = MCPServerConfig(name="test", command="npx")
        filtered = filter_env_for_server(server, ["API_KEY"])
        assert filtered == {}

    def test_allowlist_with_extra_keys(self):
        """Allowlist keys that don't exist in server env are ignored."""
        server = MCPServerConfig(
            name="test",
            command="npx",
            env={"EXISTING_KEY": "value"},
        )
        filtered = filter_env_for_server(server, ["EXISTING_KEY", "NONEXISTENT"])
        assert filtered == {"EXISTING_KEY": "value"}


class TestIsMCPEnabled:
    """is_mcp_enabled — rollback mechanism."""

    def test_enabled_config(self):
        """AC #4: Enabled config returns True."""
        config = MCPConfig(enabled=True)
        assert is_mcp_enabled(config) is True

    def test_disabled_config(self):
        """AC #4: Disabled config returns False (rollback)."""
        config = MCPConfig(enabled=False)
        assert is_mcp_enabled(config) is False

    def test_none_config(self):
        """None config returns False."""
        assert is_mcp_enabled(None) is False

    def test_rollback_then_re_enable(self):
        """Edge case: rollback then re-enable."""
        config = MCPConfig(enabled=False)
        assert is_mcp_enabled(config) is False
        config.enabled = True
        assert is_mcp_enabled(config) is True
