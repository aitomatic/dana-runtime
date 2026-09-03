"""MCPHandshakeResult dataclass tests (synchronous)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

from mcp import types
import pytest

from dana.core.mcp.protocol import MCPHandshakeResult


class TestMCPHandshakeResult:
    """MCPHandshakeResult dataclass."""

    def test_frozen(self):
        """MCPHandshakeResult is frozen (immutable)."""
        result = MCPHandshakeResult(
            server_name="test",
            server_version="1.0",
            capabilities=types.ServerCapabilities(),
        )
        with pytest.raises(FrozenInstanceError):
            result.server_name = "other"  # type: ignore[misc]

    def test_default_tools_empty(self):
        """tools defaults to empty tuple."""
        result = MCPHandshakeResult(
            server_name="test",
            server_version="1.0",
            capabilities=types.ServerCapabilities(),
        )
        assert result.tools == ()
