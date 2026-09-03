"""D5 MCP Protocol — handshake, discover_tools, call_tool.

Tests use an in-process fake MCP server via memory streams to avoid
subprocess/network dependencies.
"""

from __future__ import annotations

from mcp import types
from mcp.server import Server
from mcp.shared.memory import create_connected_server_and_client_session
import pytest

from dana.core.mcp.protocol import MCPHandshakeResult, call_tool, discover_tools, perform_handshake


def _make_server(
    name: str = "fake-server",
    version: str = "1.0.0",
    tools: list[types.Tool] | None = None,
) -> Server:
    """Create a minimal MCP server with optional tools."""
    server = Server(name=name, version=version)

    if tools is not None:

        @server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return tools

    return server


pytestmark = pytest.mark.asyncio


class TestPerformHandshake:
    """perform_handshake — MCP initialize handshake."""

    async def test_successful_handshake(self):
        """AC #1: Handshake completes with server identity and capabilities."""
        server = _make_server(tools=[])
        async with create_connected_server_and_client_session(server) as session:
            result = await perform_handshake(session)

        assert isinstance(result, MCPHandshakeResult)
        assert result.server_name == "fake-server"
        assert result.server_version == "1.0.0"
        assert result.capabilities.tools is not None

    async def test_handshake_no_tools_capability(self):
        """Handshake succeeds even when server has no tools capability."""
        server = _make_server()  # no tools handler → no tools capability
        async with create_connected_server_and_client_session(server) as session:
            result = await perform_handshake(session)

        assert result.server_name == "fake-server"
        assert result.capabilities.tools is None


class TestDiscoverTools:
    """discover_tools — tools/list integration."""

    async def test_discover_tools_returns_tools(self):
        """AC #1: tools/list returns discovered tool definitions."""
        tools = [
            types.Tool(
                name="read_file",
                description="Read a file from disk",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                    },
                    "required": ["path"],
                },
            ),
            types.Tool(
                name="write_file",
                description="Write content to a file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            ),
        ]
        server = _make_server(tools=tools)
        async with create_connected_server_and_client_session(server) as session:
            discovered = await discover_tools(session)

        assert len(discovered) == 2
        assert discovered[0].name == "read_file"
        assert discovered[1].name == "write_file"

    async def test_discover_tools_empty(self):
        """tools/list returns empty tuple when server has no tools."""
        server = _make_server(tools=[])
        async with create_connected_server_and_client_session(server) as session:
            discovered = await discover_tools(session)

        assert discovered == ()

    async def test_discover_tools_no_capability(self):
        """tools/list returns empty when server has no tools capability."""
        server = _make_server()  # no tools handler
        async with create_connected_server_and_client_session(server) as session:
            discovered = await discover_tools(session)

        assert discovered == ()


class TestCallTool:
    """call_tool — tool invocation."""

    async def test_call_tool_basic(self):
        """Call a tool and get a result."""
        server = _make_server(
            tools=[
                types.Tool(
                    name="echo",
                    description="Echo input",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                        },
                        "required": ["message"],
                    },
                ),
            ],
        )

        @server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: dict,
        ) -> list[types.TextContent]:
            msg = arguments.get("message", "")
            return [types.TextContent(type="text", text=f"Echo: {msg}")]

        async with create_connected_server_and_client_session(server) as session:
            result = await call_tool(session, "echo", {"message": "hello"})

        assert result.isError is False
        assert len(result.content) == 1
        assert result.content[0].text == "Echo: hello"
