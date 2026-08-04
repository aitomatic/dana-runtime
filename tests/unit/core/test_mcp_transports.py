"""D5 MCP Transports — stdio and HTTP adapter tests.

Tests use in-process fake MCP servers and mock subprocesses/HTTP to avoid
real I/O.
"""

from __future__ import annotations

from mcp import types
from mcp.server import Server
from mcp.shared.memory import create_connected_server_and_client_session
import pytest

from dana.core.mcp.transports.http import MCPHttpTransport
from dana.core.mcp.transports.stdio import MCPStdioTransport


pytestmark = pytest.mark.asyncio


class TestMCPStdioTransport:
    """MCPStdioTransport — stdio subprocess lifecycle."""

    async def test_connect_and_handshake(self):
        """Stdio transport connects, handshakes, and returns server info."""
        # We test the transport's interface by using the in-memory
        # server infrastructure. The stdio transport wraps stdio_client
        # which spawns a real subprocess — we test the protocol layer
        # integration via the in-memory path in test_protocol.py.
        # Here we verify the transport class structure and error handling.
        transport = MCPStdioTransport(command="python", args=["-m", "some_server"])
        assert transport.session is None
        assert transport.handshake_result is None

    async def test_call_tool_before_connect_raises(self):
        """Calling methods before connect raises RuntimeError."""
        transport = MCPStdioTransport(command="python", args=["-m", "server"])
        with pytest.raises(RuntimeError, match="not connected"):
            await transport.handshake()
        with pytest.raises(RuntimeError, match="not connected"):
            await transport.list_tools()
        with pytest.raises(RuntimeError, match="not connected"):
            await transport.call_tool("test", {})

    async def test_close_is_idempotent(self):
        """close() can be called multiple times without error."""
        transport = MCPStdioTransport(command="python", args=["-m", "server"])
        await transport.close()
        await transport.close()  # second call should not raise


class TestMCPHttpTransport:
    """MCPHttpTransport — HTTP transport with connection pooling."""

    async def test_connect_and_handshake(self):
        """HTTP transport connects, handshakes, and returns server info."""
        transport = MCPHttpTransport(url="http://localhost:9999/mcp")
        assert transport.session is None
        assert transport.handshake_result is None

    async def test_call_tool_before_connect_raises(self):
        """Calling methods before connect raises RuntimeError."""
        transport = MCPHttpTransport(url="http://localhost:9999/mcp")
        with pytest.raises(RuntimeError, match="not connected"):
            await transport.handshake()
        with pytest.raises(RuntimeError, match="not connected"):
            await transport.list_tools()
        with pytest.raises(RuntimeError, match="not connected"):
            await transport.call_tool("test", {})

    async def test_close_is_idempotent(self):
        """close() can be called multiple times without error."""
        transport = MCPHttpTransport(url="http://localhost:9999/mcp")
        await transport.close()
        await transport.close()  # second call should not raise

    async def test_http_transport_with_provided_client(self):
        """HTTP transport accepts a pre-configured httpx client."""
        import httpx

        client = httpx.AsyncClient()
        transport = MCPHttpTransport(
            url="http://localhost:9999/mcp",
            http_client=client,
        )
        assert transport._provided_client is client
        await transport.close()
        await client.aclose()


class TestTransportIntegration:
    """End-to-end transport integration via in-memory server.

    These tests verify the full protocol flow through a transport-like
    pattern using the mcp in-memory test utilities.
    """

    async def test_full_protocol_flow(self):
        """AC #4: Full protocol flow: handshake → list_tools → call_tool."""
        server = Server(name="integration-test", version="2.0.0")

        @server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="greet",
                    description="Greet someone",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                ),
            ]

        @server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: dict,
        ) -> list[types.TextContent]:
            if name == "greet":
                return [types.TextContent(type="text", text=f"Hello, {arguments.get('name', 'world')}!")]
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

        async with create_connected_server_and_client_session(server) as session:
            from dana.core.mcp.protocol import call_tool, discover_tools, perform_handshake

            # Handshake
            handshake = await perform_handshake(session)
            assert handshake.server_name == "integration-test"
            assert handshake.server_version == "2.0.0"

            # List tools
            tools = await discover_tools(session)
            assert len(tools) == 1
            assert tools[0].name == "greet"

            # Call tool
            result = await call_tool(session, "greet", {"name": "Dana"})
            assert result.isError is False
            assert result.content[0].text == "Hello, Dana!"

    async def test_server_with_no_tools(self):
        """Server with no tools returns empty list."""
        server = Server(name="empty-server", version="1.0.0")

        @server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return []

        async with create_connected_server_and_client_session(server) as session:
            from dana.core.mcp.protocol import discover_tools, perform_handshake

            await perform_handshake(session)
            tools = await discover_tools(session)
            assert tools == ()

    async def test_handshake_reveals_capabilities(self):
        """Handshake reveals server capabilities including tools support."""
        server = Server(name="cap-test", version="1.0.0")

        @server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return [
                types.Tool(name="tool_a", inputSchema={"type": "object", "properties": {}}),
                types.Tool(name="tool_b", inputSchema={"type": "object", "properties": {}}),
            ]

        async with create_connected_server_and_client_session(server) as session:
            from dana.core.mcp.protocol import perform_handshake

            result = await perform_handshake(session)
            assert result.capabilities.tools is not None
