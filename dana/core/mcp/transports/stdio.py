"""Stdio MCP transport adapter — managed subprocess lifecycle.

Per ADR-008: stdio servers default to dedicated managed processes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client

from dana.core.mcp.protocol import MCPHandshakeResult, call_tool, discover_tools, perform_handshake


logger = logging.getLogger(__name__)


class MCPStdioTransport:
    """Stdio-based MCP transport with managed subprocess lifecycle.

    Wraps ``mcp.stdio_client`` + ``ClientSession`` into a single
    async context manager that handles process start, communication,
    and graceful shutdown.

    Usage::

        async with MCPStdioTransport(command="npx", args=["@modelcontextprotocol/server-filesystem", "/path"]) as transport:
            result = await transport.handshake()
            tools = await transport.list_tools()
            response = await transport.call_tool("read_file", {"path": "/path/file.txt"})
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        client_name: str = "dana",
        client_version: str = "0.2.0",
    ) -> None:
        self._server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
            cwd=cwd,
        )
        self._client_name = client_name
        self._client_version = client_version
        self._session: ClientSession | None = None
        self._handshake_result: MCPHandshakeResult | None = None

    @property
    def session(self) -> ClientSession | None:
        """The underlying ``ClientSession``, if connected."""
        return self._session

    @property
    def handshake_result(self) -> MCPHandshakeResult | None:
        """Result of the handshake, if completed."""
        return self._handshake_result

    async def handshake(self) -> MCPHandshakeResult:
        """Perform the MCP initialize handshake.

        Returns:
            ``MCPHandshakeResult`` with server identity and capabilities.

        Raises:
            RuntimeError: If the transport is not connected.
        """
        if self._session is None:
            raise RuntimeError("Transport not connected. Use 'async with' to connect.")
        result = await perform_handshake(self._session, self._client_name, self._client_version)
        self._handshake_result = result
        return result

    async def list_tools(self) -> tuple[Any, ...]:
        """Discover tools from the server.

        Returns:
            Tuple of ``Tool`` definitions.

        Raises:
            RuntimeError: If the transport is not connected or handshake not done.
        """
        if self._session is None:
            raise RuntimeError("Transport not connected. Use 'async with' to connect.")
        return await discover_tools(self._session)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a tool on the MCP server.

        Args:
            name: The tool name to call.
            arguments: Optional arguments.

        Returns:
            ``CallToolResult`` from the server.
        """
        if self._session is None:
            raise RuntimeError("Transport not connected. Use 'async with' to connect.")
        return await call_tool(self._session, name, arguments)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[MCPStdioTransport]:
        """Connect to the stdio server and yield self.

        Manages the subprocess lifecycle: spawn, communicate, terminate.
        """
        async with stdio_client(self._server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                self._session = session
                try:
                    yield self
                finally:
                    self._session = None
                    self._handshake_result = None

    async def close(self) -> None:
        """Close the transport. No-op if already closed."""
        self._session = None
        self._handshake_result = None
