"""HTTP MCP transport adapter — Streamable HTTP and SSE with connection pooling.

Per ADR-008: HTTP connections may be pooled internally when descriptors +
credentials match.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from dana.core.mcp.protocol import MCPHandshakeResult, call_tool, discover_tools, perform_handshake


logger = logging.getLogger(__name__)


class MCPHttpTransport:
    """HTTP-based MCP transport with optional connection pooling.

    Wraps ``mcp.client.streamable_http.streamable_http_client`` +
    ``ClientSession`` into a single async context manager.

    Supports both Streamable HTTP transport and SSE transport.

    Usage::

        async with MCPHttpTransport(url="http://localhost:8080/mcp") as transport:
            result = await transport.handshake()
            tools = await transport.list_tools()
            response = await transport.call_tool("my_tool", {"arg": "val"})
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        client_name: str = "dana",
        client_version: str = "0.2.0",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._headers = headers
        self._timeout = timeout
        self._client_name = client_name
        self._client_version = client_version
        self._provided_client = http_client
        self._owned_client: httpx.AsyncClient | None = None
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
    async def connect(self) -> AsyncIterator[MCPHttpTransport]:
        """Connect to the HTTP MCP server and yield self.

        Manages the HTTP connection lifecycle.
        """
        # Create or reuse the HTTP client
        if self._provided_client is not None:
            http_client = self._provided_client
        else:
            http_client = httpx.AsyncClient(
                headers=self._headers,
                timeout=httpx.Timeout(self._timeout),
            )
            self._owned_client = http_client

        async with streamable_http_client(
            self._url,
            http_client=http_client,
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(read_stream, write_stream) as session:
                self._session = session
                try:
                    yield self
                finally:
                    self._session = None
                    self._handshake_result = None
                    if self._owned_client is not None:
                        await self._owned_client.aclose()
                        self._owned_client = None

    async def close(self) -> None:
        """Close the transport. No-op if already closed."""
        self._session = None
        self._handshake_result = None
        if self._owned_client is not None:
            await self._owned_client.aclose()
            self._owned_client = None
