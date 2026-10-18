"""MCP single-dispatch native-tool wrapper (D7.5, AC #4).

Per D7.5 Decision 2 (STAR-loop native tools canonical; D2 ToolCatalog deferred),
MCP tools are made reachable from a live host turn by registering ONE
``@named_tool``-decorated async resource method on the agent. The model calls
``call_mcp_tool(tool_name, arguments)``; the wrapper dispatches to the
per-server :class:`~dana.core.mcp.execution.MCPExecutionAdapter`.

This avoids the per-tool native-wrapper impedance mismatch (native tool
schemas are built from Python type hints; MCP tools carry arbitrary JSON
``inputSchema``) and does NOT touch the deferred D2 ``ToolCatalog``/
``ToolExecutionEngine``. Per-tool UX (model calls each MCP tool by name with
its real inputSchema) is deferred to the D7.6 "D2 Catalog Migration" story,
which wires the catalog with correct schemas for all tools incl. MCP.

No permission gating (catalog-coupled; deferred to D7.6).
"""

from __future__ import annotations

import logging
from typing import Any

from dana.common.protocols.war import named_tool
from dana.core.mcp.cancellation import MCPCancellationTracker
from dana.core.mcp.execution import MCPExecutionAdapter
from dana.core.mcp.leases import MCPLeaseManager
from dana.core.resource.base_resource import BaseResource


logger = logging.getLogger(__name__)


def _format_mcp_result(result: dict[str, Any]) -> str:
    """Render an MCPExecutionAdapter result dict as a string for the model."""
    if not result.get("success", True):
        err = result.get("result", "unknown error")
        return f"MCP tool error: {err}"
    content = result.get("result")
    if isinstance(content, str):
        return content
    # MCP content is often a list of content blocks; flatten to text.
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                text = getattr(block, "text", None)
                parts.append(text if isinstance(text, str) else str(block))
        return "\n".join(parts) if parts else ""
    return str(content) if content is not None else ""


def _make_mcp_dispatcher(adapter: MCPExecutionAdapter, tool_name: str) -> Any:
    """Build an async dispatch callable for one MCP tool (per-tool UX, A2).

    Returns an async callable yielding the execution result dict with model-facing
    text. Success, cancellation disposition, harness identity and the observed
    transport request ID survive through ToolExecutor and its durable after-hook.
    The optional second parameter forwards the harness identity to the adapter.
    Errors are returned as failed results rather than raised.
    """

    async def dispatch(arguments: dict[str, Any], tool_call_id: str | None = None) -> dict[str, Any]:
        try:
            result = await adapter.call_tool(tool_name.partition(":")[2] or tool_name, arguments or {}, tool_call_id)
        except Exception as exc:  # noqa: BLE001 — surface to the model, never crash the turn
            logger.warning("MCP dispatch failed for '%s': %s", tool_name, exc)
            result = {"success": False, "result": f"Error calling MCP tool '{tool_name}': {exc}"}
            if tool_call_id is not None:
                result["tool_call_id"] = tool_call_id
        result["result"] = _format_mcp_result(result)
        result["type"] = "mcp"
        result["target"] = tool_name
        return result

    return dispatch


class MCPDispatchResource(BaseResource):
    """A native-tool resource that dispatches to MCP tools by name (D7.5 AC #4).

    One instance per session. Holds a map from MCP tool name (both the
    namespaced ``server:tool`` and the bare ``tool`` alias) to the
    :class:`MCPExecutionAdapter` for its server. Exposes a single
    ``@named_tool("call_mcp_tool")`` async method so the STAR loop's native
    tool registry discovers + dispatches it.
    """

    object_id = "mcp"

    def __init__(
        self,
        adapters: dict[str, MCPExecutionAdapter],
        tool_descriptions: dict[str, str] | None = None,
    ) -> None:
        super().__init__(resource_id="mcp", auto_register=False)
        self._adapters = adapters
        # tool name -> "name(args): description" for the wrapper docstring
        self._tool_descriptions = dict(tool_descriptions or {})

    @property
    def available_tools(self) -> list[str]:
        """The MCP tool names this resource can dispatch to."""
        return sorted(self._adapters.keys())

    @named_tool("call_mcp_tool")
    async def call(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool by name. The tool runs on a configured MCP server.

        Args:
            tool_name: The MCP tool name. Use the namespaced form
                ``server:tool`` if ambiguous; the bare tool name works when
                unique. Available tools are listed below.
            arguments: The tool arguments as a JSON object (dict).

        Returns:
            The MCP tool result as a string, or an error message.

        Available MCP tools:
        {tools}
        """
        adapter = self._adapters.get(tool_name)
        if adapter is None:
            avail = ", ".join(self.available_tools) or "(none configured)"
            return f"Error: MCP tool '{tool_name}' not found. Available: {avail}"
        try:
            result = await adapter.call_tool(tool_name.partition(":")[2] or tool_name, arguments or {})
        except Exception as exc:  # noqa: BLE001 — surface to the model, never crash the turn
            logger.warning("MCP dispatch failed for '%s': %s", tool_name, exc)
            return f"Error calling MCP tool '{tool_name}': {exc}"
        return self._format_result(result)

    def _format_result(self, result: dict[str, Any]) -> str:
        """Render an MCPExecutionAdapter result dict as a string for the model."""
        return _format_mcp_result(result)

    def _refresh_docstring(self) -> None:
        """Inject the available-tools list into ``call``'s docstring (schema description).

        The bound method's ``__doc__`` is read-only (it proxies to the function's);
        set the underlying function's ``__doc__`` instead. There is one
        ``MCPDispatchResource`` per session, so sharing the class method's
        docstring is acceptable.
        """
        if self._tool_descriptions:
            lines = [f"  - {n}: {d}" for n, d in sorted(self._tool_descriptions.items())]
            tools_block = "\n".join(lines) if lines else "(none configured)"
        else:
            tools_block = ", ".join(self.available_tools) or "(none configured)"
        func = self.call.__func__
        func.__doc__ = (func.__doc__ or "").format(tools=tools_block) if "{tools}" in (func.__doc__ or "") else func.__doc__


class MCPWiring:
    """Holds the built MCP dispatch resource + open transports for cleanup.

    D7 follow-up 1+2 (per-tool UX): also exposes the per-MCP tool schemas
    (correct inputSchema-derived OpenAI schemas, namespaced ``server:tool``),
    the set of MCP tool names (for policy classification), and a dispatch map
    (tool_name -> async callable returning the formatted result string) so the
    model can call each MCP tool BY NAME and the ToolExecutor dispatches it.

    The payload attributes are also declared at class level (empty defaults) so the
    wiring's public shape is complete: ``unittest.mock.MagicMock(spec=MCPWiring)``
    builds its allowed attributes from the class, and a test double that omitted
    them would fail in the session's catalog wiring rather than at the assertion.
    """

    #: Per-tool MCP schemas (namespaced), the MCP names/aliases, the per-tool
    #: dispatch map, and the namespaced-MCP effect metadata (I9). Always replaced
    #: per instance in ``__init__``; declared here to complete the public shape.
    mcp_schemas: list[dict[str, Any]] = []
    mcp_names: frozenset[str] = frozenset()
    dispatch_map: dict[str, Any] = {}
    mcp_effects: dict[str, Any] = {}

    def __init__(
        self,
        resource: MCPDispatchResource,
        lease_manager: MCPLeaseManager,
        transports: list[Any],
        contexts: list[Any],
        *,
        mcp_schemas: list[dict[str, Any]] | None = None,
        mcp_names: frozenset[str] | None = None,
        dispatch_map: dict[str, Any] | None = None,
        mcp_effects: dict[str, Any] | None = None,
    ) -> None:
        self.resource = resource
        self.lease_manager = lease_manager
        self._transports = transports
        self._contexts = contexts  # asynccontextmanager instances awaiting __aexit__
        # D7 follow-up 1+2: per-tool MCP UX + per-MCP policy.
        self.mcp_schemas: list[dict[str, Any]] = list(mcp_schemas or [])
        self.mcp_names: frozenset[str] = mcp_names or frozenset()
        self.dispatch_map: dict[str, Any] = dict(dispatch_map or {})
        # I9: namespaced MCP name -> EffectMetadata derived from the server's own
        # declaration, so scheduling overlaps read-only tools and orders unknown
        # mutations per server. Empty when no tool declared anything (the executor
        # then falls back to per-server ordering from the name itself).
        self.mcp_effects: dict[str, Any] = dict(mcp_effects or {})

    async def close(self) -> None:
        """Release leases + close transports (session teardown)."""
        self.lease_manager.release_all()
        for cm in reversed(self._contexts):
            with _SuppressCtx():
                await cm.__aexit__(None, None, None)
        self._contexts.clear()
        self._transports.clear()


def _mcp_effect_metadata(tool: Any, server_name: str) -> Any:
    """Effect metadata for one discovered MCP tool, from its own declaration (I9).

    Only an explicit ``readOnlyHint=True`` annotation is trusted as read-only: the
    protocol marks absence as *unknown* (the field is ``Optional[bool]``), so an
    unannotated tool is a mutation of unknown blast radius and stays ordered
    against other mutations of the same server. A server is a natural, honest
    conflict domain: two unannotated calls to one server may share state, two
    calls to different servers do not.
    """
    from dana.core.policy.effects import Effect, EffectKind, EffectMetadata

    annotations = getattr(tool, "annotations", None)
    if annotations is not None and getattr(annotations, "readOnlyHint", None) is True:
        # A read is scoped to its *server* too: two reads of one server overlap
        # (read-only calls never conflict), while a read cannot be proved disjoint
        # from a same-server mutation - an unannotated tool's blast radius is the
        # server, so the read is ordered against it. A target-less READ would look
        # disjoint from every mutation and be scheduled straight through one.
        return EffectMetadata(effects=(Effect(kind=EffectKind.READ, target=server_name),), is_sensitive=False)
    return EffectMetadata(effects=(Effect(kind=EffectKind.EXECUTE, target=server_name),), is_sensitive=False)


class _SuppressCtx:
    """Suppress exceptions during cleanup (best-effort close)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


async def build_mcp_dispatch_resource(
    config: Any,
    transport_factory: Any | None = None,
) -> MCPWiring | None:
    """Build an :class:`MCPWiring` from an MCP configuration (D7.5 AC #4).

    For each configured server: connect the transport, perform the MCP
    handshake, discover tools, build a shared :class:`MCPExecutionAdapter`
    (one per server), and register each tool (namespaced + alias) on the
    :class:`MCPDispatchResource`. Lease lifecycle: a required-lease failure
    raises (the session cannot function); an optional-lease failure degrades
    (the server's tools are skipped, the session continues).

    Args:
        config: An ``MCPConfig`` (from ``load_mcp_config`` or
            ``load_mcp_config_from_env``). ``config.enabled`` must be True.
        transport_factory: Callable ``(MCPServerConfig) -> transport``.
            Defaults to :class:`~dana.core.mcp.transports.stdio.MCPStdioTransport`
            for stdio servers. Injected by tests with a mock transport.

    Returns:
        An :class:`MCPWiring` (with the dispatch resource + open transports),
        or ``None`` if no servers are configured.

    Raises:
        RuntimeError: If a REQUIRED server's lease fails (handshake/discovery
            error). Optional-server failures are degraded, not raised.
    """
    from dana.core.mcp.config import MCPConfig
    from dana.core.mcp.protocol import discover_tools, perform_handshake
    from dana.core.mcp.transports.stdio import MCPStdioTransport

    if not isinstance(config, MCPConfig) or not config.enabled or not config.servers:
        return None

    if transport_factory is None:

        def transport_factory(server):  # type: ignore[no-redef]
            return MCPStdioTransport(
                command=server.command or "",
                args=server.args,
                env=server.env or None,
                cwd=server.cwd,
            )

    lease_manager = MCPLeaseManager()
    cancellation_tracker = MCPCancellationTracker()
    adapters: dict[str, MCPExecutionAdapter] = {}
    descriptions: dict[str, str] = {}
    transports: list[Any] = []
    contexts: list[Any] = []
    # D7 follow-up 1+2: per-tool MCP UX (real inputSchema schemas) + dispatch map.
    mcp_schemas: list[dict[str, Any]] = []
    mcp_names: set[str] = set()
    dispatch_map: dict[str, Any] = {}
    mcp_effects: dict[str, Any] = {}

    for server in config.servers:
        lease = lease_manager.create_lease(server.name, required=True)
        try:
            transport = transport_factory(server)
            cm = transport.connect()
            await cm.__aenter__()
            contexts.append(cm)
            transports.append(transport)
            if transport.session is None:
                raise RuntimeError("transport did not establish a session")
            await perform_handshake(transport.session, client_name="dana", client_version="0.2.0")
            tools = await discover_tools(transport.session)
            adapter = MCPExecutionAdapter(transport, cancellation_tracker)
            from dana.core.mcp.schema_conversion import mcp_tool_to_catalog_entry

            for tool in tools:
                namespaced = f"{server.name}:{tool.name}"
                adapters[namespaced] = adapter
                # Bare alias only when unique (don't shadow another server's tool).
                if tool.name not in adapters:
                    adapters[tool.name] = adapter
                descriptions[namespaced] = tool.description or tool.name
                # D7 follow-up 1+2: per-tool schema (correct inputSchema) + dispatch.
                entry = mcp_tool_to_catalog_entry(tool, server.name)
                mcp_schemas.append(entry.schema)
                mcp_names.add(namespaced)
                dispatch_map[namespaced] = _make_mcp_dispatcher(adapter, namespaced)
                mcp_effects[namespaced] = _mcp_effect_metadata(tool, server.name)
            lease.activate([])
            logger.info("MCP server '%s' connected (%d tools)", server.name, len(tools))
        except Exception as exc:  # noqa: BLE001
            lease.fail(str(exc))
            # Best-effort close of the partially-open transport
            if contexts and contexts[-1] is not None:
                with _SuppressCtx():
                    await contexts.pop().__aexit__(None, None, None)
            if lease.required:
                # Roll back already-open servers + raise (required lease failed).
                for cm in reversed(contexts):
                    with _SuppressCtx():
                        await cm.__aexit__(None, None, None)
                raise RuntimeError(f"Required MCP lease failed for '{server.name}': {exc}") from exc
            # optional: degrade + continue
            logger.warning("Optional MCP lease failed for '%s': %s", server.name, exc)

    if not adapters:
        return None

    resource = MCPDispatchResource(adapters, descriptions)
    resource._refresh_docstring()
    return MCPWiring(
        resource,
        lease_manager,
        transports,
        contexts,
        mcp_schemas=mcp_schemas,
        mcp_names=frozenset(mcp_names),
        dispatch_map=dispatch_map,
        mcp_effects=mcp_effects,
    )
