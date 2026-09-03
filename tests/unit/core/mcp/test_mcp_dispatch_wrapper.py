"""D7.5 Piece C — MCP single-dispatch wrapper (AC #4: MCP reachable from a host turn).

Covers:
- MCPDispatchResource: dispatch by name, result formatting, unknown-tool + error handling.
- Native-tool discoverability: @named_tool -> extract_tool_use_methods + parse_method_signature.
- End-to-end dispatch via ToolExecutor.execute_tools_async (the STAR-loop dispatch path).
- load_mcp_config_from_env: DANA_MCP_SERVERS parsing.
- build_mcp_dispatch_resource: lease lifecycle + tool registration (mock transport).
- AgentSession._wire_mcp_tools: appends the resource when enabled + configured; no crash on failure.
"""

from __future__ import annotations

import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dana.common.utils.misc import Misc
from dana.core.mcp.cancellation import MCPCancellationTracker
from dana.core.mcp.config import MCPConfig, MCPServerConfig, load_mcp_config_from_env
from dana.core.mcp.dispatch_wrapper import MCPDispatchResource, MCPWiring, build_mcp_dispatch_resource
from dana.core.mcp.execution import MCPExecutionAdapter
from dana.core.mcp.leases import LeaseState
from dana.core.tool.tool_executor import ToolExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_adapter(result_text: str = "hello", is_error: bool = False) -> MCPExecutionAdapter:
    transport = MagicMock()
    transport.call_tool = AsyncMock(return_value=MagicMock(content=[{"type": "text", "text": result_text}], isError=is_error))
    return MCPExecutionAdapter(transport, MCPCancellationTracker())


def _mock_transport_with_tools(tools):
    """A mock MCP transport that yields ``tools`` from discover_tools."""
    transport = MagicMock()
    transport.session = AsyncMock()
    transport.call_tool = AsyncMock(return_value=MagicMock(content=[{"type": "text", "text": "ok"}], isError=False))

    @contextlib.asynccontextmanager
    async def connect():
        yield transport

    transport.connect = connect
    return transport


# ---------------------------------------------------------------------------
# MCPDispatchResource
# ---------------------------------------------------------------------------


class TestMCPDispatchResource:
    @pytest.mark.asyncio
    async def test_call_dispatches_to_adapter(self):
        adapter = _mock_adapter("hello world")
        r = MCPDispatchResource({"greet": adapter}, {"greet": "Greet a person"})
        result = await r.call("greet", {"name": "World"})
        assert result == "hello world"
        adapter._transport.call_tool.assert_awaited_once_with("greet", {"name": "World"})

    @pytest.mark.asyncio
    async def test_call_namespaced_and_alias(self):
        adapter = _mock_adapter("hi")
        r = MCPDispatchResource({"fs:greet": adapter})
        assert await r.call("fs:greet", {}) == "hi"
        # The bare alias is only registered at build time (build_mcp_dispatch_resource);
        # a directly-constructed resource only has the keys passed in.
        result = await r.call("greet", {})
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        r = MCPDispatchResource({"greet": _mock_adapter()})
        result = await r.call("nope", {})
        assert "not found" in result and "greet" in result

    @pytest.mark.asyncio
    async def test_adapter_error_is_surfaced(self):
        adapter = _mock_adapter("boom", is_error=True)
        r = MCPDispatchResource({"greet": adapter})
        result = await r.call("greet", {})
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_call_exception_does_not_propagate(self):
        transport = MagicMock()
        transport.call_tool = AsyncMock(side_effect=RuntimeError("server down"))
        adapter = MCPExecutionAdapter(transport, MCPCancellationTracker())
        r = MCPDispatchResource({"greet": adapter})
        result = await r.call("greet", {})
        assert "Error" in result and "server down" in result

    @pytest.mark.asyncio
    async def test_empty_arguments_allowed(self):
        adapter = _mock_adapter("ok")
        r = MCPDispatchResource({"ping": adapter})
        assert await r.call("ping", None) == "ok"  # type: ignore[arg-type]

    def test_available_tools_lists_names(self):
        r = MCPDispatchResource({"a": _mock_adapter(), "b": _mock_adapter()})
        assert r.available_tools == ["a", "b"]


# ---------------------------------------------------------------------------
# Native-tool discoverability (registry + schema)
# ---------------------------------------------------------------------------


class TestNativeToolDiscoverability:
    def test_extract_tool_use_methods_finds_call(self):
        r = MCPDispatchResource({"greet": _mock_adapter()})
        methods = Misc.extract_tool_use_methods(r)
        names = [name for name, _ in methods]
        assert "call" in names

    def test_parse_method_signature_reads_custom_tool_name(self):
        r = MCPDispatchResource({"greet": _mock_adapter()})
        sig = Misc.parse_method_signature(r.call)
        assert sig.tool_name == "call_mcp_tool"
        param_names = [p.name for p in sig.parameters]
        assert param_names == ["tool_name", "arguments"]

    def test_generate_tool_schema_uses_custom_name(self):
        from dana.core.tool.tool_schema import _method_signature_to_schema

        r = MCPDispatchResource({"greet": _mock_adapter()})
        methods = Misc.extract_tool_use_methods(r)
        name, method = methods[0]
        sig = Misc.parse_method_signature(method, object_id=r.object_id)
        schema = _method_signature_to_schema(sig, object_id=r.object_id, object_type="resource")
        assert schema["function"]["name"] == "call_mcp_tool"
        assert "tool_name" in schema["function"]["parameters"]["properties"]
        assert "arguments" in schema["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# End-to-end dispatch via ToolExecutor (the STAR-loop dispatch path)
# ---------------------------------------------------------------------------


class TestToolExecutorDispatch:
    @pytest.mark.asyncio
    async def test_call_mcp_tool_dispatches_through_executor(self):
        adapter = _mock_adapter("hello world")
        r = MCPDispatchResource({"fs:greet": adapter})
        executor = ToolExecutor(tool_name_registry_getter=lambda: {"call_mcp_tool": (r, "call")})
        agent = MagicMock()
        results = await executor.execute_tools_async(
            agent,
            [
                {
                    "function": "call_mcp_tool",
                    "tool_call_id": "tc-1",
                    "arguments": {"tool_name": "fs:greet", "arguments": {"name": "World"}},
                }
            ],
        )
        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["result"] == "hello world"
        assert results[0]["tool_call_id"] == "tc-1"
        adapter._transport.call_tool.assert_awaited_once_with("fs:greet", {"name": "World"})

    @pytest.mark.asyncio
    async def test_unknown_function_does_not_dispatch_to_mcp(self):
        adapter = _mock_adapter("nope")
        r = MCPDispatchResource({"greet": adapter})
        executor = ToolExecutor(tool_name_registry_getter=lambda: {"call_mcp_tool": (r, "call")})
        agent = MagicMock()
        results = await executor.execute_tools_async(
            agent,
            [{"function": "not_a_tool", "tool_call_id": "tc-2", "arguments": {}}],
        )
        # not in registry -> falls to name-parsing fallback -> class_not_found error
        assert results[0]["success"] is False
        adapter._transport.call_tool.assert_not_awaited()


# ---------------------------------------------------------------------------
# load_mcp_config_from_env
# ---------------------------------------------------------------------------


class TestLoadMcpConfigFromEnv:
    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv("DANA_MCP_SERVERS", raising=False)
        assert load_mcp_config_from_env() is None

    def test_empty_returns_none(self, monkeypatch):
        monkeypatch.setenv("DANA_MCP_SERVERS", "  ")
        assert load_mcp_config_from_env() is None

    def test_parses_servers(self, monkeypatch):
        monkeypatch.setenv(
            "DANA_MCP_SERVERS",
            json.dumps([{"name": "fs", "command": "npx", "args": ["-y", "fs-server"]}]),
        )
        config = load_mcp_config_from_env()
        assert config is not None
        assert len(config.servers) == 1
        assert config.servers[0].name == "fs"
        assert config.servers[0].command == "npx"

    def test_invalid_json_raises(self, monkeypatch):
        monkeypatch.setenv("DANA_MCP_SERVERS", "not json")
        with pytest.raises(json.JSONDecodeError):
            load_mcp_config_from_env()

    def test_non_array_raises(self, monkeypatch):
        monkeypatch.setenv("DANA_MCP_SERVERS", json.dumps({"name": "fs"}))
        with pytest.raises(ValueError, match="array"):
            load_mcp_config_from_env()


# ---------------------------------------------------------------------------
# build_mcp_dispatch_resource (lease lifecycle + tool registration, mock transport)
# ---------------------------------------------------------------------------


class TestBuildMcpDispatchResource:
    @pytest.mark.asyncio
    async def test_no_servers_returns_none(self):
        config = MCPConfig(servers=[], enabled=True)
        assert await build_mcp_dispatch_resource(config) is None

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self):
        config = MCPConfig(
            servers=[MCPServerConfig(name="fs", command="npx")],
            enabled=False,
        )
        assert await build_mcp_dispatch_resource(config) is None

    @pytest.mark.asyncio
    async def test_builds_resource_from_mock_transport(self):
        tool = MagicMock()
        tool.name = "greet"
        tool.description = "Greet"
        transport = _mock_transport_with_tools([tool])
        config = MCPConfig(servers=[MCPServerConfig(name="fs", command="npx")], enabled=True)

        with (
            patch("dana.core.mcp.protocol.discover_tools", AsyncMock(return_value=(tool,))),
            patch("dana.core.mcp.protocol.perform_handshake", AsyncMock()),
        ):
            wiring = await build_mcp_dispatch_resource(config, transport_factory=lambda s: transport)
        assert wiring is not None
        assert isinstance(wiring, MCPWiring)
        assert "fs:greet" in wiring.resource.available_tools
        assert "greet" in wiring.resource.available_tools  # alias
        # lease active
        assert wiring.lease_manager.get_lease("fs").state == LeaseState.ACTIVE
        # dispatch works
        result = await wiring.resource.call("fs:greet", {"name": "World"})
        assert result == "ok"
        await wiring.close()

    @pytest.mark.asyncio
    async def test_required_lease_failure_raises_and_cleans_up(self):
        transport = MagicMock()

        @contextlib.asynccontextmanager
        async def connect():
            raise RuntimeError("server unreachable")
            yield  # unreachable but required for a valid asynccontextmanager

        transport.connect = connect
        config = MCPConfig(servers=[MCPServerConfig(name="fs", command="npx")], enabled=True)
        with pytest.raises(RuntimeError, match="Required MCP lease failed"):
            await build_mcp_dispatch_resource(config, transport_factory=lambda s: transport)


# ---------------------------------------------------------------------------
# AgentSession wiring (_wire_mcp_tools)
# ---------------------------------------------------------------------------


class TestAgentSessionMcpWiring:
    @pytest.mark.asyncio
    async def test_wiring_appends_resource_when_enabled(self, monkeypatch):
        monkeypatch.setenv("DANA_CODE_MCP_ENABLED", "1")
        monkeypatch.setenv("DANA_MCP_SERVERS", json.dumps([{"name": "fs", "command": "npx"}]))

        from datetime import UTC, datetime
        from uuid import uuid4

        from dana.core.session.agent_session import AgentSession
        from dana.core.session.journal.models import SessionRecord
        from dana.core.session.journal.sqlite import SQLiteJournalRepository
        from dana.core.session.models import FactType, JournalFact, OwnerScope

        repo = await SQLiteJournalRepository.open(":memory:")
        scope = OwnerScope(owner_id="o", workspace="w")
        sid = "s1"
        await repo.create_session(
            SessionRecord.new(sid, scope),
            [
                JournalFact(
                    fact_id=str(uuid4()),
                    owner_scope=scope,
                    session_id=sid,
                    sequence=1,
                    fact_type=FactType.SESSION_CREATED,
                    timestamp=datetime.now(UTC),
                    correlation_id=str(uuid4()),
                    causation_id=None,
                    schema_version=1,
                    payload={},
                )
            ],
        )

        class FakeAgent:
            object_id = "fake"
            agent_type = "fake"

            def __init__(self):
                self._resources = []
                self._timeline = MagicMock(timeline=[])

            async def aquery_stream(self, *, message=None, **kw):
                from dana.core.runtime.protocols import StreamEvent, StreamEventType

                yield StreamEvent(event_type=StreamEventType.TEXT_DELTA, data="ok", iteration=0)
                yield StreamEvent(event_type=StreamEventType.DONE, data=None, iteration=0)

        session = AgentSession(owner_scope=scope, session_id=sid, repository=repo, agent_factory=FakeAgent)

        mock_resource = MCPDispatchResource({"fs:greet": _mock_adapter("hi")})
        mock_wiring = MagicMock(spec=MCPWiring)
        mock_wiring.resource = mock_resource
        mock_wiring.close = AsyncMock()

        with patch("dana.core.mcp.dispatch_wrapper.build_mcp_dispatch_resource", AsyncMock(return_value=mock_wiring)):
            await session._prepare_agent()
        assert mock_resource in session._agent._resources
        assert session._mcp_wiring is mock_wiring
        await session.dispose_mcp()
        mock_wiring.close.assert_awaited_once()
        await repo.close()

    @pytest.mark.asyncio
    async def test_wiring_failure_does_not_crash(self, monkeypatch):
        monkeypatch.setenv("DANA_CODE_MCP_ENABLED", "1")
        monkeypatch.setenv("DANA_MCP_SERVERS", json.dumps([{"name": "fs", "command": "npx"}]))

        from datetime import UTC, datetime
        from uuid import uuid4

        from dana.core.session.agent_session import AgentSession
        from dana.core.session.journal.models import SessionRecord
        from dana.core.session.journal.sqlite import SQLiteJournalRepository
        from dana.core.session.models import FactType, JournalFact, OwnerScope

        repo = await SQLiteJournalRepository.open(":memory:")
        scope = OwnerScope(owner_id="o", workspace="w")
        sid = "s2"
        await repo.create_session(
            SessionRecord.new(sid, scope),
            [
                JournalFact(
                    fact_id=str(uuid4()),
                    owner_scope=scope,
                    session_id=sid,
                    sequence=1,
                    fact_type=FactType.SESSION_CREATED,
                    timestamp=datetime.now(UTC),
                    correlation_id=str(uuid4()),
                    causation_id=None,
                    schema_version=1,
                    payload={},
                )
            ],
        )

        class FakeAgent:
            object_id = "fake"
            agent_type = "fake"

            def __init__(self):
                self._resources = []
                self._timeline = MagicMock(timeline=[])

            async def aquery_stream(self, *, message=None, **kw):
                from dana.core.runtime.protocols import StreamEvent, StreamEventType

                yield StreamEvent(event_type=StreamEventType.DONE, data=None, iteration=0)

        session = AgentSession(owner_scope=scope, session_id=sid, repository=repo, agent_factory=FakeAgent)

        with patch("dana.core.mcp.dispatch_wrapper.build_mcp_dispatch_resource", AsyncMock(side_effect=RuntimeError("boom"))):
            await session._prepare_agent()  # must not raise
        assert session._mcp_wiring is None
        assert session._agent._resources == []
        await repo.close()

    @pytest.mark.asyncio
    async def test_wiring_skipped_when_disabled(self, monkeypatch):
        monkeypatch.setenv("DANA_CODE_MCP_ENABLED", "0")
        monkeypatch.setenv("DANA_MCP_SERVERS", json.dumps([{"name": "fs", "command": "npx"}]))

        from datetime import UTC, datetime
        from uuid import uuid4

        from dana.core.session.agent_session import AgentSession
        from dana.core.session.journal.models import SessionRecord
        from dana.core.session.journal.sqlite import SQLiteJournalRepository
        from dana.core.session.models import FactType, JournalFact, OwnerScope

        repo = await SQLiteJournalRepository.open(":memory:")
        scope = OwnerScope(owner_id="o", workspace="w")
        sid = "s3"
        await repo.create_session(
            SessionRecord.new(sid, scope),
            [
                JournalFact(
                    fact_id=str(uuid4()),
                    owner_scope=scope,
                    session_id=sid,
                    sequence=1,
                    fact_type=FactType.SESSION_CREATED,
                    timestamp=datetime.now(UTC),
                    correlation_id=str(uuid4()),
                    causation_id=None,
                    schema_version=1,
                    payload={},
                )
            ],
        )

        class FakeAgent:
            object_id = "fake"
            agent_type = "fake"

            def __init__(self):
                self._resources = []
                self._timeline = MagicMock(timeline=[])

            async def aquery_stream(self, *, message=None, **kw):
                from dana.core.runtime.protocols import StreamEvent, StreamEventType

                yield StreamEvent(event_type=StreamEventType.DONE, data=None, iteration=0)

        session = AgentSession(owner_scope=scope, session_id=sid, repository=repo, agent_factory=FakeAgent)
        with patch("dana.core.mcp.dispatch_wrapper.build_mcp_dispatch_resource", AsyncMock()) as mock_build:
            await session._prepare_agent()
            mock_build.assert_not_awaited()
        assert session._mcp_wiring is None
        assert session._agent._resources == []
        await repo.close()


# ---------------------------------------------------------------------------
# D7 follow-up 1+2 — per-tool MCP UX + per-MCP policy (Approach A2)
# ---------------------------------------------------------------------------


class TestPerToolMcpUx:
    """Per-tool MCP UX: the model calls each MCP tool BY NAME (with its real
    inputSchema) and the ToolExecutor dispatches it via the MCP dispatch map.
    Per-MCP policy: each MCP tool is classified individually (EXECUTE/non-sensitive).
    """

    @pytest.mark.asyncio
    async def test_build_exposes_per_tool_schemas_names_dispatch_map(self):
        tool = MagicMock()
        tool.name = "greet"
        tool.description = "Greet a person"
        tool.inputSchema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        transport = _mock_transport_with_tools([tool])
        config = MCPConfig(servers=[MCPServerConfig(name="fs", command="npx")], enabled=True)
        with (
            patch("dana.core.mcp.protocol.discover_tools", AsyncMock(return_value=(tool,))),
            patch("dana.core.mcp.protocol.perform_handshake", AsyncMock()),
        ):
            wiring = await build_mcp_dispatch_resource(config, transport_factory=lambda s: transport)
        assert wiring is not None
        # Per-tool schemas (correct inputSchema, namespaced).
        assert len(wiring.mcp_schemas) == 1
        schema = wiring.mcp_schemas[0]
        assert schema["function"]["name"] == "fs:greet"
        assert schema["function"]["parameters"] == tool.inputSchema
        # Per-MCP names for policy classification.
        assert wiring.mcp_names == frozenset({"fs:greet"})
        # Dispatch map: namespaced name -> async callable.
        assert "fs:greet" in wiring.dispatch_map
        assert callable(wiring.dispatch_map["fs:greet"])
        await wiring.close()

    @pytest.mark.asyncio
    async def test_per_tool_dispatch_through_executor(self):
        """The model calls the MCP tool BY NAME (fs:greet) -> ToolExecutor dispatches via the MCP map."""
        adapter = _mock_adapter("hello world")
        # Build a dispatch map mirroring build_mcp_dispatch_resource's output.
        from dana.core.mcp.dispatch_wrapper import _make_mcp_dispatcher

        dispatch_map = {"fs:greet": _make_mcp_dispatcher(adapter, "fs:greet")}
        executor = ToolExecutor(mcp_dispatch_getter=lambda: dispatch_map)
        agent = MagicMock()
        results = await executor.execute_tools_async(
            agent,
            [{"function": "fs:greet", "tool_call_id": "tc-1", "arguments": {"name": "World"}}],
        )
        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["result"] == "hello world"
        assert results[0]["tool_call_id"] == "tc-1"
        adapter._transport.call_tool.assert_awaited_once_with("fs:greet", {"name": "World"})

    @pytest.mark.asyncio
    async def test_mcp_dispatch_does_not_intercept_native_tools(self):
        """A native @named_tool (call_mcp_tool) dispatches via the registry, NOT the MCP map."""

        adapter = _mock_adapter("via-wrapper")
        r = MCPDispatchResource({"fs:greet": adapter})
        mcp_map_called: list[dict] = []

        async def _mcp_dispatch(args):
            mcp_map_called.append(args)
            return "via-mcp-map"

        dispatch_map = {"fs:greet": _mcp_dispatch}
        executor = ToolExecutor(
            tool_name_registry_getter=lambda: {"call_mcp_tool": (r, "call")},
            mcp_dispatch_getter=lambda: dispatch_map,
        )
        agent = MagicMock()
        results = await executor.execute_tools_async(
            agent,
            [{"function": "call_mcp_tool", "tool_call_id": "tc-1", "arguments": {"tool_name": "fs:greet", "arguments": {"name": "World"}}}],
        )
        # call_mcp_tool is NOT in the MCP map -> dispatched via the registry (the wrapper).
        assert mcp_map_called == []
        assert results[0]["success"] is True
        assert results[0]["result"] == "via-wrapper"

    def test_per_mcp_policy_classification(self):
        """build_native_tool_catalog classifies registered MCP tools non-sensitive; unknown stay fail-cautious."""
        from dana.core.tool.native_catalog import build_native_tool_catalog

        native_tools = [
            {"type": "function", "function": {"name": "Read", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "fs:greet", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "weird:tool", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "mystery_tool", "parameters": {"type": "object"}}},
        ]
        mcp_names = frozenset({"fs:greet"})
        catalog = build_native_tool_catalog(native_tools, version=1, mcp_names=mcp_names)
        # Registered MCP tool -> EXECUTE, non-sensitive (flows to mode/grant/prompt).
        mcp_entry = catalog.get("fs:greet")
        assert mcp_entry is not None
        assert mcp_entry.effects.is_sensitive is False
        assert any(e.kind.value == "execute" for e in mcp_entry.effects.effects)
        # Unknown namespaced tool (not registered as MCP) -> fail-cautious (sensitive).
        weird = catalog.get("weird:tool")
        assert weird is not None and weird.effects.is_sensitive is True
        # Unknown non-namespaced tool -> fail-cautious (sensitive).
        mystery = catalog.get("mystery_tool")
        assert mystery is not None and mystery.effects.is_sensitive is True
        # Known native tool unchanged.
        read_entry = catalog.get("Read")
        assert read_entry is not None and read_entry.effects.is_sensitive is False

    @pytest.mark.asyncio
    async def test_agent_session_build_catalog_appends_mcp_and_wires_dispatch(self, monkeypatch):
        """_build_tool_catalog appends per-MCP schemas to _native_tools + wires the MCP dispatch getter."""
        monkeypatch.setenv("DANA_CODE_TOOL_CATALOG_ENABLED", "1")
        from datetime import UTC, datetime
        from uuid import uuid4

        from dana.core.session.agent_session import AgentSession
        from dana.core.session.journal.models import SessionRecord
        from dana.core.session.journal.sqlite import SQLiteJournalRepository
        from dana.core.session.models import FactType, JournalFact, OwnerScope

        repo = await SQLiteJournalRepository.open(":memory:")
        scope = OwnerScope(owner_id="o", workspace="w")
        sid = "pt1"
        await repo.create_session(
            SessionRecord.new(sid, scope),
            [
                JournalFact(
                    fact_id=str(uuid4()),
                    owner_scope=scope,
                    session_id=sid,
                    sequence=1,
                    fact_type=FactType.SESSION_CREATED,
                    timestamp=datetime.now(UTC),
                    correlation_id=str(uuid4()),
                    causation_id=None,
                    schema_version=1,
                    payload={},
                )
            ],
        )

        class FakeRuntime:
            def __init__(self):
                self._native_tools = [
                    {"type": "function", "function": {"name": "Read", "parameters": {"type": "object", "properties": {}}}}
                ]
                self._tool_executor = ToolExecutor()

            def _build_native_tools_if_supported(self, agent):
                return  # already populated

        class FakeAgent:
            object_id = "fake"
            agent_type = "fake"

            def __init__(self):
                self._runtime = FakeRuntime()
                self._resources = []

        adapter = _mock_adapter("hello world")
        from dana.core.mcp.dispatch_wrapper import _make_mcp_dispatcher

        mock_wiring = MagicMock(spec=MCPWiring)
        mock_wiring.mcp_schemas = [{"type": "function", "function": {"name": "fs:greet", "parameters": {"type": "object"}}}]
        mock_wiring.mcp_names = frozenset({"fs:greet"})
        mock_wiring.dispatch_map = {"fs:greet": _make_mcp_dispatcher(adapter, "fs:greet")}
        mock_wiring.close = AsyncMock()

        session = AgentSession(owner_scope=scope, session_id=sid, repository=repo, agent_factory=FakeAgent)
        session._agent = FakeAgent()
        session._mcp_wiring = mock_wiring
        await session._build_tool_catalog()

        # The per-MCP schema was appended to _native_tools (LLM sees it by name).
        names = {t["function"]["name"] for t in session._agent._runtime._native_tools}
        assert "fs:greet" in names
        # The catalog classifies the MCP tool non-sensitive (per-MCP policy).
        assert session.tool_catalog is not None
        entry = session.tool_catalog.get("fs:greet")
        assert entry is not None and entry.effects.is_sensitive is False
        # The MCP dispatch getter is wired on the ToolExecutor.
        executor = session._agent._runtime._tool_executor
        assert executor._mcp_dispatch_getter is not None
        # Dispatching the MCP tool BY NAME works end-to-end.
        results = await executor.execute_tools_async(
            session._agent,
            [{"function": "fs:greet", "tool_call_id": "tc-1", "arguments": {"name": "World"}}],
        )
        assert results[0]["success"] is True and results[0]["result"] == "hello world"
        adapter._transport.call_tool.assert_awaited_once_with("fs:greet", {"name": "World"})
        await repo.close()

    @pytest.mark.asyncio
    async def test_build_catalog_idempotent_mcp_append(self, monkeypatch):
        """Calling _build_tool_catalog twice does not duplicate the MCP schemas (idempotent)."""
        monkeypatch.setenv("DANA_CODE_TOOL_CATALOG_ENABLED", "1")
        from dana.core.mcp.dispatch_wrapper import _make_mcp_dispatcher
        from dana.core.session.agent_session import AgentSession
        from dana.core.session.journal.sqlite import SQLiteJournalRepository
        from dana.core.session.models import OwnerScope

        repo = await SQLiteJournalRepository.open(":memory:")
        scope = OwnerScope(owner_id="o", workspace="w")
        session = AgentSession(owner_scope=scope, session_id="pt2", repository=repo, agent_factory=lambda: None)

        class FakeRuntime:
            def __init__(self):
                self._native_tools = [{"type": "function", "function": {"name": "Read", "parameters": {}}}]
                self._tool_executor = ToolExecutor()

            def _build_native_tools_if_supported(self, agent):
                return

        class FakeAgent:
            def __init__(self):
                self._runtime = FakeRuntime()
                self._resources = []

        session._agent = FakeAgent()
        mock_wiring = MagicMock(spec=MCPWiring)
        mock_wiring.mcp_schemas = [{"type": "function", "function": {"name": "fs:greet", "parameters": {"type": "object"}}}]
        mock_wiring.mcp_names = frozenset({"fs:greet"})
        mock_wiring.dispatch_map = {"fs:greet": _make_mcp_dispatcher(_mock_adapter("ok"), "fs:greet")}
        mock_wiring.close = AsyncMock()
        session._mcp_wiring = mock_wiring

        await session._build_tool_catalog()
        await session._build_tool_catalog()
        names = [t["function"]["name"] for t in session._agent._runtime._native_tools]
        assert names.count("fs:greet") == 1  # not duplicated
        await repo.close()
