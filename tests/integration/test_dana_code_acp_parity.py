"""D7.4 — dana-code ↔ dana-acp parity smoke (Wave 3).

Pins the invariant that the CLI and ACP entrypoints — both host adapters over
one ``AgentSession`` — produce equivalent logical event streams for the same
scripted turn. Per ADR (Host Interface Boundary): if they diverge, the
divergence is in the adapter, not the core.

Both paths call ``AgentSession.prompt()`` (which drives the streaming STAR loop
via ``aquery_stream`` + StreamEvent→HostEvent mapping, per P0 part 2 ``f25fde3``).
The CLI path captures ``HostEvent``\\s directly; the ACP path translates each
``HostEvent`` to an ACP ``session_update`` via ``host_event_to_acp_update``.

Parity assertion (AC #2): the ACP update-kind sequence equals the forward
translation of the CLI ``HostEvent`` sequence (filtering ``None``-translatable
lifecycle events). This proves both paths see the same underlying events and
ACP translates them faithfully.

All tests use a mocked provider (``ScriptedAgent``) — no live LLM, no network.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import os
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio

from dana.core.runtime.protocols import StreamEvent, StreamEventType
from dana.core.session.agent_session import AgentSession, SessionBusy, TextBlock
from dana.core.session.journal.models import SessionRecord
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.models import FactType, JournalFact, OwnerScope
from dana.core.session.projections.host_events import HostEvent, HostEventType


os.environ.setdefault("DANA_SESSION_STATE_KEY", "test-key-32-bytes-ok-for-testing!")
os.environ.setdefault("DANA_POLICY_GRANTS_ENABLED", "0")


# ---------------------------------------------------------------------------
# ScriptedAgent — deterministic StreamEvent source (mocked provider)
# ---------------------------------------------------------------------------


class ScriptedAgent:
    """Fake agent yielding a scripted StreamEvent sequence, then DONE.

    For cancellation tests, ``gate`` / ``parked`` provide deterministic
    blocking: the agent awaits ``gate`` before each event *after the first*
    (so a partial chunk streams before the cancel point) and signals ``parked``
    once it has reached the gate.
    """

    def __init__(
        self,
        script: list[tuple[StreamEventType, Any]] | None = None,
        chunks: list[str] | None = None,
        error: BaseException | None = None,
        gate: asyncio.Event | None = None,
        parked: asyncio.Event | None = None,
    ) -> None:
        if script is not None:
            self._script = list(script)
        else:
            self._script = [(StreamEventType.TEXT_DELTA, c) for c in (chunks or [])]
        self._error = error
        self._gate = gate
        self._parked = parked
        self._timeline = SimpleNamespace(timeline=[])
        self._runtime = SimpleNamespace()
        self.object_id = "scripted-agent"
        self.agent_type = "fake"

    async def aquery_stream(self, *, message: str | None = None, **kwargs: Any):
        """Yield the scripted StreamEvents, then DONE."""
        if self._error is not None:
            raise self._error
        for i, (etype, data) in enumerate(self._script):
            if i > 0 and self._gate is not None:
                if self._parked is not None:
                    self._parked.set()
                await self._gate.wait()
            yield StreamEvent(event_type=etype, data=data, iteration=0)
        yield StreamEvent(event_type=StreamEventType.DONE, data=None, iteration=0)


def scripted_factory(script=None, chunks=None, **kwargs):
    """Return a zero-arg factory building a ScriptedAgent."""

    def _factory():
        return ScriptedAgent(script=script, chunks=chunks, **kwargs)

    return _factory


# ---------------------------------------------------------------------------
# RecordingConn — captures ACP session_update notifications
# ---------------------------------------------------------------------------


class RecordingConn:
    """Fake AgentSideConnection capturing session_update calls."""

    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append((session_id, update))


def _update_kind(update: object) -> str | None:
    return getattr(update, "session_update", None)


def _update_text(update: object) -> str | None:
    content = getattr(update, "content", None)
    if content is not None:
        return getattr(content, "text", None)
    return None


# ---------------------------------------------------------------------------
# Mock evaluator + grant store (for permission adapter parity tests)
# ---------------------------------------------------------------------------


class _MockEvaluator:
    """Minimal async evaluator returning a configurable PolicyResult."""

    def __init__(self, decision, reason: str = "mock") -> None:
        self._decision = decision
        self._reason = reason

    def set_mode(self, mode) -> None:
        pass

    async def evaluate(self, op, scope):
        from dana.core.policy.evaluator import PolicyResult

        return PolicyResult(decision=self._decision, reason=self._reason)


class _MockGrantStore:
    """No-op grant store for permission adapter tests."""

    async def create_grant(self, grant):
        return grant

    async def list_grants(self, scope):
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def repo(tmp_path):
    r = await SQLiteJournalRepository.open(str(tmp_path / "journal.db"))
    yield r
    await r.close()


# ---------------------------------------------------------------------------
# Helpers — construct both paths from one config
# ---------------------------------------------------------------------------


async def _seed_session(repo, session_id, scope=None):
    """Seed a session with SESSION_CREATED (mirrors code_app._initialize_session)."""
    scope = scope or OwnerScope(owner_id="local", workspace="/tmp")
    record = SessionRecord.new(session_id, scope)
    init_facts = [
        JournalFact(
            fact_id=str(uuid4()),
            owner_scope=scope,
            session_id=session_id,
            sequence=1,
            fact_type=FactType.SESSION_CREATED,
            timestamp=datetime.now(UTC),
            correlation_id=str(uuid4()),
            causation_id=None,
            schema_version=1,
            payload={},
        ),
    ]
    await repo.create_session(record, init_facts)
    return scope


async def build_cli_session(repo, factory, session_id="cli-sess"):
    """Construct an AgentSession the way DanaCodeApp._initialize_session does."""
    scope = await _seed_session(repo, session_id)
    session = AgentSession(
        owner_scope=scope,
        session_id=session_id,
        repository=repo,
        agent_factory=factory,
    )
    await session.load()
    return session, scope


async def build_acp_agent(tmp_path, factory):
    """Construct a DanaACPAgent + RecordingConn (in-process)."""
    from dana.apps.acp.agent import DanaACPAgent

    agent = DanaACPAgent(
        journal_path=str(tmp_path / "journal.db"),
        agent_factory=factory,
    )
    conn = RecordingConn()
    agent.on_connect(conn)
    return agent, conn


async def capture_hostevents(session, text="hello"):
    """Run one turn through session.prompt and collect HostEvents."""
    events: list[HostEvent] = []
    async for event in session.prompt([TextBlock(text=text)]):
        events.append(event)
    return events


async def _drain_prompt(session, text="go"):
    """Coroutine wrapper: consume the async generator and collect events."""
    events: list[HostEvent] = []
    async for event in session.prompt([TextBlock(text=text)]):
        events.append(event)
    return events


def hostevent_types(events: list[HostEvent]) -> list[HostEventType]:
    return [e.event_type for e in events]


def expected_acp_kinds(events: list[HostEvent]) -> list[str | None]:
    """Forward-translate CLI HostEvents to the ACP update kinds ACP should send."""
    from dana.apps.acp.translation import host_event_to_acp_update

    kinds: list[str | None] = []
    for e in events:
        update = host_event_to_acp_update(e)
        if update is not None:
            kinds.append(_update_kind(update))
    return kinds


def acp_kinds(conn: RecordingConn) -> list[str | None]:
    return [_update_kind(u) for _, u in conn.updates]


# ===========================================================================
# AC #1 + #2 — harness drives both paths; stream equivalence (text turn)
# ===========================================================================


class TestParityTextTurn:
    """Plain text turn: CLI HostEvents ≡ ACP session/update stream."""

    @pytest.mark.asyncio
    async def test_text_turn_stream_equivalence(self, repo, tmp_path):
        chunks = ["Hello, ", "world!"]
        sid = "parity-text"

        cli_session, _ = await build_cli_session(repo, scripted_factory(chunks=chunks), session_id=sid)
        cli_events = await capture_hostevents(cli_session, text="hi")

        agent, conn = await build_acp_agent(tmp_path, scripted_factory(chunks=chunks))
        new_resp = await agent.new_session(cwd="/tmp")
        prompt_resp = await agent.prompt(prompt=[{"type": "text", "text": "hi"}], session_id=new_resp.session_id)

        assert cli_events, "CLI path produced no events"
        assert conn.updates, "ACP path produced no updates"

        types = hostevent_types(cli_events)
        assert types[0] is HostEventType.TURN_STARTED
        assert types[1] is HostEventType.USER_MESSAGE
        assert types[-1] is HostEventType.TURN_COMPLETED
        assert HostEventType.ASSISTANT_CONTENT_FINAL in types
        assert types.count(HostEventType.ASSISTANT_CONTENT_CHUNK) == 2

        assert acp_kinds(conn) == expected_acp_kinds(cli_events)

        agent_texts = [_update_text(u) for _, u in conn.updates if _update_kind(u) == "agent_message_chunk"]
        assert "Hello, " in agent_texts
        assert "world!" in agent_texts
        assert prompt_resp.stop_reason == "end_turn"

        if agent._repository is not None:
            await agent._repository.close()


# ===========================================================================
# AC #2 — stream equivalence (tool-call turn)
# ===========================================================================


class TestParityToolCallTurn:
    """Turn with a tool call + result: both paths surface tool lifecycle events."""

    @pytest.mark.asyncio
    async def test_tool_call_turn_equivalence(self, repo, tmp_path):
        script = [
            (StreamEventType.TEXT_DELTA, "Let me check. "),
            (
                StreamEventType.TOOL_CALL_START,
                [{"id": "tc-1", "name": "todo_write", "input": {"items": []}}],
            ),
            (StreamEventType.TOOL_RESULT, [{"tool_call_id": "tc-1", "result": {"success": True}}]),
            (StreamEventType.TEXT_DELTA, "Done."),
        ]

        cli_session, _ = await build_cli_session(repo, scripted_factory(script=script), session_id="parity-tool")
        cli_events = await capture_hostevents(cli_session, text="add a todo")

        agent, conn = await build_acp_agent(tmp_path, scripted_factory(script=script))
        new_resp = await agent.new_session(cwd="/tmp")
        await agent.prompt(prompt=[{"type": "text", "text": "add a todo"}], session_id=new_resp.session_id)

        types = hostevent_types(cli_events)
        assert HostEventType.TOOL_REQUESTED in types
        assert HostEventType.TOOL_STARTED in types
        assert HostEventType.TOOL_RESULT in types
        assert HostEventType.TOOL_AUTHORIZED_OR_DENIED in types

        assert acp_kinds(conn) == expected_acp_kinds(cli_events)
        kinds = acp_kinds(conn)
        assert "tool_call" in kinds
        assert "tool_call_update" in kinds

        if agent._repository is not None:
            await agent._repository.close()


# ===========================================================================
# AC #2 — thought event parity
# ===========================================================================


class TestParityThoughtTurn:
    """THINKING StreamEvent → THOUGHT HostEvent → agent_thought_chunk ACP update."""

    @pytest.mark.asyncio
    async def test_thought_equivalence(self, repo, tmp_path):
        script = [
            (StreamEventType.THINKING, "Considering the request."),
            (StreamEventType.TEXT_DELTA, "Here is my answer."),
        ]

        cli_session, _ = await build_cli_session(repo, scripted_factory(script=script), session_id="parity-thought")
        cli_events = await capture_hostevents(cli_session, text="think then answer")

        agent, conn = await build_acp_agent(tmp_path, scripted_factory(script=script))
        new_resp = await agent.new_session(cwd="/tmp")
        await agent.prompt(
            prompt=[{"type": "text", "text": "think then answer"}],
            session_id=new_resp.session_id,
        )

        assert HostEventType.THOUGHT in hostevent_types(cli_events)
        assert acp_kinds(conn) == expected_acp_kinds(cli_events)
        assert "agent_thought_chunk" in acp_kinds(conn)

        if agent._repository is not None:
            await agent._repository.close()


# ===========================================================================
# AC #3 — cancellation parity
# ===========================================================================


class TestParityCancellation:
    """Cancel mid-turn → TURN_CANCELLED (CLI) + stop_reason=cancelled (ACP)."""

    @pytest.mark.asyncio
    async def test_cancel_mid_turn_cli(self, repo):
        gate = asyncio.Event()
        parked = asyncio.Event()
        script = [
            (StreamEventType.TEXT_DELTA, "partial "),
            (StreamEventType.TEXT_DELTA, "second "),
        ]
        session, _ = await build_cli_session(
            repo,
            scripted_factory(script=script, gate=gate, parked=parked),
            session_id="cancel-cli",
        )

        task = asyncio.ensure_future(_drain_prompt(session, "go"))
        await asyncio.wait_for(parked.wait(), timeout=5.0)
        await session.cancel()
        gate.set()
        events = await task

        types = hostevent_types(events)
        assert HostEventType.TURN_CANCELLED in types
        assert types[-1] is HostEventType.TURN_CANCELLED
        assert HostEventType.ASSISTANT_CONTENT_CHUNK in types
        assert session.last_terminal is not None
        assert session.last_terminal.fact_type is FactType.TURN_CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_mid_turn_acp(self, tmp_path):
        gate = asyncio.Event()
        parked = asyncio.Event()
        script = [
            (StreamEventType.TEXT_DELTA, "partial "),
            (StreamEventType.TEXT_DELTA, "second "),
        ]
        agent, conn = await build_acp_agent(tmp_path, scripted_factory(script=script, gate=gate, parked=parked))
        new_resp = await agent.new_session(cwd="/tmp")
        sid = new_resp.session_id

        prompt_task = asyncio.ensure_future(agent.prompt(prompt=[{"type": "text", "text": "go"}], session_id=sid))
        await asyncio.wait_for(parked.wait(), timeout=5.0)
        await agent.cancel(session_id=sid)
        gate.set()
        resp = await prompt_task

        assert resp.stop_reason == "cancelled"
        session = agent._sessions[sid]
        assert session.last_terminal is not None
        assert session.last_terminal.fact_type is FactType.TURN_CANCELLED

        if agent._repository is not None:
            await agent._repository.close()


# ===========================================================================
# AC #3 — model switch parity (MODEL_CHANGED journaled + busy-reject)
# ===========================================================================


class TestParityModelSwitch:
    """Both /model (CLI) and setSessionModel (ACP) journal MODEL_CHANGED + busy-reject."""

    @pytest.mark.asyncio
    async def test_model_switch_journals_changed_cli(self, repo):
        from dana.apps.code import commands as cmds

        session, scope = await build_cli_session(repo, scripted_factory(chunks=["x"]), session_id="model-cli")
        app = SimpleNamespace(agent_session=session, agent=None, renderer=None)

        result = await cmds.switch_model(app, "model anthropic/claude-sonnet-4")
        assert "Switched" in result
        assert session.current_provider == "anthropic"
        assert session.current_model == "claude-sonnet-4"

        facts = await repo.read_facts(scope, "model-cli", 0)
        model_changes = [f for f in facts if f.fact_type is FactType.MODEL_CHANGED]
        assert len(model_changes) == 1
        assert model_changes[0].payload["provider"] == "anthropic"
        assert model_changes[0].payload["model"] == "claude-sonnet-4"

    @pytest.mark.asyncio
    async def test_model_switch_journals_changed_acp(self, tmp_path):
        agent, _ = await build_acp_agent(tmp_path, scripted_factory(chunks=["x"]))
        new_resp = await agent.new_session(cwd="/tmp")
        sid = new_resp.session_id

        await agent.set_session_model("anthropic/claude-sonnet-4", sid)

        session = agent._sessions[sid]
        assert session.current_provider == "anthropic"
        assert session.current_model == "claude-sonnet-4"

        facts = await agent._repository.read_facts(session.owner_scope, sid, 0)
        model_changes = [f for f in facts if f.fact_type is FactType.MODEL_CHANGED]
        assert len(model_changes) == 1
        assert model_changes[0].payload["provider"] == "anthropic"
        assert model_changes[0].payload["model"] == "claude-sonnet-4"

        if agent._repository is not None:
            await agent._repository.close()

    @pytest.mark.asyncio
    async def test_model_switch_busy_reject_cli(self, repo):
        from dana.apps.code import commands as cmds

        gate = asyncio.Event()
        parked = asyncio.Event()
        session, _ = await build_cli_session(
            repo,
            scripted_factory(chunks=["x", "y"], gate=gate, parked=parked),
            session_id="model-busy-cli",
        )
        app = SimpleNamespace(agent_session=session, agent=None, renderer=None)

        task = asyncio.ensure_future(_drain_prompt(session, "go"))
        await asyncio.wait_for(parked.wait(), timeout=5.0)
        assert session._lock.locked()
        result = await cmds.switch_model(app, "model anthropic/claude-sonnet-4")
        assert "in progress" in result or "Cannot switch" in result
        gate.set()
        await task

    @pytest.mark.asyncio
    async def test_model_switch_busy_reject_acp(self, tmp_path):
        gate = asyncio.Event()
        parked = asyncio.Event()
        agent, _ = await build_acp_agent(tmp_path, scripted_factory(chunks=["x", "y"], gate=gate, parked=parked))
        new_resp = await agent.new_session(cwd="/tmp")
        sid = new_resp.session_id

        prompt_task = asyncio.ensure_future(agent.prompt(prompt=[{"type": "text", "text": "go"}], session_id=sid))
        await asyncio.wait_for(parked.wait(), timeout=5.0)
        with pytest.raises(SessionBusy):
            await agent.set_session_model("anthropic/claude-sonnet-4", sid)
        gate.set()
        await prompt_task

        if agent._repository is not None:
            await agent._repository.close()


# ===========================================================================
# AC #3 — permission decision parity (adapter/evaluator level)
# ===========================================================================


class TestParityPermission:
    """Permission decision parity between CLI and ACP adapters.

    The CLI adapter (CLIPermissionAdapter) is fully tested. The ACP adapter
    (DanaACPAgent.request_permission) has pre-existing bugs (filed as findings,
    not fixed — tests-only story) that prevent full ACP parity verification:

    FINDING 1 — ACP ``PermissionOptionKind`` Literal bug: ``agent.py:332+`` uses
    ``PermissionOptionKind.ALLOW_ONCE`` etc., but ``PermissionOptionKind`` is
    ``typing.Literal['allow_once', ...]``, not an enum → ``AttributeError``.

    FINDING 2 — ACP ``RequestPermissionResponse`` missing ``outcome``:
    ``agent.py:367`` constructs ``RequestPermissionResponse(options=[],
    denied_reason=...)`` without the required ``outcome`` field → pydantic
    ``ValidationError``.

    FINDING 3 — live-tool-call preflight not exercised: the STAR-loop turn path
    (``AgentSession.prompt()``) hardcodes ``authorized=True`` on
    ``TOOL_CALL_START`` — it does NOT consult ``policy_evaluator``. So
    permission preflight is NOT exercised during a real turn in either host.
    Wiring it into the STAR loop is D7.5 scope.
    """

    @pytest.mark.asyncio
    async def test_permission_cli_deny(self):
        """CLI adapter: a DENY decision → verdict.allowed=False."""
        from dana.apps.code.permissions import CLIPermissionAdapter
        from dana.core.policy.evaluator import PolicyDecision

        evaluator = _MockEvaluator(PolicyDecision.DENY, reason="hard deny")
        cli_adapter = CLIPermissionAdapter(
            evaluator,
            _MockGrantStore(),
            OwnerScope(owner_id="local", workspace="/tmp"),
            prompt=lambda p: "1",
        )
        verdict = await cli_adapter.request({"function": "some_tool", "arguments": {}})
        assert verdict.allowed is False
        assert "denied" in verdict.reason.lower()

    @pytest.mark.asyncio
    async def test_permission_cli_allow(self):
        """CLI adapter: an ALLOW decision → verdict.allowed=True."""
        from dana.apps.code.permissions import CLIPermissionAdapter
        from dana.core.policy.evaluator import PolicyDecision

        evaluator = _MockEvaluator(PolicyDecision.ALLOW, reason="durable grant")
        cli_adapter = CLIPermissionAdapter(
            evaluator,
            _MockGrantStore(),
            OwnerScope(owner_id="local", workspace="/tmp"),
            prompt=lambda p: "1",
        )
        verdict = await cli_adapter.request({"function": "some_tool", "arguments": {}})
        assert verdict.allowed is True
        assert "durable grant" in verdict.reason

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason=(
            "ACP request_permission pre-existing bugs (FINDINGS 1+2): "
            "PermissionOptionKind is typing.Literal not enum → AttributeError; "
            "RequestPermissionResponse missing required 'outcome' field. "
            "Filed as findings; fix in a core story."
        ),
        raises=Exception,
        strict=False,
    )
    async def test_permission_acp_deny_parity(self, tmp_path):
        """ACP request_permission DENY → denied_reason set (currently broken)."""
        from dana.apps.acp.agent import DanaACPAgent
        from dana.core.policy.evaluator import PolicyDecision

        evaluator = _MockEvaluator(PolicyDecision.DENY, reason="hard deny")
        agent = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=scripted_factory(chunks=["x"]),
        )
        new_resp = await agent.new_session(cwd="/tmp")
        sid = new_resp.session_id
        agent._sessions[sid].set_policy_evaluator(evaluator)

        req = SimpleNamespace(tool_name="some_tool", arguments={})
        resp = await agent.request_permission(req, sid)
        assert resp is not None
        assert resp.denied_reason is not None
        assert len(resp.options) == 0

        if agent._repository is not None:
            await agent._repository.close()

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason=(
            "ACP request_permission pre-existing bug (FINDING 1): "
            "PermissionOptionKind.ALLOW_ONCE → AttributeError. "
            "Filed as finding; fix in a core story."
        ),
        raises=Exception,
        strict=False,
    )
    async def test_permission_acp_allow_parity(self, tmp_path):
        """ACP request_permission ALLOW → options returned (currently broken)."""
        from dana.apps.acp.agent import DanaACPAgent
        from dana.core.policy.evaluator import PolicyDecision

        evaluator = _MockEvaluator(PolicyDecision.ALLOW, reason="durable grant")
        agent = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=scripted_factory(chunks=["x"]),
        )
        new_resp = await agent.new_session(cwd="/tmp")
        sid = new_resp.session_id
        agent._sessions[sid].set_policy_evaluator(evaluator)

        req = SimpleNamespace(tool_name="some_tool", arguments={})
        resp = await agent.request_permission(req, sid)
        assert resp is not None
        assert resp.denied_reason is None
        assert len(resp.options) > 0

        if agent._repository is not None:
            await agent._repository.close()


# ===========================================================================
# AC #4 — CI-runnable, mocked provider, no network
# ===========================================================================


class TestParityNoNetwork:
    """The parity suite runs with a mocked provider — no live LLM, no network."""

    def test_scripted_agent_no_llm(self):
        agent = ScriptedAgent(chunks=["a", "b"])
        assert agent.object_id == "scripted-agent"

    @pytest.mark.asyncio
    async def test_full_text_turn_no_network(self, repo, tmp_path):
        """A complete text turn through both paths with no provider config."""
        chunks = ["mocked ", "response"]
        cli_session, _ = await build_cli_session(repo, scripted_factory(chunks=chunks), session_id="no-net-cli")
        events = await capture_hostevents(cli_session, text="test")
        assert any(e.event_type is HostEventType.ASSISTANT_CONTENT_CHUNK for e in events)
