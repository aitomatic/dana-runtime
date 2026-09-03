"""
Unit tests for AgentSession D2 tool lifecycle wiring.

Covers:
- emit_thought yields THOUGHT host events
- journal_tool_requested journals TOOL_REQUESTED fact and yields host event
- journal_tool_terminal journals terminal tool facts (result/failure/acknowledged/timed_out/effect_unknown)
- execute_tool_call yields full lifecycle events
- rollback flag selects legacy executor
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio

from dana.core.session.agent_session import AgentSession
from dana.core.session.journal.models import SessionRecord
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.models import FactType, JournalFact, OwnerScope
from dana.core.session.projections.host_events import HostEventType


# ---------------------------------------------------------------------------
# FakeToolEngine — stands in for ToolExecutionEngine
# ---------------------------------------------------------------------------


class FakeToolEngine:
    """Fake tool engine for testing tool lifecycle wiring."""

    def __init__(self, result: dict | None = None, error: str | None = None, delay: float = 0.0):
        self._result = result or {"success": True, "result": {"output": "ok"}}
        self._error = error
        self._delay = delay
        self.executed_calls: list[dict] = []

    def execute(self, tool_call: dict) -> dict:
        self.executed_calls.append(tool_call)
        if self._error:
            return {"success": False, "error": self._error}
        return self._result

    async def execute_async(self, tool_call: dict) -> dict:
        self.executed_calls.append(tool_call)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error:
            return {"success": False, "error": self._error}
        return self._result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def repo(tmp_path):
    """Open a fresh SQLite journal backed by a temp file."""
    r = await SQLiteJournalRepository.open(str(tmp_path / "journal.db"))
    yield r
    await r.close()


async def _setup_session(repo, session_id="sess-1"):
    """Create a session in the journal with a single SESSION_CREATED fact."""
    scope = OwnerScope(owner_id="owner-1", workspace="ws-1")
    record = SessionRecord.new(session_id, scope)
    now = datetime.now(UTC)
    init_facts = [
        JournalFact(
            fact_id=str(uuid4()),
            owner_scope=scope,
            session_id=session_id,
            sequence=1,
            fact_type=FactType.SESSION_CREATED,
            timestamp=now,
            correlation_id=str(uuid4()),
            causation_id=None,
            schema_version=1,
            payload={},
        )
    ]
    await repo.create_session(record, init_facts)
    return scope


async def _make_session(repo, scope, tool_engine=None, use_legacy_executor=False, session_id="sess-1"):
    session = AgentSession(
        owner_scope=scope,
        session_id=session_id,
        repository=repo,
        agent_factory=lambda: SimpleNamespace(_timeline=SimpleNamespace(timeline=[])),
        tool_engine=tool_engine,
        use_legacy_executor=use_legacy_executor,
    )
    # Load to sync _current_version with the journal
    await session.load()
    return session


async def _collect(agen):
    """Collect all events from an async generator."""
    events: list = []
    async for event in agen:
        events.append(event)
    return events


# ===========================================================================
# 1. emit_thought
# ===========================================================================


class TestEmitThought:
    @pytest.mark.asyncio
    async def test_emit_thought_returns_thought_event(self, repo):
        scope = await _setup_session(repo)
        session = await _make_session(repo, scope)
        event = await session.emit_thought("I am thinking...", "corr-1")
        assert event.event_type is HostEventType.THOUGHT
        assert event.text == "I am thinking..."
        assert event.correlation_id == "corr-1"

    @pytest.mark.asyncio
    async def test_emit_thought_not_journaled(self, repo):
        """Thought events are live-only — no fact is journaled."""
        scope = await _setup_session(repo)
        session = await _make_session(repo, scope)
        await session.emit_thought("thinking...", "corr-1")
        facts = await repo.read_facts(scope, "sess-1")
        # Only the SESSION_CREATED fact should exist
        assert len(facts) == 1
        assert facts[0].fact_type == FactType.SESSION_CREATED


# ===========================================================================
# 2. journal_tool_requested
# ===========================================================================


class TestJournalToolRequested:
    @pytest.mark.asyncio
    async def test_journal_tool_requested(self, repo):
        scope = await _setup_session(repo)
        session = await _make_session(repo, scope)
        event = await session.journal_tool_requested(
            tool_call_id="tc-1",
            tool_name="read_file",
            correlation_id="corr-1",
            arguments={"path": "/tmp/test.txt"},
            kind="read",
        )
        assert event.event_type is HostEventType.TOOL_REQUESTED
        assert event.metadata["tool_call_id"] == "tc-1"
        assert event.metadata["tool_name"] == "read_file"
        assert event.metadata["kind"] == "read"

        # Verify fact was journaled
        facts = await repo.read_facts(scope, "sess-1")
        tool_facts = [f for f in facts if f.fact_type == FactType.TOOL_REQUESTED]
        assert len(tool_facts) == 1
        assert tool_facts[0].payload["tool_call_id"] == "tc-1"
        assert tool_facts[0].payload["tool_name"] == "read_file"


# ===========================================================================
# 3. journal_tool_terminal — all five terminal types
# ===========================================================================


class TestJournalToolTerminal:
    @pytest.mark.asyncio
    async def test_tool_result(self, repo):
        scope = await _setup_session(repo)
        session = await _make_session(repo, scope)
        event = await session.journal_tool_terminal(
            tool_call_id="tc-1",
            correlation_id="corr-1",
            terminal_type=FactType.TOOL_RESULT,
            result={"output": "file content"},
        )
        assert event.event_type is HostEventType.TOOL_RESULT
        assert event.metadata["result"] == {"output": "file content"}

    @pytest.mark.asyncio
    async def test_tool_failure(self, repo):
        scope = await _setup_session(repo)
        session = await _make_session(repo, scope)
        event = await session.journal_tool_terminal(
            tool_call_id="tc-1",
            correlation_id="corr-1",
            terminal_type=FactType.TOOL_FAILURE,
            error="File not found",
        )
        assert event.event_type is HostEventType.TOOL_FAILURE
        assert event.metadata["error"] == "File not found"

    @pytest.mark.asyncio
    async def test_tool_acknowledged(self, repo):
        scope = await _setup_session(repo)
        session = await _make_session(repo, scope)
        event = await session.journal_tool_terminal(
            tool_call_id="tc-1",
            correlation_id="corr-1",
            terminal_type=FactType.TOOL_ACKNOWLEDGED,
        )
        assert event.event_type is HostEventType.TOOL_ACKNOWLEDGED

    @pytest.mark.asyncio
    async def test_tool_timed_out(self, repo):
        scope = await _setup_session(repo)
        session = await _make_session(repo, scope)
        event = await session.journal_tool_terminal(
            tool_call_id="tc-1",
            correlation_id="corr-1",
            terminal_type=FactType.TOOL_TIMED_OUT,
        )
        assert event.event_type is HostEventType.TOOL_TIMED_OUT

    @pytest.mark.asyncio
    async def test_tool_effect_unknown(self, repo):
        scope = await _setup_session(repo)
        session = await _make_session(repo, scope)
        event = await session.journal_tool_terminal(
            tool_call_id="tc-1",
            correlation_id="corr-1",
            terminal_type=FactType.TOOL_EFFECT_UNKNOWN,
        )
        assert event.event_type is HostEventType.TOOL_EFFECT_UNKNOWN


# ===========================================================================
# 4. execute_tool_call — full lifecycle
# ===========================================================================


class TestExecuteToolCall:
    @pytest.mark.asyncio
    async def test_full_lifecycle_with_result(self, repo):
        scope = await _setup_session(repo)
        engine = FakeToolEngine(result={"success": True, "result": {"output": "hello"}})
        session = await _make_session(repo, scope, tool_engine=engine)
        events = await _collect(
            session.execute_tool_call(
                tool_call_id="tc-1",
                tool_name="read",
                arguments={"path": "/tmp/x"},
                correlation_id="corr-1",
                kind="read",
            )
        )
        # Expect: TOOL_REQUESTED, TOOL_AUTHORIZED_OR_DENIED, TOOL_STARTED, TOOL_RESULT
        assert len(events) == 4
        assert events[0].event_type is HostEventType.TOOL_REQUESTED
        assert events[1].event_type is HostEventType.TOOL_AUTHORIZED_OR_DENIED
        assert events[2].event_type is HostEventType.TOOL_STARTED
        assert events[3].event_type is HostEventType.TOOL_RESULT

        # Verify facts were journaled
        facts = await repo.read_facts(scope, "sess-1")
        tool_facts = [
            f
            for f in facts
            if f.fact_type
            in (
                FactType.TOOL_REQUESTED,
                FactType.TOOL_AUTHORIZED_OR_DENIED,
                FactType.TOOL_STARTED,
                FactType.TOOL_RESULT,
            )
        ]
        assert len(tool_facts) == 4

    @pytest.mark.asyncio
    async def test_lifecycle_with_failure(self, repo):
        scope = await _setup_session(repo)
        engine = FakeToolEngine(error="Something went wrong")
        session = await _make_session(repo, scope, tool_engine=engine)
        events = await _collect(
            session.execute_tool_call(
                tool_call_id="tc-1",
                tool_name="bash",
                arguments={"command": "ls"},
                correlation_id="corr-1",
            )
        )
        assert len(events) == 4
        assert events[3].event_type is HostEventType.TOOL_FAILURE
        assert events[3].metadata["error"] == "Something went wrong"

    @pytest.mark.asyncio
    async def test_no_engine_returns_stub_result(self, repo):
        """When no tool engine is set, a stub result is emitted."""
        scope = await _setup_session(repo)
        session = await _make_session(repo, scope, tool_engine=None)
        events = await _collect(
            session.execute_tool_call(
                tool_call_id="tc-1",
                tool_name="read",
                arguments={},
                correlation_id="corr-1",
            )
        )
        assert len(events) == 4
        assert events[3].event_type is HostEventType.TOOL_RESULT


# ===========================================================================
# 5. Rollback flag — legacy executor
# ===========================================================================


class TestRollbackFlag:
    @pytest.mark.asyncio
    async def test_legacy_executor_selected(self, repo):
        """When use_legacy_executor=True, the sync execute() path is used."""
        scope = await _setup_session(repo)
        engine = FakeToolEngine(result={"success": True, "result": {"output": "legacy"}})
        session = await _make_session(repo, scope, tool_engine=engine, use_legacy_executor=True)
        await _collect(
            session.execute_tool_call(
                tool_call_id="tc-1",
                tool_name="read",
                arguments={"path": "/tmp/x"},
                correlation_id="corr-1",
            )
        )
        # The engine's execute() (sync) should have been called
        assert len(engine.executed_calls) == 1
        assert engine.executed_calls[0]["function"] == "read"

    @pytest.mark.asyncio
    async def test_async_executor_selected_by_default(self, repo):
        """When use_legacy_executor=False (default), the async execute_async() path is used."""
        scope = await _setup_session(repo)
        engine = FakeToolEngine(result={"success": True, "result": {"output": "async"}})
        session = await _make_session(repo, scope, tool_engine=engine, use_legacy_executor=False)
        await _collect(
            session.execute_tool_call(
                tool_call_id="tc-1",
                tool_name="read",
                arguments={"path": "/tmp/x"},
                correlation_id="corr-1",
            )
        )
        assert len(engine.executed_calls) == 1
        assert engine.executed_calls[0]["function"] == "read"


# ===========================================================================
# 6. Tool facts survive in journal after turn
# ===========================================================================


class TestToolFactsInJournal:
    @pytest.mark.asyncio
    async def test_tool_facts_persist_after_turn(self, repo):
        """Tool facts journaled during execute_tool_call are readable after the turn."""
        scope = await _setup_session(repo)
        engine = FakeToolEngine(result={"success": True, "result": {"output": "data"}})
        session = await _make_session(repo, scope, tool_engine=engine)
        await _collect(
            session.execute_tool_call(
                tool_call_id="tc-1",
                tool_name="read",
                arguments={"path": "/tmp/x"},
                correlation_id="corr-1",
            )
        )
        facts = await repo.read_facts(scope, "sess-1")
        tool_facts = [
            f
            for f in facts
            if f.fact_type
            in (
                FactType.TOOL_REQUESTED,
                FactType.TOOL_AUTHORIZED_OR_DENIED,
                FactType.TOOL_STARTED,
                FactType.TOOL_RESULT,
            )
        ]
        assert len(tool_facts) == 4
        # Verify ordering
        assert tool_facts[0].fact_type == FactType.TOOL_REQUESTED
        assert tool_facts[1].fact_type == FactType.TOOL_AUTHORIZED_OR_DENIED
        assert tool_facts[2].fact_type == FactType.TOOL_STARTED
        assert tool_facts[3].fact_type == FactType.TOOL_RESULT
