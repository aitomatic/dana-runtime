"""
Unit tests for AgentSession — durable text turn orchestration.

Covers the Task 4 contract:
  1. input durability before model call
  2. first delta before generator completion
  3. bounded flush (chunks persisted by byte/time bound)
  4. busy conflict (second prompt while one is active)
  5. cancel (turn terminalizes as TURN_CANCELLED)
  6. exactly one terminal fact per turn (completed / cancelled / error)
  7. no post-terminal reflection (terminal is the last fact)
  8. replay host events from the journal
  9. full multi-turn roundtrip (load -> prompt -> prompt)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio

from dana.core.session.agent_session import AgentSession, SessionBusy, TextBlock, TurnTerminal
from dana.core.session.journal.models import SessionRecord
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.models import FactType, JournalFact, OwnerScope
from dana.core.session.projections.host_events import HostEventType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TERMINAL_FACT_TYPES = frozenset({FactType.TURN_COMPLETED, FactType.TURN_CANCELLED, FactType.TURN_ERROR})


# ---------------------------------------------------------------------------
# FakeAgent — stands in for STARAgent.aquery_text_stream
# ---------------------------------------------------------------------------


class FakeAgent:
    """Fake agent for AgentSession testing. Yields predefined chunks."""

    def __init__(self, chunks=None, delay=0.0, error=None, gate=None, parked=None):
        self._chunks = list(chunks or [])
        self._delay = delay
        self._error = error
        # When set, the agent awaits ``gate`` before each chunk (deterministic
        # blocking — no timing dependency). ``parked`` (if set) is signaled once
        # the agent has reached the gate, so a test can wait for it.
        self._gate = gate
        self._parked = parked
        self._timeline = SimpleNamespace(timeline=[])
        self._runtime = SimpleNamespace()
        self.object_id = "fake-agent"
        self.agent_type = "fake"

    async def aquery_text_stream(self, *, message, cancel_event, result_holder=None):
        if self._error is not None:
            raise self._error
        full_parts: list[str] = []
        for chunk in self._chunks:
            if self._delay:
                await asyncio.sleep(self._delay)
            if self._gate is not None:
                if self._parked is not None:
                    self._parked.set()
                await self._gate.wait()
            if cancel_event.is_set():
                raise asyncio.CancelledError
            full_parts.append(chunk)
            yield chunk
        if result_holder is not None:
            result_holder["full_text"] = "".join(full_parts)
            result_holder["protected_payload"] = None
            result_holder["finish_reason"] = "stop"


# ---------------------------------------------------------------------------
# Helpers / fixtures
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


def _make_session(repo, scope, agent, session_id="sess-1"):
    return AgentSession(
        owner_scope=scope,
        session_id=session_id,
        repository=repo,
        agent_factory=lambda: agent,
    )


async def _collect(agen):
    """Collect all events from an async generator."""
    events: list = []
    async for event in agen:
        events.append(event)
    return events


def _terminal_fact_types(facts):
    return [f.fact_type for f in facts if f.fact_type in _TERMINAL_FACT_TYPES]


# ===========================================================================
# 1. input durability before model call
# ===========================================================================


class TestInputDurability:
    @pytest.mark.asyncio
    async def test_input_facts_durable_before_any_chunk(self, repo):
        scope = await _setup_session(repo)
        agent = FakeAgent(chunks=["a", "b", "c"], delay=0.05)
        session = _make_session(repo, scope, agent)
        gen = session.prompt([TextBlock(text="hello")])

        # TURN_STARTED, USER_MESSAGE are yielded after the input facts are durable.
        await gen.__anext__()  # TURN_STARTED
        await gen.__anext__()  # USER_MESSAGE

        # Read the journal NOW — before the model has emitted any chunk.
        facts = await repo.read_facts(scope, "sess-1")
        fact_types = [f.fact_type for f in facts]
        assert FactType.SESSION_CREATED in fact_types
        assert FactType.TURN_STARTED in fact_types
        assert FactType.USER_CONTENT_FINAL in fact_types
        assert FactType.ASSISTANT_CONTENT_CHUNK not in fact_types

        # drain the rest
        async for _ in gen:
            pass


# ===========================================================================
# 2. first delta before generator completion
# ===========================================================================


class TestFirstDeltaBeforeCompletion:
    @pytest.mark.asyncio
    async def test_chunk_arrives_before_final(self, repo):
        scope = await _setup_session(repo)
        agent = FakeAgent(chunks=["hello", " ", "world"], delay=0.02)
        session = _make_session(repo, scope, agent)
        events = await _collect(session.prompt([TextBlock(text="hi")]))

        chunk_idx = [i for i, e in enumerate(events) if e.event_type == HostEventType.ASSISTANT_CONTENT_CHUNK]
        final_idx = [i for i, e in enumerate(events) if e.event_type == HostEventType.ASSISTANT_CONTENT_FINAL]
        assert chunk_idx, "expected at least one chunk event"
        assert final_idx, "expected an ASSISTANT_CONTENT_FINAL event"
        assert chunk_idx[0] < final_idx[0]
        terminal = session.last_terminal
        assert terminal is not None
        assert terminal.fact_type == FactType.TURN_COMPLETED


# ===========================================================================
# 3. bounded flush
# ===========================================================================


class TestBoundedFlush:
    @pytest.mark.asyncio
    async def test_chunks_flushed_to_journal_when_bound_exceeded(self, repo):
        scope = await _setup_session(repo)
        big = "x" * 2000
        # 3 * 2000 = 6000 bytes > CHUNK_FLUSH_BYTES (4096) -> at least one flush.
        agent = FakeAgent(chunks=[big, big, big])
        session = _make_session(repo, scope, agent)
        await _collect(session.prompt([TextBlock(text="hi")]))

        facts = await repo.read_facts(scope, "sess-1")
        chunk_facts = [f for f in facts if f.fact_type == FactType.ASSISTANT_CONTENT_CHUNK]
        assert len(chunk_facts) >= 1, "expected at least one flushed ASSISTANT_CONTENT_CHUNK fact"

    @pytest.mark.asyncio
    async def test_small_turn_may_have_no_chunk_facts(self, repo):
        scope = await _setup_session(repo)
        agent = FakeAgent(chunks=["hi"])
        session = _make_session(repo, scope, agent)
        await _collect(session.prompt([TextBlock(text="q")]))

        facts = await repo.read_facts(scope, "sess-1")
        final_facts = [f for f in facts if f.fact_type == FactType.ASSISTANT_CONTENT_FINAL]
        # A single small chunk need not trigger a flush, but the FINAL must exist.
        assert final_facts
        # Whatever chunks were flushed, the FINAL text reconstructs the response.
        assert final_facts[0].payload["text"] == "hi"


# ===========================================================================
# 4. busy conflict
# ===========================================================================


class TestBusyConflict:
    @pytest.mark.asyncio
    async def test_second_prompt_raises_session_busy(self, repo):
        scope = await _setup_session(repo)
        # Deterministic overlap (no timing dependency): the first turn's agent
        # parks on ``release_first`` until we release it. ``agent_parked`` tells
        # us when it has reached the gate, so the busy probe is race-free.
        release_first = asyncio.Event()
        agent_parked = asyncio.Event()
        gated = FakeAgent(chunks=["a", "b", "c"], gate=release_first, parked=agent_parked)
        session = _make_session(repo, scope, gated)

        gen1 = session.prompt([TextBlock(text="first")])
        # Drive gen1 in a task: it yields TURN_STARTED + USER_MESSAGE, then enters
        # the agent loop and parks on release_first (holding the session lock).
        task1 = asyncio.ensure_future(_collect(gen1))
        await agent_parked.wait()  # first turn is genuinely in-progress now

        gen2 = session.prompt([TextBlock(text="second")])
        with pytest.raises(SessionBusy):
            await gen2.__anext__()

        # Release the gated agent so the first turn can drain cleanly.
        release_first.set()
        await task1


# ===========================================================================
# 5. cancel
# ===========================================================================


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_terminalizes_as_turn_cancelled(self, repo):
        scope = await _setup_session(repo)
        slow = FakeAgent(chunks=["a", "b", "c", "d"], delay=0.2)
        session = _make_session(repo, scope, slow)
        gen = session.prompt([TextBlock(text="hi")])

        events = []
        events.append(await gen.__anext__())  # TURN_STARTED
        events.append(await gen.__anext__())  # USER_MESSAGE
        events.append(await gen.__anext__())  # first chunk "a"

        # Request cancellation; the agent will raise CancelledError at its next checkpoint.
        await session.cancel()

        # drain — the turn must terminalize as TURN_CANCELLED.
        events.extend(await _collect(gen))
        terminal = session.last_terminal

        assert terminal is not None
        assert terminal.fact_type == FactType.TURN_CANCELLED
        assert any(e.event_type == HostEventType.TURN_CANCELLED for e in events)

        facts = await repo.read_facts(scope, "sess-1")
        assert any(f.fact_type == FactType.TURN_CANCELLED for f in facts)


# ===========================================================================
# 6. exactly one terminal fact per turn
# ===========================================================================


class TestOneTerminal:
    @pytest.mark.asyncio
    async def test_completed_turn_has_one_terminal(self, repo):
        scope = await _setup_session(repo)
        agent = FakeAgent(chunks=["hello"])
        session = _make_session(repo, scope, agent)
        await _collect(session.prompt([TextBlock(text="hi")]))

        facts = await repo.read_facts(scope, "sess-1")
        terminals = _terminal_fact_types(facts)
        assert terminals == [FactType.TURN_COMPLETED]

    @pytest.mark.asyncio
    async def test_cancelled_turn_has_one_terminal(self, repo):
        scope = await _setup_session(repo)
        slow = FakeAgent(chunks=["a", "b"], delay=0.15)
        session = _make_session(repo, scope, slow)
        gen = session.prompt([TextBlock(text="hi")])
        await gen.__anext__()  # TURN_STARTED
        await gen.__anext__()  # USER_MESSAGE
        await gen.__anext__()  # chunk "a"
        await session.cancel()
        await _collect(gen)

        facts = await repo.read_facts(scope, "sess-1")
        terminals = _terminal_fact_types(facts)
        assert terminals == [FactType.TURN_CANCELLED]

    @pytest.mark.asyncio
    async def test_error_turn_has_one_terminal(self, repo):
        scope = await _setup_session(repo)
        agent = FakeAgent(error=RuntimeError("boom"))
        session = _make_session(repo, scope, agent)
        events = await _collect(session.prompt([TextBlock(text="hi")]))

        terminal = session.last_terminal
        assert terminal is not None
        assert terminal.fact_type == FactType.TURN_ERROR
        facts = await repo.read_facts(scope, "sess-1")
        terminals = _terminal_fact_types(facts)
        assert terminals == [FactType.TURN_ERROR]
        assert any(e.event_type == HostEventType.TURN_ERROR for e in events)


# ===========================================================================
# 7. no post-terminal reflection
# ===========================================================================


class TestNoPostTerminalReflection:
    @pytest.mark.asyncio
    async def test_terminal_is_last_fact_completed(self, repo):
        scope = await _setup_session(repo)
        agent = FakeAgent(chunks=["hello"])
        session = _make_session(repo, scope, agent)
        await _collect(session.prompt([TextBlock(text="hi")]))

        facts = await repo.read_facts(scope, "sess-1")
        assert facts[-1].fact_type == FactType.TURN_COMPLETED

    @pytest.mark.asyncio
    async def test_terminal_is_last_fact_error(self, repo):
        scope = await _setup_session(repo)
        agent = FakeAgent(error=RuntimeError("boom"))
        session = _make_session(repo, scope, agent)
        await _collect(session.prompt([TextBlock(text="hi")]))

        facts = await repo.read_facts(scope, "sess-1")
        assert facts[-1].fact_type == FactType.TURN_ERROR


# ===========================================================================
# 8. replay host events
# ===========================================================================


class TestReplayHostEvents:
    @pytest.mark.asyncio
    async def test_replay_returns_all_lifecycle_events(self, repo):
        scope = await _setup_session(repo)
        agent = FakeAgent(chunks=["hello world"])
        session = _make_session(repo, scope, agent)
        await _collect(session.prompt([TextBlock(text="hi")]))

        events = await _collect(session.replay_host_events(0))
        types = [e.event_type for e in events]
        assert HostEventType.SESSION_CREATED in types
        assert HostEventType.TURN_STARTED in types
        assert HostEventType.USER_MESSAGE in types
        assert HostEventType.ASSISTANT_CONTENT_FINAL in types
        assert HostEventType.TURN_COMPLETED in types
        assert types[-1] == HostEventType.TURN_COMPLETED

    @pytest.mark.asyncio
    async def test_replay_respects_after_sequence(self, repo):
        scope = await _setup_session(repo)
        agent = FakeAgent(chunks=["hello"])
        session = _make_session(repo, scope, agent)
        await _collect(session.prompt([TextBlock(text="hi")]))

        # SESSION_CREATED is sequence 1; replay after it excludes it.
        events = await _collect(session.replay_host_events(1))
        types = [e.event_type for e in events]
        assert HostEventType.SESSION_CREATED not in types
        assert HostEventType.TURN_STARTED in types


# ===========================================================================
# 9. full multi-turn roundtrip
# ===========================================================================


class TestFullTurnRoundtrip:
    @pytest.mark.asyncio
    async def test_two_turns_via_load(self, repo):
        scope = await _setup_session(repo)
        agent1 = FakeAgent(chunks=["answer one"])
        session1 = _make_session(repo, scope, agent1)
        await _collect(session1.prompt([TextBlock(text="question one")]))

        # Second session instance loaded from the same journal.
        agent2 = FakeAgent(chunks=["answer two"])
        session2 = AgentSession(
            owner_scope=scope,
            session_id="sess-1",
            repository=repo,
            agent_factory=lambda: agent2,
        )
        await session2.load()

        # The timeline must reflect the prior committed conversation.
        timeline_entries = agent2._timeline.timeline
        contents = [str(e.content) for e in timeline_entries]
        assert "question one" in contents
        assert "answer one" in contents

        await _collect(session2.prompt([TextBlock(text="question two")]))

        facts = await repo.read_facts(scope, "sess-1")
        completed = [f for f in facts if f.fact_type == FactType.TURN_COMPLETED]
        assert len(completed) == 2
        user_finals = [f for f in facts if f.fact_type == FactType.USER_CONTENT_FINAL]
        assert [f.payload["text"] for f in user_finals] == ["question one", "question two"]


class TestTurnTerminalShape:
    def test_turn_terminal_is_frozen(self):
        from dataclasses import FrozenInstanceError

        t = TurnTerminal(fact_type=FactType.TURN_COMPLETED, sequence=5, text="hi")
        with pytest.raises(FrozenInstanceError):
            t.text = "x"  # type: ignore[misc]
