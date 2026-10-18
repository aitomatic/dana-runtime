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
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio

from dana.core.runtime.protocols import StreamEvent, StreamEventType
from dana.core.session.agent_session import AgentSession, SessionBusy, TextBlock, TurnTerminal
from dana.core.session.journal.models import ProjectionCheckpoint, SessionRecord
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.models import FactType, JournalFact, NewJournalFact, OwnerScope
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

    async def aquery_stream(self, *, message=None, **kwargs):
        """Streaming STAR-loop stand-in: yields TEXT_DELTA per chunk, then DONE.

        Cooperative cancellation is handled by AgentSession.prompt (it checks
        its own _cancel_event between events), so this fake does not need a
        cancel_event argument.
        """
        if self._error is not None:
            raise self._error
        for chunk in self._chunks:
            if self._delay:
                await asyncio.sleep(self._delay)
            if self._gate is not None:
                if self._parked is not None:
                    self._parked.set()
                await self._gate.wait()
            yield StreamEvent(event_type=StreamEventType.TEXT_DELTA, data=chunk, iteration=0)
        yield StreamEvent(event_type=StreamEventType.DONE, data=None, iteration=0)


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

    @pytest.mark.asyncio
    async def test_cancel_interrupts_parked_provider_and_closes_stream(self, repo):
        scope = await _setup_session(repo)
        parked = asyncio.Event()
        closed = asyncio.Event()

        class ParkedAgent(FakeAgent):
            async def aquery_stream(self, **kwargs):
                try:
                    parked.set()
                    await asyncio.Event().wait()
                    yield StreamEvent(event_type=StreamEventType.DONE, data=None, iteration=0)
                finally:
                    closed.set()

        session = _make_session(repo, scope, ParkedAgent())
        consumer = asyncio.create_task(_collect(session.prompt([TextBlock(text="hi")])))
        await asyncio.wait_for(parked.wait(), 1)
        await session.cancel()
        events = await asyncio.wait_for(consumer, 1)
        assert closed.is_set()
        assert events[-1].event_type == HostEventType.TURN_CANCELLED
        facts = await repo.read_facts(scope, "sess-1")
        assert _terminal_fact_types(facts) == [FactType.TURN_CANCELLED]

    @pytest.mark.asyncio
    async def test_consumer_close_after_start_terminalizes_and_releases_session(self, repo):
        scope = await _setup_session(repo)
        session = _make_session(repo, scope, FakeAgent(chunks=["answer"]))
        stream = session.prompt([TextBlock(text="first")])
        assert (await anext(stream)).event_type == HostEventType.TURN_STARTED
        await stream.aclose()
        facts = await repo.read_facts(scope, "sess-1")
        assert _terminal_fact_types(facts) == [FactType.TURN_CANCELLED]
        await _collect(session.prompt([TextBlock(text="second")]))
        facts = await repo.read_facts(scope, "sess-1")
        assert _terminal_fact_types(facts) == [FactType.TURN_CANCELLED, FactType.TURN_COMPLETED]

    @pytest.mark.asyncio
    async def test_consumer_close_after_committed_answer_does_not_cancel(self, repo):
        scope = await _setup_session(repo)
        session = _make_session(repo, scope, FakeAgent(chunks=["answer"]))
        stream = session.prompt([TextBlock(text="hi")])
        async for event in stream:
            if event.event_type == HostEventType.ASSISTANT_CONTENT_FINAL:
                break
        await stream.aclose()
        facts = await repo.read_facts(scope, "sess-1")
        assert _terminal_fact_types(facts) == [FactType.TURN_COMPLETED]
        assert session.last_terminal.fact_type == FactType.TURN_COMPLETED

    @pytest.mark.asyncio
    async def test_cancelled_append_keeps_committed_context_exactly_once(self, repo, monkeypatch):
        scope = await _setup_session(repo)
        session = _make_session(repo, scope, FakeAgent(chunks=["answer"]))
        await session.load()
        committed = asyncio.Event()
        release = asyncio.Event()
        append = repo.append

        async def pause_after_commit(*args, **kwargs):
            result = await append(*args, **kwargs)
            committed.set()
            await release.wait()
            return result

        monkeypatch.setattr(repo, "append", pause_after_commit)
        writer = asyncio.create_task(
            session.append_fact(
                NewJournalFact(
                    fact_type=FactType.USER_CONTENT_FINAL,
                    correlation_id="external-input",
                    causation_id=None,
                    payload={"text": "retain this input once"},
                )
            )
        )
        await asyncio.wait_for(committed.wait(), 1)
        writer.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(writer, 1)

        facts = await repo.read_facts(scope, "sess-1")
        assert [fact.payload["text"] for fact in facts if fact.fact_type is FactType.USER_CONTENT_FINAL] == ["retain this input once"]
        context = session._conversation_state.view().messages
        assert [(message.role, message.content) for message in context] == [("user", "retain this input once")]

    @pytest.mark.asyncio
    async def test_tool_started_reaches_host_before_effect_finishes(self, repo):
        from dana.core.tool.tool_executor import ToolExecutor

        entered = asyncio.Event()
        release = asyncio.Event()
        cancelled = asyncio.Event()
        started_at_host = asyncio.Event()

        class Resource:
            async def wait(self):
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                return "effect completed"

        resource = Resource()
        executor = ToolExecutor(tool_name_registry_getter=lambda: {"Wait": (resource, "wait")})

        class ToolAgent(FakeAgent):
            async def aquery_stream(self, *, message=None, **kwargs):
                await executor.execute_tools_async(self, [{"function": "Wait", "arguments": {}, "tool_call_id": "blocked-effect"}])
                yield StreamEvent(event_type=StreamEventType.DONE, data=None, iteration=0)

        agent = ToolAgent()
        agent._runtime._tool_executor = executor
        scope = await _setup_session(repo)
        session = _make_session(repo, scope, agent)
        events = []

        async def consume():
            async for event in session.prompt([TextBlock(text="run the waiting tool")]):
                events.append(event.event_type)
                if event.event_type is HostEventType.TOOL_STARTED:
                    started_at_host.set()

        consumer = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(started_at_host.wait(), 1)
            await asyncio.wait_for(entered.wait(), 1)
            assert not release.is_set()
            assert not consumer.done()
            await session.cancel()
            await asyncio.wait_for(consumer, 1)
            assert cancelled.is_set()
            facts = await repo.read_facts(scope, "sess-1")
            assert [fact.fact_type for fact in facts if fact.fact_type in _TERMINAL_FACT_TYPES] == [FactType.TURN_CANCELLED]
            assert [
                fact.fact_type
                for fact in facts
                if fact.fact_type in {FactType.TOOL_RESULT, FactType.TOOL_FAILURE, FactType.TOOL_ACKNOWLEDGED, FactType.TOOL_EFFECT_UNKNOWN}
            ] == [FactType.TOOL_EFFECT_UNKNOWN]
            assert events[-1] is HostEventType.TURN_CANCELLED
        finally:
            release.set()
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            await session.aclose()
            await executor.aclose()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("decision_name", ["DENY", "NEEDS_PROMPT"])
async def test_unresolved_permission_prevents_effect_without_native_catalog(repo, tmp_path, decision_name):
    from dana.core.policy.evaluator import PolicyDecision
    from dana.core.tool.tool_executor import ToolExecutor

    effect_path = tmp_path / "unauthorized-effect.txt"

    class Resource:
        async def mutate(self):
            effect_path.write_text("committed")
            return "committed"

    class Evaluator:
        async def evaluate(self, operation, scope):
            return SimpleNamespace(decision=PolicyDecision[decision_name], reason="permission required")

    resource = Resource()
    executor = ToolExecutor(tool_name_registry_getter=lambda: {"Mutate": (resource, "mutate")})

    class ToolAgent(FakeAgent):
        async def aquery_stream(self, *, message=None, **kwargs):
            self.results = await executor.execute_tools_async(
                self, [{"function": "Mutate", "arguments": {}, "tool_call_id": "unauthorized-effect"}]
            )
            yield StreamEvent(event_type=StreamEventType.DONE, data=None, iteration=0)

    agent = ToolAgent()
    agent._runtime._tool_executor = executor
    scope = await _setup_session(repo)
    session = _make_session(repo, scope, agent)
    session._policy_evaluator = Evaluator()
    try:
        await _collect(session.prompt([TextBlock(text="attempt the mutation")]))
        # The effect never happens, and no "started" fact is ever recorded.
        assert not effect_path.exists()
        facts = await repo.read_facts(scope, "sess-1")
        assert FactType.TOOL_STARTED not in [fact.fact_type for fact in facts]
        # I5: the refusal is recoverable. The denied call carries exactly one
        # TOOL_FAILURE(policy_block) result and the turn completes, so the model
        # can adapt instead of losing the turn to a TURN_ERROR.
        failures = [fact for fact in facts if fact.fact_type is FactType.TOOL_FAILURE]
        assert len(failures) == 1
        # DENY surfaces the evaluator reason; NEEDS_PROMPT with no host callback
        # fails closed on the interactive path. Both are a truthful refusal.
        expected_reason = "permission required" if decision_name == "DENY" else "interactive permission is unavailable"
        assert expected_reason in str(failures[0].payload["result"])
        denied = [fact for fact in facts if fact.fact_type is FactType.TOOL_AUTHORIZED_OR_DENIED]
        assert [fact.payload["authorized"] for fact in denied] == [False]
        assert _terminal_fact_types(facts) == [FactType.TURN_COMPLETED]
        # The model receives the truthful blocked outcome for its call id.
        results = agent.results
        assert len(results) == 1
        assert results[0]["type"] == "policy_block"
        assert results[0]["success"] is False
        assert results[0]["tool_call_id"] == "unauthorized-effect"
    finally:
        await session.aclose()
        await executor.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_recovery_snapshot_is_not_mutated_by_abandoned_live_content(repo, native):
    from dana.core.timeline.compressed_timeline import CompressedTimeline
    from dana.core.timeline.native_message import NativeMessage, NativeToolCall
    from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType

    agent = FakeAgent()
    if native:
        agent._timeline = CompressedTimeline(compression_enabled=False)
        serialized = NativeMessage(
            role="assistant",
            content=[{"type": "text", "text": "committed"}],
            metadata={"nested": {"state": "committed"}},
            tool_calls=[NativeToolCall(id="call", name="Read", arguments={"path": "committed"})],
        ).to_dict()
    else:
        serialized = TimelineEntry(
            entry_type=TimelineEntryType.AGENT_RESPONSE,
            content=[{"type": "text", "text": "committed"}],
            metadata={"nested": {"state": "committed"}},
            tool_calls=[{"id": "call", "name": "Read", "arguments": {"path": "committed"}}],
        ).to_dict()
    session = _make_session(repo, OwnerScope(owner_id="owner", workspace="workspace"), agent)
    session._agent = agent
    session._working_context = [serialized]
    committed = deepcopy(session._working_context)

    session._restore_working_context()
    live = agent._timeline.native_messages[0] if native else agent._timeline.timeline[0]
    live.content[0]["text"] = "abandoned"
    live.metadata["nested"]["state"] = "abandoned"
    if native:
        live.tool_calls[0].arguments["path"] = "abandoned"
    else:
        live.tool_calls[0]["arguments"]["path"] = "abandoned"
    assert session._working_context == committed
    session._restore_working_context()
    recovered = agent._timeline.native_messages[0] if native else agent._timeline.timeline[0]
    assert recovered.content == [{"type": "text", "text": "committed"}]
    assert recovered.metadata["nested"] == {"state": "committed"}
    await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_malformed_working_checkpoint_reopens_from_authoritative_journal(repo, native):
    from dana.core.timeline.compressed_timeline import CompressedTimeline

    scope = await _setup_session(repo)

    def make_agent():
        agent = FakeAgent(chunks=["committed answer"])
        if native:
            agent._timeline = CompressedTimeline(compression_enabled=False)
        return agent

    first = _make_session(repo, scope, make_agent())
    first._codec = None
    reopened = _make_session(repo, scope, make_agent())
    reopened._codec = None
    try:
        await _collect(first.prompt([TextBlock(text="original question")]))
        checkpoint = await repo.load_projection_checkpoint(scope, "sess-1", "conversation-working-v1")
        assert checkpoint is not None
        data = deepcopy(checkpoint.data)
        data["state"]["working_context"] = [{}]
        await repo.save_projection_checkpoint(
            scope,
            "sess-1",
            ProjectionCheckpoint(
                projection_name=checkpoint.projection_name,
                last_sequence=checkpoint.last_sequence,
                data=data,
            ),
        )
        await reopened.load()
        assert [entry.content for entry in reopened._agent._timeline.timeline] == ["original question", "committed answer"]
        assert _terminal_fact_types(await repo.read_facts(scope, "sess-1")) == [FactType.TURN_COMPLETED]
    finally:
        await first.aclose()
        await reopened.aclose()
