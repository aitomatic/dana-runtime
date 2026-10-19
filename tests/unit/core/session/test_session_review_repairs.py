"""Regressions for the PR #32 session review findings (C3, I2, I3, I4, S6).

Each test asserts one observable contract, never an implementation detail:

* C3 - a fold failure AFTER the terminal batch is durable must leave exactly one
  terminal fact and rebuild the derived view on the next request.
* I2 - the interruption observation reaches the model on the intended recovery
  turn and is retired afterwards, without duplicating across a reopen.
* I3 - two sessions sharing one SQLite connection cannot interleave a
  transaction, and a cancellation mid-transaction cannot leave the shared
  connection inside an open transaction.
* I4 - an explicit ``cancel()`` is reported as TURN_CANCELLED, while the caller's
  own task cancellation (asyncio.timeout / TaskGroup) propagates. Interrupted
  tool commits retain exactly one truthful outcome before the turn terminal.
* S6 - terminating a turn retires its unfinished-turn bookkeeping instead of
  carrying it forever, while a still-open turn keeps what it needs.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio

from dana.common.llm.types import LLMProvider, LLMStreamChunk
from dana.common.protocols import named_tool
from dana.config.storage_config import FileStorageConfig
from dana.core.agent.star_agent import STARAgent
from dana.core.knowledge.prompts.codecs import NativeToolsCodec
from dana.core.resource.base_resource import BaseResource
from dana.core.runtime.protocols import StreamEvent, StreamEventType
from dana.core.session.agent_session import _INTERRUPTION_NOTE_KEY, AgentSession, TextBlock
from dana.core.session.journal.models import JournalConflict, SessionRecord
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.models import FactType, JournalFact, NewJournalFact, OwnerScope
from dana.core.session.projections.conversation import (
    _INTERRUPTED_OBSERVATION,
    ConversationProjectionState,
    ConversationProjector,
)
from dana.repositories.local_file_repository import (
    LocalEventRepository,
    LocalLearningRepository,
    LocalPromptRepository,
    LocalTimelineRepository,
)
from dana.repositories.repository_factory import RepositoryFactory, RepositoryType


_TERMINAL_FACTS = frozenset({FactType.TURN_COMPLETED, FactType.TURN_CANCELLED, FactType.TURN_ERROR})

# A key long enough for the AES-GCM codec used by the protected-replay path.
_KEY = b"review-repair-key-32-bytes-long!"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class ScriptedAgent:
    """Minimal agent stand-in: yields TEXT_DELTA per chunk, then DONE."""

    def __init__(self, chunks=(), *, timeline=None):
        self._chunks = list(chunks)
        self._timeline = timeline if timeline is not None else SimpleNamespace(timeline=[], _native_messages=[])
        self._runtime = SimpleNamespace()
        self._stream_updates_timeline = True

    async def aquery_stream(self, *, message=None, **kwargs):
        for chunk in self._chunks:
            yield StreamEvent(event_type=StreamEventType.TEXT_DELTA, data=chunk, iteration=0)
        yield StreamEvent(event_type=StreamEventType.DONE, data=None, iteration=0)


class GatedAgent(ScriptedAgent):
    """Yields one chunk, then blocks on ``gate`` (deterministic overlap)."""

    def __init__(self, gate, parked, **kwargs):
        super().__init__(**kwargs)
        self._gate = gate
        self._parked = parked

    async def aquery_stream(self, *, message=None, **kwargs):
        yield StreamEvent(event_type=StreamEventType.TEXT_DELTA, data="partial", iteration=0)
        self._parked.set()
        await self._gate.wait()
        yield StreamEvent(event_type=StreamEventType.DONE, data=None, iteration=0)


@pytest_asyncio.fixture
async def repo(tmp_path):
    r = await SQLiteJournalRepository.open(str(tmp_path / "journal.db"))
    yield r
    await r.close()


def _scope(tag: str = "a") -> OwnerScope:
    return OwnerScope(owner_id=f"owner-{tag}", workspace=f"ws-{tag}")


async def _seed(repo, session_id: str, scope: OwnerScope | None = None) -> OwnerScope:
    scope = scope or _scope()
    record = SessionRecord.new(session_id, scope)
    await repo.create_session(
        record,
        [
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
            )
        ],
    )
    return scope


def _make_session(repo, scope, agent, session_id="sess-1") -> AgentSession:
    return AgentSession(owner_scope=scope, session_id=session_id, repository=repo, agent_factory=lambda: agent)


async def _collect(agen):
    return [event async for event in agen]


# ---------------------------------------------------------------------------
# C3: post-commit fold failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_postcommit_fold_failure_keeps_single_terminal_and_rebuilds(repo, monkeypatch):
    """A derived-fold failure after a durable terminal cannot add a TURN_ERROR."""
    scope = await _seed(repo, "sess-1")
    agent = ScriptedAgent(chunks=["hello"])
    session = _make_session(repo, scope, agent)

    # Inject the failure at the *post-commit* fold boundary: the terminal batch
    # is already durable when this raises.
    original = AgentSession._advance_conversation
    calls = {"n": 0}

    def failing(self, facts):
        calls["n"] += 1
        if any(f.fact_type is FactType.TURN_COMPLETED for f in facts):
            raise RuntimeError("injected projection failure")
        return original(self, facts)

    monkeypatch.setattr(AgentSession, "_advance_conversation", failing)
    events = await _collect(session.prompt([TextBlock(text="hi")]))

    types = [e.event_type.value for e in events]
    assert "turn_error" not in types, f"failed fold must not surface as a turn error: {types}"
    assert session.last_terminal is not None
    assert session.last_terminal.fact_type is FactType.TURN_COMPLETED

    facts = await repo.read_facts(scope, "sess-1")
    terminals = [f.fact_type for f in facts if f.fact_type in _TERMINAL_FACTS]
    assert terminals == [FactType.TURN_COMPLETED], f"exactly one terminal expected, got {terminals}"
    assert calls["n"] >= 1

    # The next request rebuilds the derived state from the authoritative journal.
    monkeypatch.setattr(AgentSession, "_advance_conversation", original)
    view = ConversationProjector().project(facts)
    assert [m.role for m in view.messages] == ["user", "assistant"]
    assert view.messages[1].content == "hello"

    agent._chunks = ["second"]
    events2 = await _collect(session.prompt([TextBlock(text="again")]))
    assert [e.event_type.value for e in events2][:2] == ["turn_started", "user_message"]
    assert session.last_terminal is not None
    assert session.last_terminal.fact_type is FactType.TURN_COMPLETED

    facts2 = await repo.read_facts(scope, "sess-1")
    terminals2 = [f.fact_type for f in facts2 if f.fact_type in _TERMINAL_FACTS]
    assert terminals2 == [FactType.TURN_COMPLETED, FactType.TURN_COMPLETED], f"got {terminals2}"
    # The rebuilt working set carries the committed history, not a lost turn.
    assert [m.content for m in ConversationProjector().project(facts2).messages if m.role == "user"] == ["hi", "again"]


# ---------------------------------------------------------------------------
# I3: one lock per shared SQLite connection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_connection_transactions_do_not_interleave(repo):
    """Concurrent appends on one shared connection may not interleave.

    Each writer reads the version, then inserts its batch, then bumps the
    version. Without a per-connection lock two appends can both pass the version
    check and insert the SAME sequence number, so the journal would gain
    duplicate/gapped sequences. With the lock the check and the insert are one
    unit: exactly one writer wins and the loser gets a truthful
    ``JournalConflict``.
    """
    scope = await _seed(repo, "sess-a", _scope("a"))

    def batch(tag: str) -> list[NewJournalFact]:
        return [
            NewJournalFact(
                fact_type=FactType.USER_CONTENT_FINAL,
                correlation_id=tag,
                causation_id=None,
                payload={"text": tag},
            )
        ]

    for version in range(1, 6):
        results = await asyncio.gather(
            repo.append(scope, "sess-a", version, batch(f"a{version}")),
            repo.append(scope, "sess-a", version, batch(f"b{version}")),
            return_exceptions=True,
        )
        winners = [r for r in results if not isinstance(r, BaseException)]
        losers = [r for r in results if isinstance(r, BaseException)]
        assert len(winners) == 1, f"exactly one append may win version {version}: {results}"
        assert len(losers) == 1 and isinstance(losers[0], JournalConflict), losers

    facts = await repo.read_facts(scope, "sess-a")
    sequences = [f.sequence for f in facts]
    assert sequences == list(range(1, len(facts) + 1)), f"duplicate or gapped sequences: {sequences}"
    header = await repo.load_session(scope, "sess-a")
    assert header.version == len(facts)


@pytest.mark.asyncio
async def test_cancelled_append_leaves_no_open_transaction(repo):
    """A cancelled append must not strand the shared connection in a transaction."""
    scope = await _seed(repo, "sess-1")

    class _SlowConnection:
        """Wrapper that parks on the first insert of a batch."""

        def __init__(self, db, parked, gate):
            self._db = db
            self._parked = parked
            self._gate = gate
            self._armed = False

        def __getattr__(self, name):
            return getattr(self._db, name)

        async def execute(self, sql, params=()):
            if not self._armed and sql.strip().upper().startswith("INSERT INTO SESSION_FACTS"):
                self._armed = True
                self._parked.set()
                await self._gate.wait()
            return await self._db.execute(sql, params)

    parked, gate = asyncio.Event(), asyncio.Event()
    real_db = repo._db
    repo._db = _SlowConnection(real_db, parked, gate)

    task = asyncio.create_task(
        repo.append(
            scope,
            "sess-1",
            1,
            [NewJournalFact(fact_type=FactType.USER_CONTENT_FINAL, correlation_id="c1", causation_id=None, payload={"text": "x"})],
        )
    )
    await parked.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.set()
    repo._db = real_db

    # The connection is usable and the aborted batch left nothing behind.
    facts = await repo.read_facts(scope, "sess-1")
    assert [f.fact_type for f in facts] == [FactType.SESSION_CREATED]
    appended = await repo.append(
        scope,
        "sess-1",
        1,
        [NewJournalFact(fact_type=FactType.USER_CONTENT_FINAL, correlation_id="c2", causation_id=None, payload={"text": "y"})],
    )
    assert appended.new_version == 2


# ---------------------------------------------------------------------------
# I4: caller cancellation propagates, explicit cancel() does not
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_caller_cancellation_propagates_after_durable_terminal(repo):
    """asyncio.timeout/TaskGroup cancellation must not look like a normal return."""
    scope = await _seed(repo, "sess-1")
    gate, parked = asyncio.Event(), asyncio.Event()
    session = _make_session(repo, scope, GatedAgent(gate, parked, chunks=["partial"]))
    gen = session.prompt([TextBlock(text="hi")])

    await gen.__anext__()  # TURN_STARTED
    await gen.__anext__()  # USER_MESSAGE
    await gen.__anext__()  # ASSISTANT_CONTENT_CHUNK

    # Drive one more step: the stream parks on the gate, then cancel the caller.
    step = asyncio.create_task(gen.__anext__())
    await parked.wait()
    step.cancel()
    with pytest.raises(asyncio.CancelledError):
        await step
    gate.set()

    # The durable outcome exists exactly once and is truthful.
    assert session.last_terminal is not None
    assert session.last_terminal.fact_type is FactType.TURN_CANCELLED
    facts = await repo.read_facts(scope, "sess-1")
    terminals = [f.fact_type for f in facts if f.fact_type in _TERMINAL_FACTS]
    assert terminals == [FactType.TURN_CANCELLED]


@pytest.mark.asyncio
async def test_internal_stream_cancellation_does_not_cancel_consumer(repo):
    scope = await _seed(repo, "sess-1")

    class InternallyCancelledAgent(ScriptedAgent):
        async def aquery_stream(self, *, message=None, **kwargs):
            yield StreamEvent(event_type=StreamEventType.TEXT_DELTA, data="partial", iteration=0)
            raise asyncio.CancelledError

    session = _make_session(repo, scope, InternallyCancelledAgent())
    events = await _collect(session.prompt([TextBlock(text="run")]))
    assert events[-1].event_type.value == "turn_cancelled"
    assert asyncio.current_task().cancelling() == 0
    facts = await repo.read_facts(scope, "sess-1")
    assert [f.fact_type for f in facts if f.fact_type in _TERMINAL_FACTS] == [FactType.TURN_CANCELLED]


@pytest.mark.asyncio
async def test_postcommit_observer_failure_preserves_committed_version_and_rebuild(repo):
    scope = await _seed(repo, "sess-1")
    session = _make_session(repo, scope, ScriptedAgent(chunks=["ok"]))
    await session.load()

    def fail_observation(facts, cancelled):
        raise ValueError("synthetic post-commit observation failure")

    session._tool_commit_observer = fail_observation
    await session.append_fact(
        NewJournalFact(
            fact_type=FactType.MODEL_CHANGED,
            correlation_id="model-change",
            causation_id=None,
            payload={"model": "synthetic-model"},
        )
    )
    facts = await repo.read_facts(scope, "sess-1")
    assert facts[-1].fact_type is FactType.MODEL_CHANGED
    assert session.version == facts[-1].sequence
    assert session._conversation_state is None
    session._tool_commit_observer = None
    await _collect(session.prompt([TextBlock(text="continue")]))
    facts = await repo.read_facts(scope, "sess-1")
    assert [f.fact_type for f in facts if f.fact_type in _TERMINAL_FACTS] == [FactType.TURN_COMPLETED]


# ---------------------------------------------------------------------------
# I4: caller cancellation while a durable append is parked
# ---------------------------------------------------------------------------


class _ParkAppendRepository:
    """Delegating repository that parks the batch carrying a target fact type."""

    def __init__(self, inner, target: FactType):
        self._inner = inner
        self._target = target
        self.parked = asyncio.Event()
        self.gate = asyncio.Event()
        self.committed = asyncio.Event()
        self._armed = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def append(self, *args: Any, **kwargs: Any) -> Any:
        facts = args[3] if len(args) > 3 else kwargs.get("facts", [])
        selected = self._armed and any(getattr(fact, "fact_type", None) is self._target for fact in facts)
        if selected:
            self._armed = False
            self.parked.set()
            await self.gate.wait()
        result = await self._inner.append(*args, **kwargs)
        if selected:
            self.committed.set()
        return result


class _ToolCommitProvider(LLMProvider):
    """Synthetic native-tool traffic through the real STAR loop, without I/O."""

    name = "session-tool-commit-regression"
    model = "synthetic"
    fingerprint = "session-tool-commit-regression:synthetic:offline"

    def __init__(self):
        self.requests = 0

    @property
    def supports_native_tools(self):
        return True

    def prepare_tools(self, tools):
        return list(tools)

    async def stream(self, messages, tools=None, **kwargs):
        self.requests += 1
        if self.requests == 1:
            yield LLMStreamChunk(
                type="tool_use",
                tool_call={"id": "effect-1", "name": "commit_effect", "input": {"text": "committed-effect"}},
            )
        else:
            yield LLMStreamChunk(type="text_delta", content="Recovered after cancellation.")


class _ToolCommitResource(BaseResource):
    """Record an observable effect before either returning or raising."""

    def __init__(self):
        super().__init__(resource_type="session_commit_probe", resource_id="session-commit-tools", auto_register=False)
        self.calls = []
        self.fail = False

    @named_tool(name="commit_effect")
    async def commit_effect(self, text: str = "") -> str:
        """Record the requested effect and return its acknowledgement."""
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("Synthetic failure after the recorded effect")
        return f"acknowledged:{text}"


@pytest_asyncio.fixture
async def real_tool_session(repo, tmp_path, monkeypatch):
    """Isolate every repository and disable optional external integrations."""
    monkeypatch.setenv("DANA_MCP_SERVERS", "")
    monkeypatch.setenv("DANA_MOCK_LLM", "false")
    factory = RepositoryFactory()
    config = FileStorageConfig(workspace_folder=str(tmp_path))
    for kind, creator in (
        (RepositoryType.TIMELINE, LocalTimelineRepository),
        (RepositoryType.EVENT, LocalEventRepository),
        (RepositoryType.LEARNING, LocalLearningRepository),
        (RepositoryType.PROMPT, LocalPromptRepository),
    ):
        factory.register(kind, creator, config)
    provider = _ToolCommitProvider()
    resource = _ToolCommitResource()
    agent = STARAgent(
        agent_id=f"session-commit-{uuid4()}",
        agent_type="session_commit_regression",
        llm_provider_instance=provider,
        codec=NativeToolsCodec,
        repository_factory=factory,
        auto_register=False,
        enable_skills=False,
        enable_assistant=False,
        enable_web_search=False,
        enable_code_execution=False,
        compress_timeline=False,
        identity_override="Synthetic session cancellation regression.",
    ).with_resources(resource)
    monkeypatch.setattr(agent._runtime, "get_runtime_context_fields", lambda: ["timestamp", "timezone", "user"])
    scope = await _seed(repo, "sess-1")
    session = _make_session(repo, scope, agent)
    try:
        yield session, resource, scope
    finally:
        await session.aclose()
        await provider.aclose()


@pytest.mark.asyncio
async def test_caller_cancel_during_durable_terminal_append_keeps_single_terminal(repo):
    """Cancel while the terminal batch commits: one terminal, and a raise.

    The batch is already durable when the caller's task is cancelled, so it must
    not be re-appended (the turn would end with both TURN_COMPLETED and
    TURN_CANCELLED); and the cancellation must still reach the caller instead of
    being reported as a normal turn completion.
    """
    scope = await _seed(repo, "sess-1")
    slow = _ParkAppendRepository(repo, FactType.TURN_COMPLETED)
    session = _make_session(slow, scope, ScriptedAgent(chunks=["ok"]))

    task = asyncio.create_task(_collect(session.prompt([TextBlock(text="hi")])))
    await slow.parked.wait()
    task.cancel()
    slow.gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    facts = await repo.read_facts(scope, "sess-1")
    terminals = [f.fact_type for f in facts if f.fact_type in _TERMINAL_FACTS]
    assert terminals == [FactType.TURN_COMPLETED], f"exactly one terminal expected, got {terminals}"


@pytest.mark.asyncio
async def test_caller_cancel_during_durable_start_append_terminalizes_the_turn(repo):
    """Cancel while the turn-start batch commits: the turn still gets one terminal."""
    scope = await _seed(repo, "sess-1")
    slow = _ParkAppendRepository(repo, FactType.TURN_STARTED)
    session = _make_session(slow, scope, ScriptedAgent(chunks=["ok"]))

    task = asyncio.create_task(_collect(session.prompt([TextBlock(text="hi")])))
    await slow.parked.wait()
    task.cancel()
    slow.gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    facts = await repo.read_facts(scope, "sess-1")
    terminals = [f.fact_type for f in facts if f.fact_type in _TERMINAL_FACTS]
    assert terminals == [FactType.TURN_CANCELLED], f"exactly one terminal expected, got {terminals}"


@pytest.mark.asyncio
async def test_explicit_cancel_returns_turn_cancelled_event(repo):
    """cancel() is an owned outcome: the caller keeps iterating, no raise."""
    scope = await _seed(repo, "sess-1")
    gate, parked = asyncio.Event(), asyncio.Event()
    session = _make_session(repo, scope, GatedAgent(gate, parked, chunks=["partial"]))
    gen = session.prompt([TextBlock(text="hi")])

    await gen.__anext__()  # TURN_STARTED
    await gen.__anext__()  # USER_MESSAGE
    await gen.__anext__()  # ASSISTANT_CONTENT_CHUNK

    step = asyncio.create_task(gen.__anext__())
    await parked.wait()
    await session.cancel()
    gate.set()
    assert (await step).event_type.value == "turn_cancelled"
    events = [event async for event in gen]

    assert [e.event_type.value for e in events] == []
    assert session.last_terminal is not None
    assert session.last_terminal.fact_type is FactType.TURN_CANCELLED


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["explicit", "caller", "both"])
@pytest.mark.parametrize(
    "boundary",
    [
        FactType.TOOL_REQUESTED,
        FactType.TOOL_AUTHORIZED_OR_DENIED,
        FactType.TOOL_STARTED,
        FactType.TOOL_RESULT,
        FactType.TOOL_FAILURE,
    ],
)
async def test_tool_commit_cancellation_preserves_one_truthful_outcome(repo, real_tool_session, boundary, mode):
    """Cancellation settles durable tool facts before classifying the consumer."""
    session, resource, scope = real_tool_session
    resource.fail = boundary is FactType.TOOL_FAILURE
    slow = _ParkAppendRepository(repo, boundary)
    session._repository = slow
    events = []

    async def consume():
        async for event in session.prompt([TextBlock(text="Perform the synthetic effect.")]):
            events.append(event)

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(slow.parked.wait(), 3)
        effect_completed = boundary in {FactType.TOOL_RESULT, FactType.TOOL_FAILURE}
        assert resource.calls == (["committed-effect"] if effect_completed else [])
        if mode in {"explicit", "both"}:
            await session.cancel()
        if mode in {"caller", "both"}:
            task.cancel()
        # Give cancellation a scheduling turn while the real append is held.
        await asyncio.sleep(0)
        assert not task.done(), "The consumer must retain the unresolved commit"
        assert not slow.committed.is_set()
        slow.gate.set()
        if mode == "explicit":
            await asyncio.wait_for(task, 3)
            assert events[-1].event_type.value == "turn_cancelled"
        else:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
        assert slow.committed.is_set()

        facts = await repo.read_facts(scope, "sess-1")
        tool_facts = [fact for fact in facts if fact.payload.get("tool_call_id") == "effect-1"]
        assert sum(fact.fact_type is FactType.TOOL_REQUESTED for fact in tool_facts) == 1
        tool_terminal_types = {
            FactType.TOOL_RESULT,
            FactType.TOOL_FAILURE,
            FactType.TOOL_ACKNOWLEDGED,
            FactType.TOOL_TIMED_OUT,
            FactType.TOOL_EFFECT_UNKNOWN,
        }
        tool_terminals = [fact for fact in tool_facts if fact.fact_type in tool_terminal_types]
        expected = boundary if effect_completed else FactType.TOOL_ACKNOWLEDGED
        assert [fact.fact_type for fact in tool_terminals] == [expected]
        if boundary is FactType.TOOL_RESULT:
            assert tool_terminals[0].payload["result"] == "acknowledged:committed-effect"
        if boundary is FactType.TOOL_FAILURE:
            assert "Synthetic failure after the recorded effect" in str(tool_terminals[0].payload)
        turn_terminals = [fact for fact in facts if fact.fact_type in _TERMINAL_FACTS]
        assert [fact.fact_type for fact in turn_terminals] == [FactType.TURN_CANCELLED]
        assert tool_terminals[0].sequence < turn_terminals[0].sequence
        assert [fact.sequence for fact in facts] == list(range(1, len(facts) + 1))
        assert (await repo.load_session(scope, "sess-1")).version == len(facts)
        if mode == "explicit":
            visible_terminals = [event for event in events if event.event_type.value == expected.value]
            assert [event.sequence for event in visible_terminals] == [tool_terminals[0].sequence]

        recovered = await asyncio.wait_for(_collect(session.prompt([TextBlock(text="Continue on a new turn.")])), 3)
        assert recovered[-1].event_type.value == "turn_completed"
        assert resource.calls == (["committed-effect"] if effect_completed else [])
        later = await repo.read_facts(scope, "sess-1")
        assert [fact.fact_type for fact in later if fact.fact_type in _TERMINAL_FACTS] == [
            FactType.TURN_CANCELLED,
            FactType.TURN_COMPLETED,
        ]
        assert [fact for fact in later if fact.fact_type in tool_terminal_types] == tool_terminals
    finally:
        slow.gate.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# ---------------------------------------------------------------------------
# I2: the interruption note reaches the model once, then is retired
# ---------------------------------------------------------------------------


def _native_timeline():
    from dana.core.timeline.native_message import NativeMessage

    return SimpleNamespace(timeline=[], _native_messages=[NativeMessage(role="user", content="earlier")])


def _agent_with(timeline):
    agent = SimpleNamespace(_timeline=timeline)
    return agent


def test_reconcile_injects_exactly_one_note_and_retires_it():
    from dana.core.session.agent_session import AgentSession

    timeline = _native_timeline()

    class _Stub:
        pass

    stub = _Stub()
    stub._agent = _agent_with(timeline)
    timeline.add_entry = None  # force the entries-list path

    AgentSession._reconcile_interruption_note(stub, _INTERRUPTED_OBSERVATION)
    notes = [e for e in timeline.timeline if (e.metadata or {}).get(_INTERRUPTION_NOTE_KEY)]
    assert len(notes) == 1
    assert notes[0].content == _INTERRUPTED_OBSERVATION

    # Idempotent while the observation stays current.
    AgentSession._reconcile_interruption_note(stub, _INTERRUPTED_OBSERVATION)
    assert len([e for e in timeline.timeline if (e.metadata or {}).get(_INTERRUPTION_NOTE_KEY)]) == 1

    # Retired once the observation clears.
    AgentSession._reconcile_interruption_note(stub, None)
    assert not [e for e in timeline.timeline if (e.metadata or {}).get(_INTERRUPTION_NOTE_KEY)]


def test_interruption_note_is_not_persisted_in_working_context():
    from dana.core.session.agent_session import AgentSession

    timeline = _native_timeline()
    stub = SimpleNamespace(
        _agent=_agent_with(timeline),
        _conversation_state=ConversationProjectionState(),
        _working_context=None,
        _working_format=None,
        _hydrated_tail_count=0,
        _needs_hydration=True,
    )
    timeline.add_entry = None
    AgentSession._reconcile_interruption_note(stub, _INTERRUPTED_OBSERVATION)
    AgentSession._capture_working_context(stub)

    assert stub._working_context is not None
    assert all(not (item.get("metadata") or {}).get(_INTERRUPTION_NOTE_KEY) for item in stub._working_context)


def test_reopened_legacy_note_is_replaced_not_duplicated():
    """A note restored without the marker cannot duplicate the derived one."""
    from dana.core.session.agent_session import AgentSession

    timeline = _native_timeline()
    timeline.add_entry = None
    stub = SimpleNamespace(_agent=_agent_with(timeline))

    # Reopen re-derives exactly one note per request, even when the retained
    # projection already carried one from the previous request.
    for _ in range(3):
        AgentSession._reconcile_interruption_note(stub, _INTERRUPTED_OBSERVATION)
        assert len([e for e in timeline.timeline if (e.metadata or {}).get(_INTERRUPTION_NOTE_KEY)]) == 1


@pytest.mark.asyncio
async def test_interruption_note_reaches_model_then_is_retired_on_next_turn(repo):
    """I2 end-to-end: the observation is visible on the recovery turn, then retired."""
    from dana.core.session.legacy_timeline_migration import recover_interrupted_turns

    scope = await _seed(repo, "sess-1")
    # An interrupted turn: TURN_STARTED with no terminal.
    await repo.append(
        scope,
        "sess-1",
        1,
        [
            NewJournalFact(
                fact_type=FactType.TURN_STARTED,
                correlation_id="interrupted-1",
                causation_id=None,
                payload={"prompt_summary": "was doing something"},
            )
        ],
    )
    assert await recover_interrupted_turns(repo, scope, "sess-1") == 1

    timeline = _native_timeline()
    seen: list[list[Any]] = []

    class CapturingAgent(ScriptedAgent):
        async def aquery_stream(self, *, message=None, **kwargs):
            seen.append([e for e in timeline.timeline if (e.metadata or {}).get(_INTERRUPTION_NOTE_KEY)])
            async for event in super().aquery_stream(message=message, **kwargs):
                yield event

    session = _make_session(repo, scope, CapturingAgent(chunks=["recovered"], timeline=timeline))
    events = await _collect(session.prompt([TextBlock(text="continue")]))

    # The recovery turn is the one that carries the note (exactly once), and the
    # observation is the truthful interrupted-turn text.
    assert seen, "the agent must have been invoked"
    assert len(seen[0]) == 1, f"exactly one note expected during the recovery turn: {seen[0]}"
    assert seen[0][0].content == _INTERRUPTED_OBSERVATION
    assert session.last_terminal is not None
    assert session.last_terminal.fact_type is FactType.TURN_COMPLETED
    assert [e.event_type.value for e in events][-1] == "turn_completed"

    # The completed recovery turn clears the observation, so the note is retired
    # and the derived working context no longer carries it.
    final_facts = await repo.read_facts(scope, "sess-1")
    state = ConversationProjectionState()
    state.apply(final_facts)
    assert state.interruption_observation is None
    assert [m.content for m in state.view().messages if m.role == "user"] == ["continue"]
    assert [e for e in timeline.timeline if (e.metadata or {}).get(_INTERRUPTION_NOTE_KEY)] == []
    assert session._working_context is not None
    assert all(not (item.get("metadata") or {}).get(_INTERRUPTION_NOTE_KEY) for item in session._working_context)

    # A second recovery pass finds nothing left to recover: no duplicate note.
    assert await recover_interrupted_turns(repo, scope, "sess-1") == 0


@pytest.mark.asyncio
async def test_recovery_note_expires_across_reopen_and_checkpoint(repo):
    """The note must not survive a real reopen through the derived checkpoint."""
    from dana.core.session.legacy_timeline_migration import recover_interrupted_turns

    scope = await _seed(repo, "sess-1")
    await repo.append(
        scope,
        "sess-1",
        1,
        [
            NewJournalFact(
                fact_type=FactType.TURN_STARTED,
                correlation_id="interrupted-1",
                causation_id=None,
                payload={"prompt_summary": "was doing something"},
            )
        ],
    )
    await recover_interrupted_turns(repo, scope, "sess-1")

    first_timeline = _native_timeline()
    first = _make_session(repo, scope, ScriptedAgent(chunks=["recovered"], timeline=first_timeline))
    await _collect(first.prompt([TextBlock(text="continue")]))
    assert (await repo.load_projection_checkpoint(scope, "sess-1", "conversation-working-v1")) is not None

    # Reopen: a brand-new session over the same journal, with a fresh timeline.
    reopened_timeline = _native_timeline()
    reopened = _make_session(repo, scope, ScriptedAgent(chunks=["again"], timeline=reopened_timeline))
    await reopened.load()

    state = reopened._conversation_state
    assert state is not None
    assert state.interruption_observation is None, "the recovery turn already cleared the observation"
    assert [e for e in reopened_timeline.timeline if (e.metadata or {}).get(_INTERRUPTION_NOTE_KEY)] == []
    assert all(not (item.get("metadata") or {}).get(_INTERRUPTION_NOTE_KEY) for item in (reopened._working_context or []))

    # A further turn still never injects a stale observation.
    await _collect(reopened.prompt([TextBlock(text="third")]))
    assert [e for e in reopened_timeline.timeline if (e.metadata or {}).get(_INTERRUPTION_NOTE_KEY)] == []


@pytest.mark.asyncio
async def test_note_visible_on_native_request_path_and_retired_after_recovery(repo):
    """The note must reach `to_llm_messages()` even after `set_context`, then go.

    Production uses a real ``CompressedTimeline```, where ``set_context`` (called
    from ``build_prompt``) replaces the CONTEXT *entry* while the native mirror
    keeps its own objects. The note is therefore asserted on the provider-facing
    message list - not only on the entries list - both while it is current and
    after the recovery turn has completed.
    """
    from dana.core.session.legacy_timeline_migration import recover_interrupted_turns
    from dana.core.timeline.compressed_timeline import CompressedTimeline

    scope = await _seed(repo, "sess-1")
    await repo.append(
        scope,
        "sess-1",
        1,
        [
            NewJournalFact(
                fact_type=FactType.TURN_STARTED,
                correlation_id="interrupted-1",
                causation_id=None,
                payload={"prompt_summary": "was doing something"},
            )
        ],
    )
    await recover_interrupted_turns(repo, scope, "sess-1")

    timeline = CompressedTimeline(max_context_tokens=20000, compression_enabled=False)
    seen: list[list[Any]] = []

    class TimelineAgent(ScriptedAgent):
        async def aquery_stream(self, *, message=None, **kwargs):
            # Exactly what build_prompt would hand the provider for this turn.
            timeline.set_context({"timestamp": "2026-10-19T00:00:00Z", "user": "synthetic"})
            seen.append(list(timeline.to_llm_messages()))
            async for event in super().aquery_stream(message=message, **kwargs):
                yield event

    agent = TimelineAgent(chunks=["recovered"], timeline=timeline)
    session = _make_session(repo, scope, agent)
    await _collect(session.prompt([TextBlock(text="continue")]))

    assert seen, "the agent must have been invoked"
    first_request = seen[0]
    notes = [m for m in first_request if m.role == "system" and _INTERRUPTED_OBSERVATION in str(m.content)]
    assert len(notes) == 1, f"the recovery turn's request must carry exactly one note: {[m.role for m in first_request]}"

    # The note is retired on the provider path too - no stale native mirror left.
    after = timeline.to_llm_messages()
    assert not [m for m in after if _INTERRUPTED_OBSERVATION in str(m.content)]

    # And it stays gone across a reopen that re-derives from the journal.
    reopened_timeline = CompressedTimeline(max_context_tokens=20000, compression_enabled=False)
    reopened = _make_session(repo, scope, ScriptedAgent(chunks=["again"], timeline=reopened_timeline))
    await reopened.load()
    assert not [m for m in reopened_timeline.to_llm_messages() if _INTERRUPTED_OBSERVATION in str(m.content)]
    assert not [e for e in reopened_timeline.timeline if (e.metadata or {}).get(_INTERRUPTION_NOTE_KEY)]


# ---------------------------------------------------------------------------
# I9: shared executor owner isolation for the durable hook seam
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_executor_resolves_hooks_per_invoking_agent():
    """Two agents on one executor must never resolve each other's hooks.

    The durable hooks close over one turn (owner scope, correlation id, journal
    target). On a shared executor a single global trio would let the last session
    to register overwrite the other's, so a tool dispatched for A would journal
    into B. Owner-scoped registration + per-call resolution keeps each turn's
    facts on the invoking agent, and an unknown agent falls back to the global
    trio (standalone callers).
    """
    from dana.core.tool.tool_executor import ToolExecutor

    executor = ToolExecutor()
    global_before = lambda call, reason: None  # noqa: E731
    global_after = lambda call, result: None  # noqa: E731
    executor.set_execution_hooks(before=global_before, after=global_after)

    agent_a, agent_b = SimpleNamespace(name="a"), SimpleNamespace(name="b")
    a_before, a_after, a_authorize = (lambda c, r: None), (lambda c, r: None), (lambda c, r: None)
    b_before, b_after, b_authorize = (lambda c, r: None), (lambda c, r: None), (lambda c, r: None)
    executor.set_execution_hooks(before=a_before, after=a_after, authorize=a_authorize, agent=agent_a)
    executor.set_execution_hooks(before=b_before, after=b_after, authorize=b_authorize, agent=agent_b)

    resolved_a = executor._hooks_for(agent_a)
    resolved_b = executor._hooks_for(agent_b)
    assert (resolved_a.before, resolved_a.after, resolved_a.authorize) == (a_before, a_after, a_authorize)
    assert (resolved_b.before, resolved_b.after, resolved_b.authorize) == (b_before, b_after, b_authorize)

    # An unknown agent keeps the legacy global trio.
    unknown = executor._hooks_for(SimpleNamespace(name="unknown"))
    assert unknown.before is global_before and unknown.after is global_after

    # Retiring one owner leaves the other and the global trio intact.
    executor.set_execution_hooks(agent=agent_a)
    assert executor._hooks_for(agent_a).before is global_before
    still_b = executor._hooks_for(agent_b)
    assert still_b.before is b_before and still_b.after is b_after
    executor.close()


@pytest.mark.asyncio
async def test_session_hooks_survive_another_sessions_turn_on_shared_executor(repo):
    """A concurrent session's turn must not clobber this session's live hooks.

    Two sessions share one executor (the shape the owner-aware effect catalog
    exists for). Session A parks mid-turn; session B runs a whole turn; A's hooks
    must still be installed afterwards - B's registration and its teardown must
    touch only B's own entry.
    """
    from dana.core.tool.tool_executor import ToolExecutor

    executor = ToolExecutor()
    scope_a, scope_b = _scope("a"), _scope("b")
    await _seed(repo, "sess-a", scope_a)
    await _seed(repo, "sess-b", scope_b)

    class RuntimeAgent(ScriptedAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._runtime = SimpleNamespace(_tool_executor=executor, _native_tools=[])

    gate, parked = asyncio.Event(), asyncio.Event()
    agent_b = RuntimeAgent(chunks=["b"])
    agent_a = GatedAgent(gate, parked, chunks=["a"])
    # The executor is resolved from the agent's runtime by the turn loop.
    agent_a._runtime = SimpleNamespace(_tool_executor=executor, _native_tools=[])
    session_a = _make_session(repo, scope_a, agent_a, session_id="sess-a")
    session_b = _make_session(repo, scope_b, agent_b, session_id="sess-b")

    async def drive_a() -> None:
        async for _ in session_a.prompt([TextBlock(text="hi")]):
            pass

    task_a = asyncio.create_task(drive_a())
    await parked.wait()
    assert executor._hooks_for(session_a._agent).after is not None, "A's live turn owns its hooks"

    # B runs and finishes a complete turn, installing and then retiring its own.
    await _collect(session_b.prompt([TextBlock(text="hi")]))
    assert executor._hooks_for(session_b._agent).after is None, "B retired only its own hooks"
    assert executor._hooks_for(session_a._agent).after is not None, "B's turn must not clobber A's live hooks"

    gate.set()
    await task_a
    executor.close()


# ---------------------------------------------------------------------------
# I9: owner-scoped effect catalog / MCP effect map
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_effect_catalog_is_resolved_per_invoking_agent(repo):
    """A shared executor must never schedule from another session's catalog."""
    from dana.core.tool.tool_executor import ToolExecutor

    scope_a = await _seed(repo, "sess-a", _scope("a"))
    scope_b = await _seed(repo, "sess-b", _scope("b"))

    shared_executor = ToolExecutor(tool_name_registry_getter=lambda: {})

    class RuntimeAgent(ScriptedAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._runtime = SimpleNamespace(_tool_executor=shared_executor, _native_tools=[])

    agent_a = RuntimeAgent(chunks=["a"])
    agent_b = RuntimeAgent(chunks=["b"])
    session_a = _make_session(repo, scope_a, agent_a, session_id="sess-a")
    session_b = _make_session(repo, scope_b, agent_b, session_id="sess-b")

    await session_a._prepare_agent()
    await session_b._prepare_agent()

    catalog_a = agent_a._session_effect_catalog
    catalog_b = agent_b._session_effect_catalog
    assert catalog_a is not None and catalog_b is not None
    assert catalog_a is not catalog_b, "each session must publish its own catalog"

    # The executor resolves the catalog from the *invoking* agent, so an
    # overlapping dispatch for B cannot be scheduled from A's metadata.
    assert shared_executor._effect_catalog(agent_a) is catalog_a
    assert shared_executor._effect_catalog(agent_b) is catalog_b
    assert shared_executor._effect_catalog(None) is None

    # Teardown clears the owner's scheduling artifacts, not another session's.
    await session_a.aclose()
    assert agent_a._session_effect_catalog is None
    assert agent_b._session_effect_catalog is catalog_b
    await session_b.aclose()
    assert agent_b._session_effect_catalog is None
    await shared_executor.aclose()


@pytest.mark.asyncio
async def test_mcp_effect_resolver_is_owner_aware(repo):
    """The MCP effect map follows the invoking agent, not the last writer."""
    from dana.core.policy.effects import Effect, EffectKind, EffectMetadata
    from dana.core.tool.tool_executor import ToolExecutor

    scope = await _seed(repo, "sess-a", _scope("a"))
    shared_executor = ToolExecutor(tool_name_registry_getter=lambda: {})

    read_effects = EffectMetadata(effects=(Effect(kind=EffectKind.READ),))

    class Wiring:
        def __init__(self) -> None:
            self.dispatch_map: dict[str, Any] = {}
            self.mcp_schemas: list[dict[str, Any]] = []
            self.mcp_names = frozenset()
            self.mcp_effects = {"srv:read": read_effects}

        async def close(self) -> None:
            return None

    wiring = Wiring()

    class RuntimeAgent(ScriptedAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._runtime = SimpleNamespace(_tool_executor=shared_executor, _native_tools=[])

    agent = RuntimeAgent(chunks=["a"])
    session = _make_session(repo, scope, agent, session_id="sess-a")
    session._mcp_wiring = wiring
    await session._prepare_agent()

    # The agent owns its map, and the executor consults it for that agent only.
    assert agent._session_mcp_effects == {"srv:read": read_effects}
    assert shared_executor._mcp_effect_resolver is not None
    assert shared_executor._mcp_effect_resolver(agent) == {"srv:read": read_effects}
    assert shared_executor._mcp_effect_resolver(None) is None

    await session.aclose()
    assert agent._session_mcp_effects is None
    await shared_executor.aclose()


# ---------------------------------------------------------------------------
# S6: unfinished-turn state is retired with the turn
# ---------------------------------------------------------------------------


def _fact(fact_type: FactType, seq: int, correlation_id: str = "t1", payload=None) -> JournalFact:
    return JournalFact(
        fact_id=f"f{seq}-{correlation_id}",
        owner_scope=_scope(),
        session_id="sess-1",
        sequence=seq,
        fact_type=fact_type,
        timestamp=datetime(2026, 7, 16, 12, 0, 0),
        correlation_id=correlation_id,
        causation_id=None,
        schema_version=1,
        payload=payload if payload is not None else {},
    )


def test_terminated_turn_retires_unfinished_bookkeeping():
    state = ConversationProjectionState()
    state.apply(
        [
            _fact(FactType.USER_CONTENT_FINAL, 1, payload={"text": "q"}),
            _fact(FactType.TOOL_REQUESTED, 2, payload={"tool_call_id": "x", "tool_name": "read_file", "arguments": {}}),
            _fact(FactType.ASSISTANT_CONTENT_FINAL, 3, payload={"text": "a"}),
            _fact(FactType.TURN_COMPLETED, 4),
        ]
    )
    assert state._pending_final == {}
    assert state._pending_native == {}
    assert state._turn_starts == {}
    assert state._tool_requests == {}, "a finished turn must not keep unresolved tool requests"
    assert state.open_correlation_ids == ()
    assert state.view().interruption_observation is None


def test_open_turn_keeps_its_unfinished_bookkeeping():
    state = ConversationProjectionState()
    state.apply(
        [
            _fact(FactType.USER_CONTENT_FINAL, 1, payload={"text": "q"}),
            _fact(FactType.TOOL_REQUESTED, 2, payload={"tool_call_id": "x", "tool_name": "read_file", "arguments": {}}),
        ]
    )
    assert state.open_correlation_ids == ("t1",)
    assert list(state._tool_requests) == [("t1", "x")], "an in-flight turn must keep its pending request"

    # A later interruption retires it: the partial output is never promoted, and
    # the observation (not a pending map) carries the visibility.
    state.apply([_fact(FactType.TURN_INTERRUPTED, 3)])
    assert state._tool_requests == {}
    assert state.open_correlation_ids == ()
    assert state.view().interruption_observation == _INTERRUPTED_OBSERVATION


def test_later_terminal_clears_observation_but_keeps_message_history():
    state = ConversationProjectionState()
    state.apply([_fact(FactType.USER_CONTENT_FINAL, 1, correlation_id="t1", payload={"text": "q"})])
    state.apply([_fact(FactType.TURN_INTERRUPTED, 2, correlation_id="t1")])
    assert state.interruption_observation == _INTERRUPTED_OBSERVATION

    state.apply(
        [
            _fact(FactType.USER_CONTENT_FINAL, 3, correlation_id="t2", payload={"text": "q2"}),
            _fact(FactType.ASSISTANT_CONTENT_FINAL, 4, correlation_id="t2", payload={"text": "a2"}),
            _fact(FactType.TURN_COMPLETED, 5, correlation_id="t2"),
        ]
    )
    view = state.view()
    assert view.interruption_observation is None
    assert [(m.role, m.content) for m in view.messages] == [("user", "q"), ("user", "q2"), ("assistant", "a2")]
