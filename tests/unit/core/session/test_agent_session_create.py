"""
D8 AgentSession ergonomics — flat imports, create() convenience, idempotent
resume, hello-world flow, and timeline/journal identity reconciliation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dana.core.runtime.protocols import StreamEvent, StreamEventType
import dana.core.session as session_pkg
from dana.core.session import AgentSession, FactType, TextBlock
from dana.core.session.projections.host_events import HostEventType


EXAMPLE_PATH = Path(__file__).parents[4] / "docs" / "examples" / "host_hello.py"


class FakeAgent:
    """Minimal mock agent — yields canned StreamEvents per prompt (D8 pattern)."""

    def __init__(self, chunks=("Hello from mock dana!",)):
        self._chunks = list(chunks)

    async def aquery_stream(self, *, message=None, **kwargs):
        for chunk in self._chunks:
            yield StreamEvent(event_type=StreamEventType.TEXT_DELTA, data=chunk, iteration=0)
        yield StreamEvent(event_type=StreamEventType.DONE, data=None, iteration=0)


class RecordingAgent(FakeAgent):
    """FakeAgent that tracks set_session_id calls (identity reconciliation)."""

    def __init__(self):
        super().__init__()
        self._session_id = "agent-internal-uuid"
        self.relabels: list[tuple[str, bool]] = []

    def set_session_id(self, session_id: str, reload_timeline: bool = False) -> None:
        self.relabels.append((session_id, reload_timeline))
        self._session_id = session_id


async def _drain(session: AgentSession) -> list:
    events = []
    async for event in session.prompt([TextBlock(text="hello")]):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# AC 1: flat imports — the five D8 names live on dana.core.session alone
# ---------------------------------------------------------------------------


def test_flat_imports():
    for name in ("AgentSession", "TextBlock", "FactType", "JournalFact", "SessionRecord"):
        assert hasattr(session_pkg, name), f"dana.core.session.{name} missing"
        assert name in session_pkg.__all__


def test_deep_import_paths_still_work():
    # Backward compat: ACP / dana-code deep imports unchanged.
    from dana.core.session.agent_session import AgentSession as A1
    from dana.core.session.journal.models import SessionRecord as S1
    from dana.core.session.models import FactType as F1
    from dana.core.session.models import JournalFact as J1

    assert A1 is AgentSession
    assert S1 is session_pkg.SessionRecord
    assert F1 is FactType
    assert J1 is session_pkg.JournalFact


# ---------------------------------------------------------------------------
# AC 2: create() with defaults + explicit id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_explicit_id(tmp_path):
    session = await AgentSession.create(
        session_id="eval-1",
        journal_path=str(tmp_path / "journal.db"),
        agent_factory=FakeAgent,
    )
    assert session.session_id == "eval-1"
    assert session.version == 1  # SESSION_CREATED appended
    facts = await session._repository.read_facts(session.owner_scope, "eval-1")
    assert [f.fact_type for f in facts] == [FactType.SESSION_CREATED]
    await session._repository.close()


@pytest.mark.asyncio
async def test_create_defaults(tmp_path):
    session = await AgentSession.create(journal_path=str(tmp_path / "journal.db"), agent_factory=FakeAgent)
    assert session.session_id  # generated uuid
    assert session.owner_scope.owner_id  # USER/local + cwd
    assert session.version == 1
    await session._repository.close()


# ---------------------------------------------------------------------------
# AC 3: double create with the same id resumes (no duplicate SESSION_CREATED)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_is_idempotent_on_existing_session(tmp_path):
    path = str(tmp_path / "journal.db")
    first = await AgentSession.create(session_id="same", journal_path=path, agent_factory=FakeAgent)
    await _drain(first)

    second = await AgentSession.create(session_id="same", journal_path=path, agent_factory=FakeAgent)
    facts = await second._repository.read_facts(second.owner_scope, "same")
    created = [f for f in facts if f.fact_type is FactType.SESSION_CREATED]
    assert len(created) == 1, "resume must not duplicate SESSION_CREATED"
    # Resumed session continues from the journal version (turn appended facts).
    assert second.version == first.version
    terminal = second.last_terminal
    assert terminal is None  # fresh AgentSession; the turn is replayable though
    events = [e.event_type async for e in second.replay_host_events(0)]
    assert HostEventType.TURN_COMPLETED in events
    await first._repository.close()
    await second._repository.close()


# ---------------------------------------------------------------------------
# AC 4: hello-world flow — create -> prompt -> consume HostEvent stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hello_world_flow(tmp_path):
    session = await AgentSession.create(session_id="hello", journal_path=str(tmp_path / "journal.db"), agent_factory=FakeAgent)
    types = [e.event_type for e in await _drain(session)]
    assert types[0] is HostEventType.TURN_STARTED
    assert HostEventType.USER_MESSAGE in types
    assert HostEventType.ASSISTANT_CONTENT_FINAL in types
    assert types[-1] is HostEventType.TURN_COMPLETED
    assert session.last_terminal is not None
    assert session.last_terminal.text == "Hello from mock dana!"
    await session._repository.close()


# ---------------------------------------------------------------------------
# AC 5: identity reconciliation — timeline persistence keys the journal id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_session_id_reconciled_with_journal(tmp_path):
    session = await AgentSession.create(session_id="eval-1", journal_path=str(tmp_path / "journal.db"), agent_factory=RecordingAgent)
    await _drain(session)

    agent = session._agent
    # Pure relabel (no timeline reload): the journal is the authority.
    assert agent.relabels == [("eval-1", False)]
    # Both identity systems now carry the same id: journal facts and the
    # timeline-save key (agent._session_id).
    assert agent._session_id == session.session_id == "eval-1"
    facts = await session._repository.read_facts(session.owner_scope, "eval-1")
    assert all(f.session_id == "eval-1" for f in facts)
    await session._repository.close()


def test_hello_world_example_is_compact():
    lines = [ln for ln in EXAMPLE_PATH.read_text().splitlines() if ln.strip()]
    assert len(lines) <= 20  # host logic (create -> prompt -> print) is 3 lines; rest is mock + boilerplate


def test_hello_world_example_runs_mock(monkeypatch, tmp_path, capsys):
    import runpy

    monkeypatch.setenv("DANA_MOCK_LLM", "1")
    monkeypatch.setenv("DANA_CODE_JOURNAL", str(tmp_path / "journal.db"))
    runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
    assert "Hello from mock dana!" in capsys.readouterr().out
