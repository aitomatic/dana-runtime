"""Integration tests for legacy Timeline migration and crash recovery.

Covers:
- Crash recovery: after input, during output, completed-turn, multiple turns.
- Migration of every TimelineEntryType (text converted, non-text skipped).
- Idempotency via content-addressed source fingerprint.
- Corrupt-entry handling (must not crash).
- Compact (TIMELINE_SUMMARY) session migration.
- Compatibility Timeline projection behind the rollback flag.
- Conversation View projection after migration.
"""

from __future__ import annotations

from datetime import UTC, datetime
import uuid

import pytest
import pytest_asyncio

from dana.core.session import FactType, JournalFact, NewJournalFact, OwnerScope
from dana.core.session.journal import SessionRecord, SQLiteJournalRepository
from dana.core.session.legacy_timeline_migration import (
    MigrationResult,
    journal_facts_to_timeline_entries,
    migrate_legacy_timeline,
    recover_interrupted_turns,
)
from dana.core.session.projections.conversation import ConversationProjector
from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType


# ---------------------------------------------------------------------------
# Shared scope + helpers
# ---------------------------------------------------------------------------

SCOPE = OwnerScope(owner_id="owner-1", workspace="ws-1")


def _new_fact(
    fact_type: FactType,
    correlation_id: str = "turn-1",
    payload: dict | None = None,
) -> NewJournalFact:
    return NewJournalFact(
        fact_type=fact_type,
        correlation_id=correlation_id,
        causation_id=None,
        payload=payload or {},
    )


def _to_journal_facts(session_id: str, scope: OwnerScope, facts: list[NewJournalFact]) -> list[JournalFact]:
    """Promote NewJournalFacts to durable JournalFacts at sequence 1..N for create_session."""
    out: list[JournalFact] = []
    now = datetime.now(UTC)
    for offset, nf in enumerate(facts):
        out.append(
            JournalFact(
                fact_id=str(uuid.uuid4()),
                owner_scope=scope,
                session_id=session_id,
                sequence=offset + 1,
                fact_type=nf.fact_type,
                timestamp=now,
                correlation_id=nf.correlation_id,
                causation_id=nf.causation_id,
                schema_version=nf.schema_version,
                payload=nf.payload,
            )
        )
    return out


async def _create_session(repo: SQLiteJournalRepository, session_id: str, facts: list[NewJournalFact]) -> None:
    record = SessionRecord.new(session_id=session_id, owner_scope=SCOPE)
    await repo.create_session(record, _to_journal_facts(session_id, SCOPE, facts))


async def _version(repo: SQLiteJournalRepository, session_id: str) -> int:
    record = await repo.load_session(SCOPE, session_id)
    return record.version


@pytest_asyncio.fixture
async def repo(tmp_path):
    r = await SQLiteJournalRepository.open(str(tmp_path / "test.db"))
    yield r
    await r.close()


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crash_after_input_recovery(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-crash-input"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init", {"reason": "init"})])
    v = await _version(repo, session_id)
    # Turn started + user content, NO terminal fact — crash after input.
    await repo.append(
        SCOPE,
        session_id,
        expected_version=v,
        facts=[
            _new_fact(FactType.TURN_STARTED, "turn-1", {}),
            _new_fact(FactType.USER_CONTENT_FINAL, "turn-1", {"text": "hello"}),
        ],
    )

    recovered = await recover_interrupted_turns(repo, SCOPE, session_id)
    assert recovered == 1

    facts = await repo.read_facts(SCOPE, session_id)
    interrupted = [f for f in facts if f.fact_type is FactType.TURN_INTERRUPTED]
    assert len(interrupted) == 1
    assert interrupted[0].correlation_id == "turn-1"

    # Recovery is idempotent: running again finds the turn already terminated.
    recovered_again = await recover_interrupted_turns(repo, SCOPE, session_id)
    assert recovered_again == 0


@pytest.mark.asyncio
async def test_crash_during_output_recovery(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-crash-output"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init", {"reason": "init"})])
    v = await _version(repo, session_id)
    # Turn started, user content, streamed chunks + final, NO terminal — crash during output.
    await repo.append(
        SCOPE,
        session_id,
        expected_version=v,
        facts=[
            _new_fact(FactType.TURN_STARTED, "turn-1", {}),
            _new_fact(FactType.USER_CONTENT_FINAL, "turn-1", {"text": "hello"}),
            _new_fact(FactType.ASSISTANT_CONTENT_CHUNK, "turn-1", {"text": "par", "index": 0}),
            _new_fact(FactType.ASSISTANT_CONTENT_FINAL, "turn-1", {"text": "partial response"}),
        ],
    )

    recovered = await recover_interrupted_turns(repo, SCOPE, session_id)
    assert recovered == 1

    facts = await repo.read_facts(SCOPE, session_id)
    assert any(f.fact_type is FactType.TURN_INTERRUPTED for f in facts)

    # ConversationProjector must EXCLUDE the partial assistant text from messages.
    view = ConversationProjector().project(facts)
    roles = [m.role for m in view.messages]
    assert roles == ["user"]
    assert all(m.content != "partial response" for m in view.messages)
    # The interruption observation must be surfaced to the next model turn.
    assert view.interruption_observation is not None


@pytest.mark.asyncio
async def test_completed_turn_not_recovered(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-complete"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init", {"reason": "init"})])
    v = await _version(repo, session_id)
    await repo.append(
        SCOPE,
        session_id,
        expected_version=v,
        facts=[
            _new_fact(FactType.TURN_STARTED, "turn-1", {}),
            _new_fact(FactType.USER_CONTENT_FINAL, "turn-1", {"text": "hi"}),
            _new_fact(FactType.ASSISTANT_CONTENT_FINAL, "turn-1", {"text": "hey"}),
            _new_fact(FactType.TURN_COMPLETED, "turn-1", {}),
        ],
    )
    before = await repo.read_facts(SCOPE, session_id)

    recovered = await recover_interrupted_turns(repo, SCOPE, session_id)
    assert recovered == 0

    after = await repo.read_facts(SCOPE, session_id)
    assert len(after) == len(before)
    assert not any(f.fact_type is FactType.TURN_INTERRUPTED for f in after)


@pytest.mark.asyncio
async def test_multiple_interrupted_turns(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-multi"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init", {"reason": "init"})])
    v = await _version(repo, session_id)
    await repo.append(
        SCOPE,
        session_id,
        expected_version=v,
        facts=[
            _new_fact(FactType.TURN_STARTED, "turn-1", {}),
            _new_fact(FactType.USER_CONTENT_FINAL, "turn-1", {"text": "one"}),
            _new_fact(FactType.TURN_STARTED, "turn-2", {}),
            _new_fact(FactType.USER_CONTENT_FINAL, "turn-2", {"text": "two"}),
        ],
    )

    recovered = await recover_interrupted_turns(repo, SCOPE, session_id)
    assert recovered == 2

    facts = await repo.read_facts(SCOPE, session_id)
    interrupted = sorted(f.correlation_id for f in facts if f.fact_type is FactType.TURN_INTERRUPTED)
    assert interrupted == ["turn-1", "turn-2"]


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_user_message(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-migrate-user"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init", {"reason": "init"})])
    entries = [TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="hello world")]

    result = await migrate_legacy_timeline(repo, SCOPE, session_id, entries)

    assert isinstance(result, MigrationResult)
    assert result.already_migrated is False
    # USER_CONTENT_FINAL + LEGACY_TIMELINE_MIGRATED marker at minimum.
    assert result.appended >= 2
    assert result.skipped_entries == 0

    facts = await repo.read_facts(SCOPE, session_id)
    user_facts = [f for f in facts if f.fact_type is FactType.USER_CONTENT_FINAL]
    assert len(user_facts) == 1
    assert user_facts[0].payload["text"] == "hello world"
    assert any(f.fact_type is FactType.LEGACY_TIMELINE_MIGRATED for f in facts)


@pytest.mark.asyncio
async def test_migration_agent_response(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-migrate-pair"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init", {"reason": "init"})])
    entries = [
        TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="q"),
        TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content="a"),
    ]

    result = await migrate_legacy_timeline(repo, SCOPE, session_id, entries)
    assert result.already_migrated is False
    assert result.skipped_entries == 0

    facts = await repo.read_facts(SCOPE, session_id)
    types = [f.fact_type for f in facts]
    assert FactType.USER_CONTENT_FINAL in types
    assert FactType.ASSISTANT_CONTENT_FINAL in types
    assert FactType.TURN_COMPLETED in types
    assert FactType.LEGACY_TIMELINE_MIGRATED in types

    # Conversation view pairs the user + assistant into committed messages.
    view = ConversationProjector().project(facts)
    assert [m.role for m in view.messages] == ["user", "assistant"]
    assert [m.content for m in view.messages] == ["q", "a"]


@pytest.mark.asyncio
async def test_migration_every_entry_type(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-migrate-all"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init", {"reason": "init"})])
    entries = [
        TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="u"),
        TimelineEntry(entry_type=TimelineEntryType.AGENT_THOUGHTS, content="thinking"),
        TimelineEntry(entry_type=TimelineEntryType.TOOL_CALL, content="some tool"),
        TimelineEntry(entry_type=TimelineEntryType.FAILED_TOOL_CALL, content="bad tool"),
        TimelineEntry(entry_type=TimelineEntryType.SUB_AGENT_RESPONSE, content="sub"),
        TimelineEntry(entry_type=TimelineEntryType.RESOURCE_RESULT, content="res"),
        TimelineEntry(entry_type=TimelineEntryType.WORKFLOW_RESULT, content="wf"),
        TimelineEntry(entry_type=TimelineEntryType.UNKNOWN_TOOL_CALL, content="unknown"),
        TimelineEntry(entry_type=TimelineEntryType.AGENT_LEARNING, content="learn"),
        TimelineEntry(entry_type=TimelineEntryType.TIMELINE_SUMMARY, content="summary"),
        TimelineEntry(entry_type=TimelineEntryType.CONTEXT, content="ctx"),
        TimelineEntry(entry_type=TimelineEntryType.TODO_LIST, content="todos"),
        TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content="a"),
    ]

    result = await migrate_legacy_timeline(repo, SCOPE, session_id, entries)
    assert result.already_migrated is False
    # 11 non-text entries skipped (everything except USER_MESSAGE and AGENT_RESPONSE).
    assert result.skipped_entries == 11

    facts = await repo.read_facts(SCOPE, session_id)
    assert len([f for f in facts if f.fact_type is FactType.USER_CONTENT_FINAL]) == 1
    assert len([f for f in facts if f.fact_type is FactType.ASSISTANT_CONTENT_FINAL]) == 1


@pytest.mark.asyncio
async def test_migration_idempotent(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-idem"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init", {"reason": "init"})])
    entries = [
        TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="q"),
        TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content="a"),
    ]

    first = await migrate_legacy_timeline(repo, SCOPE, session_id, entries)
    assert first.already_migrated is False
    assert first.appended > 0

    facts_after_first = await repo.read_facts(SCOPE, session_id)

    second = await migrate_legacy_timeline(repo, SCOPE, session_id, entries)
    assert second.already_migrated is True
    assert second.appended == 0
    assert second.source_hash == first.source_hash

    # No new facts appended on the second (idempotent) run.
    facts_after_second = await repo.read_facts(SCOPE, session_id)
    assert len(facts_after_second) == len(facts_after_first)


@pytest.mark.asyncio
async def test_migration_different_source_not_idempotent(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-diff"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init", {"reason": "init"})])
    entries_a = [TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="a")]
    entries_b = [TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="b")]

    first = await migrate_legacy_timeline(repo, SCOPE, session_id, entries_a)
    assert first.already_migrated is False

    second = await migrate_legacy_timeline(repo, SCOPE, session_id, entries_b)
    assert second.already_migrated is False
    assert second.source_hash != first.source_hash

    # Two distinct migration markers — one per source fingerprint.
    facts = await repo.read_facts(SCOPE, session_id)
    markers = [f for f in facts if f.fact_type is FactType.LEGACY_TIMELINE_MIGRATED]
    assert len(markers) == 2


@pytest.mark.asyncio
async def test_migration_corrupt_entry(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-corrupt"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init", {"reason": "init"})])
    entries = [
        TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content=None),  # malformed
        TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content=12345),  # unexpected type
    ]

    # Must not crash; content converted safely.
    result = await migrate_legacy_timeline(repo, SCOPE, session_id, entries)
    assert result.already_migrated is False
    assert result.skipped_entries == 0

    facts = await repo.read_facts(SCOPE, session_id)
    user_facts = [f for f in facts if f.fact_type is FactType.USER_CONTENT_FINAL]
    assert len(user_facts) == 1
    # None is safely converted to an empty string (not the literal "None").
    assert user_facts[0].payload["text"] == ""


@pytest.mark.asyncio
async def test_compact_session_migration(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-compact"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init", {"reason": "init"})])
    entries = [
        TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="old q"),
        TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content="old a"),
        TimelineEntry(entry_type=TimelineEntryType.TIMELINE_SUMMARY, content="[summary] old convo"),
        TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="new q"),
        TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content="new a"),
    ]

    result = await migrate_legacy_timeline(repo, SCOPE, session_id, entries)
    assert result.skipped_entries == 1  # only the TIMELINE_SUMMARY was skipped

    facts = await repo.read_facts(SCOPE, session_id)
    user_facts = [f for f in facts if f.fact_type is FactType.USER_CONTENT_FINAL]
    asst_facts = [f for f in facts if f.fact_type is FactType.ASSISTANT_CONTENT_FINAL]
    assert len(user_facts) == 2
    assert len(asst_facts) == 2
    # Text entries around the summary are preserved in order.
    assert [str(f.payload["text"]) for f in user_facts] == ["old q", "new q"]


# ---------------------------------------------------------------------------
# Compatibility projection + conversation view
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compatibility_timeline_projection(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-compat"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init", {"reason": "init"})])
    entries = [
        TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="q"),
        TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content="a"),
    ]
    await migrate_legacy_timeline(repo, SCOPE, session_id, entries)

    facts = await repo.read_facts(SCOPE, session_id)
    projected = journal_facts_to_timeline_entries(facts)

    # Only text entries project back; chunks/terminals/marker are skipped.
    assert [e.entry_type for e in projected] == [TimelineEntryType.USER_MESSAGE, TimelineEntryType.AGENT_RESPONSE]
    assert [e.content for e in projected] == ["q", "a"]


@pytest.mark.asyncio
async def test_conversation_view_after_migration(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-view"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init", {"reason": "init"})])
    entries = [
        TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="what is 2+2?"),
        TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content="4"),
        TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="thanks!"),
        TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content="you're welcome"),
    ]
    await migrate_legacy_timeline(repo, SCOPE, session_id, entries)

    facts = await repo.read_facts(SCOPE, session_id)
    view = ConversationProjector().project(facts)
    assert [m.role for m in view.messages] == ["user", "assistant", "user", "assistant"]
    assert [m.content for m in view.messages] == ["what is 2+2?", "4", "thanks!", "you're welcome"]
    assert view.interruption_observation is None
