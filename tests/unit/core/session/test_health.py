"""Unit tests for the Session Journal health check.

Covers:
  1. Healthy empty journal — all checks pass with zero counts.
  2. Sessions counted correctly by status.
  3. Database error short-circuits the report with ``database_connectivity=False``.
  4. Interrupted turns are detected (read-only; no mutation of the journal).
  5. The report carries no owner_id, workspace, session_id, or payload text —
     the redaction contract.
  6. Projection lag and legacy migration markers are reported.
"""

from __future__ import annotations

from datetime import UTC, datetime
import uuid

import pytest
import pytest_asyncio

from dana.core.session.health import JournalHealthReport, check_journal_health
from dana.core.session.journal.models import ProjectionCheckpoint, SessionRecord
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.models import FactType, JournalFact, NewJournalFact, OwnerScope


# ---------------------------------------------------------------------------
# Shared scope + helpers
# ---------------------------------------------------------------------------

OWNER_ID = "owner-health"
WORKSPACE = "ws-health"
SCOPE = OwnerScope(owner_id=OWNER_ID, workspace=WORKSPACE)
SECRET_NEEDLE = OWNER_ID  # reused for redaction sweep


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
    r = await SQLiteJournalRepository.open(str(tmp_path / "health.db"))
    yield r
    await r.close()


# ---------------------------------------------------------------------------
# 1. Healthy empty journal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_healthy_journal_empty(repo: SQLiteJournalRepository) -> None:
    report = await check_journal_health(repo, SCOPE)

    assert isinstance(report, JournalHealthReport)
    assert report.database_connectivity is True
    assert report.total_sessions == 0
    assert report.active_sessions == 0
    assert report.archived_sessions == 0
    assert report.interrupted_turns_detected == 0
    assert report.projection_lag_max == 0
    assert report.legacy_migration_markers == 0
    assert report.errors == []


# ---------------------------------------------------------------------------
# 2. Sessions counted by status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sessions_counted_by_status(repo: SQLiteJournalRepository) -> None:
    # Two ACTIVE sessions, one ARCHIVED session.
    await _create_session(repo, "sess-a", [_new_fact(FactType.SESSION_CREATED, "init-a")])
    await _create_session(repo, "sess-b", [_new_fact(FactType.SESSION_CREATED, "init-b")])
    await _create_session(repo, "sess-c", [_new_fact(FactType.SESSION_CREATED, "init-c")])
    await repo.archive_session(SCOPE, "sess-c")

    report = await check_journal_health(repo, SCOPE)

    assert report.database_connectivity is True
    assert report.total_sessions == 3
    assert report.active_sessions == 2
    assert report.archived_sessions == 1


# ---------------------------------------------------------------------------
# 3. Database connectivity failure
# ---------------------------------------------------------------------------


class _BoomRepository:
    """Minimal stand-in that raises on every operation; used to exercise the
    connectivity-shortcut path."""

    async def list_sessions(self, scope: OwnerScope) -> list[SessionRecord]:
        raise RuntimeError("simulated connectivity failure")


@pytest.mark.asyncio
async def test_database_error_handled() -> None:
    report = await check_journal_health(_BoomRepository(), SCOPE)  # type: ignore[arg-type]

    assert report.database_connectivity is False
    assert report.total_sessions == 0
    assert report.active_sessions == 0
    assert report.archived_sessions == 0
    assert report.interrupted_turns_detected == 0
    assert report.projection_lag_max == 0
    assert report.legacy_migration_markers == 0
    assert len(report.errors) == 1
    assert "database_connectivity" in report.errors[0]
    assert "RuntimeError" in report.errors[0]


# ---------------------------------------------------------------------------
# 4. Interrupted turn detection (read-only — no journal mutation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupted_turn_detected(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-interrupted"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init")])
    v = await _version(repo, session_id)
    # Turn started + user content, NO terminal fact — interrupted.
    await repo.append(
        SCOPE,
        session_id,
        expected_version=v,
        facts=[
            _new_fact(FactType.TURN_STARTED, "turn-1", {}),
            _new_fact(FactType.USER_CONTENT_FINAL, "turn-1", {"text": "hello"}),
        ],
    )
    facts_before = await repo.read_facts(SCOPE, session_id)

    report = await check_journal_health(repo, SCOPE)

    assert report.interrupted_turns_detected == 1

    # Read-only contract: the journal is unchanged by the health check.
    facts_after = await repo.read_facts(SCOPE, session_id)
    assert [f.fact_id for f in facts_after] == [f.fact_id for f in facts_before]
    assert not any(f.fact_type is FactType.TURN_INTERRUPTED for f in facts_after)


@pytest.mark.asyncio
async def test_completed_turn_not_interrupted(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-complete"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init")])
    v = await _version(repo, session_id)
    await repo.append(
        SCOPE,
        session_id,
        expected_version=v,
        facts=[
            _new_fact(FactType.TURN_STARTED, "turn-1", {}),
            _new_fact(FactType.USER_CONTENT_FINAL, "turn-1", {"text": "hi"}),
            _new_fact(FactType.TURN_COMPLETED, "turn-1", {}),
        ],
    )

    report = await check_journal_health(repo, SCOPE)
    assert report.interrupted_turns_detected == 0


# ---------------------------------------------------------------------------
# 5. Redaction — no owner/session identifiers or payload content leak
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_redacts_identifiers_and_payloads(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-leaky-123"
    distinctive_payload = "SUPER-SECRET-PAYLOAD-TEXT"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init")])
    v = await _version(repo, session_id)
    await repo.append(
        SCOPE,
        session_id,
        expected_version=v,
        facts=[
            _new_fact(FactType.TURN_STARTED, "turn-1", {}),
            _new_fact(FactType.USER_CONTENT_FINAL, "turn-1", {"text": distinctive_payload}),
        ],
    )

    report = await check_journal_health(repo, SCOPE)

    blob = repr(report) + "".join(report.errors)
    for needle in (OWNER_ID, WORKSPACE, session_id, "sess-leaky", distinctive_payload, "SUPER-SECRET"):
        assert needle not in blob, f"redaction violation: {needle!r} appears in report"


# ---------------------------------------------------------------------------
# 6. Projection lag + legacy migration markers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_projection_lag_reported(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-lag"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init")])
    v = await _version(repo, session_id)
    # Append two more facts so version advances to 3, then checkpoint at 1.
    await repo.append(
        SCOPE,
        session_id,
        expected_version=v,
        facts=[
            _new_fact(FactType.TURN_STARTED, "turn-1", {}),
            _new_fact(FactType.USER_CONTENT_FINAL, "turn-1", {"text": "hi"}),
        ],
    )
    await repo.save_projection_checkpoint(
        SCOPE,
        session_id,
        ProjectionCheckpoint(projection_name="conversation", last_sequence=1, data={}),
    )

    report = await check_journal_health(repo, SCOPE)
    # session_journals.version == 3, conversation checkpoint at 1 -> lag 2.
    assert report.projection_lag_max == 2


@pytest.mark.asyncio
async def test_legacy_migration_markers_counted(repo: SQLiteJournalRepository) -> None:
    session_id = "sess-migrated"
    await _create_session(repo, session_id, [_new_fact(FactType.SESSION_CREATED, "init")])
    v = await _version(repo, session_id)
    await repo.append(
        SCOPE,
        session_id,
        expected_version=v,
        facts=[
            _new_fact(FactType.USER_CONTENT_FINAL, "legacy-turn-1", {"text": "q"}),
            _new_fact(FactType.LEGACY_TIMELINE_MIGRATED, "migration-abc12345", {"source_hash": "abc"}),
        ],
    )

    report = await check_journal_health(repo, SCOPE)
    assert report.legacy_migration_markers == 1
