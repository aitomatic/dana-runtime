"""
Parameterized contract tests for the Session Journal persistence layer.

Every test runs against BOTH backends:

* ``sqlite``  — a fresh on-disk SQLite database per test (under ``tmp_path``).
* ``postgres`` — a real PostgreSQL instance reachable via the
  ``DANA_TEST_POSTGRES_DSN`` environment variable. When the DSN is absent the
  postgres cases skip, UNLESS ``CI=true`` is set, in which case they fail —
  the real-database contract is exercised on the master-PR CI lane
  (see ``pr-lint-and-test.yml``); the develop/PR fast lane deselects the
  postgres parametrizations via ``-k "not postgres"``.

The two adapters are required to implement EQUIVALENT domain semantics; this
module is the single source of truth for that equivalence. Backend-specific
features (WAL, JSONB, row locks) must not leak through the public interface.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import os
import uuid

import pytest
import pytest_asyncio

from dana.core.session import FactType, JournalFact, JSONValue, NewJournalFact, OwnerScope
from dana.core.session.journal import (
    AppendResult,
    JournalConflict,
    JournalError,
    JournalRepository,
    ProjectionCheckpoint,
    SessionNotFound,
    SessionRecord,
    SessionStatus,
    SQLiteJournalRepository,
)
from dana.core.session.journal.postgres import PostgresJournalRepository


# ---------------------------------------------------------------------------
# Backend fixture — parameterized over sqlite + postgres
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def repository(request: pytest.FixtureRequest, tmp_path):
    """Yield a fresh JournalRepository for each backend, cleaned between tests."""
    if request.param == "sqlite":
        repo = await SQLiteJournalRepository.open(str(tmp_path / "test.db"))
    elif request.param == "postgres":
        dsn = os.environ.get("DANA_TEST_POSTGRES_DSN", "")
        if not dsn:
            if os.environ.get("CI") == "true":
                pytest.fail("CI=true but DANA_TEST_POSTGRES_DSN not set")
            pytest.skip("DANA_TEST_POSTGRES_DSN not set")
        repo = await PostgresJournalRepository.open(dsn)
        # Wipe all journal tables so each test starts from a known-empty state.
        await repo._db.execute("DELETE FROM projection_checkpoints")
        await repo._db.execute("DELETE FROM session_facts")
        await repo._db.execute("DELETE FROM session_journals")
    else:  # pragma: no cover - defensive
        pytest.fail(f"unknown backend {request.param}")

    yield repo

    await repo.close()


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def journal_factory(request: pytest.FixtureRequest, tmp_path):
    """Yield a factory opening fresh repos on a SHARED backend database.

    Each invocation returns a NEW repository instance pointing at the same
    underlying database, so callers can exercise true cross-instance
    concurrency (two connections racing on the same session).
    """
    repos: list = []

    if request.param == "sqlite":
        path = str(tmp_path / "test.db")

        async def factory() -> JournalRepository:
            repo = await SQLiteJournalRepository.open(path)
            repos.append(repo)
            return repo

    elif request.param == "postgres":
        dsn = os.environ.get("DANA_TEST_POSTGRES_DSN", "")
        if not dsn:
            if os.environ.get("CI") == "true":
                pytest.fail("CI=true but DANA_TEST_POSTGRES_DSN not set")
            pytest.skip("DANA_TEST_POSTGRES_DSN not set")
        # Wipe once on setup using a scratch connection.
        scratch = await PostgresJournalRepository.open(dsn)
        try:
            await scratch._db.execute("DELETE FROM projection_checkpoints")
            await scratch._db.execute("DELETE FROM session_facts")
            await scratch._db.execute("DELETE FROM session_journals")
        finally:
            await scratch.close()

        async def factory() -> JournalRepository:
            repo = await PostgresJournalRepository.open(dsn)
            repos.append(repo)
            return repo

    else:  # pragma: no cover - defensive
        pytest.fail(f"unknown backend {request.param}")

    yield factory

    for repo in repos:
        await repo.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scope(tag: str = "a") -> OwnerScope:
    return OwnerScope(owner_id=f"owner-{tag}", workspace=f"ws-{tag}")


def _new_fact(
    fact_type: FactType = FactType.USER_CONTENT_FINAL,
    payload: dict[str, JSONValue] | None = None,
) -> NewJournalFact:
    return NewJournalFact(
        fact_type=fact_type,
        correlation_id="corr-1",
        causation_id=None,
        payload=payload if payload is not None else {"role": "user", "content": "hello"},
    )


def _seed_record(session_id: str = "sess-1", scope: OwnerScope | None = None) -> tuple[SessionRecord, list[NewJournalFact]]:
    record = SessionRecord.new(session_id=session_id, owner_scope=scope or _scope())
    facts = [_new_fact(FactType.SESSION_CREATED, {"reason": "init"})]
    return record, facts


# ---------------------------------------------------------------------------
# Contract: create + load
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_and_load(repository: JournalRepository) -> None:
    record, facts = _seed_record()
    created = await repository.create_session(record, _to_journal_facts_for_create(record, facts))
    assert created.version == 1
    assert created.status == SessionStatus.ACTIVE

    loaded = await repository.load_session(record.owner_scope, record.session_id)
    assert loaded.session_id == record.session_id
    assert loaded.owner_scope == record.owner_scope
    assert loaded.version == 1
    assert loaded.status == SessionStatus.ACTIVE

    stored = await repository.read_facts(record.owner_scope, record.session_id)
    assert len(stored) == 1
    assert stored[0].fact_type == FactType.SESSION_CREATED
    assert stored[0].sequence == 1
    assert stored[0].owner_scope == record.owner_scope


# ---------------------------------------------------------------------------
# Contract: ordered batch append
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ordered_batch_append(repository: JournalRepository) -> None:
    record, seed = _seed_record()
    await repository.create_session(record, _to_journal_facts_for_create(record, seed))

    batch = [
        _new_fact(FactType.USER_CONTENT_FINAL, {"i": 1}),
        _new_fact(FactType.ASSISTANT_CONTENT_FINAL, {"i": 2}),
        _new_fact(FactType.TURN_COMPLETED, {"i": 3}),
    ]
    result = await repository.append(record.owner_scope, record.session_id, expected_version=1, facts=batch)
    assert isinstance(result, AppendResult)
    assert result.new_version == 4
    assert len(result.appended_facts) == 3

    sequences = [f.sequence for f in result.appended_facts]
    assert sequences == [2, 3, 4]

    facts = await repository.read_facts(record.owner_scope, record.session_id)
    assert [f.sequence for f in facts] == [1, 2, 3, 4]
    assert [f.fact_type for f in facts] == [
        FactType.SESSION_CREATED,
        FactType.USER_CONTENT_FINAL,
        FactType.ASSISTANT_CONTENT_FINAL,
        FactType.TURN_COMPLETED,
    ]


# ---------------------------------------------------------------------------
# Contract: version conflict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_version_conflict(repository: JournalRepository) -> None:
    record, seed = _seed_record()
    await repository.create_session(record, _to_journal_facts_for_create(record, seed))

    # Correct version is 1; pass a stale 99 to force a conflict.
    with pytest.raises(JournalConflict) as exc_info:
        await repository.append(
            record.owner_scope,
            record.session_id,
            expected_version=99,
            facts=[_new_fact()],
        )
    conflict = exc_info.value
    assert conflict.session_id == record.session_id
    assert conflict.expected_version == 99
    assert conflict.actual_version == 1


# ---------------------------------------------------------------------------
# Contract: atomic metadata update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_atomic_metadata(repository: JournalRepository) -> None:
    record, seed = _seed_record()
    await repository.create_session(record, _to_journal_facts_for_create(record, seed))

    metadata = {"title": "first turn", "tags": ["intro", "demo"]}
    result = await repository.append(
        record.owner_scope,
        record.session_id,
        expected_version=1,
        facts=[_new_fact(FactType.USER_CONTENT_FINAL, {"content": "hi"})],
        metadata=metadata,
    )
    assert result.new_version == 2

    loaded = await repository.load_session(record.owner_scope, record.session_id)
    assert dict(loaded.metadata) == metadata
    # Facts advanced atomically with the metadata change.
    facts = await repository.read_facts(record.owner_scope, record.session_id)
    assert len(facts) == 2


# ---------------------------------------------------------------------------
# Contract: read_after (after_sequence)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_after(repository: JournalRepository) -> None:
    record, seed = _seed_record()
    await repository.create_session(record, _to_journal_facts_for_create(record, seed))
    await repository.append(
        record.owner_scope,
        record.session_id,
        expected_version=1,
        facts=[_new_fact(), _new_fact(), _new_fact()],
    )
    # sequences are now 1..4; reading after 2 returns only 3,4.
    tail = await repository.read_facts(record.owner_scope, record.session_id, after_sequence=2)
    assert [f.sequence for f in tail] == [3, 4]


# ---------------------------------------------------------------------------
# Contract: projection checkpoints do not affect journal facts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_projection_checkpoint(repository: JournalRepository) -> None:
    record, seed = _seed_record()
    await repository.create_session(record, _to_journal_facts_for_create(record, seed))

    checkpoint = ProjectionCheckpoint(
        projection_name="timeline",
        last_sequence=1,
        data={"cursor": "abc"},
    )
    await repository.save_projection_checkpoint(record.owner_scope, record.session_id, checkpoint)

    loaded = await repository.load_projection_checkpoint(record.owner_scope, record.session_id, "timeline")
    assert loaded is not None
    assert loaded.projection_name == "timeline"
    assert loaded.last_sequence == 1
    assert dict(loaded.data) == {"cursor": "abc"}

    # Unknown projection returns None.
    missing = await repository.load_projection_checkpoint(record.owner_scope, record.session_id, "nope")
    assert missing is None

    # Journal facts are untouched.
    facts = await repository.read_facts(record.owner_scope, record.session_id)
    assert len(facts) == 1


# ---------------------------------------------------------------------------
# Contract: OwnerScope isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_owner_isolation(repository: JournalRepository) -> None:
    scope_a = _scope("a")
    scope_b = _scope("b")
    record, seed = _seed_record(scope=scope_a)
    await repository.create_session(record, _to_journal_facts_for_create(record, seed))

    # Same session_id, different owner scope — must not be visible.
    with pytest.raises(SessionNotFound):
        await repository.load_session(scope_b, record.session_id)
    with pytest.raises(SessionNotFound):
        await repository.read_facts(scope_b, record.session_id)
    with pytest.raises(SessionNotFound):
        await repository.append(scope_b, record.session_id, expected_version=1, facts=[_new_fact()])

    # list_sessions under scope B does not see scope A's session.
    sessions_b = await repository.list_sessions(scope_b)
    assert sessions_b == []


# ---------------------------------------------------------------------------
# Contract: archive + purge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_archive_and_purge(repository: JournalRepository) -> None:
    record, seed = _seed_record()
    await repository.create_session(record, _to_journal_facts_for_create(record, seed))

    archived = await repository.archive_session(record.owner_scope, record.session_id)
    assert archived.status == SessionStatus.ARCHIVED

    # Archived sessions remain loadable; list_sessions excludes only DELETED.
    loaded = await repository.load_session(record.owner_scope, record.session_id)
    assert loaded.status == SessionStatus.ARCHIVED
    listed = await repository.list_sessions(record.owner_scope)
    assert record.session_id in [s.session_id for s in listed]

    await repository.purge_session(record.owner_scope, record.session_id)
    with pytest.raises(SessionNotFound):
        await repository.load_session(record.owner_scope, record.session_id)
    with pytest.raises(SessionNotFound):
        await repository.read_facts(record.owner_scope, record.session_id)
    # After purge, the session no longer appears in list_sessions.
    listed_after = await repository.list_sessions(record.owner_scope)
    assert record.session_id not in [s.session_id for s in listed_after]


# ---------------------------------------------------------------------------
# Contract: stale version on append must conflict (sequential)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_version_rejected(repository: JournalRepository) -> None:
    record, seed = _seed_record()
    await repository.create_session(record, _to_journal_facts_for_create(record, seed))

    # Writer 1 commits first, advancing version 1 -> 2.
    await repository.append(
        record.owner_scope,
        record.session_id,
        expected_version=1,
        facts=[_new_fact(FactType.USER_CONTENT_FINAL, {"who": "w1"})],
    )

    # Writer 2 still holds the stale expected_version=1 and must be rejected.
    with pytest.raises(JournalConflict) as exc_info:
        await repository.append(
            record.owner_scope,
            record.session_id,
            expected_version=1,
            facts=[_new_fact(FactType.USER_CONTENT_FINAL, {"who": "w2"})],
        )
    assert exc_info.value.actual_version == 2
    assert exc_info.value.expected_version == 1


# ---------------------------------------------------------------------------
# Contract: two concurrent writers on separate connections — exactly one wins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_writers_one_wins(journal_factory) -> None:
    # Two independent repository instances pointing at the SAME database.
    repo_a = await journal_factory()
    repo_b = await journal_factory()
    scope = _scope()
    record, seed = _seed_record(scope=scope)
    await repo_a.create_session(record, _to_journal_facts_for_create(record, seed))

    # Both writers race with the SAME expected_version=1. Exactly one must
    # succeed; the other must raise JournalConflict. No other outcome is valid.
    async def writer(repo: JournalRepository, who: str):
        return await repo.append(
            scope,
            record.session_id,
            expected_version=1,
            facts=[_new_fact(FactType.USER_CONTENT_FINAL, {"who": who})],
        )

    results = await asyncio.gather(
        writer(repo_a, "w1"),
        writer(repo_b, "w2"),
        return_exceptions=True,
    )

    successes = [r for r in results if not isinstance(r, BaseException)]
    conflicts = [r for r in results if isinstance(r, JournalConflict)]
    other_failures = [r for r in results if isinstance(r, BaseException) and not isinstance(r, JournalConflict)]

    assert len(successes) == 1, f"expected exactly one success, got {successes!r}; results={results!r}"
    assert len(conflicts) == 1, f"expected exactly one JournalConflict, got {conflicts!r}; results={results!r}"
    assert not other_failures, f"unexpected non-conflict failures: {other_failures!r}"

    # The one success advanced the version to 2.
    assert successes[0].new_version == 2

    # The durable state reflects exactly one winner (version 2, two facts).
    durable = await repo_a.load_session(scope, record.session_id)
    assert durable.version == 2
    facts = await repo_a.read_facts(scope, record.session_id)
    assert len(facts) == 2


# ---------------------------------------------------------------------------
# Contract: edge cases — empty batch, missing sessions, duplicate create
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_empty_batch_raises(repository: JournalRepository) -> None:
    record, seed = _seed_record()
    await repository.create_session(record, _to_journal_facts_for_create(record, seed))

    with pytest.raises(ValueError):
        await repository.append(
            record.owner_scope,
            record.session_id,
            expected_version=1,
            facts=[],
        )

    # Version is untouched by the rejected empty append.
    loaded = await repository.load_session(record.owner_scope, record.session_id)
    assert loaded.version == 1


@pytest.mark.asyncio
async def test_append_missing_session_raises(repository: JournalRepository) -> None:
    scope = _scope()
    with pytest.raises(SessionNotFound):
        await repository.append(scope, "nope", expected_version=0, facts=[_new_fact()])


@pytest.mark.asyncio
async def test_read_facts_missing_session_raises(repository: JournalRepository) -> None:
    scope = _scope()
    with pytest.raises(SessionNotFound):
        await repository.read_facts(scope, "nope")


@pytest.mark.asyncio
async def test_duplicate_create_raises(repository: JournalRepository) -> None:
    record, seed = _seed_record()
    await repository.create_session(record, _to_journal_facts_for_create(record, seed))

    with pytest.raises(JournalError):
        await repository.create_session(record, _to_journal_facts_for_create(record, seed))


@pytest.mark.asyncio
async def test_archive_missing_session_raises(repository: JournalRepository) -> None:
    scope = _scope()
    with pytest.raises(SessionNotFound):
        await repository.archive_session(scope, "nope")


@pytest.mark.asyncio
async def test_purge_missing_session_raises(repository: JournalRepository) -> None:
    scope = _scope()
    with pytest.raises(SessionNotFound):
        await repository.purge_session(scope, "nope")


# ---------------------------------------------------------------------------
# Internal helper — create_session expects durable JournalFacts with identity
# ---------------------------------------------------------------------------


def _to_journal_facts_for_create(record: SessionRecord, facts: list[NewJournalFact]) -> list[JournalFact]:
    """Promote NewJournalFacts to durable JournalFacts at sequence 1..N for create_session."""
    out: list[JournalFact] = []
    for offset, nf in enumerate(facts):
        out.append(
            JournalFact(
                fact_id=str(uuid.uuid4()),
                owner_scope=record.owner_scope,
                session_id=record.session_id,
                sequence=offset + 1,
                fact_type=nf.fact_type,
                timestamp=datetime.now(UTC),
                correlation_id=nf.correlation_id,
                causation_id=nf.causation_id,
                schema_version=nf.schema_version,
                payload=nf.payload,
                protected_payload=nf.protected_payload,
            )
        )
    return out
