"""Regression tests: cancelling a caller must never abandon a write unit.

Finding N1 for the PR #32 session review.

``SQLiteJournalRepository`` runs every write as one ``BEGIN IMMEDIATE`` ..
``COMMIT`` unit on a single shared ``aiosqlite`` connection. Before this fix
the transaction began with an unprotected ``await self._db.execute("BEGIN
IMMEDIATE")``:

* cancelling the caller while that ``BEGIN`` was *queued* on the aiosqlite
  worker thread abandoned the statement - it still executed, so the shared
  connection was handed back sitting inside an open transaction and the next
  writer died with ``cannot start a transaction within a transaction``;
* cancelling at ``COMMIT`` produced ``cannot rollback - no transaction is
  active`` from the unconditional ``ROLLBACK``, which replaced the caller's
  ``CancelledError`` with an ``OperationalError``.

The repair turns the whole unit into a shielded inner task that owns the
connection lock through the physical ``BEGIN``, the body and the ``COMMIT``
(or a conditional ``ROLLBACK``), waits for it to settle, and only then re-raises
the caller's cancellation.

These tests drive the REAL aiosqlite worker thread: the connection is parked
inside a genuine statement (either a queued ``BEGIN``, or - via a sqlite3 trace
hook - immediately before a body statement / the ``COMMIT`` itself), so the
cancellation provably lands at that stage rather than at a plausible-looking
suspension point in repository code.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from datetime import UTC, datetime
import threading
from typing import Any
import uuid

import pytest
import pytest_asyncio

from dana.core.session.journal.models import JournalConflict, ProjectionCheckpoint, SessionNotFound, SessionRecord
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.models import FactType, JournalFact, NewJournalFact, OwnerScope


SCOPE = OwnerScope(owner_id="owner-n1", workspace="ws-n1")

# Bounded so a regression that hangs is reported as a failure, not a hung suite.
_SETTLE_BUDGET = 20.0


def _seed_fact(session_id: str, sequence: int = 1) -> JournalFact:
    return JournalFact(
        fact_id=str(uuid.uuid4()),
        owner_scope=SCOPE,
        session_id=session_id,
        sequence=sequence,
        fact_type=FactType.SESSION_CREATED,
        timestamp=datetime.now(UTC),
        correlation_id=str(uuid.uuid4()),
        causation_id=None,
        schema_version=1,
        payload={},
    )


def _batch(tag: str, count: int = 1) -> list[NewJournalFact]:
    return [
        NewJournalFact(
            fact_type=FactType.USER_CONTENT_FINAL,
            correlation_id=f"{tag}-{index}",
            causation_id=None,
            payload={"text": tag},
        )
        for index in range(count)
    ]


@pytest_asyncio.fixture
async def repo(tmp_path) -> AsyncIterator[SQLiteJournalRepository]:
    r = await SQLiteJournalRepository.open(str(tmp_path / "journal.db"))
    try:
        yield r
    finally:
        await asyncio.wait_for(r.close(), _SETTLE_BUDGET)


async def _seed(repo: SQLiteJournalRepository, session_id: str) -> None:
    await asyncio.wait_for(
        repo.create_session(SessionRecord.new(session_id, SCOPE), [_seed_fact(session_id)]),
        _SETTLE_BUDGET,
    )


async def _state(repo: SQLiteJournalRepository, session_id: str) -> tuple[int, list[int], list[FactType]]:
    """The durable outcome of ``session_id``: version, sequences, fact types."""
    record = await asyncio.wait_for(repo.load_session(SCOPE, session_id), _SETTLE_BUDGET)
    facts = await asyncio.wait_for(repo.read_facts(SCOPE, session_id), _SETTLE_BUDGET)
    return record.version, [f.sequence for f in facts], [f.fact_type for f in facts]


async def _settle(task: asyncio.Task[Any]) -> str:
    """Wait for a cancelled task to settle and report how it ended."""
    try:
        await asyncio.wait_for(asyncio.shield(task), _SETTLE_BUDGET)
        return "returned"
    except asyncio.CancelledError:
        return "cancelled"
    except TimeoutError:  # pragma: no cover - a regression symptom, not a code path
        return "timed-out"
    except BaseException as exc:  # noqa: BLE001 - the failure mode under test
        return f"{type(exc).__name__}: {exc}"


class _StatementWatch:
    """``repo._db`` stand-in that flags when a targeted statement reaches the queue.

    ``SQLiteJournalRepository`` reads ``self._db`` on every statement, so
    substituting this proxy lets a test watch the real call stream without
    touching repository code.
    """

    def __init__(self, db: Any, needle: str, seen: asyncio.Event) -> None:
        self._db = db
        self._needle = needle
        self._seen = seen

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)

    async def execute(self, sql: str, params: tuple[object, ...] = ()) -> Any:
        if sql.strip().upper().startswith(self._needle):
            self._seen.set()
        return await self._db.execute(sql, params)

    async def commit(self) -> None:
        # ``COMMIT`` is issued through ``Connection.commit``, not ``execute``.
        self._seen.set()
        await self._db.commit()


async def _hold_worker(db: Any) -> tuple[asyncio.Task[Any], threading.Event]:
    """Block the aiosqlite worker thread inside one real statement.

    Returns ``(blocker, release)``. While the handler runs, every statement the
    repository queues is provably still QUEUED (not yet executed): that is what
    makes "cancel while BEGIN is queued" deterministic instead of racy.
    """
    blocked = threading.Event()
    release = threading.Event()

    def handler() -> None:
        blocked.set()
        release.wait(_SETTLE_BUDGET * 2)
        return None

    await db.set_progress_handler(handler, 1)
    blocker = asyncio.create_task(db.execute("SELECT count(*) FROM session_journals"))
    assert await asyncio.to_thread(blocked.wait, _SETTLE_BUDGET), "aiosqlite worker never entered the progress handler"
    return blocker, release


class _TracePark:
    """Park the worker thread immediately before a targeted statement executes."""

    def __init__(self, db: Any, needle: str) -> None:
        self._db = db
        self._needle = needle
        self.entered = threading.Event()
        self.release = threading.Event()

    async def __aenter__(self) -> _TracePark:
        await self._db.set_trace_callback(self._handler)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.release.set()
        await self._db.set_trace_callback(None)

    def _handler(self, sql: str) -> None:
        if sql.strip().upper().startswith(self._needle):
            self.entered.set()
            self.release.wait(_SETTLE_BUDGET * 2)

    async def wait_entered(self) -> bool:
        return await asyncio.wait_for(asyncio.to_thread(self.entered.wait, _SETTLE_BUDGET), _SETTLE_BUDGET + 5)


async def _cancel_queued_begin(
    repo: SQLiteJournalRepository,
    write: Callable[[], Coroutine[Any, Any, Any]],
    *,
    repeats: int = 1,
) -> str:
    """Cancel ``write`` while its real ``BEGIN IMMEDIATE`` sits in the queue.

    The cancellation is delivered BEFORE the worker is released, so the BEGIN is
    provably still queued at that instant; then the worker is released and the
    caller's task must settle.
    """
    real_db = repo._db
    seen = asyncio.Event()
    repo._db = _StatementWatch(real_db, "BEGIN", seen)
    blocker, release = await _hold_worker(real_db)
    task = asyncio.create_task(write())
    parked = False
    try:
        try:
            await asyncio.wait_for(seen.wait(), _SETTLE_BUDGET)
            parked = True
        finally:
            if not parked:
                task.cancel()
        if parked:
            for _ in range(repeats):
                task.cancel()
                await asyncio.sleep(0)
    finally:
        repo._db = real_db
        release.set()
    outcome = await _settle(task)
    await asyncio.wait_for(blocker, _SETTLE_BUDGET)
    await real_db.set_progress_handler(None, 0)
    assert parked, "the write unit never reached BEGIN"
    return outcome


async def _cancel_executing(
    repo: SQLiteJournalRepository,
    needle: str,
    write: Callable[[], Coroutine[Any, Any, Any]],
    *,
    repeats: int = 1,
) -> str:
    """Cancel ``write`` while ``needle`` is executing on the worker thread.

    The worker is parked immediately before ``needle`` runs, the caller's task is
    cancelled there, and only then is the worker released - so the unit is left
    to finish on its own (the property under test).
    """
    park = _TracePark(repo._db, needle)
    await park.__aenter__()
    task = asyncio.create_task(write())
    parked = False
    try:
        try:
            parked = await park.wait_entered()
        finally:
            if not parked:
                task.cancel()
        if parked:
            for _ in range(repeats):
                task.cancel()
                await asyncio.sleep(0)
    finally:
        await park.__aexit__()
    outcome = await _settle(task)
    assert parked, f"the write unit never reached {needle}"
    return outcome


# ---------------------------------------------------------------------------
# N1 regression: real aiosqlite BEGIN queued when the caller is cancelled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_caller_while_begin_is_queued_leaves_connection_usable(repo: SQLiteJournalRepository) -> None:
    """The headline N1 regression.

    A cancellation delivered while the write unit's real ``BEGIN IMMEDIATE`` is
    queued must (a) still surface as ``CancelledError``, (b) leave the shared
    connection outside any transaction, (c) leave no half-applied batch, and (d)
    not break the next writer - on this session or on a second session sharing
    the connection.
    """
    await _seed(repo, "sess-n1")

    outcome = await _cancel_queued_begin(repo, lambda: repo.append(SCOPE, "sess-n1", 1, _batch("n1")))

    assert outcome == "cancelled"
    assert repo._db.in_transaction is False, "connection was handed back inside an open transaction"

    # The whole unit was unwound: nothing from the aborted batch is durable, so
    # the session is exactly as the seed left it.
    version, sequences, fact_types = await _state(repo, "sess-n1")
    assert version == 1
    assert sequences == [1]
    assert fact_types == [FactType.SESSION_CREATED]

    # The next writer on the SAME session still works and does not double-apply.
    appended = await asyncio.wait_for(repo.append(SCOPE, "sess-n1", 1, _batch("next")), _SETTLE_BUDGET)
    assert appended.new_version == 2

    # And a second session sharing the connection is unaffected.
    await _seed(repo, "sess-n1b")
    other = await asyncio.wait_for(repo.append(SCOPE, "sess-n1b", 1, _batch("other")), _SETTLE_BUDGET)
    assert other.new_version == 2
    assert repo._db.in_transaction is False


@pytest.mark.asyncio
async def test_repeated_cancellation_during_queued_write_keeps_ownership(repo: SQLiteJournalRepository) -> None:
    """Repeated cancellation must not abandon the unit mid-transaction.

    The connection is restored exactly once, by the write unit itself, so extra
    cancellations of the caller cannot cut the restoration short and leave the
    next writer staring at an open transaction.
    """
    await _seed(repo, "sess-rep")

    outcome = await _cancel_queued_begin(repo, lambda: repo.append(SCOPE, "sess-rep", 1, _batch("rep")), repeats=3)

    assert outcome == "cancelled"
    assert repo._db.in_transaction is False
    version, sequences, fact_types = await _state(repo, "sess-rep")
    assert (version, sequences) == (1, [1])
    assert fact_types == [FactType.SESSION_CREATED]

    appended = await asyncio.wait_for(repo.append(SCOPE, "sess-rep", 1, _batch("after")), _SETTLE_BUDGET)
    assert appended.new_version == 2


# ---------------------------------------------------------------------------
# N1: cancellation at the body / commit stages of the same unit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancellation_mid_batch_does_not_poison_connection(repo: SQLiteJournalRepository) -> None:
    """Cancelling while the batch INSERT is executing keeps the unit's ownership."""
    await _seed(repo, "sess-body")

    outcome = await _cancel_executing(repo, "INSERT INTO SESSION_FACTS", lambda: repo.append(SCOPE, "sess-body", 1, _batch("b", 2)))

    assert outcome == "cancelled", f"caller cancellation was replaced by {outcome!r}"
    assert repo._db.in_transaction is False

    # Neither of the two facts may survive a cancelled unit.
    version, sequences, fact_types = await _state(repo, "sess-body")
    assert (version, sequences) == (1, [1])
    assert fact_types == [FactType.SESSION_CREATED]

    appended = await asyncio.wait_for(repo.append(SCOPE, "sess-body", 1, _batch("after")), _SETTLE_BUDGET)
    assert appended.new_version == 2


@pytest.mark.asyncio
async def test_cancellation_at_commit_propagates_cancellation_not_rollback_error(repo: SQLiteJournalRepository) -> None:
    """A cancellation at ``COMMIT`` must not surface as a spurious ROLLBACK error.

    The worker may already have applied the commit, so the durable outcome is
    legitimately either-or; what must never happen is the caller being handed
    ``cannot rollback - no transaction is active`` instead of its own
    ``CancelledError``, or the connection being left inside a transaction.
    """
    await _seed(repo, "sess-commit")

    outcome = await _cancel_executing(repo, "COMMIT", lambda: repo.append(SCOPE, "sess-commit", 1, _batch("commit")))

    assert outcome == "cancelled", f"caller cancellation was replaced by {outcome!r}"
    assert repo._db.in_transaction is False

    version, sequences, fact_types = await _state(repo, "sess-commit")
    # Commit is a single atomic step: the batch landed whole, or not at all.
    assert version in (1, 2)
    assert sequences == list(range(1, version + 1))
    if version == 2:
        assert fact_types[-1] is FactType.USER_CONTENT_FINAL

    appended = await asyncio.wait_for(repo.append(SCOPE, "sess-commit", version, _batch("after")), _SETTLE_BUDGET)
    assert appended.new_version == version + 1


@pytest.mark.asyncio
async def test_queued_second_session_cannot_enter_before_cleanup_finishes(repo: SQLiteJournalRepository) -> None:
    """The lock must span the unwind *and* the restoration, not just the unit.

    A writer is abandoned at its queued ``BEGIN``; a competing reader and a
    competing writer are queued on the connection lock while it is still held.
    The statements the worker then executes are recorded in order, which is what
    settles the question: if the lock were released when the unit unwound (and
    only re-acquired by the restoration), the queued competitor would be served
    inside the gap - its ``BEGIN IMMEDIATE`` would run against the still-open
    transaction, and the ``ROLLBACK`` would end up physically executing *after*
    it. With the lock owned by the outer frame, the rollback is the first thing
    the worker does after the abandoned ``BEGIN``, and the competitors only then
    run, normally.
    """
    await _seed(repo, "sess-lock")
    await _seed(repo, "sess-follower")

    real_db = repo._db
    seen = asyncio.Event()
    repo._db = _StatementWatch(real_db, "BEGIN", seen)
    executed: list[str] = []

    def _record(sql: str) -> None:
        executed.append(sql.strip().upper())

    # Installed BEFORE the worker is parked: every statement queueing onto the
    # worker blocks while the progress handler holds it.
    await real_db.set_trace_callback(_record)
    blocker, release = await _hold_worker(real_db)
    writer = asyncio.create_task(repo.append(SCOPE, "sess-lock", 1, _batch("aborted")))
    parked = False
    try:
        try:
            await asyncio.wait_for(seen.wait(), _SETTLE_BUDGET)
            parked = True
        finally:
            if not parked:
                writer.cancel()
        assert parked, "the write unit never reached BEGIN"

        # Queued only now, while the writer provably owns the lock.
        reader: asyncio.Task[Any] = asyncio.create_task(repo.load_session(SCOPE, "sess-follower"))
        follower: asyncio.Task[Any] = asyncio.create_task(repo.append(SCOPE, "sess-follower", 1, _batch("follower")))
        for _turn in range(4):
            await asyncio.sleep(0)

        writer.cancel()
        await asyncio.sleep(0)
        executed.clear()  # only what runs after the cancellation matters
        assert not reader.done() and not follower.done(), "a queued competitor completed before the cleanup ran"
    finally:
        repo._db = real_db
        release.set()
    outcome = await _settle(writer)
    assert outcome == "cancelled"

    await asyncio.wait_for(blocker, _SETTLE_BUDGET)
    await real_db.set_progress_handler(None, 0)

    # Both competitors then complete normally: no nested BEGIN, no uncommitted read.
    assert real_db.in_transaction is False
    record = await asyncio.wait_for(reader, _SETTLE_BUDGET)
    assert record.version == 1
    appended = await asyncio.wait_for(follower, _SETTLE_BUDGET)
    assert appended.new_version == 2
    await real_db.set_trace_callback(None)

    # Ordering proof, on the now-frozen statement log: the abandoned
    # transaction's rollback physically ran BEFORE either queued competitor was
    # allowed to touch the connection. Had the lock been released when the unit
    # unwound, a competitor's statements would appear between the abandoned
    # BEGIN and the rollback.
    assert "ROLLBACK" in executed, f"no rollback was executed: {executed}"
    competitor_begins = [index for index, sql in enumerate(executed[1:], start=1) if sql.startswith("BEGIN IMMEDIATE")]
    assert competitor_begins, f"the queued competitor never ran: {executed}"
    assert executed.index("ROLLBACK") < competitor_begins[0], f"a competitor entered before the rollback: {executed}"

    version, sequences, _ = await _state(repo, "sess-lock")
    assert (version, sequences) == (1, [1])


@pytest.mark.asyncio
async def test_cancellation_while_begin_executes_keeps_unit_ownership(repo: SQLiteJournalRepository) -> None:
    """The BEGIN already running on the worker thread must be completed, not orphaned."""
    await _seed(repo, "sess-begin-exec")

    outcome = await _cancel_executing(repo, "BEGIN", lambda: repo.append(SCOPE, "sess-begin-exec", 1, _batch("beg")))

    assert outcome == "cancelled"
    assert repo._db.in_transaction is False
    version, sequences, fact_types = await _state(repo, "sess-begin-exec")
    assert version in (1, 2)
    assert sequences == list(range(1, version + 1))

    appended = await asyncio.wait_for(repo.append(SCOPE, "sess-begin-exec", version, _batch("after")), _SETTLE_BUDGET)
    assert appended.new_version == version + 1


# ---------------------------------------------------------------------------
# N1: every write method is covered, and errors keep the conditional rollback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_create_session_leaves_connection_usable(repo: SQLiteJournalRepository) -> None:
    """``create_session`` is the other headline write path (agent start-up)."""
    outcome = await _cancel_queued_begin(
        repo,
        lambda: repo.create_session(SessionRecord.new("sess-new", SCOPE), [_seed_fact("sess-new")]),
    )

    assert outcome == "cancelled"
    assert repo._db.in_transaction is False

    # The cancelled create was unwound, so the id is free and the retry works.
    with pytest.raises(SessionNotFound):
        await asyncio.wait_for(repo.load_session(SCOPE, "sess-new"), _SETTLE_BUDGET)

    record = await asyncio.wait_for(repo.create_session(SessionRecord.new("sess-new", SCOPE), [_seed_fact("sess-new")]), _SETTLE_BUDGET)
    assert record.version == 1

    added = await asyncio.wait_for(repo.append(SCOPE, "sess-new", 1, _batch("after-create")), _SETTLE_BUDGET)
    assert added.new_version == 2


@pytest.mark.asyncio
async def test_cancelled_checkpoint_write_leaves_connection_usable(repo: SQLiteJournalRepository) -> None:
    """``save_projection_checkpoint`` must obey the same ownership rule."""
    await _seed(repo, "sess-cp")

    outcome = await _cancel_executing(
        repo,
        "COMMIT",
        lambda: repo.save_projection_checkpoint(
            SCOPE, "sess-cp", ProjectionCheckpoint(projection_name="conversation-working-v1", last_sequence=7, data={"k": 1})
        ),
    )

    assert outcome == "cancelled"
    assert repo._db.in_transaction is False

    await asyncio.wait_for(
        repo.save_projection_checkpoint(
            SCOPE, "sess-cp", ProjectionCheckpoint(projection_name="conversation-working-v1", last_sequence=8, data={"k": 2})
        ),
        _SETTLE_BUDGET,
    )
    stored = await asyncio.wait_for(repo.load_projection_checkpoint(SCOPE, "sess-cp", "conversation-working-v1"), _SETTLE_BUDGET)
    assert stored is not None
    assert stored.last_sequence == 8
    assert dict(stored.data) == {"k": 2}


@pytest.mark.asyncio
async def test_cancelled_purge_and_archive_keep_connection_usable(repo: SQLiteJournalRepository) -> None:
    """``purge_session`` and ``archive_session`` are write units too."""
    await _seed(repo, "sess-purge")
    outcome = await _cancel_executing(repo, "COMMIT", lambda: repo.purge_session(SCOPE, "sess-purge"))
    assert outcome == "cancelled"
    assert repo._db.in_transaction is False
    # The commit may already have been applied by the worker before the caller's
    # cancellation was delivered, so deletion is legitimately either-or. What is
    # not negotiable: the session is entirely gone, or entirely intact.
    try:
        version, sequences, _ = await _state(repo, "sess-purge")
        assert (version, sequences) == (1, [1])
    except SessionNotFound:
        pass

    await _seed(repo, "sess-archive")
    outcome = await _cancel_executing(repo, "COMMIT", lambda: repo.archive_session(SCOPE, "sess-archive"))
    assert outcome == "cancelled"
    assert repo._db.in_transaction is False
    archived = await asyncio.wait_for(repo.load_session(SCOPE, "sess-archive"), _SETTLE_BUDGET)
    assert archived.status.value in ("active", "archived")
    await asyncio.wait_for(
        repo.save_projection_checkpoint(SCOPE, "sess-archive", ProjectionCheckpoint(projection_name="p", last_sequence=1, data={})),
        _SETTLE_BUDGET,
    )


@pytest.mark.asyncio
async def test_error_paths_still_roll_back_and_keep_the_lock(repo: SQLiteJournalRepository) -> None:
    """A raise inside the unit must still roll back, and must not wedge the lock.

    Covers the conditional ROLLBACK: ``JournalConflict``/``SessionNotFound``
    raised by the body, and a duplicate ``create_session``.
    """
    await _seed(repo, "sess-err")

    with pytest.raises(JournalConflict):
        await asyncio.wait_for(repo.append(SCOPE, "sess-err", 5, _batch("stale")), _SETTLE_BUDGET)
    assert repo._db.in_transaction is False

    with pytest.raises(SessionNotFound):
        await asyncio.wait_for(repo.append(SCOPE, "missing", 1, _batch("ghost")), _SETTLE_BUDGET)
    assert repo._db.in_transaction is False

    with pytest.raises(Exception) as excinfo:
        await asyncio.wait_for(repo.create_session(SessionRecord.new("sess-err", SCOPE), [_seed_fact("sess-err")]), _SETTLE_BUDGET)
    assert type(excinfo.value).__name__ == "JournalError"
    assert repo._db.in_transaction is False

    with pytest.raises(SessionNotFound):
        await asyncio.wait_for(repo.purge_session(SCOPE, "missing"), _SETTLE_BUDGET)
    assert repo._db.in_transaction is False

    # The single shared connection/lock is still healthy and still optimistic.
    appended = await asyncio.wait_for(repo.append(SCOPE, "sess-err", 1, _batch("ok")), _SETTLE_BUDGET)
    assert appended.new_version == 2
    with pytest.raises(JournalConflict):
        await asyncio.wait_for(repo.append(SCOPE, "sess-err", 1, _batch("stale-again")), _SETTLE_BUDGET)
    version, sequences, _ = await _state(repo, "sess-err")
    assert (version, sequences) == (2, [1, 2])
