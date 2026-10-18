"""
SQLite adapter for the Session Journal.

Uses ``aiosqlite`` with WAL mode for concurrent reader/writer tolerance and
``BEGIN IMMEDIATE`` transactions to serialize writers. Every append performs
an optimistic-version check + the inserts + the version advance inside a
single ``BEGIN IMMEDIATE`` transaction, so two concurrent writers cannot
interleave: the second to acquire the write lock observes the new version and
raises :class:`~dana.core.session.journal.models.JournalConflict`.

Every write runs as one cancellation-safe unit
(:meth:`SQLiteJournalRepository._write_unit`). The unit owns the connection
until it has either committed or been unwound and rolled back, so a caller that
is cancelled (``asyncio.timeout``/``TaskGroup``) can never hand the shared
connection back sitting inside an open transaction, and the caller always
observes its own cancellation rather than a database error.

OwnerScope maps to the ``(owner_id, workspace)`` composite column pair; a
session_id is only ever resolved within that pair, giving owner isolation for
free at the index level.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
import json
from typing import TypeVar
import uuid

import aiosqlite
import structlog

from dana.core.session.journal.models import (
    AppendResult,
    JournalConflict,
    JournalError,
    ProjectionCheckpoint,
    SessionNotFound,
    SessionRecord,
    SessionStatus,
)
from dana.core.session.journal.schema import SCHEMA_VERSION, SQLITE_DDL
from dana.core.session.models import (
    ArtifactRef,
    FactType,
    JournalFact,
    JSONValue,
    NewJournalFact,
    OwnerScope,
)


async def _fetchone(db: aiosqlite.Connection, sql: str, params: tuple[object, ...] = ()) -> aiosqlite.Row | None:
    """Run ``sql`` and return a single row (or None). Closes the cursor."""
    cursor = await db.execute(sql, params)
    try:
        return await cursor.fetchone()
    finally:
        await cursor.close()


async def _fetchall(db: aiosqlite.Connection, sql: str, params: tuple[object, ...] = ()) -> list[aiosqlite.Row]:
    """Run ``sql`` and return all rows. Closes the cursor."""
    cursor = await db.execute(sql, params)
    try:
        return await cursor.fetchall()
    finally:
        await cursor.close()


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _serialize_json(value: Mapping[str, JSONValue] | None) -> str:
    return json.dumps(dict(value) if value is not None else {})


def _deserialize_json(value: str | None) -> dict[str, JSONValue]:
    if not value:
        return {}
    return json.loads(value)


def _serialize_artifact_refs(refs: Sequence[ArtifactRef] | None) -> str | None:
    if not refs:
        return None
    return json.dumps([{"uri": r.uri, "media_type": r.media_type, "size": r.size, "sha256": r.sha256} for r in refs])


def _deserialize_artifact_refs(value: str | None) -> tuple[ArtifactRef, ...]:
    if not value:
        return ()
    raw = json.loads(value)
    return tuple(ArtifactRef(uri=r["uri"], media_type=r["media_type"], size=r["size"], sha256=r["sha256"]) for r in raw)


_T = TypeVar("_T")

logger = structlog.get_logger(__name__)


class SQLiteJournalRepository:
    """JournalRepository backed by an on-disk SQLite database (aiosqlite)."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db
        # I3: aiosqlite serializes individual statements, not transactional
        # *units*. Two sessions sharing one connection would otherwise be able to
        # interleave a BEGIN..ROLLBACK with another writer's statements, exposing
        # uncommitted rows to a reader or losing part of a batch. Every statement
        # this repository issues goes through this lock, so one transaction - and
        # its rollback - is atomic with respect to every other operation on the
        # same connection. Distinct connections (and external writers) are
        # unaffected: SQLite's own file locking governs those.
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, path: str) -> SQLiteJournalRepository:
        """Open (or create) the SQLite database at ``path`` and initialize the schema."""
        db = await aiosqlite.connect(path)
        try:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA foreign_keys=ON")
            for stmt in SQLITE_DDL:
                await db.execute(stmt)
            await cls._ensure_schema_version(db)
            await db.commit()
        except BaseException:
            await db.close()
            raise
        return cls(db)

    @staticmethod
    async def _ensure_schema_version(db: aiosqlite.Connection) -> None:
        row = await _fetchone(db, "SELECT value FROM journal_meta WHERE key='schema_version'")
        if row is None:
            await db.execute(
                "INSERT INTO journal_meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        else:
            current = int(row["value"])
            if current == SCHEMA_VERSION:
                return
            if current == 1 and SCHEMA_VERSION == 2:
                # Migration v1 → v2: add artifact_refs column
                await db.execute("ALTER TABLE session_facts ADD COLUMN artifact_refs TEXT")
                await db.execute(
                    "UPDATE journal_meta SET value=? WHERE key='schema_version'",
                    (str(SCHEMA_VERSION),),
                )
                return
            raise JournalError(f"SQLite session journal schema version mismatch: file is v{current}, runtime expects v{SCHEMA_VERSION}")

    # ------------------------------------------------------------------
    # Internal: row <-> domain mappers
    # ------------------------------------------------------------------

    @staticmethod
    def _scope_key(scope: OwnerScope) -> tuple[str, str]:
        return (scope.owner_id, scope.workspace)

    @staticmethod
    def _row_to_session(row: aiosqlite.Row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            owner_scope=OwnerScope(owner_id=row["owner_id"], workspace=row["workspace"]),
            version=row["version"],
            status=SessionStatus(row["status"]),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            metadata=_deserialize_json(row["metadata"]),
        )

    @staticmethod
    def _row_to_fact(row: aiosqlite.Row) -> JournalFact:
        return JournalFact(
            fact_id=row["fact_id"],
            owner_scope=OwnerScope(owner_id=row["owner_id"], workspace=row["workspace"]),
            session_id=row["session_id"],
            sequence=row["sequence"],
            fact_type=FactType(row["fact_type"]),
            timestamp=_parse_dt(row["timestamp"]),
            correlation_id=row["correlation_id"],
            causation_id=row["causation_id"],
            schema_version=row["schema_version"],
            payload=_deserialize_json(row["payload"]),
            protected_payload=row["protected_payload"],
            artifact_refs=_deserialize_artifact_refs(row["artifact_refs"]),
        )

    async def _insert_facts(
        self,
        db: aiosqlite.Connection,
        scope: OwnerScope,
        session_id: str,
        facts: Sequence[JournalFact],
    ) -> None:
        # Each JournalFact already carries its assigned sequence (1..N for
        # create_session; expected_version+1.. for append). We persist that
        # sequence verbatim so callers control ordering deterministically.
        for fact in facts:
            await db.execute(
                """
                INSERT INTO session_facts
                    (fact_id, owner_id, workspace, session_id, sequence, fact_type, timestamp,
                     correlation_id, causation_id, schema_version, payload, protected_payload, artifact_refs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.fact_id,
                    scope.owner_id,
                    scope.workspace,
                    session_id,
                    fact.sequence,
                    fact.fact_type.value,
                    _iso(fact.timestamp),
                    fact.correlation_id,
                    fact.causation_id,
                    fact.schema_version,
                    _serialize_json(fact.payload),
                    fact.protected_payload,
                    _serialize_artifact_refs(fact.artifact_refs),
                ),
            )

    async def _require_session_row(self, scope: OwnerScope, session_id: str) -> aiosqlite.Row:
        """Row lookup for callers that already hold the connection ownership.

        Safe to call from inside a write unit (:meth:`_write_unit`) or from a
        method that holds ``self._lock``; it never acquires the lock itself.
        """
        row = await _fetchone(
            self._db,
            "SELECT * FROM session_journals WHERE owner_id=? AND workspace=? AND session_id=?",
            (*self._scope_key(scope), session_id),
        )
        if row is None:
            raise SessionNotFound(scope, session_id)
        return row

    # ------------------------------------------------------------------
    # Internal: cancellation-safe write units
    # ------------------------------------------------------------------

    async def _write_unit(self, work: Callable[[], Awaitable[_T]]) -> _T:
        """Run ``work`` as one complete, cancellation-safe transaction unit.

        ``work`` is invoked with ``self._lock`` held and after the ``BEGIN
        IMMEDIATE`` has physically executed; it must therefore not touch
        ``self._lock`` itself (use the lock-free ``_require_session_row`` /
        ``_fetchone`` helpers). It returns before ``COMMIT``.

        The lock is acquired here, in the *outer* frame, and is held for the
        entire lifetime of the unit plus its cleanup - including the rollback
        that :meth:`_abandon` has to perform after a cancellation. That is what
        makes the ownership airtight: a competing reader or writer that was
        already queued on the lock can never slip into the gap between "the
        unit unwound" and "the connection was put back outside a transaction",
        so it can never observe an uncommitted row or trip a nested
        ``BEGIN IMMEDIATE``.

        The unit runs in its own task, so cancelling the *caller*
        (``asyncio.timeout`` / ``TaskGroup`` / teardown) cannot abandon the
        transaction half-done: the unit either commits, or - if the caller was
        cancelled - is unwound and the connection restored before the caller's
        cancellation is re-raised and the lock released. The connection is
        therefore never handed back inside an open transaction, no matter where
        the cancellation lands.
        """
        async with self._lock:
            unit = asyncio.create_task(self._run_unit(work))
            try:
                return await asyncio.shield(unit)
            except asyncio.CancelledError:
                # Still holding the lock: _restore must not (and does not)
                # reacquire it, it just waits for the unit to unwind and then
                # restores the connection in place.
                await self._abandon(unit)
                raise

    async def _run_unit(self, work: Callable[[], Awaitable[_T]]) -> _T:
        """Execute one ``BEGIN IMMEDIATE`` .. ``COMMIT`` unit.

        Caller contract: ``self._lock`` is already held by :meth:`_write_unit`
        and is not released here, so this coroutine never touches the lock.
        """
        # Deliberately NOT inside a try: if this BEGIN fails (``database is
        # locked``) the unit opened no transaction of its own, and a ROLLBACK
        # here would mask that error ("cannot rollback - no transaction is
        # active") and could silently undo a transaction a previous, abandoned
        # unit left behind. Cancellation during the BEGIN is resolved by
        # _abandon once the statement queue has drained.
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            result = await work()
            await self._db.commit()
        except BaseException as exc:
            # Cancellation is unwound by _abandon instead: a statement the body
            # queued may still be waiting on the worker thread (a BEGIN that has
            # not executed yet is invisible to ``in_transaction``), so the
            # rollback has to happen after the queue is flushed.
            if not isinstance(exc, asyncio.CancelledError) and self._db.in_transaction:
                await self._db.execute("ROLLBACK")
            raise
        return result

    async def _abandon(self, unit: asyncio.Task[_T]) -> None:
        """Unwind a write unit whose caller was cancelled, then restore the connection.

        Called with ``self._lock`` held by :meth:`_write_unit`, which keeps
        holding it until this returns: the lock therefore spans the unwind, the
        queue barrier and the conditional rollback as one indivisible region.

        Runs off the cancelled caller's stack (hence in its own task) so that
        repeated cancellation of the caller cannot interrupt the restoration
        half-way. Returns only once the connection is out of any transaction.
        """
        await self._settle(asyncio.create_task(self._restore(unit)))

    async def _restore(self, unit: asyncio.Task[_T]) -> None:
        """Cancel ``unit``, wait for it to unwind, and roll back what it left open.

        ``self._lock`` is held by the caller for the whole call, so this method
        deliberately never acquires it - acquiring it here would open exactly
        the window (unit released, restoration not yet run) that this repair
        exists to close.
        """
        unit.cancel()
        try:
            await unit
        except BaseException:
            # The unit's own outcome is irrelevant here: the caller was cancelled
            # and will observe that, not the abandoned unit's error.
            pass
        try:
            # FIFO barrier: every statement queued before the cancellation has
            # now executed, so ``in_transaction`` finally reflects the truth
            # (a BEGIN that was still queued when the unit unwound is only
            # visible after this point).
            await self._db.execute("SELECT 1")
        except BaseException:
            logger.warning(
                "session journal connection unusable after a cancelled write; unable to inspect transaction state",
                exc_info=True,
            )
            return
        if not self._db.in_transaction:
            return
        try:
            await self._db.execute("ROLLBACK")
        except BaseException:
            logger.warning(
                "session journal rollback after a cancelled write failed; the connection may remain in a transaction",
                exc_info=True,
            )

    async def _settle(self, task: asyncio.Task[object]) -> None:
        """Wait for a restoration task to finish, absorbing repeated cancellation.

        Each cancellation of the caller is delivered at the ``shield`` await, so
        the loop simply keeps waiting for the same (uncancelled) task instead of
        abandoning the connection restoration half-way.
        """
        while True:
            try:
                await asyncio.shield(task)
                return
            except asyncio.CancelledError:
                if task.done():
                    return
            except BaseException:
                return

    # ------------------------------------------------------------------
    # Protocol: create / append / read / load
    # ------------------------------------------------------------------

    async def create_session(self, record: SessionRecord, facts: Sequence[JournalFact]) -> SessionRecord:
        """Persist a new session header + its initial ordered facts.

        The caller provides fully-formed JournalFacts with their sequences
        already assigned; the repository stores them verbatim and sets the
        session ``version = len(facts)``. Raises
        :class:`~dana.core.session.journal.models.JournalError` if the session
        already exists.
        """
        scope = record.owner_scope

        async def work() -> SessionRecord:
            existing = await _fetchone(
                self._db,
                "SELECT session_id FROM session_journals WHERE owner_id=? AND workspace=? AND session_id=?",
                (*self._scope_key(scope), record.session_id),
            )
            if existing is not None:
                raise JournalError(f"session {record.session_id!r} already exists for {scope.owner_id!r}/{scope.workspace!r}")

            now = _now()
            await self._db.execute(
                """
                INSERT INTO session_journals
                    (owner_id, workspace, session_id, version, status, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope.owner_id,
                    scope.workspace,
                    record.session_id,
                    len(facts),
                    SessionStatus.ACTIVE.value,
                    _iso(now),
                    _iso(now),
                    _serialize_json(record.metadata),
                ),
            )
            await self._insert_facts(self._db, scope, record.session_id, facts)
            # Read back from the row this unit just wrote: a nested locked load
            # would deadlock against the lock the unit already holds.
            return self._row_to_session(await self._require_session_row(scope, record.session_id))

        return await self._write_unit(work)

    async def append(
        self,
        scope: OwnerScope,
        session_id: str,
        expected_version: int,
        facts: Sequence[NewJournalFact],
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> AppendResult:
        if not facts:
            raise ValueError("cannot append an empty batch of facts")

        async def work() -> AppendResult:
            row = await _fetchone(
                self._db,
                "SELECT version FROM session_journals WHERE owner_id=? AND workspace=? AND session_id=?",
                (*self._scope_key(scope), session_id),
            )
            if row is None:
                # No row yet -> raise SessionNotFound, NOT a conflict (matches
                # the owner-isolation contract: a missing session is missing,
                # not a version skew).
                raise SessionNotFound(scope, session_id)
            actual_version = row["version"]
            if actual_version != expected_version:
                raise JournalConflict(session_id, expected_version, actual_version)

            now = _now()
            durable: list[JournalFact] = []
            for offset, nf in enumerate(facts):
                seq = expected_version + 1 + offset
                durable.append(
                    JournalFact(
                        fact_id=str(uuid.uuid4()),
                        owner_scope=scope,
                        session_id=session_id,
                        sequence=seq,
                        fact_type=nf.fact_type,
                        timestamp=now,
                        correlation_id=nf.correlation_id,
                        causation_id=nf.causation_id,
                        schema_version=nf.schema_version,
                        payload=nf.payload,
                        protected_payload=nf.protected_payload,
                        artifact_refs=nf.artifact_refs,
                    )
                )
            await self._insert_facts(self._db, scope, session_id, durable)

            new_version = expected_version + len(facts)
            if metadata is not None:
                await self._db.execute(
                    "UPDATE session_journals SET version=?, updated_at=?, metadata=? WHERE owner_id=? AND workspace=? AND session_id=?",
                    (new_version, _iso(now), _serialize_json(metadata), *self._scope_key(scope), session_id),
                )
            else:
                await self._db.execute(
                    "UPDATE session_journals SET version=?, updated_at=? WHERE owner_id=? AND workspace=? AND session_id=?",
                    (new_version, _iso(now), *self._scope_key(scope), session_id),
                )
            return AppendResult(new_version=new_version, appended_facts=tuple(durable))

        return await self._write_unit(work)

    async def read_facts(self, scope: OwnerScope, session_id: str, after_sequence: int = 0) -> list[JournalFact]:
        async with self._lock:
            # Existence check first so SessionNotFound is raised for unknown sessions.
            await self._require_session_row(scope, session_id)
            rows = await _fetchall(
                self._db,
                "SELECT * FROM session_facts WHERE owner_id=? AND workspace=? AND session_id=? AND sequence > ? ORDER BY sequence ASC",
                (*self._scope_key(scope), session_id, after_sequence),
            )
            return [self._row_to_fact(r) for r in rows]

    async def load_session(self, scope: OwnerScope, session_id: str) -> SessionRecord:
        async with self._lock:
            row = await self._require_session_row(scope, session_id)
            return self._row_to_session(row)

    # ------------------------------------------------------------------
    # Protocol: list / archive / purge
    # ------------------------------------------------------------------

    async def list_sessions(self, scope: OwnerScope) -> list[SessionRecord]:
        async with self._lock:
            rows = await _fetchall(
                self._db,
                "SELECT * FROM session_journals WHERE owner_id=? AND workspace=? AND status != ? ORDER BY created_at ASC",
                (scope.owner_id, scope.workspace, SessionStatus.DELETED.value),
            )
            return [self._row_to_session(r) for r in rows]

    async def archive_session(self, scope: OwnerScope, session_id: str) -> SessionRecord:
        async def work() -> SessionRecord:
            # Existence check first -> SessionNotFound for unknown sessions.
            await self._require_session_row(scope, session_id)
            await self._db.execute(
                "UPDATE session_journals SET status=?, updated_at=? WHERE owner_id=? AND workspace=? AND session_id=?",
                (SessionStatus.ARCHIVED.value, _iso(_now()), *self._scope_key(scope), session_id),
            )
            # Read back from the row this unit just wrote: a nested locked load
            # would deadlock against the lock the unit already holds.
            return self._row_to_session(await self._require_session_row(scope, session_id))

        return await self._write_unit(work)

    async def purge_session(self, scope: OwnerScope, session_id: str) -> None:
        async def work() -> None:
            # Existence check INSIDE the transaction so a concurrent purge
            # between check and BEGIN cannot silently no-op.
            await self._require_session_row(scope, session_id)
            # Deleting the header row cascades to session_facts (FK ON DELETE
            # CASCADE); clean checkpoints too (no FK on that table).
            await self._db.execute(
                "DELETE FROM projection_checkpoints WHERE owner_id=? AND workspace=? AND session_id=?",
                (*self._scope_key(scope), session_id),
            )
            await self._db.execute(
                "DELETE FROM session_journals WHERE owner_id=? AND workspace=? AND session_id=?",
                (*self._scope_key(scope), session_id),
            )

        await self._write_unit(work)

    # ------------------------------------------------------------------
    # Protocol: projection checkpoints
    # ------------------------------------------------------------------

    async def save_projection_checkpoint(self, scope: OwnerScope, session_id: str, checkpoint: ProjectionCheckpoint) -> None:
        """Upsert a projection checkpoint, refusing to move a checkpoint backwards.

        The write is a no-op when the stored checkpoint has advanced *past* the
        incoming ``last_sequence`` (a stale writer that resumed from an older
        cursor), so a slower process can never overwrite a projection that is
        further ahead. An equal cursor still overwrites: the latest writer for
        the same cursor wins. The conflict target is the OwnerScope-scoped
        primary key, so owner isolation is unchanged.
        """
        now = _now()
        ts = checkpoint.updated_at if checkpoint.updated_at is not None else now

        async def work() -> None:
            await self._db.execute(
                """
                INSERT INTO projection_checkpoints
                    (owner_id, workspace, session_id, projection_name, last_sequence, updated_at, data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, workspace, session_id, projection_name) DO UPDATE SET
                    last_sequence = excluded.last_sequence,
                    updated_at    = excluded.updated_at,
                    data          = excluded.data
                WHERE excluded.last_sequence >= projection_checkpoints.last_sequence
                """,
                (
                    scope.owner_id,
                    scope.workspace,
                    session_id,
                    checkpoint.projection_name,
                    checkpoint.last_sequence,
                    _iso(ts),
                    _serialize_json(checkpoint.data),
                ),
            )

        await self._write_unit(work)

    async def load_projection_checkpoint(self, scope: OwnerScope, session_id: str, projection_name: str) -> ProjectionCheckpoint | None:
        async with self._lock:
            row = await _fetchone(
                self._db,
                "SELECT * FROM projection_checkpoints WHERE owner_id=? AND workspace=? AND session_id=? AND projection_name=?",
                (*self._scope_key(scope), session_id, projection_name),
            )
            if row is None:
                return None
            return ProjectionCheckpoint(
                projection_name=row["projection_name"],
                last_sequence=row["last_sequence"],
                data=_deserialize_json(row["data"]),
                updated_at=_parse_dt(row["updated_at"]),
            )

    # ------------------------------------------------------------------
    # Protocol: lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        async with self._lock:
            await self._db.close()
