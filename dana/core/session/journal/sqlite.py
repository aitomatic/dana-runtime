"""
SQLite adapter for the Session Journal.

Uses ``aiosqlite`` with WAL mode for concurrent reader/writer tolerance and
``BEGIN IMMEDIATE`` transactions to serialize writers. Every append performs
an optimistic-version check + the inserts + the version advance inside a
single ``BEGIN IMMEDIATE`` transaction, so two concurrent writers cannot
interleave: the second to acquire the write lock observes the new version and
raises :class:`~dana.core.session.journal.models.JournalConflict`.

OwnerScope maps to the ``(owner_id, workspace)`` composite column pair; a
session_id is only ever resolved within that pair, giving owner isolation for
free at the index level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
import uuid

import aiosqlite

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


class SQLiteJournalRepository:
    """JournalRepository backed by an on-disk SQLite database (aiosqlite)."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

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
        row = await _fetchone(
            self._db,
            "SELECT * FROM session_journals WHERE owner_id=? AND workspace=? AND session_id=?",
            (*self._scope_key(scope), session_id),
        )
        if row is None:
            raise SessionNotFound(scope, session_id)
        return row

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
        # aiosqlite has no async context-manager transaction; manage BEGIN/COMMIT/ROLLBACK manually.
        await self._db.execute("BEGIN IMMEDIATE")
        try:
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
            await self._db.commit()
        except BaseException:
            await self._db.execute("ROLLBACK")
            raise

        return await self.load_session(scope, record.session_id)

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
        await self._db.execute("BEGIN IMMEDIATE")
        try:
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
            await self._db.commit()
        except BaseException:
            await self._db.execute("ROLLBACK")
            raise

        return AppendResult(new_version=new_version, appended_facts=tuple(durable))

    async def read_facts(self, scope: OwnerScope, session_id: str, after_sequence: int = 0) -> list[JournalFact]:
        # Existence check first so SessionNotFound is raised for unknown sessions.
        await self._require_session_row(scope, session_id)
        rows = await _fetchall(
            self._db,
            "SELECT * FROM session_facts WHERE owner_id=? AND workspace=? AND session_id=? AND sequence > ? ORDER BY sequence ASC",
            (*self._scope_key(scope), session_id, after_sequence),
        )
        return [self._row_to_fact(r) for r in rows]

    async def load_session(self, scope: OwnerScope, session_id: str) -> SessionRecord:
        row = await self._require_session_row(scope, session_id)
        return self._row_to_session(row)

    # ------------------------------------------------------------------
    # Protocol: list / archive / purge
    # ------------------------------------------------------------------

    async def list_sessions(self, scope: OwnerScope) -> list[SessionRecord]:
        rows = await _fetchall(
            self._db,
            "SELECT * FROM session_journals WHERE owner_id=? AND workspace=? AND status != ? ORDER BY created_at ASC",
            (scope.owner_id, scope.workspace, SessionStatus.DELETED.value),
        )
        return [self._row_to_session(r) for r in rows]

    async def archive_session(self, scope: OwnerScope, session_id: str) -> SessionRecord:
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            # Existence check first -> SessionNotFound for unknown sessions.
            await self._require_session_row(scope, session_id)
            await self._db.execute(
                "UPDATE session_journals SET status=?, updated_at=? WHERE owner_id=? AND workspace=? AND session_id=?",
                (SessionStatus.ARCHIVED.value, _iso(_now()), *self._scope_key(scope), session_id),
            )
            await self._db.commit()
        except BaseException:
            await self._db.execute("ROLLBACK")
            raise
        # Re-read to reflect the updated row in a single source of truth.
        return await self.load_session(scope, session_id)

    async def purge_session(self, scope: OwnerScope, session_id: str) -> None:
        await self._db.execute("BEGIN IMMEDIATE")
        try:
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
            await self._db.commit()
        except BaseException:
            await self._db.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Protocol: projection checkpoints
    # ------------------------------------------------------------------

    async def save_projection_checkpoint(self, scope: OwnerScope, session_id: str, checkpoint: ProjectionCheckpoint) -> None:
        now = _now()
        ts = checkpoint.updated_at if checkpoint.updated_at is not None else now
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            await self._db.execute(
                """
                INSERT INTO projection_checkpoints
                    (owner_id, workspace, session_id, projection_name, last_sequence, updated_at, data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, workspace, session_id, projection_name) DO UPDATE SET
                    last_sequence = excluded.last_sequence,
                    updated_at    = excluded.updated_at,
                    data          = excluded.data
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
            await self._db.commit()
        except BaseException:
            await self._db.execute("ROLLBACK")
            raise

    async def load_projection_checkpoint(self, scope: OwnerScope, session_id: str, projection_name: str) -> ProjectionCheckpoint | None:
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
        await self._db.close()
