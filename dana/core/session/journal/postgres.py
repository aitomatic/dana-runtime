"""
PostgreSQL adapter for the Session Journal.

Uses ``asyncpg`` with a single connection (a pool is an internal optimization
deferred to a later phase per the design — YAGNI for Phase 01). Writers are
serialized via ``SELECT ... FOR UPDATE`` inside a transaction: the version
check locks the session header row so a second concurrent append blocks until
the first commits, then observes the new version and raises
:class:`~dana.core.session.journal.models.JournalConflict`.

JSON columns are JSONB (binary, indexable). A connection-level codec maps
JSONB <-> Python ``dict``/``list`` via ``json.dumps``/``json.loads`` so the
public value types stay plain JSON-safe Python objects — no asyncpg/SQLAlchemy
types leak through the interface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
import uuid

import asyncpg

from dana.core.session.journal.models import (
    AppendResult,
    JournalConflict,
    JournalError,
    ProjectionCheckpoint,
    SessionNotFound,
    SessionRecord,
    SessionStatus,
)
from dana.core.session.journal.schema import POSTGRES_DDL, SCHEMA_VERSION
from dana.core.session.models import (
    ArtifactRef,
    FactType,
    JournalFact,
    JSONValue,
    NewJournalFact,
    OwnerScope,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _serialize_artifact_refs(refs: Sequence[ArtifactRef] | None) -> list[dict[str, object]] | None:
    if not refs:
        return None
    return [{"uri": r.uri, "media_type": r.media_type, "size": r.size, "sha256": r.sha256} for r in refs]


def _deserialize_artifact_refs(value: list[dict[str, object]] | None) -> tuple[ArtifactRef, ...]:
    if not value:
        return ()
    return tuple(ArtifactRef(uri=r["uri"], media_type=r["media_type"], size=r["size"], sha256=r["sha256"]) for r in value)


class PostgresJournalRepository:
    """JournalRepository backed by PostgreSQL (asyncpg, single connection)."""

    def __init__(self, db: asyncpg.Connection) -> None:
        self._db = db

    @classmethod
    async def open(cls, dsn: str) -> PostgresJournalRepository:
        """Connect to ``dsn`` and initialize the schema (idempotent)."""
        db = await asyncpg.connect(dsn=dsn)
        try:
            # Map JSONB columns <-> Python dict/list so value types stay plain.
            await db.set_type_codec(
                "jsonb",
                encoder=json.dumps,
                decoder=json.loads,
                schema="pg_catalog",
            )
            for stmt in POSTGRES_DDL:
                await db.execute(stmt)
            await cls._ensure_schema_version(db)
        except BaseException:
            await db.close()
            raise
        return cls(db)

    @staticmethod
    async def _ensure_schema_version(db: asyncpg.Connection) -> None:
        current = await db.fetchval("SELECT value FROM journal_meta WHERE key='schema_version'")
        if current is None:
            await db.execute(
                "INSERT INTO journal_meta (key, value) VALUES ('schema_version', $1)",
                str(SCHEMA_VERSION),
            )
        else:
            if int(current) != SCHEMA_VERSION:
                raise JournalError(f"Postgres session journal schema version mismatch: db is v{current}, runtime expects v{SCHEMA_VERSION}")

    # ------------------------------------------------------------------
    # Internal: row <-> domain mappers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_session(row: asyncpg.Record) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            owner_scope=OwnerScope(owner_id=row["owner_id"], workspace=row["workspace"]),
            version=row["version"],
            status=SessionStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=dict(row["metadata"]) if row["metadata"] else {},
        )

    @staticmethod
    def _row_to_fact(row: asyncpg.Record) -> JournalFact:
        return JournalFact(
            fact_id=row["fact_id"],
            owner_scope=OwnerScope(owner_id=row["owner_id"], workspace=row["workspace"]),
            session_id=row["session_id"],
            sequence=row["sequence"],
            fact_type=FactType(row["fact_type"]),
            timestamp=row["timestamp"],
            correlation_id=row["correlation_id"],
            causation_id=row["causation_id"],
            schema_version=row["schema_version"],
            payload=dict(row["payload"]) if row["payload"] else {},
            protected_payload=row["protected_payload"],
            artifact_refs=_deserialize_artifact_refs(row["artifact_refs"]),
        )

    async def _insert_facts(self, scope: OwnerScope, session_id: str, facts: Sequence[JournalFact]) -> None:
        for fact in facts:
            await self._db.execute(
                """
                INSERT INTO session_facts
                    (fact_id, owner_id, workspace, session_id, sequence, fact_type, timestamp,
                     correlation_id, causation_id, schema_version, payload, protected_payload, artifact_refs)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                fact.fact_id,
                scope.owner_id,
                scope.workspace,
                session_id,
                fact.sequence,
                fact.fact_type.value,
                fact.timestamp,
                fact.correlation_id,
                fact.causation_id,
                fact.schema_version,
                dict(fact.payload),
                fact.protected_payload,
                _serialize_artifact_refs(fact.artifact_refs),
            )

    async def _require_session_row(self, scope: OwnerScope, session_id: str) -> asyncpg.Record:
        row = await self._db.fetchrow(
            "SELECT * FROM session_journals WHERE owner_id=$1 AND workspace=$2 AND session_id=$3",
            scope.owner_id,
            scope.workspace,
            session_id,
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
        async with self._db.transaction():
            existing = await self._db.fetchval(
                "SELECT session_id FROM session_journals WHERE owner_id=$1 AND workspace=$2 AND session_id=$3",
                scope.owner_id,
                scope.workspace,
                record.session_id,
            )
            if existing is not None:
                raise JournalError(f"session {record.session_id!r} already exists for {scope.owner_id!r}/{scope.workspace!r}")

            now = _now()
            try:
                await self._db.execute(
                    """
                    INSERT INTO session_journals
                        (owner_id, workspace, session_id, version, status, created_at, updated_at, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    scope.owner_id,
                    scope.workspace,
                    record.session_id,
                    len(facts),
                    SessionStatus.ACTIVE.value,
                    now,
                    now,
                    dict(record.metadata),
                )
            except asyncpg.exceptions.UniqueViolationError:
                # Race: a concurrent create_session passed the existence
                # check too and won the PK insert first. Surface the
                # documented public exception, never the asyncpg one.
                raise JournalError(f"session {record.session_id!r} already exists for {scope.owner_id!r}/{scope.workspace!r}") from None
            await self._insert_facts(scope, record.session_id, facts)

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
        async with self._db.transaction():
            # FOR UPDATE locks the header row: a second concurrent append blocks
            # here until this transaction commits, then sees the new version.
            row = await self._db.fetchrow(
                "SELECT version FROM session_journals WHERE owner_id=$1 AND workspace=$2 AND session_id=$3 FOR UPDATE",
                scope.owner_id,
                scope.workspace,
                session_id,
            )
            if row is None:
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
                    )
                )
            await self._insert_facts(scope, session_id, durable)

            new_version = expected_version + len(facts)
            if metadata is not None:
                await self._db.execute(
                    "UPDATE session_journals SET version=$1, updated_at=$2, metadata=$3 "
                    "WHERE owner_id=$4 AND workspace=$5 AND session_id=$6",
                    new_version,
                    now,
                    dict(metadata),
                    scope.owner_id,
                    scope.workspace,
                    session_id,
                )
            else:
                await self._db.execute(
                    "UPDATE session_journals SET version=$1, updated_at=$2 WHERE owner_id=$3 AND workspace=$4 AND session_id=$5",
                    new_version,
                    now,
                    scope.owner_id,
                    scope.workspace,
                    session_id,
                )

        return AppendResult(new_version=new_version, appended_facts=tuple(durable))

    async def read_facts(self, scope: OwnerScope, session_id: str, after_sequence: int = 0) -> list[JournalFact]:
        await self._require_session_row(scope, session_id)
        rows = await self._db.fetch(
            "SELECT * FROM session_facts WHERE owner_id=$1 AND workspace=$2 AND session_id=$3 AND sequence > $4 ORDER BY sequence ASC",
            scope.owner_id,
            scope.workspace,
            session_id,
            after_sequence,
        )
        return [self._row_to_fact(r) for r in rows]

    async def load_session(self, scope: OwnerScope, session_id: str) -> SessionRecord:
        row = await self._require_session_row(scope, session_id)
        return self._row_to_session(row)

    # ------------------------------------------------------------------
    # Protocol: list / archive / purge
    # ------------------------------------------------------------------

    async def list_sessions(self, scope: OwnerScope) -> list[SessionRecord]:
        rows = await self._db.fetch(
            "SELECT * FROM session_journals WHERE owner_id=$1 AND workspace=$2 AND status != $3 ORDER BY created_at ASC",
            scope.owner_id,
            scope.workspace,
            SessionStatus.DELETED.value,
        )
        return [self._row_to_session(r) for r in rows]

    async def archive_session(self, scope: OwnerScope, session_id: str) -> SessionRecord:
        async with self._db.transaction():
            await self._require_session_row(scope, session_id)
            await self._db.execute(
                "UPDATE session_journals SET status=$1, updated_at=$2 WHERE owner_id=$3 AND workspace=$4 AND session_id=$5",
                SessionStatus.ARCHIVED.value,
                _now(),
                scope.owner_id,
                scope.workspace,
                session_id,
            )
        return await self.load_session(scope, session_id)

    async def purge_session(self, scope: OwnerScope, session_id: str) -> None:
        async with self._db.transaction():
            # Existence check INSIDE the transaction so a concurrent purge
            # between check and BEGIN cannot silently no-op.
            await self._require_session_row(scope, session_id)
            # Deleting the header row cascades to session_facts (FK ON DELETE
            # CASCADE); clean checkpoints too (no FK on that table).
            await self._db.execute(
                "DELETE FROM projection_checkpoints WHERE owner_id=$1 AND workspace=$2 AND session_id=$3",
                scope.owner_id,
                scope.workspace,
                session_id,
            )
            await self._db.execute(
                "DELETE FROM session_journals WHERE owner_id=$1 AND workspace=$2 AND session_id=$3",
                scope.owner_id,
                scope.workspace,
                session_id,
            )

    # ------------------------------------------------------------------
    # Protocol: projection checkpoints
    # ------------------------------------------------------------------

    async def save_projection_checkpoint(self, scope: OwnerScope, session_id: str, checkpoint: ProjectionCheckpoint) -> None:
        now = _now()
        ts = checkpoint.updated_at if checkpoint.updated_at is not None else now
        async with self._db.transaction():
            await self._db.execute(
                """
                INSERT INTO projection_checkpoints
                    (owner_id, workspace, session_id, projection_name, last_sequence, updated_at, data)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (owner_id, workspace, session_id, projection_name) DO UPDATE SET
                    last_sequence = EXCLUDED.last_sequence,
                    updated_at    = EXCLUDED.updated_at,
                    data          = EXCLUDED.data
                """,
                scope.owner_id,
                scope.workspace,
                session_id,
                checkpoint.projection_name,
                checkpoint.last_sequence,
                ts,
                dict(checkpoint.data),
            )

    async def load_projection_checkpoint(self, scope: OwnerScope, session_id: str, projection_name: str) -> ProjectionCheckpoint | None:
        row = await self._db.fetchrow(
            "SELECT * FROM projection_checkpoints WHERE owner_id=$1 AND workspace=$2 AND session_id=$3 AND projection_name=$4",
            scope.owner_id,
            scope.workspace,
            session_id,
            projection_name,
        )
        if row is None:
            return None
        return ProjectionCheckpoint(
            projection_name=row["projection_name"],
            last_sequence=row["last_sequence"],
            data=dict(row["data"]) if row["data"] else {},
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------
    # Protocol: lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self._db.close()
