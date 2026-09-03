"""
Session Journal repository — supporting value types and exceptions.

These types are backend-agnostic: both the SQLite and PostgreSQL adapters
produce and consume the same :class:`SessionRecord`, :class:`AppendResult`,
and :class:`ProjectionCheckpoint` values and raise the same exception
hierarchy. Backend-specific types (aiosqlite connections, asyncpg pools,
SQLAlchemy models) never appear in this module or in the
:class:`~dana.core.session.journal.protocol.JournalRepository` interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from dana.core.session.models import JournalFact, JSONValue, OwnerScope


class SessionStatus(Enum):
    """Lifecycle state of a Session Journal, stored as a lowercase string."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class JournalError(Exception):
    """Base exception for Session Journal persistence failures."""


class SessionNotFound(JournalError):
    """Raised when no session exists for the given OwnerScope + session_id."""

    def __init__(self, owner_scope: OwnerScope, session_id: str) -> None:
        self.owner_scope = owner_scope
        self.session_id = session_id
        super().__init__(f"session {session_id!r} not found for owner {owner_scope.owner_id!r}/{owner_scope.workspace!r}")


class JournalConflict(JournalError):
    """Raised on optimistic-concurrency mismatch during append.

    ``expected_version`` is the version the caller assumed; ``actual_version``
    is the version currently durable in the journal (``None`` only if the
    session vanished mid-transaction).
    """

    def __init__(self, session_id: str, expected_version: int | None, actual_version: int | None) -> None:
        self.session_id = session_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(f"journal conflict for session {session_id!r}: expected version {expected_version}, actual {actual_version}")


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """The durable header row of a Session Journal.

    ``version`` is the high-water mark equal to the highest assigned fact
    sequence (0 before any facts are appended). ``metadata`` is an arbitrary
    JSON-safe mapping updated atomically with appends.
    """

    session_id: str
    owner_scope: OwnerScope
    version: int
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    @staticmethod
    def new(session_id: str, owner_scope: OwnerScope) -> SessionRecord:
        """Build a fresh ACTIVE record at version 0 with empty metadata.

        Callers pass the returned record to ``create_session`` along with the
        initial facts; the repository assigns the real ``version`` /
        timestamps on persist.
        """
        now = datetime.now(UTC)
        return SessionRecord(
            session_id=session_id,
            owner_scope=owner_scope,
            version=0,
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            metadata={},
        )


@dataclass(frozen=True, slots=True)
class AppendResult:
    """Result of a successful ordered batch append."""

    new_version: int
    appended_facts: tuple[JournalFact, ...]


@dataclass(frozen=True, slots=True)
class ProjectionCheckpoint:
    """A named cursor + opaque JSON blob saved by a projection.

    Checkpoints are stored OUT-OF-BAND of journal facts: writing one never
    changes any fact and never advances the session version.
    """

    projection_name: str
    last_sequence: int
    data: Mapping[str, JSONValue] = field(default_factory=dict)
    updated_at: datetime | None = None
