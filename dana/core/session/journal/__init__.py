"""Session Journal persistence package — backend-agnostic repository contract.

Re-exports the public value types, exceptions, and the
:class:`~dana.core.session.journal.protocol.JournalRepository` protocol. The
SQLite adapter is always importable (pure-Python + stdlib sqlite3 underneath).
The PostgreSQL adapter requires ``asyncpg`` and is imported explicitly from
``dana.core.session.journal.postgres`` by callers that need it.
"""

from __future__ import annotations

from dana.core.session.journal.models import (
    AppendResult,
    JournalConflict,
    JournalError,
    ProjectionCheckpoint,
    SessionNotFound,
    SessionRecord,
    SessionStatus,
)
from dana.core.session.journal.protocol import JournalRepository
from dana.core.session.journal.sqlite import SQLiteJournalRepository


__all__ = [
    "AppendResult",
    "JournalConflict",
    "JournalError",
    "JournalRepository",
    "ProjectionCheckpoint",
    "SessionNotFound",
    "SessionRecord",
    "SessionStatus",
    "SQLiteJournalRepository",
]
