"""
Session Journal repository protocol — the backend-agnostic persistence contract.

A :class:`JournalRepository` is the sole durable authority for the ordered
facts of a Dana agent session. Two reference implementations exist
(:class:`~dana.core.session.journal.sqlite.SQLiteJournalRepository` and
:class:`~dana.core.session.journal.postgres.PostgresJournalRepository`); both
MUST satisfy this protocol with equivalent domain semantics. The contract is
enforced by the parameterized suite in
``tests/integration/test_session_journal_contract.py``.

All operations are scoped by :class:`~dana.core.session.models.OwnerScope`
(``owner_id`` + ``workspace``); a session_id is only meaningful within its
owner scope and is invisible to any other scope.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from dana.core.session.journal.models import AppendResult, ProjectionCheckpoint, SessionRecord
from dana.core.session.models import JournalFact, JSONValue, NewJournalFact, OwnerScope


@runtime_checkable
class JournalRepository(Protocol):
    """Durable, owner-scoped store for Session Journal facts.

    Implementations MUST be safe to call from a single async task. Concurrency
    across writers is controlled by optimistic versioning: every append
    declares the ``expected_version`` it observed; if the durable version
    differs, :class:`~dana.core.session.journal.models.JournalConflict` is
    raised and no facts are persisted.
    """

    async def create_session(self, record: SessionRecord, facts: Sequence[JournalFact]) -> SessionRecord:
        """Persist a new session header + its initial ordered facts.

        The caller provides fully-formed :class:`JournalFact` objects with
        their sequences already assigned; the repository stores them verbatim
        and sets the session ``version = len(facts)``. ``record.version`` is
        ignored. Returns a refreshed :class:`SessionRecord` reflecting the
        durable version, status, and timestamps. Raises
        :class:`~dana.core.session.journal.models.JournalError` if a session
        with the same (owner_id, workspace, session_id) already exists.
        """
        ...

    async def append(
        self,
        scope: OwnerScope,
        session_id: str,
        expected_version: int,
        facts: Sequence[NewJournalFact],
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> AppendResult:
        """Atomically append an ordered batch and advance the session version.

        The batch is assigned sequences ``expected_version + 1 ..`` and the
        session version is advanced to ``expected_version + len(facts)``. If
        ``metadata`` is provided it replaces the session metadata in the same
        transaction. Raises
        :class:`~dana.core.session.journal.models.JournalConflict` when the
        durable version is not ``expected_version``.
        """
        ...

    async def read_facts(self, scope: OwnerScope, session_id: str, after_sequence: int = 0) -> list[JournalFact]:
        """Read facts with ``sequence > after_sequence`` in ascending order.

        Raises :class:`~dana.core.session.journal.models.SessionNotFound` when
        the session does not exist in ``scope``.
        """
        ...

    async def load_session(self, scope: OwnerScope, session_id: str) -> SessionRecord:
        """Load the session header. Raises
        :class:`~dana.core.session.journal.models.SessionNotFound` when
        the session does not exist in ``scope``.
        """
        ...

    async def list_sessions(self, scope: OwnerScope) -> list[SessionRecord]:
        """List sessions within the OwnerScope, excluding DELETED sessions."""
        ...

    async def archive_session(self, scope: OwnerScope, session_id: str) -> SessionRecord:
        """Set the session status to ARCHIVED. Returns the refreshed record."""
        ...

    async def purge_session(self, scope: OwnerScope, session_id: str) -> None:
        """Permanently delete a session and all of its facts."""
        ...

    async def save_projection_checkpoint(self, scope: OwnerScope, session_id: str, checkpoint: ProjectionCheckpoint) -> None:
        """Upsert a projection checkpoint without changing any journal fact."""
        ...

    async def load_projection_checkpoint(self, scope: OwnerScope, session_id: str, projection_name: str) -> ProjectionCheckpoint | None:
        """Load a projection checkpoint, or ``None`` if not found."""
        ...

    async def close(self) -> None:
        """Close the underlying database connection."""
        ...
