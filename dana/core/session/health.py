"""Operational health checks for the Session Journal system.

A health check is **read-only**: it never appends facts and never mutates a
checkpoint. The aggregate report carries only counts and booleans — owner_ids,
workspace names, session_ids, payloads, and protected payloads are never
included. This redaction is verified by ``tests/unit/core/session/test_health.py``.

Five operational concerns are covered:

1. **Database connectivity** — ``list_sessions`` round-trips; a failure here
   short-circuits the report with ``database_connectivity=False``.
2. **Journal conflicts** — structural: optimistic concurrency is exercised on
   every append; a healthy repository surfaces :class:`JournalConflict` on
   version skew rather than silently overwriting. The contract test in
   ``tests/integration/test_session_journal_contract.py`` is the authoritative
   check; this health endpoint only confirms the repository is reachable.
3. **Projection lag** — the maximum gap between a session's durable
   ``version`` and the ``last_sequence`` of its named projection checkpoints.
4. **Interrupted recovery** — the count of started-but-unterminated turns
   observed across the scope's sessions (detection only; recovery is performed
   by :func:`~dana.core.session.legacy_timeline_migration.recover_interrupted_turns`).
5. **Migration parity** — the count of ``LEGACY_TIMELINE_MIGRATED`` markers
   present, so operators can confirm legacy imports landed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from dana.core.session.journal.protocol import JournalRepository
from dana.core.session.models import FactType, OwnerScope


# Projection checkpoint names tracked by the Dana runtime. The lag check
# queries each and reports the maximum gap. A projection that has no
# checkpoint yet (e.g. a freshly created session) is not counted as lag —
# the projector simply has not run.
_TRACKED_PROJECTIONS: tuple[str, ...] = ("conversation", "timeline")

# Terminal fact types that close a started turn. Mirrors the constant in
# ``legacy_timeline_migration``; duplicated here to keep the health check
# strictly read-only (importing the private frozenset would couple the
# contract). Any of these sharing a ``correlation_id`` with a ``TURN_STARTED``
# marks the turn as terminated.
_TERMINAL_FACT_TYPES: frozenset[FactType] = frozenset(
    {
        FactType.TURN_COMPLETED,
        FactType.TURN_CANCELLED,
        FactType.TURN_ERROR,
        FactType.TURN_INTERRUPTED,
    }
)


@dataclass(frozen=True, slots=True)
class JournalHealthReport:
    """Aggregated, redacted health status of a Session Journal scope.

    No field exposes owner_ids, workspace names, session_ids, payloads, or
    protected payloads. Counts are aggregate only.
    """

    database_connectivity: bool
    total_sessions: int
    active_sessions: int
    archived_sessions: int
    interrupted_turns_detected: int
    projection_lag_max: int
    legacy_migration_markers: int
    errors: list[str] = field(default_factory=list)


def _count_interrupted(facts: Iterable[Any]) -> int:
    """Read-only count of started-but-unterminated turns from journal facts.

    A turn is *interrupted* when a ``TURN_STARTED`` fact has no matching
    terminal fact (``TURN_COMPLETED``, ``TURN_CANCELLED``, ``TURN_ERROR``, or
    ``TURN_INTERRUPTED``) sharing its ``correlation_id``. This mirrors
    :func:`~dana.core.session.legacy_timeline_migration.recover_interrupted_turns`
    detection but performs NO append — health checks are read-only.
    """
    started: set[str] = set()
    terminated: set[str] = set()
    for fact in facts:
        if fact.fact_type is FactType.TURN_STARTED:
            started.add(fact.correlation_id)
        elif fact.fact_type in _TERMINAL_FACT_TYPES:
            terminated.add(fact.correlation_id)
    return len(started - terminated)


async def check_journal_health(
    repository: JournalRepository,
    scope: OwnerScope,
) -> JournalHealthReport:
    """Run all health checks and return an aggregated, redacted report.

    The function is **read-only**: it never calls ``append``,
    ``create_session``, ``save_projection_checkpoint``, or any other mutating
    operation. All payloads are redacted — only counts and booleans are
    returned.

    A failure on the initial ``list_sessions`` call short-circuits the report
    with ``database_connectivity=False``; subsequent per-session errors are
    captured in ``errors`` without aborting the scan.
    """
    errors: list[str] = []

    # 1. Database connectivity (and the session listing we need anyway).
    try:
        sessions = await repository.list_sessions(scope)
    except Exception as e:
        errors.append(f"database_connectivity: {type(e).__name__}")
        return JournalHealthReport(
            database_connectivity=False,
            total_sessions=0,
            active_sessions=0,
            archived_sessions=0,
            interrupted_turns_detected=0,
            projection_lag_max=0,
            legacy_migration_markers=0,
            errors=errors,
        )

    # Local imports keep the module import-free of the journal.models cycle
    # at module load; SessionStatus lives in journal.models.
    from dana.core.session.journal.models import SessionStatus

    active = sum(1 for s in sessions if s.status is SessionStatus.ACTIVE)
    archived = sum(1 for s in sessions if s.status is SessionStatus.ARCHIVED)

    interrupted_total = 0
    legacy_markers = 0
    max_lag = 0

    for record in sessions:
        # 3. Interrupted recovery detection + 5. migration parity scan.
        # Both read the same fact stream; combine the pass to amortize I/O.
        try:
            facts = await repository.read_facts(scope, record.session_id)
        except Exception as e:
            errors.append(f"read_facts: {type(e).__name__}")
            continue

        interrupted_total += _count_interrupted(facts)
        legacy_markers += sum(1 for f in facts if f.fact_type is FactType.LEGACY_TIMELINE_MIGRATED)

        # 4. Projection lag: max(version - checkpoint.last_sequence) across
        # tracked projections. A missing checkpoint contributes no lag (the
        # projector has not yet persisted progress for this session).
        for projection_name in _TRACKED_PROJECTIONS:
            try:
                checkpoint = await repository.load_projection_checkpoint(scope, record.session_id, projection_name)
            except Exception as e:
                errors.append(f"load_projection_checkpoint:{projection_name}: {type(e).__name__}")
                continue
            if checkpoint is None:
                continue
            lag = record.version - checkpoint.last_sequence
            if lag > max_lag:
                max_lag = lag

    return JournalHealthReport(
        database_connectivity=True,
        total_sessions=len(sessions),
        active_sessions=active,
        archived_sessions=archived,
        interrupted_turns_detected=interrupted_total,
        projection_lag_max=max_lag,
        legacy_migration_markers=legacy_markers,
        errors=errors,
    )
