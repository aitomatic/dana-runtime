"""
Legacy Timeline migration and crash recovery for Session Journals.

Three responsibilities:

1. **Crash recovery** (:func:`recover_interrupted_turns`): detects turns that
   have a ``TURN_STARTED`` fact but no terminal fact (``TURN_COMPLETED``,
   ``TURN_CANCELLED``, ``TURN_ERROR``, ``TURN_INTERRUPTED``) and appends a typed
   ``TURN_INTERRUPTED`` fact for each. This excludes partial assistant output
   from the Conversation View (the projector holds it as pending and never
   promotes it without a matching ``TURN_COMPLETED``) while preserving it in the
   Host Event View, and surfaces an interruption observation to the next model
   turn.

2. **Legacy migration** (:func:`migrate_legacy_timeline`): imports legacy
   ``TimelineEntry`` sessions into a Session Journal. Idempotent via a
   content-addressed source hash stored on a ``LEGACY_TIMELINE_MIGRATED``
   marker fact — re-migrating the identical source is a no-op. D1 is text-only:
   only ``USER_MESSAGE`` and ``AGENT_RESPONSE`` are converted to journal facts;
   tools, thoughts, summaries, and ephemeral context are skipped.

3. **Compatibility projection** (:func:`journal_facts_to_timeline_entries`):
   projects journal facts back into the legacy ``TimelineEntry`` shape, used
   behind the ``DANA_SESSION_JOURNAL_AUTHORITY=0`` rollback flag so the journal
   can serve as the sole authority while legacy readers still consume Timeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json

from structlog import get_logger

from dana.core.session.journal.protocol import JournalRepository
from dana.core.session.models import FactType, JournalFact, NewJournalFact, OwnerScope
from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType


logger = get_logger()


# Terminal fact types that close a turn (any of these makes a started turn
# count as recovered/finished and prevents double-recovery).
_TERMINAL_FACT_TYPES: frozenset[FactType] = frozenset(
    {
        FactType.TURN_COMPLETED,
        FactType.TURN_CANCELLED,
        FactType.TURN_ERROR,
        FactType.TURN_INTERRUPTED,
    }
)


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """Result of a legacy timeline migration.

    Attributes:
        already_migrated: True when an identical source was migrated before
            (content-addressed source hash matched an existing marker); no
            facts were appended in that case.
        appended: Number of journal facts appended in this call (0 when the
            source was already migrated). Includes the migration marker.
        source_hash: SHA-256 of the canonical JSON of the source entries — the
            content-addressed idempotency key.
        skipped_entries: Count of source entries that were not text (tools,
            thoughts, summaries, ephemeral context, etc.) and therefore not
            converted for D1.
    """

    already_migrated: bool
    appended: int
    source_hash: str
    skipped_entries: int


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


async def recover_interrupted_turns(
    repository: JournalRepository,
    scope: OwnerScope,
    session_id: str,
) -> int:
    """Detect and recover interrupted turns in a session journal.

    A turn is *interrupted* when it has a ``TURN_STARTED`` fact but no matching
    terminal fact (``TURN_COMPLETED``, ``TURN_CANCELLED``, ``TURN_ERROR``, or
    ``TURN_INTERRUPTED``) sharing its ``correlation_id``. For each such turn a
    typed ``TURN_INTERRUPTED`` fact is appended.

    Recovery is idempotent: the appended ``TURN_INTERRUPTED`` is itself a
    terminal fact, so a second call finds no remaining interrupted turns.

    Returns the count of recovered turns.
    """
    facts = await repository.read_facts(scope, session_id)
    if not facts:
        return 0

    current_version = max(f.sequence for f in facts)

    started_turns: set[str] = set()
    terminated_turns: set[str] = set()
    for fact in facts:
        if fact.fact_type is FactType.TURN_STARTED:
            started_turns.add(fact.correlation_id)
        elif fact.fact_type in _TERMINAL_FACT_TYPES:
            terminated_turns.add(fact.correlation_id)

    interrupted = started_turns - terminated_turns
    if not interrupted:
        return 0

    # Deterministic order so repeated recovery of the same state is stable.
    recovery_facts = [
        NewJournalFact(
            fact_type=FactType.TURN_INTERRUPTED,
            correlation_id=corr_id,
            causation_id=None,
            payload={"recovery": "crash_recovery"},
        )
        for corr_id in sorted(interrupted)
    ]
    await repository.append(scope, session_id, current_version, recovery_facts)
    logger.info(
        "recovered interrupted turns",
        session_id=session_id,
        recovered=len(interrupted),
        correlation_ids=sorted(interrupted),
    )
    return len(interrupted)


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------


def _canonical_json(entries: Sequence[TimelineEntry]) -> str:
    """Canonical representation for source fingerprinting, excluding volatile timestamps.

    ``to_dict()`` embeds ``timestamp``, which for programmatic entries without an
    explicit timestamp differs on each construction and would defeat idempotency.
    We strip ``timestamp`` so the fingerprint is stable regardless of how the
    entries were constructed. ``sort_keys=True`` gives a stable key order;
    ``ensure_ascii=False`` keeps non-ASCII text readable; ``default=str`` is a
    safety net for any value ``TimelineEntry.to_dict`` did not already sanitize.
    """
    stripped = []
    for e in entries:
        d = e.to_dict()
        d.pop("timestamp", None)  # Exclude volatile field for stable fingerprinting
        stripped.append(d)
    return json.dumps(stripped, sort_keys=True, ensure_ascii=False, default=str)


def _source_hash(entries: Sequence[TimelineEntry]) -> str:
    return hashlib.sha256(_canonical_json(entries).encode()).hexdigest()


def _text_of(content: object) -> str:
    """Safely coerce legacy entry content to a journal text payload.

    ``None`` (corrupt/missing content) becomes an empty string rather than the
    literal ``"None"``; everything else is stringified so unexpected types
    cannot crash migration.
    """
    if content is None:
        return ""
    return str(content)


async def migrate_legacy_timeline(
    repository: JournalRepository,
    scope: OwnerScope,
    session_id: str,
    source_entries: Sequence[TimelineEntry],
) -> MigrationResult:
    """Migrate legacy timeline entries into a session journal.

    Idempotent: a content-addressed source hash is recorded on a
    ``LEGACY_TIMELINE_MIGRATED`` marker fact; re-migrating the identical source
    is detected and returns ``already_migrated=True`` with nothing appended.

    Only text entries are converted for D1:

    - ``USER_MESSAGE`` -> ``USER_CONTENT_FINAL`` (starts a legacy turn).
    - ``AGENT_RESPONSE`` -> ``ASSISTANT_CONTENT_FINAL`` + ``TURN_COMPLETED``
      (a committed turn).

    Non-text entries (thoughts, tools, summaries, ephemeral context, etc.) are
    skipped and counted in ``skipped_entries``.

    The session MUST already exist in ``scope``; this function never creates
    one. Raises :class:`~dana.core.session.journal.models.SessionNotFound`
    otherwise.
    """
    source_hash = _source_hash(source_entries)

    facts = await repository.read_facts(scope, session_id)

    # Idempotency: a prior migration of the identical source is a no-op.
    for fact in facts:
        if fact.fact_type is FactType.LEGACY_TIMELINE_MIGRATED and fact.payload.get("source_hash") == source_hash:
            logger.info("legacy timeline already migrated", session_id=session_id, source_hash=source_hash)
            return MigrationResult(already_migrated=True, appended=0, source_hash=source_hash, skipped_entries=0)

    new_facts: list[NewJournalFact] = []
    skipped = 0
    turn_counter = 0

    for entry in source_entries:
        if entry.entry_type is TimelineEntryType.USER_MESSAGE:
            turn_counter += 1
            corr_id = f"legacy-turn-{turn_counter}"
            new_facts.append(
                NewJournalFact(
                    fact_type=FactType.USER_CONTENT_FINAL,
                    correlation_id=corr_id,
                    causation_id=None,
                    payload={"text": _text_of(entry.content)},
                )
            )
        elif entry.entry_type is TimelineEntryType.AGENT_RESPONSE:
            # Pairs with the most recent user message's turn (same correlation_id)
            # so the Conversation projector promotes this to a committed message.
            corr_id = f"legacy-turn-{turn_counter}"
            new_facts.append(
                NewJournalFact(
                    fact_type=FactType.ASSISTANT_CONTENT_FINAL,
                    correlation_id=corr_id,
                    causation_id=corr_id,
                    payload={"text": _text_of(entry.content)},
                )
            )
            new_facts.append(
                NewJournalFact(
                    fact_type=FactType.TURN_COMPLETED,
                    correlation_id=corr_id,
                    causation_id=corr_id,
                    payload={},
                )
            )
        else:
            # D1 is text-only: thoughts, tools, summaries, context, learnings,
            # todos are deferred to later phases.
            skipped += 1

    # Record the content-addressed migration marker so re-runs are idempotent.
    new_facts.append(
        NewJournalFact(
            fact_type=FactType.LEGACY_TIMELINE_MIGRATED,
            correlation_id=f"migration-{source_hash[:8]}",
            causation_id=None,
            payload={"source_hash": source_hash, "source_count": len(source_entries)},
        )
    )

    current_version = max((f.sequence for f in facts), default=0)
    await repository.append(scope, session_id, current_version, new_facts)
    logger.info(
        "migrated legacy timeline",
        session_id=session_id,
        source_hash=source_hash,
        appended=len(new_facts),
        skipped=skipped,
    )

    return MigrationResult(
        already_migrated=False,
        appended=len(new_facts),
        source_hash=source_hash,
        skipped_entries=skipped,
    )


# ---------------------------------------------------------------------------
# Compatibility projection (journal -> legacy Timeline)
# ---------------------------------------------------------------------------


def journal_facts_to_timeline_entries(facts: Sequence[JournalFact]) -> list[TimelineEntry]:
    """Project journal facts back to TimelineEntry format for legacy compatibility.

    Used behind the ``DANA_SESSION_JOURNAL_AUTHORITY=0`` rollback flag to
    regenerate a compatibility Timeline from journal facts so legacy readers
    keep working after the journal becomes the sole authority.

    Only the text-bearing facts round-trip; chunks, terminal facts, lifecycle
    facts, and the migration marker are intentionally skipped.

    .. note::

        The compatibility projection includes all assistant finals regardless of
        terminal status, which differs from ``ConversationProjector``'s
        committed-turn gating. This is intentional for the rollback projection
        — partial output should remain visible in the legacy Timeline format.
    """
    entries: list[TimelineEntry] = []
    for fact in facts:
        if fact.fact_type is FactType.USER_CONTENT_FINAL:
            entries.append(
                TimelineEntry(
                    entry_type=TimelineEntryType.USER_MESSAGE,
                    content=str(fact.payload.get("text", "")),
                    timestamp=fact.timestamp,
                )
            )
        elif fact.fact_type is FactType.ASSISTANT_CONTENT_FINAL:
            entries.append(
                TimelineEntry(
                    entry_type=TimelineEntryType.AGENT_RESPONSE,
                    content=str(fact.payload.get("text", "")),
                    timestamp=fact.timestamp,
                )
            )
        # All other fact types (chunks, terminals, lifecycle, migration marker)
        # have no legacy text representation and are skipped.
    return entries
