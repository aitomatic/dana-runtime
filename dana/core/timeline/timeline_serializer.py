"""
Timeline serializer mixin for CompressedTimeline.

Provides TimelineSerializerMixin with repository-agnostic persistence for
CompressedTimeline: read_since, save, load_from_entries, and supporting
private helpers. All persistence goes through the repository protocol —
no direct file I/O, no `open()`, no `Path(...)`, no `glob`, no access to
any repository private attribute.

Compaction model (GH-1):
  Until the first compaction fires, `save(session_id)` writes to the
  caller-supplied base session id. When `_apply_compression` stamps
  `_last_compression_at` to a new timestamp, the next `save()` mints a
  fresh logical session id of the form `{base}__compact__{YYYYMMDDTHHMMSS}`
  and redirects writes there. Subsequent saves keep updating the same
  compact session until the next compaction rolls forward. Full audit
  retention — old compact sessions remain stored.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from structlog import get_logger

from dana.core.timeline.native_message import (
    COMPRESSED_CONTEXT_KEY,
    COMPRESSION_TIMESTAMP_KEY,
    NATIVE_SOURCE_ENTRY_KEY,
    NativeMessage,
)
from dana.core.timeline.timeline import TimelineEntry


if TYPE_CHECKING:
    from dana.core.timeline.compressed_timeline import CompressedTimeline

logger = get_logger()

_COMPACT_TOKEN = "__compact__"
# Microsecond precision prevents session-id collisions when two compactions
# fire within the same wall-clock second (stamping `_last_compression_at` from
# tests or from a fast LLM path). Collision would cause `repo.save` to
# overwrite the earlier compact session, breaking full audit retention.
_COMPACT_TS_FORMAT = "%Y%m%dT%H%M%S_%f"


class TimelineSerializerMixin:
    """
    Mixin providing repository-agnostic persistence logic for CompressedTimeline.

    Expects the following attributes on self (provided by CompressedTimeline):
        _repository: repository instance (TimelineRepositoryProtocol) or None
        _agent: BaseAgent or None
        _native_messages: list[NativeMessage]
        timeline: list[TimelineEntry]
        _last_compression_at: datetime | None
        _active_compact_session_id: str | None
        _active_compact_compression_at: datetime | None
        _timeline_entry_to_native_message: callable
        _native_message_to_timeline_entry: callable
    """

    # ------------------------------------------------------------------
    # Compact-session id helpers (pure-string, no I/O)
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_compact_suffix(session_id: str) -> str:
        """Return base session_id by removing any ``__compact__<ts>`` suffix."""
        idx = session_id.find(_COMPACT_TOKEN)
        return session_id[:idx] if idx >= 0 else session_id

    @staticmethod
    def _parse_ts_from_compact_id(session_id: str) -> datetime | None:
        """Extract datetime from ``{base}__compact__{YYYYMMDDTHHMMSS}``.

        Returns ``None`` on a plain base id or malformed timestamp.
        """
        idx = session_id.rfind(_COMPACT_TOKEN)
        if idx < 0:
            return None
        raw = session_id[idx + len(_COMPACT_TOKEN) :]
        try:
            return datetime.strptime(raw, _COMPACT_TS_FORMAT)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # read_since + resume rehydration
    # ------------------------------------------------------------------

    def read_since(self: CompressedTimeline, checkpoint: int) -> Iterator[TimelineEntry]:
        """
        Read timeline entries since checkpoint, with compression-aware loading.

        Discovers the latest compact session via ``repo.list_sessions`` first,
        then redirects the read through that session id (so post-compaction
        state is returned, not the base-session history). Subsequent saves
        continue writing to the discovered compact session until the next
        compaction rolls forward. Native messages are recomputed from entries
        (no longer persisted).

        Args:
            checkpoint: Starting index for reading entries

        Yields:
            TimelineEntry objects since checkpoint
        """
        # Step 1: discover latest compact session (mutates active-session state).
        self._rehydrate_active_compact_session()

        # Step 2: read from compact session if discovered, else delegate to base.
        if self._active_compact_session_id is not None:
            all_entries = self._read_from_compact_session(checkpoint)
        else:
            all_entries = list(super().read_since(checkpoint))  # type: ignore[misc]

        # Compression-aware cutoff: drop entries older than the newest
        # compressed-context marker, if any.
        cutoff_idx = self._cut_at_compression_marker(all_entries, lambda entry: COMPRESSED_CONTEXT_KEY in entry.metadata)
        result_entries = all_entries[cutoff_idx:]

        # Native messages: recompute from entries — not persisted anymore.
        # Install through the accounting seam so the incremental token total
        # and compaction generation track the rehydrated projection. The
        # aliasing-risk flag is cleared only if no installed entry still shares
        # caller-mutable content or tool arguments with the repository read.
        self._install_built_projection(result_entries)

        yield from result_entries

    def _read_from_compact_session(self: CompressedTimeline, checkpoint: int) -> list[TimelineEntry]:
        """Read entries from the active compact session with checkpoint slicing.

        Mirrors the base ``Timeline.read_since`` semantics (negative-checkpoint
        handling) but targets ``_active_compact_session_id`` instead of the
        agent's base session id.
        """
        assert self._repository is not None
        assert self._active_compact_session_id is not None
        all_entries = list(self._repository.read_session_entries(self._active_compact_session_id))
        if checkpoint < 0:
            checkpoint = max(0, len(all_entries) + checkpoint)
        return all_entries[checkpoint:]

    def _rehydrate_active_compact_session(self: CompressedTimeline) -> None:
        """Discover the latest compacted session for this agent, if any, and
        adopt it as the active write target. No-op when the repo is absent,
        returns an empty list, or raises. Silent fallback is intentional —
        external repos without ``list_sessions`` support (via default mixin)
        simply stay on the base session id.
        """
        if self._repository is None or self._agent is None:
            return
        session_id = getattr(self._agent, "_session_id", None)
        if not session_id:
            return
        base = self._strip_compact_suffix(session_id)
        try:
            candidates = self._repository.list_sessions(prefix=f"{base}{_COMPACT_TOKEN}")
        except Exception as e:
            logger.warning("list_sessions_failed", error=str(e))
            return
        if not candidates:
            return
        latest = sorted(candidates)[-1]  # ISO timestamps sort lexicographically
        self._active_compact_session_id = latest
        self._active_compact_compression_at = self._parse_ts_from_compact_id(latest)
        logger.info("compact_session_adopted", session_id=latest)

    def rehydrate(self: CompressedTimeline) -> None:
        """Reload this timeline from the agent's current session on disk.

        Thin wrapper over ``read_since``: the entries are assigned and the
        native projection + token accounting re-anchored by ``read_since``
        itself (as part of the same projection swap). Compaction-snapshot aware.

        No-op for in-memory timelines (no repository or no agent) — ``read_since``
        would otherwise raise ``ValueError``. A session id that was never saved
        naturally yields an empty timeline (no error).
        """
        if self._repository is None or self._agent is None:
            return
        # Consume the generator; read_since installs the loaded projection.
        for _ in self.read_since(0):
            pass

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save(self: CompressedTimeline, session_id: str) -> None:
        """
        Save timeline entries through the repository protocol.

        Mints a new compact session id whenever a compaction has fired since
        the last save. Pre-compaction writes go to the caller-supplied base
        session id. No direct file I/O.

        Ephemeral entries (e.g. CONTEXT) are excluded from persistence.
        Native messages are NOT persisted — they are recomputed from entries
        on load. For a journal-managed session the opaque provider replay
        carriers are withheld from the snapshot (``_snapshot_entries``).

        Args:
            session_id: Caller-supplied session identifier. If it already
                contains the ``__compact__`` token (e.g. resumed from a
                previously-compacted id), it is tolerantly stripped to the
                base before deriving a fresh compact id.
        """
        if self._repository is None:
            raise ValueError("Cannot save timeline: repository is None. Initialize Timeline with repository or agent.")

        base_session_id = self._strip_compact_suffix(session_id)
        if base_session_id != session_id:
            logger.warning(
                "session_id_contained_compact_token",
                original=session_id,
                base=base_session_id,
            )

        # Roll to a new compact session if a compaction has fired since the
        # last save.
        if self._last_compression_at is not None and self._last_compression_at != self._active_compact_compression_at:
            ts = self._last_compression_at.strftime(_COMPACT_TS_FORMAT)
            self._active_compact_session_id = f"{base_session_id}{_COMPACT_TOKEN}{ts}"
            self._active_compact_compression_at = self._last_compression_at
            logger.info(
                "compact_session_rolled",
                session_id=self._active_compact_session_id,
                compression_at=self._last_compression_at.isoformat(),
            )

        target = self._active_compact_session_id or base_session_id

        persistent_entries = [e for e in self.timeline if not e.ephemeral]
        self._repository.save(target, self._snapshot_entries(persistent_entries))

        logger.info(
            "compressed_timeline_saved",
            session_id=target,
            entries=len(persistent_entries),
            is_compact=(self._active_compact_session_id is not None),
        )

    def _snapshot_entries(self: CompressedTimeline, entries: list[TimelineEntry]) -> list[TimelineEntry]:
        """Return the entries to hand the repository for this save.

        For a journal-managed session the opaque provider replay carriers
        (``reasoning_items``, which can carry the provider's ``encrypted_content``
        blob) are withheld from the snapshot: the journal persists that payload
        envelope-encrypted, so writing the same bytes into this plaintext
        snapshot would defeat the journal's encryption boundary. Every other
        metadata key — fingerprint, response id, compressed context — is kept,
        as is every entry without carriers.

        Entries with carriers are copied, never mutated: in-flight replay, the
        compaction marker bookkeeping and the session's protected-payload build
        all read the live metadata. Agents without the journal marker (the plain
        legacy API) are passed through untouched.
        """
        agent = self._agent
        if agent is None or not getattr(agent, "_journal_owned_protected_replay", False):
            return entries

        snapshot: list[TimelineEntry] = []
        for entry in entries:
            metadata = entry.metadata
            if isinstance(metadata, dict) and metadata.get("reasoning_items"):
                snapshot.append(replace(entry, metadata={key: value for key, value in metadata.items() if key != "reasoning_items"}))
            else:
                snapshot.append(entry)
        return snapshot

    # ------------------------------------------------------------------
    # load_from_entries (repo-free; works on caller-supplied entries)
    # ------------------------------------------------------------------

    def load_from_entries(
        self: CompressedTimeline,
        entries: list[TimelineEntry] | list[dict[str, Any]],
        native_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Load timeline from entries, supporting both legacy and native message formats.

        This method does not access the repository — it operates on
        caller-supplied entries only. Native messages are accepted for
        backward compatibility with callers that used to persist them;
        the serializer no longer persists them itself.

        Loading is a full replacement of the model-facing projection: the
        timeline entries and the native messages derived from them are
        re-established from the supplied data, never carried over from the
        projection being replaced, and a compaction target that belongs to a
        different session is dropped (see ``_clear_stale_compact_target``) so the
        next ``save`` cannot write this session's history into another session's
        snapshot.

        Args:
            entries: List of TimelineEntry objects or dicts (legacy format)
            native_messages: Optional list of native message dicts
        """
        self._clear_stale_compact_target()

        if not entries and not native_messages:
            self._install_built_projection([])
            return

        first_entry = entries[0] if entries else None

        if isinstance(first_entry, dict):
            if "role" in first_entry and "type" not in first_entry:
                # Native format - load as NativeMessage directly
                self._load_from_native_format(entries)  # type: ignore[arg-type]
                return

        # Legacy format: normalize dicts to TimelineEntry
        timeline_entries: list[TimelineEntry] = []
        for entry in entries:
            if isinstance(entry, dict):
                timeline_entries.append(TimelineEntry.from_dict(entry))
            else:
                timeline_entries.append(entry)  # type: ignore[arg-type]

        if native_messages:
            self._load_timeline_entries_legacy(timeline_entries)
            # The supplied native list is the projection of the same journal
            # that produced ``entries``, so it is cut at the compression marker
            # by the same rule. Keeping pre-marker messages here would hand the
            # provider the compressed-away history alongside the summary.
            native = [NativeMessage.from_dict(msg) for msg in native_messages]
            cut = self._native_compression_cutoff(native)
            retained = native[cut:]
            # Messages decoded from dicts still share any mutable value those
            # dicts held (e.g. a multimodal content list), so this install does
            # not clear the aliasing-risk flag.
            self._install_model_state(self.timeline, retained)
            # The supplied native messages are the authoritative model-facing
            # projection for this load, so tag the ones matching live entries and
            # leave the rest untagged: a later compaction keeps those objects
            # (preserving caller edits and replay metadata) and rebuilds only the
            # entries the supplied list does not cover.
            self._tag_native_mirrors(self.timeline, retained)
            logger.info(
                f"Loaded {len(self.timeline)} timeline entries with {len(self._native_messages)} native messages from separate storage"
                f"{f' (skipped {cut} older messages)' if cut else ''}"
            )
        else:
            self._load_timeline_entries_legacy(timeline_entries)
            self._install_built_projection(self.timeline)
            logger.info(f"Loaded and converted {len(self.timeline)} legacy timeline entries to native format")

    def _clear_stale_compact_target(self: CompressedTimeline) -> None:
        """Drop compact-session bookkeeping that belongs to another session.

        A load replaces the projection this instance models. If the active
        compact write target was adopted for a different session, keeping it
        means the next ``save`` writes the newly loaded session's history into
        the previous session's snapshot. Same-session bookkeeping (a compaction
        that fired on this instance and has a pending roll) is left alone so the
        compact-roll semantics of ``save`` are unchanged.
        """
        agent = self._agent
        session_id = getattr(agent, "_session_id", None) if agent is not None else None
        if not session_id or self._active_compact_session_id is None:
            return
        if self._strip_compact_suffix(self._active_compact_session_id) == self._strip_compact_suffix(session_id):
            return
        logger.info(
            "stale_compact_target_cleared",
            stale_session_id=self._active_compact_session_id,
            session_id=session_id,
        )
        self._last_compression_at = None
        self._active_compact_session_id = None
        self._active_compact_compression_at = None

    def _load_timeline_entries_legacy(self: CompressedTimeline, entries: list[TimelineEntry]) -> None:
        """Load timeline entries with compression-aware cutoff."""
        if not entries:
            self.timeline = []
            return

        cutoff = self._cut_at_compression_marker(entries, lambda entry: COMPRESSED_CONTEXT_KEY in entry.metadata)
        self.timeline = entries[cutoff:]

        if cutoff:
            logger.info(f"Loaded {len(self.timeline)} entries with compressed context (skipped {cutoff} older entries)")
        else:
            logger.info(f"Loaded all {len(self.timeline)} entries (no compressed context found)")

    @staticmethod
    def _cut_at_compression_marker(items: list[Any], has_marker: Callable[[Any], bool]) -> int:
        """Return the index of the newest item carrying the compression marker.

        ``_apply_compression`` stamps the marker on the oldest retained item, so
        everything before it is compressed-away history: it must not be handed
        to the model again, in either representation (entries or native
        messages). Returns 0 when no item carries a marker (nothing was
        compacted), which keeps the whole list.
        """
        for index in range(len(items) - 1, -1, -1):
            if has_marker(items[index]):
                return index
        return 0

    def _native_compression_cutoff(self: CompressedTimeline, messages: list[NativeMessage]) -> int:
        """Keep the explicit summary and retained messages from the newest compaction."""
        cut = self._cut_at_compression_marker(messages, lambda msg: COMPRESSED_CONTEXT_KEY in msg.metadata)
        if cut:
            summary, retained = messages[cut - 1], messages[cut]
            stamp = retained.metadata.get(COMPRESSION_TIMESTAMP_KEY)
            if (
                summary.role == "system"
                and stamp is not None
                and summary.metadata.get(COMPRESSION_TIMESTAMP_KEY) == stamp
                and summary.metadata.get(COMPRESSED_CONTEXT_KEY) == retained.metadata[COMPRESSED_CONTEXT_KEY]
            ):
                cut -= 1
        return cut

    def _tag_native_mirrors(self: CompressedTimeline, entries: list[TimelineEntry], messages: list[NativeMessage]) -> int:
        """Tag each native message with the entry at the same index.

        A hydration path that receives both representations decodes them
        independently, so the pairing has to be re-established before a later
        compaction can keep those message objects (preserving caller edits made
        through the mutable public API, and their reasoning/replay metadata)
        instead of rebuilding a fresh projection from the entries.

        The two lists are the same projection, so with a whole-projection load they
        are index-aligned — except that a compaction prepends a summary message,
        which has no entry. The alignment is therefore established from the tail
        (the offset is at most the extra summary cohort), giving a single linear
        pass with no repeated search. Decoded message objects are this timeline's
        own (``NativeMessage.from_dict`` builds fresh instances), so no value
        guessing is needed.

        Returns:
            Number of messages tagged.
        """
        offset = len(messages) - len(entries)
        tagged = 0
        for index, entry in enumerate(entries):
            message_index = index + offset
            if not 0 <= message_index < len(messages):
                continue
            message = messages[message_index]
            metadata = message.metadata
            if metadata is None:
                metadata = message.metadata = {}
            metadata[NATIVE_SOURCE_ENTRY_KEY] = entry
            tagged += 1
        return tagged

    def _load_from_native_format(self: CompressedTimeline, native_data: list[dict[str, Any]]) -> None:
        """Load timeline from native message format (dicts with 'role' key)."""
        native = [NativeMessage.from_dict(msg_dict) for msg_dict in native_data]
        cutoff = self._native_compression_cutoff(native)
        retained = native[cutoff:]
        # Reconstruct TimelineEntry for backward compatibility, then install.
        # The entries are rebuilt from the decoded messages, so this install owns
        # them; the decoded native messages themselves may still share mutable
        # values held by the input dicts, so the aliasing-risk flag is not cleared
        # by this path.
        entries = [self._native_message_to_timeline_entry(msg) for msg in retained]
        # Tag each message with the entry it mirrors, exactly as the in-memory
        # builder does, so a later compaction keeps these very message objects
        # (with any edit a caller made through the mutable public API) instead of
        # rebuilding a fresh projection from the entries.
        for message, entry in zip(retained, entries, strict=True):
            # Each message was decoded from the same payload as its entry, and a
            # compaction's extra summary message is already dropped by the cutoff
            # above, so this pairing is exact and needs no value comparison.
            message.metadata[NATIVE_SOURCE_ENTRY_KEY] = entry
        self._install_model_state(entries, retained)

        if cutoff:
            logger.info(f"Loaded {len(self._native_messages)} native messages with compressed context (skipped {cutoff} older messages)")
        else:
            logger.info(f"Loaded all {len(self._native_messages)} native messages (no compressed context found)")

    def _native_message_to_timeline_entry(self: CompressedTimeline, msg: NativeMessage) -> TimelineEntry:
        """Convert NativeMessage back to TimelineEntry for backward compatibility."""
        from dana.core.timeline.timeline import TimelineEntryType

        entry_type: TimelineEntryType
        tool_calls: list[dict[str, Any]] | None = None
        tool_call_id: str | None = msg.tool_call_id

        if msg.role == "user":
            entry_type = TimelineEntryType.USER_MESSAGE
        elif msg.role == "system":
            if isinstance(msg.content, str) and (msg.content.startswith("[SUMMARY]") or COMPRESSED_CONTEXT_KEY in msg.metadata):
                entry_type = TimelineEntryType.TIMELINE_SUMMARY
            else:
                entry_type = TimelineEntryType.CONTEXT
        elif msg.role == "tool":
            entry_type = TimelineEntryType.RESOURCE_RESULT
        elif msg.role == "assistant":
            if msg.tool_calls:
                entry_type = TimelineEntryType.TOOL_CALL
                tool_calls = [tc.to_dict() for tc in msg.tool_calls]
            else:
                entry_type = TimelineEntryType.AGENT_RESPONSE
        else:
            entry_type = TimelineEntryType.AGENT_RESPONSE

        entry_metadata = {key: value for key, value in msg.metadata.items() if key != NATIVE_SOURCE_ENTRY_KEY}
        return TimelineEntry(
            entry_type=entry_type,
            content=msg.content,
            timestamp=msg.timestamp,
            metadata=entry_metadata,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
        )
