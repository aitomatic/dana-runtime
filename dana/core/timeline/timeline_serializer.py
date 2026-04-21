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

from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from structlog import get_logger

from dana.core.timeline.native_message import (
    COMPRESSED_CONTEXT_KEY,
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
        cutoff_idx = 0
        for i, entry in enumerate(reversed(all_entries)):
            if COMPRESSED_CONTEXT_KEY in entry.metadata:
                cutoff_idx = len(all_entries) - i - 1
                break

        result_entries = all_entries[cutoff_idx:]

        # Native messages: recompute from entries — not persisted anymore.
        self._native_messages = [self._timeline_entry_to_native_message(e) for e in result_entries]

        for entry in result_entries:
            yield entry

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
        on load.

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
        self._repository.save(target, persistent_entries)

        logger.info(
            "compressed_timeline_saved",
            session_id=target,
            entries=len(persistent_entries),
            is_compact=(self._active_compact_session_id is not None),
        )

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

        Args:
            entries: List of TimelineEntry objects or dicts (legacy format)
            native_messages: Optional list of native message dicts
        """
        if not entries and not native_messages:
            self.timeline = []
            self._native_messages = []
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
            self._native_messages = [NativeMessage.from_dict(msg) for msg in native_messages]
            logger.info(
                f"Loaded {len(self.timeline)} timeline entries with {len(self._native_messages)} native messages from separate storage"
            )
        else:
            self._load_timeline_entries_legacy(timeline_entries)
            self._native_messages = [self._timeline_entry_to_native_message(entry) for entry in self.timeline]
            logger.info(f"Loaded and converted {len(self.timeline)} legacy timeline entries to native format")

    def _load_timeline_entries_legacy(self: CompressedTimeline, entries: list[TimelineEntry]) -> None:
        """Load timeline entries with compression-aware cutoff."""
        if not entries:
            self.timeline = []
            return

        entries_to_load: list[TimelineEntry] = []
        found_compressed = False

        for entry in reversed(entries):
            entries_to_load.insert(0, entry)
            if COMPRESSED_CONTEXT_KEY in entry.metadata:
                found_compressed = True
                break

        if found_compressed:
            self.timeline = entries_to_load
            logger.info(
                f"Loaded {len(entries_to_load)} entries with compressed context "
                f"(skipped {len(entries) - len(entries_to_load)} older entries)"
            )
        else:
            self.timeline = entries
            logger.info(f"Loaded all {len(entries)} entries (no compressed context found)")

    def _load_from_native_format(self: CompressedTimeline, native_data: list[dict[str, Any]]) -> None:
        """Load timeline from native message format (dicts with 'role' key)."""
        self._native_messages = []
        entries_to_load: list[NativeMessage] = []
        found_compressed = False

        for msg_dict in reversed(native_data):
            msg = NativeMessage.from_dict(msg_dict)
            entries_to_load.insert(0, msg)
            if COMPRESSED_CONTEXT_KEY in msg.metadata:
                found_compressed = True
                break

        if found_compressed:
            self._native_messages = entries_to_load
            logger.info(
                f"Loaded {len(entries_to_load)} native messages with compressed context "
                f"(skipped {len(native_data) - len(entries_to_load)} older messages)"
            )
        else:
            self._native_messages = [NativeMessage.from_dict(d) for d in native_data]
            logger.info(f"Loaded all {len(native_data)} native messages (no compressed context found)")

        # Reconstruct TimelineEntry for backward compatibility
        self.timeline = [self._native_message_to_timeline_entry(msg) for msg in self._native_messages]

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

        return TimelineEntry(
            entry_type=entry_type,
            content=msg.content,
            timestamp=msg.timestamp,
            metadata=msg.metadata.copy(),
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
        )
