"""
Compressed Timeline with intelligent context management.

This module provides a Timeline subclass that implements progressive compression
using LLM-based summarization, similar to Claude Code's compression technique.

Key features:
- Token-based compression triggers
- Stores compressed context in entry metadata for efficient loading
- Progressive compression that preserves recent context
- Native message format for provider-agnostic LLM integration
"""

from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import dataclass
from datetime import datetime
import json
from typing import TYPE_CHECKING, Any

from structlog import get_logger

from dana.common.llm.types import LLMMessage
from dana.core.timeline.compact_trigger import resolve_trigger_tokens
from dana.core.timeline.compression_engine import CompressionMixin
from dana.core.timeline.native_message import (
    COMPRESSED_CONTEXT_KEY,
    COMPRESSED_ENTRIES_COUNT_KEY,
    COMPRESSION_TIMESTAMP_KEY,
    NATIVE_SOURCE_ENTRY_KEY,
    NativeMessage,
    NativeMessageRole,
    NativeToolCall,
)
from dana.core.timeline.timeline import (
    Timeline,
    TimelineConfig,
    TimelineEntry,
    TimelineEntryType,
)
from dana.core.timeline.timeline_serializer import TimelineSerializerMixin
from dana.core.timeline.token_limiting_helpers import (
    apply_token_limit_to_messages,
    estimate_messages_tokens,
)
from dana.repositories.repository_factory import DEFAULT_REPOSITORY_FACTORY, RepositoryFactory


if TYPE_CHECKING:
    from dana.core.agent.base_agent import BaseAgent

logger = get_logger()

# Sentinel recorded as a message's dependency snapshot when a shared payload
# cannot be copied for later comparison. Such a message is re-estimated on every
# read (bounded to that message), which is exact and never a whole-history scan.
_UNSNAPSHOTABLE = object()

# Re-export type alias so existing imports still work
# (e.g. `from dana.core.agent.compressed_timeline import NativeMessageRole`)
__all__ = [
    "CompressedTimeline",
    "CompressedTimelineConfig",
    "NativeMessage",
    "NativeMessageRole",
    "NativeToolCall",
    "COMPRESSED_CONTEXT_KEY",
    "COMPRESSION_TIMESTAMP_KEY",
    "COMPRESSED_ENTRIES_COUNT_KEY",
    "CompressionMixin",
    "TimelineSerializerMixin",
]


def _same_token_dependency(live: Any, snapshot: Any) -> bool:
    """Compare counted payloads without conflating equal, differently rendered values."""
    if live is snapshot:
        return True
    if type(live) is not type(snapshot):
        return False
    if isinstance(live, dict):
        return len(live) == len(snapshot) and all(
            _same_token_dependency(key, old_key) and _same_token_dependency(value, old_value)
            for (key, value), (old_key, old_value) in zip(live.items(), snapshot.items(), strict=True)
        )
    if isinstance(live, (list, tuple)):
        return len(live) == len(snapshot) and all(
            _same_token_dependency(value, old_value) for value, old_value in zip(live, snapshot, strict=True)
        )
    if isinstance(live, (str, bytes, int, bool, type(None))):
        return live == snapshot
    return repr(live) == repr(snapshot)


@dataclass
class CompressedTimelineConfig(TimelineConfig):
    """Configuration for compressed timeline with enhanced context management."""

    # Maximum tokens before compression is triggered
    max_tokens_until_compression: int = 80000

    # Maximum number of recent entries to keep uncompressed
    max_recent_entries_to_keep: int = 20

    # Token cutoff for recent entries (default 30% of max_tokens_until_compression)
    # Set to 0 or None to use the default calculation
    cutoff_when_token_reach: int | None = None

    # Phase 2 (P6) — cheap shrink: stub old tool_result content before full
    # summary. Off by default (opt-in).
    enable_cheap_shrink_tool_results: bool = False

    # Number of most-recent entries to exclude from shrink eligibility.
    cheap_shrink_keep_recent: int = 10

    # Phase 3 (P2) — reactive compaction kill switch.
    enable_reactive_compact: bool = True

    def __post_init__(self) -> None:
        """Calculate default cutoff if not specified."""
        if self.cutoff_when_token_reach is None or self.cutoff_when_token_reach == 0:
            self.cutoff_when_token_reach = int(0.3 * self.max_tokens_until_compression)


class CompressedTimeline(CompressionMixin, TimelineSerializerMixin, Timeline):
    """
    Timeline with intelligent compression and context management.

    This Timeline subclass implements the TimelineProtocol interface and provides
    progressive compression using LLM-based summarization. It tracks token usage
    and compresses older entries when thresholds are reached, storing the compressed
    context in metadata for efficient session loading.

    Compression triggers when:
    - Total tokens exceed max_tokens_until_compression, OR
    - Number of entries exceeds a threshold

    Compression behavior:
    1. Iterate through entries from latest to oldest
    2. Count tokens until reaching cutoff_when_token_reach OR max_recent_entries_to_keep
    3. Compress all older entries using LLM summarization
    4. Store compressed context in metadata of the oldest kept entry
    5. Remove compressed entries from the timeline

    Loading optimization:
    - When loading entries, check for compressed context in metadata
    - If found, load the compressed context and skip older entries
    """

    def __init__(
        self,
        max_tokens_until_compression: int | None = None,
        max_recent_entries_to_keep: int = 20,
        cutoff_when_token_reach: int | None = None,
        agent: BaseAgent | None = None,
        repository_factory: RepositoryFactory = DEFAULT_REPOSITORY_FACTORY,
        llm_call_fn: Callable[[str], str] | None = None,
        llm_call_async_fn: Callable[[str], Any] | None = None,
        compression_enabled: bool = True,
        system_tokens_fn: Callable[[], int] | None = None,
        tools_tokens_fn: Callable[[], int] | None = None,
        max_context_tokens: int | None = None,
    ):
        """
        Initialize the CompressedTimeline.

        Two independent knobs:
        - ``max_tokens_until_compression`` — the compression TRIGGER. When the
          estimated token count (messages + system + tools) crosses this, the
          next ``needs_compression()`` returns True. If None, resolves from the
          ``DANA_COMPACT_TRIGGER_TOKENS`` env var (default 150k).
        - ``max_context_tokens`` — the LLM context-window BUDGET used by
          ``to_llm_messages()`` for sliding-window token limiting. Independent
          from the trigger. If None, falls back to the resolved trigger (backward
          compat with the pre-split behavior where they were the same number).

        Args:
            max_tokens_until_compression: Compression trigger threshold. None →
                env (DANA_COMPACT_TRIGGER_TOKENS) or default.
            max_recent_entries_to_keep: Maximum number of recent entries to preserve
            cutoff_when_token_reach: Token cutoff for recent entries (default 30% of trigger)
            agent: Agent instance (can be None, for backward compatibility)
            repository_factory: Repository factory to create the repository
            llm_call_fn: Synchronous function to call LLM for compression
            llm_call_async_fn: Async function to call LLM for compression
            compression_enabled: Whether compression is enabled (default True).
                Set to False to disable compression and behave like plain Timeline.
            system_tokens_fn: Optional callback returning system-prompt token estimate.
            tools_tokens_fn: Optional callback returning tools-schema token estimate.
            max_context_tokens: LLM context-window budget for ``to_llm_messages()``.
                None → falls back to the resolved trigger.
        """
        # Resolve trigger: explicit value wins, else env-resolved.
        # _explicit flag preserved so needs_compression() can re-check env at
        # resolve time when the caller deferred.
        explicit_trigger = max_tokens_until_compression
        effective_trigger = explicit_trigger if explicit_trigger is not None else resolve_trigger_tokens()

        # Resolve context-window budget independently. If unspecified, fall back
        # to the trigger to preserve the legacy "one knob does both" behavior
        # for callers that haven't been updated yet.
        effective_context_tokens = max_context_tokens if max_context_tokens is not None else effective_trigger

        # Calculate cutoff if not specified (tied to trigger, not budget —
        # that's the legitimate coupling).
        if cutoff_when_token_reach is None or cutoff_when_token_reach == 0:
            cutoff_when_token_reach = int(0.3 * effective_trigger)

        # Create config — context budget and trigger are now stored separately.
        self._compressed_config = CompressedTimelineConfig(
            max_context_tokens=effective_context_tokens,
            max_tokens_until_compression=effective_trigger,
            max_recent_entries_to_keep=max_recent_entries_to_keep,
            cutoff_when_token_reach=cutoff_when_token_reach,
            compression_enabled=compression_enabled,
        )

        # Track whether threshold was explicitly set so needs_compression()
        # can prefer env at resolve time when caller deferred.
        self._explicit_max_tokens_until_compression: int | None = explicit_trigger

        # Initialize parent — pass the context budget (not the trigger) so
        # to_llm_messages() sliding-window limits use the right number.
        super().__init__(
            max_context_tokens=effective_context_tokens,
            agent=agent,
            repository_factory=repository_factory,
            config=self._compressed_config,
        )

        # Store LLM call functions for compression
        self._llm_call_fn = llm_call_fn
        self._llm_call_async_fn = llm_call_async_fn

        # Optional token-count callbacks folded into needs_compression() estimate.
        self._system_tokens_fn = system_tokens_fn
        self._tools_tokens_fn = tools_tokens_fn

        # Internal storage for native message format
        self._native_messages: list[NativeMessage] = []

        import asyncio
        import threading

        # Guards the two model-facing representations (``timeline`` and
        # ``_native_messages``) and their derived state: the incremental token
        # total and ``_timeline_revision``. Reentrant so compaction, which holds
        # it while composing a commit, can call mutators that take it too.
        self._state_lock: threading.RLock = threading.RLock()

        # Incremental token accounting: the total is updated by exact deltas on
        # append/replacement/shrink, and by one recomputation at hydration and
        # compaction boundaries. ``needs_compression()`` therefore never
        # rescans retained history. The recorded marker (source list identity,
        # length) is what makes a cached total trustworthy; see
        # ``get_native_token_estimate``.
        self._native_token_total: int = 0
        self._native_token_source: list[NativeMessage] | None = None
        self._native_token_len: int = 0

        # Per-message token cache backing the incremental total: one entry per
        # message in the list the total was computed for, as
        # ``(message, dependency_snapshot | None, tokens)``. The snapshot is what a
        # read compares to observe a caller's in-place edit of a shared payload or a
        # rebound counted attribute, so only the messages that actually changed are
        # re-estimated. See ``_refresh_aliased_token_caches``.
        self._token_message_caches: list[tuple[NativeMessage, Any, int]] = []

        # True once a caller has been handed the live message *objects* (the
        # ``native_messages`` getter). A held object lets the caller rebind even a
        # counted attribute that was immutable, so while this is set every cached
        # message carries a dependency snapshot. Distinct from
        # ``_native_aliasing_risk``, which is also set when only the caller's
        # ``add_entry`` payload objects are shared: those are covered by the
        # snapshot on the message that holds them, without widening the check to
        # messages that hold nothing a caller can reach.
        self._message_objects_escaped: bool = False

        # True once the model-facing native state has been handed to a caller that
        # can mutate it in place — through the ``native_messages`` getter, or by
        # handing ``add_entry`` content/tool-argument objects it still owns. Such a
        # change (a same-length content edit, or a nested tool-argument dict edit)
        # cannot be observed from the list or from the message objects, and escaped
        # message objects survive into new lists (compaction, hydration and
        # ``load_from_entries`` reuse the very same ``NativeMessage`` instances).
        # This flag is the conservative read signal used by the install/ownership
        # predicates; the per-message dependency snapshots are what actually observe
        # a caller's edit, so it never forces a re-scan of retained history by itself.
        # It is sticky for the timeline instance and is only cleared by building a
        # projection this timeline rebuilt without sharing caller-mutable state
        # (see ``_install_built_projection``).
        self._native_aliasing_risk: bool = False

        # Bumped whenever the model-facing projection is replaced wholesale
        # (hydration, load, compaction commit). A compaction plan captured before
        # a (possibly awaited) summarizer call is validated against this, so a
        # stale summary cannot commit over a projection that replaced the one it
        # was planned against. In-place mutations (append, context swap, shrink)
        # are caught by the plan's object-identity alignment at commit instead.
        self._compaction_generation: int = 0

        # Phase 2/3 — all mutators of compression state acquire this lock. A
        # threading lock mirrors for sync paths that run off-loop.
        self._compact_lock: asyncio.Lock = asyncio.Lock()
        self._compact_sync_lock: threading.Lock = threading.Lock()

        # Phase 3 — circuit breaker state.
        self._consecutive_compact_failures: int = 0
        self._compaction_disabled: bool = False
        self._circuit_opened_at: datetime | None = None

        # The partition described by the last ``build_compression_prompt``, as
        # ``(entries_to_compress, plan_generation)``. Hosts that own their
        # summarizer call (prompt → LLM → ``compress_old_entries``) commit exactly
        # this partition, so a summary is never applied to entries the summarizer
        # did not see. Consumed (and cleared) by ``compress_old_entries``;
        # invalidated by any projection swap.
        self._pending_compression_plan: tuple[list[TimelineEntry], int] | None = None

        # Compact-session persistence (repo-agnostic, post-GH-1):
        # `_last_compression_at` is stamped by ``_apply_compression``. On the next
        # ``save()``, the serializer detects a new compression event and mints a
        # fresh logical session id of the form ``{base}__compact__{ISO-ts}`` which
        # then receives all writes until the next compaction. Pre-compaction writes
        # go to the caller-supplied base session id. Full audit retention — old
        # compact sessions are never deleted.
        self._last_compression_at: datetime | None = None
        self._active_compact_session_id: str | None = None
        self._active_compact_compression_at: datetime | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_tokens_until_compression(self) -> int:
        """Get max tokens threshold."""
        return self._compressed_config.max_tokens_until_compression

    @property
    def max_recent_entries_to_keep(self) -> int:
        """Get max recent entries to keep."""
        return self._compressed_config.max_recent_entries_to_keep

    @property
    def cutoff_when_token_reach(self) -> int:
        """Get token cutoff for recent entries."""
        return self._compressed_config.cutoff_when_token_reach or int(0.3 * self._compressed_config.max_tokens_until_compression)

    # ------------------------------------------------------------------
    # Token-count callback helper (used by needs_compression)
    # ------------------------------------------------------------------

    def _safe_call_tokens_fn(self, fn: Callable[[], int] | None) -> int:
        """Invoke a tokens callback safely. Returns 0 on None / raise / invalid."""
        if fn is None:
            return 0
        try:
            v = fn()
            return int(v) if v is not None and v >= 0 else 0
        except Exception:
            logger.debug("tokens callback raised; treating as 0", exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Incremental token accounting (single source of truth: _native_token_total)
    # ------------------------------------------------------------------
    #
    # ``_native_messages`` is the representation the provider sees, so its token
    # total is what ``needs_compression()`` compares against the trigger. That
    # total is maintained incrementally:
    #
    #   * appends, in-place content replacements, and caller edits fold exact
    #     per-message deltas (``_fold_token_delta``, ``_refresh_aliased_token_caches``),
    #   * hydration, load, and compaction-replacement install a whole new list
    #     and recompute once (``_install_model_state``).
    #
    # The marker (source list identity + length) catches rebinding and trimming.
    # An edit a caller makes in place is caught per message instead: each cache
    # entry remembers the payload state its token count was computed from, and a
    # read re-checks only the messages that have such a dependency, re-estimating
    # (and folding) exactly those that changed. That covers both supported edit
    # routes — the payload objects ``add_entry`` shares (a multimodal content list,
    # tool-call argument dicts) and a message object taken through the public
    # ``native_messages`` getter — so a tool-using history, the ordinary case,
    # observes an in-place argument edit without re-estimating the retained history.
    # Mutable dependencies require recursive, representation-sensitive comparison.
    # Verification cost scales with retained mutable payload size, even when no
    # message needs a new token estimate.

    def _token_marker_valid(self) -> bool:
        """True when the cached token total still describes the live native list.

        The marker is the cheap structural check — same list object, same length,
        and every cached message still at its own index (see
        ``_refresh_aliased_token_caches``). An escaped projection is *not* part of
        this check: escaped edits are observed per message by the dependency
        snapshots, so escaping never forces a full-history recomputation.
        """
        return self._native_token_source is self._native_messages and self._native_token_len == len(self._native_messages)

    def _anchor_token_marker(self) -> None:
        """Record which native list the cached token total was computed for."""
        self._native_token_source = self._native_messages
        self._native_token_len = len(self._native_messages)

    def _fold_token_delta(self, delta: int, *, marker_was_valid: bool) -> None:
        """Fold an exact token delta from one mutation into the cached total.

        ``marker_was_valid`` must be captured via ``_token_marker_valid()``
        *before* the mutation mutated the list, because the mutation itself may
        invalidate the marker by rebinding the list. When the marker was already
        stale the delta is dropped: the next ``get_native_token_estimate()``
        recomputes from the live list instead of trusting a partial sum.
        """
        if not marker_was_valid:
            return
        self._native_token_total = max(0, self._native_token_total + delta)
        self._anchor_token_marker()

    def _apply_message_token_delta(self, message: NativeMessage, fresh_tokens: int, *, marker_was_valid: bool) -> None:
        """Record that ``message`` now costs ``fresh_tokens``, folding the delta.

        Used by mutators that change one message's counted content in place outside
        the append path (cheap-shrink stubbing). The cache entry is updated in the
        same step as the total, so a later read does not re-detect the same change
        and fold it a second time.
        """
        if not marker_was_valid:
            return
        for index, (cached, _deps, tokens) in enumerate(self._token_message_caches):
            if cached is message:
                self._native_token_total = max(0, self._native_token_total + fresh_tokens - tokens)
                self._token_message_caches[index] = (message, self._capture_message_dependencies(message), fresh_tokens)
                self._anchor_token_marker()
                return
        # Not in the cache (a cold list): fold from zero, the read will reconcile.
        self._native_token_total = max(0, self._native_token_total + fresh_tokens)
        self._anchor_token_marker()

    def _dependency_sources(self, message: NativeMessage) -> list[Any]:
        """The live values a message's token count is computed from.

        Mirrors ``_estimate_native_message_tokens``: the content, and for each tool
        call its id, name and arguments mapping. These are exactly the values a
        caller can change in place on a message the timeline shares with them.
        """
        sources: list[Any] = [message.content]
        if message.tool_calls:
            sources.append(len(message.tool_calls))
            for call in message.tool_calls:
                sources.extend((call.id, call.name, call.arguments))
        return sources

    def _message_needs_dependency_check(self, message: NativeMessage) -> bool:
        """True when ``message`` carries state a check on read must verify.

        A message whose counted payloads are all immutable is skipped: nothing a
        caller does to their own bindings can change it (the fast path for ordinary
        text-only history). A message with a shared mutable payload — the content
        list of a multimodal turn, or tool-call arguments — is checked. Once a caller
        holds the message *objects* themselves, a counted attribute can be rebound,
        so every message is checked then.
        """
        if self._message_objects_escaped:
            return True
        return self._is_caller_owned_value(message.content) or bool(message.tool_calls)

    def _capture_message_dependencies(self, message: NativeMessage) -> Any:
        """Snapshot the dependency state of ``message`` for later comparison.

        A fixed-length tuple mirrors ``_dependency_sources``. Comparisons recurse
        through mutable payloads and therefore scale with their size;
        ``None`` is stored when the message needs no check at all. Each slot
        holds a private copy of a mutable payload (so an in-place edit of the caller's
        object compares unequal) or the value itself for an immutable one (so an
        unchanged string or id compares by identity rather than by text).
        ``_UNSNAPSHOTABLE`` marks a tuple holding at least one payload that could not
        be copied: that message is re-estimated on every read — bounded to it, never
        a whole-history scan.
        """
        if not self._message_needs_dependency_check(message):
            return None
        uncopyable = False
        captured: list[Any] = []
        for value in self._dependency_sources(message):
            if self._is_caller_owned_value(value):
                try:
                    captured.append(copy.deepcopy(value))
                except Exception:
                    logger.debug("message payload could not be snapshotted; re-estimating it per read", exc_info=True)
                    captured.append(value)
                    uncopyable = True
            else:
                captured.append(value)
        return _UNSNAPSHOTABLE if uncopyable else tuple(captured)

    def _record_message_token_cache(self, message: NativeMessage, tokens: int) -> None:
        """Record one message's token count and dependency snapshot."""
        self._token_message_caches.append((message, self._capture_message_dependencies(message), tokens))

    def _mark_projection_escaped(self) -> None:
        """Record that the live message objects have escaped to a caller.

        The public ``native_messages`` getter hands out the message objects
        themselves, so from now on any one of them can have a counted attribute
        rebound — including a ``content`` that was an immutable ``str``. Every
        cached message is re-armed with a dependency snapshot here, once per escape
        event; the per-read cost stays a compare per message, never a re-estimation
        of retained history.
        """
        with self._state_lock:
            self._native_aliasing_risk = True
            if self._message_objects_escaped:
                return
            self._message_objects_escaped = True
            self._token_message_caches = [
                (message, deps if deps is not None else self._capture_message_dependencies(message), tokens)
                for message, deps, tokens in self._token_message_caches
            ]

    def _dependencies_changed(self, message: NativeMessage, deps: Any) -> bool:
        """True when a message's live dependency state no longer matches its snapshot.

        Immutable dependencies use identity when unchanged; mutable payloads use
        type-sensitive comparisons because equal values such as False and 0 have
        different counted representations.
        """
        if deps is _UNSNAPSHOTABLE:
            return True
        if not message.tool_calls:
            # Fast path: content is the only dependency. A ``str`` compares by
            # identity when unchanged, so the ordinary text-only message allocates
            # nothing on read; a shared content list compares by value.
            if len(deps) != 1:
                return True
            snapshot = deps[0]
            return not _same_token_dependency(message.content, snapshot)
        sources = self._dependency_sources(message)
        if len(sources) != len(deps):
            return True
        for live, snapshot in zip(sources, deps, strict=True):
            if live is snapshot:
                continue
            if not _same_token_dependency(live, snapshot):
                return True
        return False

    def _refresh_aliased_token_caches(self) -> None:
        """Fold per-message deltas for the messages a caller changed.

        One pass over the cache: a message still at its own index whose dependencies
        still match is skipped; one that moved, was replaced, or whose shared payload
        was edited is re-estimated on its own and its delta folded. So an edited
        message costs one message estimate, and the retained history is never
        re-estimated as a whole. A text-only history that has never escaped holds no
        dependency snapshots and does no work beyond this pointer pass.
        """
        live = self._native_messages
        for index, (message, deps, tokens) in enumerate(self._token_message_caches):
            live_message = live[index]
            if live_message is message and (deps is None or not self._dependencies_changed(message, deps)):
                continue
            fresh = self._estimate_native_message_tokens(live_message)
            self._native_token_total = max(0, self._native_token_total + fresh - tokens)
            self._token_message_caches[index] = (live_message, self._capture_message_dependencies(live_message), fresh)

    def _recompute_token_total(self) -> None:
        """Recompute the cached token total from the live native list (one scan).

        Records each message's token count and dependency snapshot in the same
        pass, so a later read only checks the messages that actually depend on
        caller-mutable state.
        """
        total = 0
        caches: list[tuple[NativeMessage, Any, int]] = []
        for message in self._native_messages:
            tokens = self._estimate_native_message_tokens(message)
            total += tokens
            caches.append((message, self._capture_message_dependencies(message), tokens))
        self._native_token_total = total
        self._token_message_caches = caches
        self._anchor_token_marker()

    def get_native_token_estimate(self) -> int:
        """Estimated tokens of the retained model working set.

        Returns the incrementally maintained total when its marker still describes
        the live list, folding the deltas of the messages whose shared payload or
        dependency a caller changed since the last read; otherwise recomputes once
        and re-anchors. This is the entry point for the compression trigger: the
        ordinary append-only turn — including a tool-using one — re-estimates at
        most the messages that changed, never the retained history.
        """
        with self._state_lock:
            if not self._token_marker_valid():
                self._recompute_token_total()
            else:
                self._refresh_aliased_token_caches()
            return self._native_token_total

    def _rebase_token_caches(self) -> None:
        """Re-align the per-message cache with ``_native_messages`` after a rebind.

        Used by the mutators that replace the list rather than append to it
        (``set_context``, ``clear_old_entries``). Entries for messages still live
        are kept as they are; only messages the cache does not know yet are
        tokenized. This is an O(len(list)) identity pass with no estimation of the
        retained history, so it does not reintroduce a per-turn recount.
        """
        by_id = {id(message): (message, deps, tokens) for message, deps, tokens in self._token_message_caches}
        rebased: list[tuple[NativeMessage, Any, int]] = []
        for message in self._native_messages:
            cached = by_id.get(id(message))
            if cached is None:
                cached = (message, self._capture_message_dependencies(message), self._estimate_native_message_tokens(message))
            rebased.append(cached)
        self._token_message_caches = rebased
        self._native_token_total = sum(tokens for _message, _deps, tokens in rebased)

    def _install_model_state(self, entries: list[TimelineEntry], messages: list[NativeMessage]) -> None:
        """Replace both model-facing representations as a single projection swap.

        Used by hydration/load and by compaction commit. Both lists are replaced
        together so ``timeline`` and ``_native_messages`` cannot drift apart, the
        token total and the per-message cache are recomputed exactly once, and
        ``_compaction_generation`` is bumped so any compaction plan captured
        against the previous projection refuses to commit over the new one.

        The aliasing-risk flag is set — never cleared — when a message still shares
        caller-mutable state: compaction reuses the very same ``NativeMessage``
        objects it retained, and a caller may hold references to them from the list
        that was just replaced, so any counted attribute of theirs may be rebound.
        Setting it widens the per-read dependency check to every message
        (``_message_needs_dependency_check``); it does not cost a history rescan.
        """
        with self._state_lock:
            self.timeline = entries
            self._native_messages = messages
            # A prompt plan describes a projection that no longer exists.
            self._pending_compression_plan = None
            if self._projection_shares_caller_state(entries, messages):
                self._native_aliasing_risk = True
            self._recompute_token_total()
            self._compaction_generation += 1

    def _install_built_projection(self, entries: list[TimelineEntry]) -> None:
        """Install a projection built here, clearing the aliasing risk if provable.

        The entries are converted to *new* ``NativeMessage`` instances (via
        ``_timeline_entry_to_native_message``) rather than reusing caller-supplied
        message objects, and the token total is recomputed from those instances.
        The aliasing-risk flag is cleared only when nothing installed shares a
        caller-mutable object (see ``_projection_shares_caller_state``) — a plain
        string-content entry is safe, a multimodal content list or tool arguments
        are not, whether they were handed over as live entry objects or as a dict
        this timeline decoded (the decoded nested objects are the caller's). A clear
        flag narrows the per-read dependency check back to messages that actually
        hold a shared payload (``_message_needs_dependency_check``).

        Used by hydration and load, where the projection is rebuilt from a
        journal read instead of mutated in place.
        """
        fresh_messages = [self._timeline_entry_to_native_message(entry) for entry in entries]
        # The messages here are derived from the entries, so scanning the entries
        # alone covers everything the messages share.
        shares_caller_state = any(self._entry_is_externally_owned(entry) for entry in entries)
        with self._state_lock:
            self.timeline = entries
            self._native_messages = fresh_messages
            self._pending_compression_plan = None
            self._native_aliasing_risk = shares_caller_state
            # Freshly built message objects are not held by any caller, so only a
            # shared caller entry can keep the per-message check armed.
            self._message_objects_escaped = shares_caller_state
            self._recompute_token_total()
            self._compaction_generation += 1

    # ------------------------------------------------------------------
    # LLM call function setters
    # ------------------------------------------------------------------

    def set_llm_call_fn(self, fn: Callable[[str], str]) -> None:
        """Set the synchronous LLM call function."""
        self._llm_call_fn = fn

    def set_llm_call_async_fn(self, fn: Callable[[str], Any]) -> None:
        """Set the async LLM call function."""
        self._llm_call_async_fn = fn

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def add_entry(self, entry: TimelineEntry) -> None:
        """
        Add entry to timeline, converting to native message format.

        Overrides the parent add_entry() to convert TimelineEntry to NativeMessage
        before storing internally. The original entry is still added to self.timeline
        for backward compatibility.

        Mapping:
        - USER_MESSAGE -> role='user'
        - AGENT_RESPONSE -> role='assistant'
        - AGENT_THOUGHTS -> role='assistant'
        - TOOL_CALL -> role='assistant' with tool_calls
        - RESOURCE_RESULT -> role='tool'
        - WORKFLOW_RESULT -> role='tool'
        - TIMELINE_SUMMARY -> role='system'
        - CONTEXT -> role='system'
        - Other -> role='assistant' (default)

        Args:
            entry: TimelineEntry to add
        """
        native_msg = self._timeline_entry_to_native_message(entry)
        with self._state_lock:
            if self._entry_is_externally_owned(entry):
                # The caller still owns mutable content/tool-argument objects that
                # this message now shares. They are not copied away (the mutable API
                # contract is preserved): the message records a snapshot of them, so
                # an in-place edit of the caller's is observed per message on read
                # rather than by distrusting (and rescanning) all retained history.
                self._native_aliasing_risk = True
            marker_was_valid = self._token_marker_valid()
            # Add to parent timeline for backward compatibility
            super().add_entry(entry)
            # Convert to native message and store
            self._native_messages.append(native_msg)
            tokens = self._estimate_native_message_tokens(native_msg)
            if marker_was_valid:
                # Record this message's token count together with the dependencies a
                # later read compares to detect an in-place edit of the caller's.
                self._record_message_token_cache(native_msg, tokens)
            self._fold_token_delta(tokens, marker_was_valid=marker_was_valid)

    @staticmethod
    def _is_caller_owned_value(value: Any) -> bool:
        """True when a value is a mutable object the caller can still mutate in place.

        Scalars, tuples and ``None`` are values this timeline can safely own or
        simply copy: nothing a caller does to their own binding changes what was
        stored. Dicts, lists, sets and arbitrary objects can be mutated in place
        (including nested argument dicts), so they are treated as aliased.
        """
        return isinstance(value, dict | list | set | bytearray) or (
            value is not None and not isinstance(value, str | bytes | int | float | bool | tuple)
        )

    def _entry_is_externally_owned(self, entry: TimelineEntry) -> bool:
        """True when ``entry`` still shares caller-mutable state with this timeline.

        ``add_entry`` keeps the caller's object alive in ``timeline`` and shares its
        ``content`` (a ``str`` is immutable, but multimodal content is a list of
        dicts) and its tool-argument structures with the native message. A caller
        holding the entry — the ordinary ``tl.add_entry(TimelineEntry(...))`` call —
        can therefore mutate text or arguments behind the accounting's back, so the
        entry counts as aliased until the caller proves sole ownership by copying
        (``TimelineEntry.from_dict(entry.to_dict())``) or building fresh values.

        ``content`` is checked even when empty: the native message stores that very
        object, so a later ``blocks.append(...)`` by the caller would be invisible.
        ``tool_calls`` is only shared when truthy — a falsy list is never converted,
        the native message gets ``tool_calls=None`` instead.
        """
        if self._is_caller_owned_value(entry.content):
            return True
        return bool(entry.tool_calls) and self._is_caller_owned_value(entry.tool_calls)

    def _message_shares_caller_state(self, message: NativeMessage) -> bool:
        """True when a native message still holds a caller-mutable object.

        ``NativeMessage.from_dict`` keeps the decoded nested ``content`` and
        ``arguments`` objects by reference, so a hydration path that decodes
        caller-supplied dicts can install messages whose nested lists or argument
        dicts the caller still holds.
        """
        if self._is_caller_owned_value(message.content):
            return True
        if message.tool_calls:
            for call in message.tool_calls:
                if self._is_caller_owned_value(call.arguments):
                    return True
        return False

    def _projection_shares_caller_state(self, entries: list[TimelineEntry], messages: list[NativeMessage]) -> bool:
        """True when any installed entry or native message shares caller-mutable state."""
        return any(self._entry_is_externally_owned(entry) for entry in entries) or any(
            self._message_shares_caller_state(message) for message in messages
        )

    def _timeline_entry_to_native_message(self, entry: TimelineEntry) -> NativeMessage:
        """
        Convert a TimelineEntry to NativeMessage format.

        Mapping rules:
        - USER_MESSAGE -> role='user'
        - AGENT_RESPONSE, AGENT_THOUGHTS, AGENT_LEARNING, TODO_LIST -> role='assistant'
        - TOOL_CALL -> role='assistant' with tool_calls
        - RESOURCE_RESULT, WORKFLOW_RESULT -> role='tool' with tool_call_id
        - SUB_AGENT_RESPONSE w/ tool_call_id -> role='tool' (sub-agent invoked as native tool)
        - SUB_AGENT_RESPONSE w/o tool_call_id -> role='assistant' (legacy XML flow)
        - UNKNOWN_TOOL_CALL / FAILED_TOOL_CALL w/ tool_call_id -> role='tool'
        - TIMELINE_SUMMARY, CONTEXT -> role='system'
        - Other -> role='assistant' (default)

        Args:
            entry: TimelineEntry to convert

        Returns:
            NativeMessage in provider-agnostic format
        """
        role: NativeMessageRole
        tool_calls: list[NativeToolCall] | None = None
        tool_call_id: str | None = None

        # Map entry type to role
        # Compare by .value (string) instead of enum identity to handle the case where
        # the same TimelineEntryType enum is loaded under two module paths
        # (e.g., dana.core.agent.timeline vs dana_agent.dana.core.agent.timeline)
        # due to namespace package merging with editable installs.
        entry_type_value = entry.entry_type.value if hasattr(entry.entry_type, "value") else str(entry.entry_type)

        if entry_type_value == TimelineEntryType.USER_MESSAGE.value:
            role = "user"
        elif entry_type_value in (
            TimelineEntryType.TIMELINE_SUMMARY.value,
            TimelineEntryType.CONTEXT.value,
        ):
            role = "system"
        elif entry_type_value in (
            TimelineEntryType.RESOURCE_RESULT.value,
            TimelineEntryType.WORKFLOW_RESULT.value,
        ):
            role = "tool"
            tool_call_id = entry.tool_call_id
        elif entry_type_value == TimelineEntryType.TOOL_CALL.value:
            role = "assistant"
            # Convert entry.tool_calls to NativeToolCall list
            if entry.tool_calls:
                tool_calls = []
                for tc in entry.tool_calls:
                    # Handle both dict and object formats
                    if isinstance(tc, dict):
                        # Support multiple formats:
                        # 1. Runtime format: {"function": "name", "arguments": {...}, "tool_call_id": "..."}
                        # 2. OpenAI nested: {"id": "...", "function": {"name": "...", "arguments": "..."}}
                        # 3. Direct format: {"id": "...", "name": "...", "arguments": {...}}
                        tc_id = tc.get("id", "") or tc.get("tool_call_id", "")

                        # Get function name - handle both string and nested dict formats
                        tc_function = tc.get("function")
                        if isinstance(tc_function, str):
                            # Runtime format: "function" is the name string
                            tc_name = tc_function
                        elif isinstance(tc_function, dict):
                            # OpenAI nested format: "function" is {"name": "...", "arguments": "..."}
                            tc_name = tc_function.get("name", "")
                        else:
                            # Direct format: "name" key exists
                            tc_name = tc.get("name", "")

                        # Get arguments - handle both direct and nested formats
                        tc_args = tc.get("arguments")
                        if tc_args is None and isinstance(tc_function, dict):
                            tc_args = tc_function.get("arguments", {})
                        # Handle string arguments (JSON string from OpenAI format)
                        if isinstance(tc_args, str):
                            try:
                                tc_args = json.loads(tc_args)
                            except json.JSONDecodeError:
                                tc_args = {"raw": tc_args}
                        # Default to empty dict if still None
                        if tc_args is None:
                            tc_args = {}
                    else:
                        # Object with attributes
                        tc_id = getattr(tc, "id", "")
                        tc_name = getattr(tc, "name", "") or getattr(getattr(tc, "function", None), "name", "")
                        tc_args = getattr(tc, "arguments", {}) or getattr(getattr(tc, "function", None), "arguments", {})
                        if isinstance(tc_args, str):
                            try:
                                tc_args = json.loads(tc_args)
                            except json.JSONDecodeError:
                                tc_args = {"raw": tc_args}

                    tool_calls.append(NativeToolCall(id=tc_id, name=tc_name, arguments=tc_args))
        elif (
            entry_type_value
            in (
                TimelineEntryType.UNKNOWN_TOOL_CALL.value,
                TimelineEntryType.FAILED_TOOL_CALL.value,
                TimelineEntryType.SUB_AGENT_RESPONSE.value,
            )
            and entry.tool_call_id
        ):
            # Tool execution results / errors with a tool_call_id must be role="tool"
            # so the LLM API can match them to their corresponding tool_calls.
            # Without this, the API rejects with "tool_call_ids did not have response messages".
            # SUB_AGENT_RESPONSE is produced by `_record_tool_results` for tool_type="agent"
            # the same way RESOURCE_RESULT/WORKFLOW_RESULT are produced for resources/workflows,
            # so it must be normalized to the same role when a tool_call_id is present.
            role = "tool"
            tool_call_id = entry.tool_call_id
        else:
            # Default: AGENT_RESPONSE, AGENT_THOUGHTS, AGENT_LEARNING, TODO_LIST,
            # SUB_AGENT_RESPONSE (without tool_call_id — legacy XML flow),
            # UNKNOWN_TOOL_CALL/FAILED_TOOL_CALL (without tool_call_id)
            role = "assistant"

        metadata = entry.metadata.copy() if entry.metadata else {}
        # Internal backlink so a compaction commit can map a retained entry to the
        # exact message that mirrors it (see NATIVE_SOURCE_ENTRY_KEY).
        metadata[NATIVE_SOURCE_ENTRY_KEY] = entry
        # Mirror the entry's temporary-entry signal so both projections agree on
        # what is disposable. Without this, an ephemeral entry a caller handed to
        # ``add_entry`` (the ``TimelineEntry.ephemeral`` contract, not the
        # metadata convention ``set_context`` sets on its own entry) would be
        # dropped from ``timeline`` but linger in the provider-facing native list
        # for the rest of the session (I2).
        if entry.ephemeral:
            metadata["ephemeral"] = True

        return NativeMessage(
            role=role,
            content=entry.content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            metadata=metadata,
            timestamp=entry.timestamp,
        )

    # ------------------------------------------------------------------
    # Native messages property
    # ------------------------------------------------------------------

    @property
    def native_messages(self) -> list[NativeMessage]:
        """
        Get the list of native messages.

        Returns:
            List of NativeMessage objects in chronological order

        This returns the live backing list, which callers legitimately mutate in
        place (that has always been the contract). Because a returned message object
        lets such a caller rebind even a payload attribute, the read records that the
        projection has escaped (``_native_aliasing_risk``) and re-arms every cached
        message with a dependency snapshot, so a later edit is observed per message
        rather than by distrusting and re-scanning the whole history.
        """
        self._mark_projection_escaped()
        return self._native_messages

    # ------------------------------------------------------------------
    # LLM message conversion
    # ------------------------------------------------------------------

    def to_llm_messages(
        self,
        max_tokens: int | None = None,
        default_role: str = "user",  # noqa: ARG002 - kept for TimelineProtocol compatibility
        separate_latest_user: bool = False,
    ) -> list[LLMMessage]:
        """
        Convert native messages to LLM messages for provider consumption.

        Returns individual messages from the native storage, preserving the
        original message structure. If compression has occurred, a summary
        system message is prepended. Token limiting is applied via sliding
        window on native messages, keeping tool_call/tool_result pairs together.

        Args:
            max_tokens: Maximum tokens to include (overrides max_context_tokens)
            default_role: Default role for entries without specific mapping. Unused
                in this implementation since NativeMessage has explicit roles, but
                kept for TimelineProtocol compatibility.
            separate_latest_user: If True, separates latest user message

        Returns:
            List of LLMMessage objects in chronological order
        """
        token_limit = max_tokens or self.max_context_tokens

        # Convert native messages to LLMMessage
        messages = [msg.to_llm_message() for msg in self._native_messages]

        # If we have compressed context, prepend it as a system message
        compressed_context = self.get_compressed_context()
        if compressed_context:
            # Check if we already have a summary message (to avoid duplication)
            has_summary = any(
                isinstance(msg.content, str)
                and (msg.content.startswith("[Previous context summary]") or msg.content.startswith("[SUMMARY]"))
                for msg in messages
            )

            if not has_summary:
                summary_message = LLMMessage(
                    role="system",
                    content=f"[Previous context summary] {compressed_context}",
                )
                # Insert at the beginning (or after other system messages)
                insert_idx = 0
                for i, msg in enumerate(messages):
                    if msg.role == "system":
                        insert_idx = i + 1
                    else:
                        break
                messages.insert(insert_idx, summary_message)

        # Handle separate_latest_user: separate the latest user message
        if separate_latest_user and messages:
            latest_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].role == "user":
                    latest_user_idx = i
                    break

            if latest_user_idx is not None:
                latest_user_msg = messages[latest_user_idx]
                context_messages = messages[:latest_user_idx] + messages[latest_user_idx + 1 :]

                # Apply token limit to context
                if estimate_messages_tokens(context_messages) > token_limit:
                    context_messages = apply_token_limit_to_messages(context_messages, token_limit)

                # Append latest user message at the end
                context_messages.append(latest_user_msg)
                return context_messages

        # Apply token limit if needed
        if estimate_messages_tokens(messages) > token_limit:
            messages = apply_token_limit_to_messages(messages, token_limit)

        return messages

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def set_context(self, context: dict[str, Any]) -> None:
        """
        Set or replace the ephemeral runtime context entry.

        Overrides the parent set_context() to also update the native messages list.
        This removes any existing CONTEXT entries from both self.timeline and
        self._native_messages, then adds a fresh one to both. Only the
        runtime-context slot this method owns (``role == "system"`` carrying the
        ``ephemeral`` flag) is retired: a non-ephemeral CONTEXT entry is a caller's
        durable state whose lifetime its owner controls, and the session's
        recovery-note reconciliation depends on it outliving a ``set_context`` that
        runs before the recovery request is sent.

        Args:
            context: Dictionary with context info (e.g., timestamp, user, timezone)
        """
        # Remove existing CONTEXT native messages
        with self._state_lock:
            marker_was_valid = self._token_marker_valid()
            removed_tokens = 0
            dropped: list[NativeMessage] = []
            for msg in self._native_messages:
                if msg.role == "system" and msg.metadata.get("ephemeral", False):
                    removed_tokens += self._estimate_native_message_tokens(msg)
                else:
                    dropped.append(msg)
            self._native_messages = dropped
            self._fold_token_delta(-removed_tokens, marker_was_valid=marker_was_valid)
            self._rebase_token_caches()

            # Call parent implementation (handles timeline and creates entry)
            super().set_context(context)

            # The parent inserts the context entry at the beginning of self.timeline
            # Get the context entry that was just added
            context_entry = self.timeline[0] if self.timeline else None
            if context_entry and context_entry.entry_type == TimelineEntryType.CONTEXT:
                # Convert to native message and insert at the beginning
                native_msg = self._timeline_entry_to_native_message(context_entry)
                native_msg.metadata["ephemeral"] = True
                marker_was_valid = self._token_marker_valid()
                self._native_messages.insert(0, native_msg)
                tokens = self._estimate_native_message_tokens(native_msg)
                self._fold_token_delta(tokens, marker_was_valid=marker_was_valid)
                if marker_was_valid:
                    # Record the fresh context message in the cache. Its content is
                    # this timeline's own immutable string, so a snapshot is only
                    # taken if the message *objects* have already escaped to a caller.
                    self._token_message_caches.insert(0, (native_msg, self._capture_message_dependencies(native_msg), tokens))

    def clear_old_entries(self, before_timestamp: datetime) -> int:
        """
        Remove entries before timestamp from both timeline and native messages.

        Overrides the parent clear_old_entries() to also remove corresponding
        native messages, maintaining consistency between the two lists.

        Args:
            before_timestamp: Remove entries before this timestamp

        Returns:
            Number of entries removed
        """
        with self._state_lock:
            original_count = len(self.timeline)

            # Remove from timeline
            self.timeline = [entry for entry in self.timeline if entry.timestamp >= before_timestamp]

            # Remove from native messages, folding the exact removed-token delta.
            marker_was_valid = self._token_marker_valid()
            removal_delta = -sum(
                self._estimate_native_message_tokens(msg) for msg in self._native_messages if msg.timestamp < before_timestamp
            )
            self._native_messages = [msg for msg in self._native_messages if msg.timestamp >= before_timestamp]
            self._fold_token_delta(removal_delta, marker_was_valid=marker_was_valid)
            self._rebase_token_caches()

            removed_count = original_count - len(self.timeline)
            return removed_count

    # ------------------------------------------------------------------
    # Default LLM call function helpers (used by CompressionMixin)
    # ------------------------------------------------------------------

    def _get_default_llm_call_fn(self) -> Callable[[str], str] | None:
        """
        Get a default LLM call function using the agent's runtime.

        Returns a function that wraps the prompt string in an LLMMessage
        and calls the agent's runtime.call_llm method.

        Returns:
            Callable that takes a prompt string and returns the LLM response,
            or None if agent or runtime is not available.
        """
        if self._agent is None:
            return None

        runtime = getattr(self._agent, "_runtime", None)
        if runtime is None or not hasattr(runtime, "call_llm"):
            return None

        def default_call_fn(prompt: str) -> str:
            messages = [LLMMessage(role="user", content=prompt)]
            return runtime.call_llm(messages)

        return default_call_fn

    def _get_default_llm_call_async_fn(self) -> Callable[[str], Any] | None:
        """
        Get a default async LLM call function using the agent's runtime.

        Returns a function that wraps the prompt string in an LLMMessage
        and calls the agent's runtime.call_llm_async method.

        Returns:
            Callable that takes a prompt string and returns an awaitable LLM response,
            or None if agent or runtime is not available.
        """
        if self._agent is None:
            return None

        runtime = getattr(self._agent, "_runtime", None)
        if runtime is None or not hasattr(runtime, "call_llm_async"):
            return None

        async def default_call_async_fn(prompt: str) -> str:
            messages = [LLMMessage(role="user", content=prompt)]
            return await runtime.call_llm_async(messages)

        return default_call_async_fn

    # ------------------------------------------------------------------
    # Token estimation (used by CompressionMixin)
    # ------------------------------------------------------------------

    def _estimate_native_message_tokens(self, message: NativeMessage) -> int:
        """
        Estimate token count for a single NativeMessage.

        Uses a character-based heuristic (4 characters per token), which is
        a reasonable approximation for most LLM tokenizers.

        Args:
            message: NativeMessage to estimate

        Returns:
            Estimated token count
        """
        # Base content tokens
        content = message.content
        total = len(content) // 4 if isinstance(content, str) else len(str(content)) // 4

        # Add tokens for tool_calls if present
        if message.tool_calls:
            for tc in message.tool_calls:
                # Estimate tokens for tool call structure: id, name, arguments
                total += len(tc.id) // 4
                total += len(tc.name) // 4
                total += len(str(tc.arguments)) // 4

        return total

    def _estimate_native_messages_list_tokens(self, messages: list[NativeMessage]) -> int:
        """
        Estimate token count for a list of NativeMessage objects.

        Args:
            messages: List of NativeMessage objects

        Returns:
            Estimated token count
        """
        total = 0
        for msg in messages:
            try:
                total += self._estimate_native_message_tokens(msg)
            except Exception as e:
                logger.error(f"Error estimating token count for native message: {e}")
                continue
        return total

    def _estimate_entries_tokens(self, entries: list[TimelineEntry]) -> int:
        """
        Estimate token count for timeline entries.

        Args:
            entries: List of TimelineEntry objects

        Returns:
            Estimated token count
        """
        total = 0
        for entry in entries:
            # Rough estimation: 4 characters per token
            try:
                total += self._estimate_entry_tokens(entry)
            except Exception as e:
                logger.error(f"Error estimating token count for entry: {e}")
                continue
        return int(total)
