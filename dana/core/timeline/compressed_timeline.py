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

        # True once the model-facing native state has been handed to a caller that
        # can mutate it in place — through the ``native_messages`` getter, or by
        # handing ``add_entry`` content/tool-argument objects it still owns. Such a
        # change (even a same-length content edit, or a nested tool-argument dict
        # edit) cannot be observed from the list or from the message objects, and
        # escaped message objects survive into new lists (compaction, hydration and
        # ``load_from_entries`` reuse the very same ``NativeMessage`` instances).
        # While this is set, the cached total is not trusted and reads recompute.
        # It is sticky for the timeline instance, and is only cleared by building a
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
    #   * appends and in-place content replacements fold an exact delta
    #     (``_fold_token_delta``),
    #   * hydration, load, and compaction-replacement install a whole new list
    #     and recompute once (``_install_model_state``).
    #
    # The cached total is trusted only while its marker still describes the live
    # list *and* the model-facing state has never been aliased: same list object,
    # same length, and ``_native_aliasing_risk`` clear. A list this size can be
    # mutated in place by a caller holding one of its messages, and those message
    # objects survive into later lists, so once the state has escaped the cached
    # total is not trusted and reads recompute. Direct rebinding or trimming of
    # ``_native_messages`` fails the marker and recomputes the same way. A stale
    # cached value is never returned.

    def _token_marker_valid(self) -> bool:
        """True when the cached token total still describes the live native list."""
        if self._native_aliasing_risk:
            return False
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

    def _recompute_token_total(self) -> None:
        """Recompute the cached token total from the live native list (one scan)."""
        self._native_token_total = self._estimate_native_messages_list_tokens(self._native_messages)
        self._anchor_token_marker()

    def get_native_token_estimate(self) -> int:
        """Estimated tokens of the retained model working set.

        Returns the incrementally maintained total when its marker is still
        valid, otherwise recomputes once and re-anchors. This is the entry point
        for the compression trigger: it never rescans retained history on the
        ordinary append-only path.
        """
        with self._state_lock:
            if not self._token_marker_valid():
                self._recompute_token_total()
            return self._native_token_total

    def _install_model_state(self, entries: list[TimelineEntry], messages: list[NativeMessage]) -> None:
        """Replace both model-facing representations as a single projection swap.

        Used by hydration/load and by compaction commit. Both lists are replaced
        together so ``timeline`` and ``_native_messages`` cannot drift apart, the
        token total is recomputed exactly once, and ``_compaction_generation`` is
        bumped so any compaction plan captured against the previous projection
        refuses to commit over the new one.

        This sets — never clears — the aliasing-risk flag: compaction reuses the
        very same ``NativeMessage`` objects it retained, and a caller may hold
        references to them from the list that was just replaced, so the new
        projection is not provably unaliased. See ``_install_built_projection``.
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
        this timeline decoded (the decoded nested objects are the caller's).

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
        aliased = self._entry_is_externally_owned(entry)
        with self._state_lock:
            if aliased:
                # The caller still owns mutable content/tool-argument objects that
                # this message now shares; an in-place edit of theirs would be
                # invisible to any marker, so accounting must stop trusting the
                # cached total. Not a copy: the mutable API contract is preserved.
                self._native_aliasing_risk = True
            marker_was_valid = self._token_marker_valid()
            # Add to parent timeline for backward compatibility
            super().add_entry(entry)
            # Convert to native message and store
            self._native_messages.append(native_msg)
            self._fold_token_delta(
                self._estimate_native_message_tokens(native_msg),
                marker_was_valid=marker_was_valid,
            )

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
        place (that has always been the contract). Because such an edit — a
        same-length content replacement, or a nested tool-argument dict — cannot be
        observed from the list or the message objects, the read marks the timeline's
        token accounting untrusted so estimates recompute from live content. The
        mark is sticky for this instance: escaped message objects survive into later
        projections, so only a projection this timeline rebuilt without carrying
        caller-mutable state clears it (see ``_install_built_projection``).
        """
        with self._state_lock:
            self._native_aliasing_risk = True
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
        self._native_messages, then adds a fresh one to both.

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
                self._fold_token_delta(
                    self._estimate_native_message_tokens(native_msg),
                    marker_was_valid=marker_was_valid,
                )

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
