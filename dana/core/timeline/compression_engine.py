"""
Compression engine mixin for CompressedTimeline.

Provides CompressionMixin with all compression-related methods:
needs_compression, get_entries_to_keep_and_compress, compress, compress_async,
_apply_compression, build_compression_prompt, etc.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING

from structlog import get_logger

from dana.core.timeline.compact_trigger import resolve_trigger_tokens
from dana.core.timeline.native_message import (
    COMPRESSED_CONTEXT_KEY,
    COMPRESSED_ENTRIES_COUNT_KEY,
    COMPRESSION_TIMESTAMP_KEY,
    NATIVE_SOURCE_ENTRY_KEY,
    NativeMessage,
)
from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType


if TYPE_CHECKING:
    from dana.core.timeline.compressed_timeline import CompressedTimeline

logger = get_logger()

# Literal stub content for client-side tool_result shrinking (Phase 2).
# Idempotency is detected by exact content-string equality — no metadata flag.
SHRINK_STUB_CONTENT = "[cleared for context budget]"


class CompressionMixin:
    """
    Mixin providing LLM-based compression logic for CompressedTimeline.

    Expects the following attributes on self (provided by CompressedTimeline):
        _compressed_config: CompressedTimelineConfig
        _native_messages: list[NativeMessage]
        timeline: list[TimelineEntry]
        _llm_call_fn: Callable | None
        _llm_call_async_fn: Callable | None
        cutoff_when_token_reach: int
        max_recent_entries_to_keep: int
    """

    # ------------------------------------------------------------------
    # Token estimation helpers
    # ------------------------------------------------------------------

    def _estimate_entry_tokens(self: CompressedTimeline, entry: TimelineEntry) -> int:
        """
        Estimate token count for a single entry.

        Args:
            entry: TimelineEntry to estimate

        Returns:
            Estimated token count
        """
        # Rough estimation: 4 characters per token
        content = entry.content
        if isinstance(content, list):
            # Multimodal content: estimate text parts only
            total = 0
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += len(block["text"])
            return int(total / 4)
        return int(len(str(content)) / 4)

    # ------------------------------------------------------------------
    # Needs compression check
    # ------------------------------------------------------------------

    def needs_compression(self: CompressedTimeline) -> bool:
        """
        Check if timeline compression is needed.

        Compression triggers when `(messages_est + system_est + tools_est) >=
        trigger`. Trigger precedence: explicit `max_tokens_until_compression`
        passed at construction wins; otherwise fall back to env-resolved
        `resolve_trigger_tokens()` (default 150000, env DANA_COMPACT_TRIGGER_TOKENS).

        ``messages_est`` comes from the timeline's incremental accounting
        (``get_native_token_estimate``), so an ordinary append-only turn is
        O(new message) here rather than a full retained-history recount.

        Optional `system_tokens_fn` / `tools_tokens_fn` callbacks fold system
        prompt and tools-schema size into the estimate. Callbacks are invoked
        defensively via `_safe_call_tokens_fn`.
        """
        if not self._compressed_config.compression_enabled:
            return False

        # Need at least some entries to compress
        if len(self._native_messages) <= self._compressed_config.max_recent_entries_to_keep:
            return False

        messages_est = self.get_native_token_estimate()
        sys_est = self._safe_call_tokens_fn(self._system_tokens_fn)
        tools_est = self._safe_call_tokens_fn(self._tools_tokens_fn)

        explicit = self._explicit_max_tokens_until_compression
        trigger = explicit if explicit is not None else resolve_trigger_tokens()

        total = messages_est + sys_est + tools_est
        if total >= trigger:
            logger.info(
                "compression_needed",
                reason="threshold",
                tokens_est=total,
                threshold=trigger,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Partition logic: which messages/entries to keep vs. compress
    # ------------------------------------------------------------------

    def get_entries_to_keep_and_compress(
        self: CompressedTimeline,
    ) -> tuple[list[TimelineEntry], list[TimelineEntry]]:
        """
        Determine which entries to keep and which to compress.

        This method now derives its result from native message partitioning to ensure
        consistency. The partitioning is based on native message token estimation
        which is more accurate than TimelineEntry-based estimation.

        Iterates from latest to oldest, keeping entries until:
        - Token count reaches cutoff_when_token_reach, OR
        - Entry count reaches max_recent_entries_to_keep

        Ensures tool_call/tool_result pairs are not split.

        Returns:
            Tuple of (entries_to_keep, entries_to_compress)
        """
        if not self.timeline:
            return [], []

        # Use native message partitioning as the source of truth
        native_to_keep, _ = self.get_native_messages_to_keep_and_compress()

        # The number of entries to keep should match the number of native messages to keep
        # because each TimelineEntry corresponds to one NativeMessage
        entries_to_keep_count = len(native_to_keep)

        # Get entries from the end of timeline (most recent)
        if entries_to_keep_count >= len(self.timeline):
            entries_to_keep = self.timeline[:]
            entries_to_compress = []
        else:
            entries_to_keep = self.timeline[-entries_to_keep_count:] if entries_to_keep_count > 0 else []
            entries_to_compress = self.timeline[:-entries_to_keep_count] if entries_to_keep_count > 0 else self.timeline[:]

        return entries_to_keep, entries_to_compress

    def _ensure_tool_pair_integrity(self: CompressedTimeline, entries: list[TimelineEntry]) -> list[TimelineEntry]:
        """
        Ensure tool_call and tool_result pairs are kept together.

        If entries start with tool results without their tool_calls,
        we need to expand backwards in the original timeline.

        Args:
            entries: List of entries to check

        Returns:
            Entries with complete tool pairs
        """
        if not entries:
            return entries

        # Check if first entries are orphaned tool results
        first_idx = 0
        while first_idx < len(entries) and entries[first_idx].tool_call_id:
            first_idx += 1

        if first_idx == 0:
            # No orphaned tool results at the start
            return entries

        # Find the tool_call for these orphaned results in the original timeline
        timeline_idx = self.timeline.index(entries[0]) if entries[0] in self.timeline else -1
        if timeline_idx <= 0:
            return entries

        # Look backwards in the original timeline for the tool_call
        additional_entries = []
        for i in range(timeline_idx - 1, -1, -1):
            entry = self.timeline[i]
            additional_entries.insert(0, entry)
            if entry.tool_calls:
                # Found the tool_call, stop looking
                break

        return additional_entries + entries

    def _remove_forward_orphans(self: CompressedTimeline, entries: list[TimelineEntry]) -> list[TimelineEntry]:
        """Drop any `tool_result` entry whose matching `tool_use` is absent.

        Used after truncation in `reactive_compact` (Phase 3). Matches
        existing pair-integrity philosophy: drop rather than repair.
        Returns a new list without the orphaned tool_results.
        """
        if not entries:
            return entries

        tool_use_ids: set[str] = set()
        for e in entries:
            if getattr(e, "tool_calls", None):
                for tc in e.tool_calls or []:
                    if isinstance(tc, dict):
                        tc_id = tc.get("id") or tc.get("tool_call_id") or ""
                    else:
                        tc_id = getattr(tc, "id", "") or ""
                    if tc_id:
                        tool_use_ids.add(tc_id)

        return [e for e in entries if not (e.tool_call_id and e.tool_call_id not in tool_use_ids)]

    # ------------------------------------------------------------------
    # Phase 2 — cheap shrink of old tool_result content
    # ------------------------------------------------------------------

    def _is_tool_result_entry(self: CompressedTimeline, entry: TimelineEntry) -> bool:
        """Return True if entry represents a tool-result carrying a tool_call_id."""
        if not entry.tool_call_id:
            return False
        etv = entry.entry_type.value if hasattr(entry.entry_type, "value") else str(entry.entry_type)
        return etv in (
            TimelineEntryType.RESOURCE_RESULT.value,
            TimelineEntryType.WORKFLOW_RESULT.value,
            TimelineEntryType.UNKNOWN_TOOL_CALL.value,
            TimelineEntryType.FAILED_TOOL_CALL.value,
        )

    def cheap_shrink_tool_results(self: CompressedTimeline) -> bool:
        """Client-side stub old tool_result bodies to a fixed literal.

        Preserves `tool_call_id` (so tool_use/tool_result pair integrity
        holds) and leaves recent entries untouched. Guarded by a predictive
        gate: if stubbing would not drop us below the configured trigger,
        leave the timeline unmutated and return False (caller falls through
        to full `compress()`).

        Idempotent via content-string equality — re-running is a no-op.

        Returns:
            True iff the timeline was mutated AND post-shrink estimate is
            below the trigger.
        """
        keep_recent = self._compressed_config.cheap_shrink_keep_recent
        trigger = (
            self._explicit_max_tokens_until_compression
            if self._explicit_max_tokens_until_compression is not None
            else resolve_trigger_tokens()
        )

        if len(self._native_messages) <= keep_recent:
            return False

        # Pre-shrink tokens come from the incremental accounting — same estimate
        # as needs_compression, without rescanning the retained history.
        pre_tokens = self.get_native_token_estimate()
        if pre_tokens < trigger:
            return False

        # Build set of tool_call_ids referenced by pending tool_use in the
        # kept-recent window — do NOT stub results of calls still in flight.
        recent_cutoff = len(self._native_messages) - keep_recent
        recent_tool_use_ids: set[str] = set()
        for msg in self._native_messages[recent_cutoff:]:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = getattr(tc, "id", "") or ""
                    if tc_id:
                        recent_tool_use_ids.add(tc_id)

        # Identify stub-eligible TimelineEntry objects — older than keep_recent,
        # tool-result kind, not already stubbed.
        eligible_indices: list[int] = []
        predicted_savings = 0
        if len(self.timeline) > keep_recent:
            for idx in range(len(self.timeline) - keep_recent):
                entry = self.timeline[idx]
                if not self._is_tool_result_entry(entry):
                    continue
                if entry.content == SHRINK_STUB_CONTENT:
                    continue
                if entry.tool_call_id in recent_tool_use_ids:
                    # Call still referenced by pending tool_use in kept window.
                    continue
                eligible_indices.append(idx)
                content_len = len(entry.content) if isinstance(entry.content, str) else len(str(entry.content))
                predicted_savings += content_len // 4

        if not eligible_indices:
            return False

        # Predictive gate — if shrink alone cannot drop us below trigger,
        # bail without mutating. Avoids vacuous summary over stubs.
        if (pre_tokens - predicted_savings) >= trigger:
            return False

        # Stub eligible TimelineEntry and matching NativeMessage in place,
        # folding the exact token delta of each replacement into the
        # incremental accounting (the stub is shorter than the payload).
        with self._state_lock:
            stubbed_ids: set[str] = set()
            for idx in eligible_indices:
                entry = self.timeline[idx]
                entry.content = SHRINK_STUB_CONTENT
                if entry.tool_call_id:
                    stubbed_ids.add(entry.tool_call_id)

            for nm in self._native_messages:
                if nm.role == "tool" and nm.tool_call_id and nm.tool_call_id in stubbed_ids and nm.content != SHRINK_STUB_CONTENT:
                    nm.content = SHRINK_STUB_CONTENT
                    after = self._estimate_native_message_tokens(nm)
                    marker_was_valid = self._token_marker_valid()
                    self._apply_message_token_delta(nm, after, marker_was_valid=marker_was_valid)

            # Pair integrity safety net — tool_call_ids preserved so this is a no-op
            # in practice, but runs for defense in depth.
            self._ensure_tool_pair_integrity(self.timeline)

        post_tokens = self.get_native_token_estimate()
        logger.info(
            "cheap_shrink_tool_results",
            entries_stubbed=len(eligible_indices),
            tokens_before=pre_tokens,
            tokens_after=post_tokens,
            below_threshold=post_tokens < trigger,
        )
        return post_tokens < trigger

    def get_native_messages_to_keep_and_compress(
        self: CompressedTimeline,
    ) -> tuple[list[NativeMessage], list[NativeMessage]]:
        """
        Determine which native messages to keep and which to compress.

        Iterates from latest to oldest, keeping messages until:
        - Token count reaches cutoff_when_token_reach, OR
        - Message count reaches max_recent_entries_to_keep

        Ensures tool_call/tool_result pairs are not split.

        Returns:
            Tuple of (messages_to_keep, messages_to_compress)
        """
        if not self._native_messages:
            return [], []

        messages_to_keep: list[NativeMessage] = []
        current_tokens = 0
        message_count = 0

        # Iterate from latest to oldest
        for msg in reversed(self._native_messages):
            msg_tokens = self._estimate_native_message_tokens(msg)

            # Check if we've hit our limits
            would_exceed_tokens = (current_tokens + msg_tokens) > self.cutoff_when_token_reach
            would_exceed_entries = message_count >= self.max_recent_entries_to_keep

            if would_exceed_tokens or would_exceed_entries:
                # Check if we need to include this message to keep tool pairs together
                if msg.tool_calls and messages_to_keep:
                    # This is a tool_call message, check if we have its results in kept messages
                    has_orphaned_results = any(
                        m.tool_call_id is not None
                        for m in messages_to_keep
                        if not any(kept.tool_calls for kept in messages_to_keep if kept.timestamp < m.timestamp)
                    )
                    if has_orphaned_results:
                        # Include this tool_call to avoid orphaning results
                        messages_to_keep.insert(0, msg)
                        current_tokens += msg_tokens
                        message_count += 1
                        continue
                break

            messages_to_keep.insert(0, msg)
            current_tokens += msg_tokens
            message_count += 1

        # Ensure we don't break tool_call/tool_result pairs at the boundary
        messages_to_keep = self._ensure_native_message_tool_pair_integrity(messages_to_keep)

        # Everything not kept should be compressed
        kept_set = set(id(m) for m in messages_to_keep)
        messages_to_compress = [m for m in self._native_messages if id(m) not in kept_set]

        return messages_to_keep, messages_to_compress

    def _ensure_native_message_tool_pair_integrity(self: CompressedTimeline, messages: list[NativeMessage]) -> list[NativeMessage]:
        """
        Ensure tool_call and tool_result pairs are kept together for native messages.

        If messages start with tool results without their tool_calls,
        we need to expand backwards in the original native message list.

        Args:
            messages: List of NativeMessage to check

        Returns:
            Messages with complete tool pairs
        """
        if not messages:
            return messages

        # Check if first messages are orphaned tool results
        first_idx = 0
        while first_idx < len(messages) and messages[first_idx].tool_call_id:
            first_idx += 1

        if first_idx == 0:
            # No orphaned tool results at the start
            return messages

        # Find the tool_call for these orphaned results in the original native messages list
        try:
            msg_idx = self._native_messages.index(messages[0])
        except ValueError:
            return messages

        if msg_idx <= 0:
            return messages

        # Look backwards in the original native messages for the tool_call
        additional_messages: list[NativeMessage] = []
        for i in range(msg_idx - 1, -1, -1):
            msg = self._native_messages[i]
            additional_messages.insert(0, msg)
            if msg.tool_calls:
                # Found the tool_call, stop looking
                break

        return additional_messages + messages

    # ------------------------------------------------------------------
    # Compression prompt construction
    # ------------------------------------------------------------------

    def build_compression_prompt(self: CompressedTimeline) -> str | None:
        """
        Build a prompt for LLM-based compression using Claude Code technique.

        The prompt is designed to create a concise summary that preserves:
        - Key decisions and outcomes
        - Important facts and context
        - Tool calls and their results
        - User preferences and constraints

        Returns:
            Prompt string for summarization, or None if compression not needed
        """
        plan = self._plan_compression()
        if plan is None:
            return None
        entries_to_compress, prompt, plan_generation = plan

        # Remember what this prompt describes so a commit after the summary call
        # (``compress_old_entries`` — the pattern used by hosts that own their own
        # summarizer call) validates against it instead of re-partitioning a
        # timeline that may have moved on. See ``_pending_compression_plan``.
        with self._state_lock:
            self._pending_compression_plan = (entries_to_compress, plan_generation)

        return prompt

    def _render_compression_prompt(self: CompressedTimeline, entries_to_compress: list[TimelineEntry]) -> str:
        """Render the summarization prompt for an explicit compress partition.

        Split out of ``build_compression_prompt`` so compaction can render the
        prompt for the very partition it plans to commit, instead of
        re-deriving (and re-scanning) the partition a second time.
        """
        # Format entries for compression prompt
        formatted_entries = self._format_entries_for_compression(entries_to_compress)

        # Claude Code style compression prompt
        return f"""You are compressing conversation history to preserve important context.

Create a concise summary that captures:
1. Key facts, decisions, and outcomes
2. Important tool calls and their results
3. User requirements and constraints
4. Any ongoing tasks or pending items

The summary should allow the conversation to continue seamlessly without losing critical context.

IMPORTANT: Be concise but comprehensive. Focus on information that would be needed to continue the conversation intelligently.

Conversation history to compress:
{formatted_entries}

Respond with ONLY a JSON object containing the summary:
{{"summary": "Your compressed context summary here"}}"""

    def _format_entries_for_compression(self: CompressedTimeline, entries: list[TimelineEntry]) -> str:
        """
        Format timeline entries for the compression prompt.

        HIGH-1 fix: entries whose content was previously replaced by
        ``SHRINK_STUB_CONTENT`` (from ``cheap_shrink_tool_results``) are
        rendered as a compact identity-only marker rather than the literal
        stub string. Summarizing the literal ``"[cleared for context budget]"``
        produces vacuous summaries on reload after a prior shrink.

        Args:
            entries: List of entries to format

        Returns:
            Formatted string representation
        """
        role_map = {
            TimelineEntryType.USER_MESSAGE: "User",
            TimelineEntryType.AGENT_RESPONSE: "Assistant",
            TimelineEntryType.AGENT_THOUGHTS: "Assistant (thinking)",
            TimelineEntryType.TOOL_CALL: "Tool call",
            TimelineEntryType.RESOURCE_RESULT: "Tool result",
            TimelineEntryType.WORKFLOW_RESULT: "Workflow result",
            TimelineEntryType.SUB_AGENT_RESPONSE: "Sub-agent",
            TimelineEntryType.AGENT_LEARNING: "Learning",
            TimelineEntryType.TIMELINE_SUMMARY: "Previous summary",
            TimelineEntryType.TODO_LIST: "Todo list",
        }

        formatted_parts = []
        for entry in entries:
            content = entry.content

            # HIGH-1: skip or tag already-shrunk tool_result entries so the LLM
            # doesn't produce a summary consisting of "the agent cleared tool
            # results". Preserve call identity via tool_call_id when present.
            if isinstance(content, str) and content == SHRINK_STUB_CONTENT:
                tc_id = entry.tool_call_id or "unknown"
                formatted_parts.append(f"[Tool result id={tc_id}: previously cleared — content unavailable]")
                continue

            if isinstance(content, list):
                # Multimodal content: extract text parts for compression
                text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and "text" in b]
                content = " ".join(text_parts) if text_parts else "[multimodal content]"
            if len(content) > 1000:
                content = content[:1000] + "... [truncated]"

            role = role_map.get(entry.entry_type, entry.entry_type.value)
            formatted_parts.append(f"[{role}] {content}")

        return "\n\n".join(formatted_parts)

    # ------------------------------------------------------------------
    # Core compress / compress_async methods
    # ------------------------------------------------------------------

    def compress(self: CompressedTimeline) -> int:
        """
        Perform compression on the timeline.

        This method:
        1. Determines entries to keep and compress
        2. Calls LLM to generate summary of entries to compress
        3. Stores compressed context in metadata of oldest kept entry
        4. Removes compressed entries

        The summarizer call is not made under a lock (the default call function
        re-enters the runtime, so holding ``_compact_sync_lock`` across it could
        self-deadlock); the commit that follows is validated against the planned
        partition, so a concurrent mutation from another thread cannot be
        silently dropped.

        Returns:
            Number of entries compressed

        Raises:
            RuntimeError: If LLM call function is not set and no agent runtime available
        """
        # Use provided function or fall back to agent runtime
        llm_call_fn = self._llm_call_fn or self._get_default_llm_call_fn()
        if llm_call_fn is None:
            raise RuntimeError(
                "LLM call function not set and no agent runtime available. "
                "Use set_llm_call_fn() to provide a compression function or ensure an agent with runtime is provided."
            )

        if not self.needs_compression():
            return 0

        # Phase 2 — cheap shrink: if enabled and predicted savings are enough,
        # stub old tool_results and skip full summary. Predictive gate ensures
        # we never summarize over stubs.
        if self._compressed_config.enable_cheap_shrink_tool_results:
            with self._compact_sync_lock:
                if self.cheap_shrink_tool_results():
                    return 0

        plan = self._plan_compression()
        if plan is None:
            return 0

        entries_to_compress, prompt, plan_generation = plan

        # Call LLM to get summary. Deliberately not under a lock: the default
        # call function re-enters the runtime (and so this timeline's
        # compaction paths on PromptTooLongError), which a held sync lock would
        # turn into a self-deadlock. Concurrent commits are resolved by the
        # plan validation in ``_apply_compression`` instead.
        try:
            response = llm_call_fn(prompt)
            summary = self._extract_summary_from_response(response)
        except Exception as e:
            logger.error(f"Failed to compress timeline: {e}")
            return 0

        if not summary:
            logger.warning("LLM returned empty summary, skipping compression")
            return 0

        # Commit against the plan, validating that nothing else mutated the
        # model-facing state since the partition was captured.
        compressed_count = self._apply_compression(
            entries_to_compress,
            summary,
            plan_generation=plan_generation,
        )

        logger.info(
            f"Compressed {compressed_count} timeline entries",
            compressed_count=compressed_count,
            remaining_entries=len(self.timeline),
            summary_length=len(summary),
        )

        return compressed_count

    async def compress_async(self: CompressedTimeline) -> int:
        """
        Perform compression on the timeline asynchronously.

        The summarizer is awaited, so other tasks can mutate the timeline while
        it runs. The plan captured before the await is committed afterwards
        through the validated ``_apply_compression`` commit: entries appended
        meanwhile are retained, and a plan whose projection was replaced
        wholesale (hydration/load) is discarded instead of overwriting it.

        Returns:
            Number of entries compressed

        Raises:
            RuntimeError: If async LLM call function is not set and no agent runtime available
        """
        # Use provided function or fall back to agent runtime
        llm_call_async_fn = self._llm_call_async_fn or self._get_default_llm_call_async_fn()
        if llm_call_async_fn is None:
            raise RuntimeError(
                "Async LLM call function not set and no agent runtime available. "
                "Use set_llm_call_async_fn() to provide a compression function or ensure an agent with runtime is provided."
            )

        if not self.needs_compression():
            return 0

        # Phase 2 — cheap shrink under async lock.
        if self._compressed_config.enable_cheap_shrink_tool_results:
            async with self._compact_lock:
                if self.cheap_shrink_tool_results():
                    return 0

        plan = self._plan_compression()
        if plan is None:
            return 0

        entries_to_compress, prompt, plan_generation = plan

        # Call LLM to get summary
        try:
            response = await llm_call_async_fn(prompt)
            summary = self._extract_summary_from_response(response)
        except Exception as e:
            logger.error(f"Failed to compress timeline: {e}")
            return 0

        if not summary:
            logger.warning("LLM returned empty summary, skipping compression")
            return 0

        # Commit under the async lock, validating the captured plan.
        async with self._compact_lock:
            compressed_count = self._apply_compression(
                entries_to_compress,
                summary,
                plan_generation=plan_generation,
            )

        logger.info(
            f"Compressed {compressed_count} timeline entries",
            compressed_count=compressed_count,
            remaining_entries=len(self.timeline),
            summary_length=len(summary),
        )

        return compressed_count

    def _plan_compression(
        self: CompressedTimeline,
    ) -> tuple[list[TimelineEntry], str, int] | None:
        """Capture the compress partition, its prompt, and the projection generation.

        All three come from a single read of the live state, so the summary is
        generated for exactly the entries the commit will replace, and the commit
        can tell whether the projection it was planned against still exists.
        Returns ``None`` when nothing should be compressed.

        The retained set is deliberately *not* captured: the commit retains every
        live entry the summarizer did not see, which is what makes entries
        appended while the summarizer ran survive.
        """
        with self._state_lock:
            _, entries_to_compress = self.get_entries_to_keep_and_compress()
            if not entries_to_compress:
                return None
            prompt = self._render_compression_prompt(entries_to_compress)
            return entries_to_compress, prompt, self._compaction_generation

    # ------------------------------------------------------------------
    # Apply compression & extract summary
    # ------------------------------------------------------------------

    def _apply_compression(
        self: CompressedTimeline,
        entries_to_compress: list[TimelineEntry],
        summary: str,
        *,
        plan_generation: int | None = None,
    ) -> int:
        """
        Commit a compaction: store the summary and install the retained projection.

        This method updates both the legacy TimelineEntry list (``self.timeline``)
        and the native message list (``self._native_messages``) so the two cannot
        drift apart.

        The commit is exact: only the entries the summarizer actually saw
        (``entries_to_compress``, matched by object identity) are replaced by the
        summary; every other live entry — including ones appended while an awaited
        summarizer ran — is retained. A commit planned against a projection that
        has since been replaced wholesale (``plan_generation`` mismatch) is
        discarded with a warning rather than resurrecting the replaced history.

        Args:
            entries_to_compress: Entries the summary was generated from; defines
                exactly what this commit may remove.
            summary: The compressed summary
            plan_generation: ``_compaction_generation`` observed when the partition
                was planned. ``None`` for callers that plan and commit without
                suspending (they get the identity check alone).

        Returns:
            Number of entries compressed (0 when the commit was discarded)
        """
        compression_timestamp = datetime.now()

        with self._state_lock:
            # Any recorded prompt plan describes a partition this commit
            # supersedes; drop it so it cannot be replayed later.
            self._pending_compression_plan = None

            if plan_generation is not None and plan_generation != self._compaction_generation:
                # The projection was replaced wholesale (hydration/load/another
                # commit) while this summary was being produced. Committing would
                # resurrect the replaced history.
                logger.warning(
                    "compaction_commit_discarded",
                    reason="projection_replaced",
                    planned_entries=len(entries_to_compress),
                    live_entries=len(self.timeline),
                )
                return 0

            # Exact by object identity, and count-aware: if the same entry object
            # is live more than once (a caller re-adding an existing entry) only
            # the planned occurrences are removed, so an append during the
            # summarizer is never mistaken for a planned one.
            planned_remaining: dict[int, int] = {}
            for entry in entries_to_compress:
                key = id(entry)
                planned_remaining[key] = planned_remaining.get(key, 0) + 1

            live = self.timeline
            retained_flags: list[bool] = []
            new_keep: list[TimelineEntry] = []
            for entry in live:
                key = id(entry)
                if planned_remaining.get(key, 0) > 0:
                    planned_remaining[key] -= 1
                    retained_flags.append(False)
                    continue
                retained_flags.append(True)
                new_keep.append(entry)

            compressed_count = len(live) - len(new_keep)
            if compressed_count == 0:
                logger.warning(
                    "compaction_commit_discarded",
                    reason="planned_entries_no_longer_present",
                    planned_entries=len(entries_to_compress),
                    live_entries=len(live),
                )
                return 0

            # Stamp so the next save() rolls the active snapshot to a new
            # `timeline-after-compress-{ts}.json` file. See
            # ``CompressedTimeline.save`` for the snapshot rollover logic.
            self._last_compression_at = compression_timestamp

            # Compose summary metadata from scratch (no reasoning_items / encrypted_content
            # leak from compressed-away entries — those are stale state that doesn't
            # belong in the summary anyway, and the summary text already encodes outcomes).
            # Explicit dict construction here is the no-leak guarantee.
            summary_metadata = {
                COMPRESSED_CONTEXT_KEY: summary,
                COMPRESSION_TIMESTAMP_KEY: compression_timestamp.isoformat(),
                COMPRESSED_ENTRIES_COUNT_KEY: compressed_count,
            }

            # Create summary as a NativeMessage with role='system'
            summary_native_message = NativeMessage(
                role="system",
                content=f"[SUMMARY] {summary}",
                metadata=dict(summary_metadata),
                timestamp=compression_timestamp,
            )

            if not new_keep:
                # Edge case: create a summary entry if nothing to keep
                summary_entry = TimelineEntry(
                    entry_type=TimelineEntryType.TIMELINE_SUMMARY,
                    content=summary,
                    timestamp=entries_to_compress[0].timestamp if entries_to_compress else compression_timestamp,
                    metadata=dict(summary_metadata),
                )
                self._install_model_state([summary_entry], [summary_native_message])
                return compressed_count

            # Store compressed context in metadata of the oldest retained entry.
            # Ephemeral carriers are skipped: they are excluded from snapshots and
            # the next ``set_context`` replaces them, either of which would take
            # the marker (and so the reload cutoff) with it. Any retained entries
            # before that carrier are ephemeral too, so cutting at the carrier on
            # reload drops nothing that was persisted.
            carrier_idx = self._summary_carrier_index(retained_flags, live)
            oldest_kept = live[carrier_idx]
            oldest_kept.metadata[COMPRESSED_CONTEXT_KEY] = summary
            oldest_kept.metadata[COMPRESSION_TIMESTAMP_KEY] = compression_timestamp.isoformat()
            oldest_kept.metadata[COMPRESSED_ENTRIES_COUNT_KEY] = compressed_count

            # Map the retained entries onto their native mirrors by identity, not
            # by position or tail length (see ``_rebuild_native_mirrors``). The
            # marker then goes on the carrier's own mirror.
            mirrors = self._rebuild_native_mirrors(live, retained_flags)
            carrier_message = mirrors[retained_flags[: carrier_idx + 1].count(True) - 1]

            message_metadata = carrier_message.metadata
            if message_metadata is None:
                message_metadata = carrier_message.metadata = {}
            message_metadata[COMPRESSED_CONTEXT_KEY] = summary
            message_metadata[COMPRESSION_TIMESTAMP_KEY] = compression_timestamp.isoformat()
            message_metadata[COMPRESSED_ENTRIES_COUNT_KEY] = compressed_count

            self._install_model_state(new_keep, [summary_native_message] + mirrors)

        return compressed_count

    @staticmethod
    def _suffix_retention_flags(live: list[TimelineEntry], retained: list[TimelineEntry]) -> list[bool]:
        """Which live occurrences survive a right-truncating retention step.

        ``reactive_compact`` keeps a suffix of ``live`` (plus, via
        ``_ensure_tool_pair_integrity``, a contiguous run of entries immediately
        before it). Matching by identity *set* would wrongly mark an earlier
        removed occurrence of an entry that is retained later as retained, and
        that removed occurrence would then consume a mirror that belongs to the
        retained one. Counting occurrences from the right instead marks exactly
        the retained occurrences, without assuming the two lists are aligned.

        Returns:
            One flag per ``live`` entry, in order.
        """
        remaining: dict[int, int] = {}
        for entry in retained:
            key = id(entry)
            remaining[key] = remaining.get(key, 0) + 1

        flags: list[bool] = [False] * len(live)
        for index in range(len(live) - 1, -1, -1):
            key = id(live[index])
            if remaining.get(key, 0) > 0:
                remaining[key] -= 1
                flags[index] = True
        return flags

    def _rebuild_native_mirrors(
        self: CompressedTimeline,
        live: list[TimelineEntry],
        retained_flags: list[bool],
    ) -> list[NativeMessage]:
        """Replace the native projection with the mirrors of the retained entries.

        The two model-facing lists cannot be assumed index-aligned: a compaction
        prepends a summary message, ``set_context`` prepends a context entry to
        both, and callers may inject into either. So each retained entry is mapped
        to the message built from it (``NATIVE_SOURCE_ENTRY_KEY`` backlink, matched
        by identity and count-aware for entries added twice), which preserves that
        message's reasoning items, tool calls and replay state.

        A retained entry with no tagged mirror (messages that never came through
        ``_timeline_entry_to_native_message``, e.g. decoded from a snapshot that
        dropped the backlink) is rebuilt from its entry — the conservative path:
        nothing retained is lost, no compressed-away message is resurrected, and
        the cost is that rebuild only.

        Returns:
            The installed mirrors, in retained-entry order. Callers hold
            ``_state_lock`` (it is an RLock) while they replace the entry
            projection alongside this one.
        """
        messages_by_entry_id: dict[int, deque[NativeMessage]] = {}
        for message in self._native_messages:
            source_entry = (message.metadata or {}).get(NATIVE_SOURCE_ENTRY_KEY)
            if source_entry is None:
                continue
            messages_by_entry_id.setdefault(id(source_entry), deque()).append(message)

        mirrors: list[NativeMessage] = []
        rebuilt_any = False
        for index, entry in enumerate(live):
            # One mirror is consumed per *live occurrence*, including occurrences
            # being compressed away: the same entry object can be live more than
            # once (a caller re-adding it), and skipping the removed occurrences
            # would hand a retained later occurrence the earlier, compressed-away
            # occurrence's message.
            queue = messages_by_entry_id.get(id(entry))
            message = queue.popleft() if queue else None
            if not retained_flags[index]:
                continue
            if message is None:
                message = self._timeline_entry_to_native_message(entry)
                rebuilt_any = True
            mirrors.append(message)

        if rebuilt_any:
            logger.warning(
                "compaction_native_mirror_rebuilt",
                reason="retained_entry_without_mirror",
                retained_entries=len(mirrors),
            )

        self._native_messages = mirrors
        return mirrors

    def _summary_carrier_index(self: CompressedTimeline, retained_flags: list[bool], live: list[TimelineEntry]) -> int:
        """Index in ``live`` of the entry that carries the compaction marker.

        The oldest retained entry — except that a retained ephemeral entry (the
        runtime CONTEXT prepended by ``set_context``) is never used as the carrier:
        ephemeral entries do not survive persistence and the next ``set_context``
        replaces them, so a marker placed there would be silently lost.
        """
        for index, keep in enumerate(retained_flags):
            if keep and not live[index].ephemeral:
                return index
        # Everything retained is ephemeral: fall back to the first retained entry.
        return retained_flags.index(True)

    def _extract_summary_from_response(self, response: str) -> str | None:
        """
        Extract summary from LLM response.

        Handles both JSON format {"summary": "..."} and plain text.

        Args:
            response: LLM response string

        Returns:
            Extracted summary or None if extraction fails
        """
        import json

        response = response.strip()

        # Try JSON extraction first
        try:
            # Handle potential markdown code blocks
            if response.startswith("```"):
                # Extract content between code blocks
                lines = response.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                response = "\n".join(json_lines)

            data = json.loads(response)
            if isinstance(data, dict) and "summary" in data:
                return data["summary"]
        except json.JSONDecodeError:
            pass

        # If not JSON, use the response as-is (but clean it up)
        if response and not response.startswith("{"):
            return response

        return None

    # ------------------------------------------------------------------
    # Public compressed context accessors
    # ------------------------------------------------------------------

    def get_compressed_context(self: CompressedTimeline) -> str | None:
        """
        Get the compressed context from the timeline if available.

        Looks for compressed context metadata in both native messages and
        timeline entries (for backward compatibility).

        Returns:
            Compressed context string or None if not available
        """
        # First check native messages (preferred source after compression)
        for msg in self._native_messages:
            if COMPRESSED_CONTEXT_KEY in msg.metadata:
                return msg.metadata[COMPRESSED_CONTEXT_KEY]

        # Fall back to timeline entries (for backward compatibility)
        for entry in self.timeline:
            if COMPRESSED_CONTEXT_KEY in entry.metadata:
                return entry.metadata[COMPRESSED_CONTEXT_KEY]
        return None

    def has_compressed_context(self: CompressedTimeline) -> bool:
        """
        Check if timeline has compressed context.

        Returns:
            True if compressed context exists in any entry's metadata
        """
        return self.get_compressed_context() is not None

    def compress_old_entries(self: CompressedTimeline, summary: str) -> int:
        """
        Compress old timeline entries into a summary entry.

        This method implements the TimelineProtocol interface for compression.
        It replaces old entries with a summary, updating both self.timeline
        and self._native_messages to maintain consistency.

        For CompressedTimeline, this method uses the same logic as _apply_compression()
        but accepts an externally provided summary (e.g., from LLM-based summarization).
        The caller owns the summarizer call (typically ``build_compression_prompt``
        → host LLM call → this method). It commits the partition that prompt was
        built from, recorded by ``build_compression_prompt``, so a summary is never
        applied to entries it did not include. Falls back to planning here when no
        prompt was built first, and ``_apply_compression`` validates either way.

        Args:
            summary: The summary text to use for the compressed entries

        Returns:
            Number of entries that were compressed
        """
        with self._compact_sync_lock:
            pending = self._pending_compression_plan
            self._pending_compression_plan = None
            if pending is not None:
                # The recorded prompt partition is what the summary was generated
                # from; commit exactly that.
                entries_to_compress, plan_generation = pending
                return self._apply_compression(entries_to_compress, summary, plan_generation=plan_generation)

            _, entries_to_compress = self.get_entries_to_keep_and_compress()
            if not entries_to_compress:
                return 0
            return self._apply_compression(entries_to_compress, summary)

    # ------------------------------------------------------------------
    # Phase 3 — reactive compaction + circuit breaker
    # ------------------------------------------------------------------

    def _circuit_cooldown_seconds(self: CompressedTimeline) -> int:
        """Circuit-breaker cooldown from env `DANA_CIRCUIT_COOLDOWN_SECONDS` (default 300)."""
        import os

        raw = os.getenv("DANA_CIRCUIT_COOLDOWN_SECONDS")
        if not raw:
            return 300
        try:
            v = int(raw)
            return max(1, v)
        except (ValueError, TypeError):
            return 300

    def reset_circuit(self: CompressedTimeline) -> None:
        """Force-close the compaction circuit (ops escape hatch)."""
        self._consecutive_compact_failures = 0
        self._compaction_disabled = False
        self._circuit_opened_at = None

    def _check_circuit_and_probe(self: CompressedTimeline) -> None:
        """Raise `CompactCircuitOpenError` if circuit open and cooldown unexpired.

        If cooldown has elapsed, enter half-open state (disabled=False) so ONE
        attempt can run. Failure re-opens; success closes via reset_circuit().
        """
        from dana.common.llm.types import CompactCircuitOpenError

        if not self._compaction_disabled:
            return
        opened_at = self._circuit_opened_at
        if opened_at is None:
            return
        cooldown = self._circuit_cooldown_seconds()
        elapsed = (datetime.now() - opened_at).total_seconds()
        if elapsed < cooldown:
            remaining = cooldown - elapsed
            raise CompactCircuitOpenError(f"compaction circuit open; cooldown remaining: {remaining:.0f}s")
        # Half-open: allow one probe.
        self._compaction_disabled = False

    def reactive_compact(self: CompressedTimeline, attempt: int) -> None:
        """Drop old kept entries progressively, prune forward orphans, re-summarize.

        Drop counts: attempt 1 → 5, 2 → 10, 3 → 20. Always runs the full
        summary path (never shrink-bypass). Raises `CompactCircuitOpenError`
        when the circuit is open and cooldown not elapsed.
        """
        from dana.common.llm.types import CompactCircuitOpenError

        with self._compact_sync_lock:
            self._check_circuit_and_probe()

            drop_count = {1: 5, 2: 10, 3: 20}.get(attempt, 20)
            if len(self.timeline) <= drop_count:
                drop_count = max(0, len(self.timeline) - 1)

            if drop_count <= 0:
                # Nothing useful to drop — treat as attempt failure.
                self._consecutive_compact_failures += 1
                if self._consecutive_compact_failures >= 3:
                    self._compaction_disabled = True
                    self._circuit_opened_at = datetime.now()
                    raise CompactCircuitOpenError(f"reactive_compact cannot drop entries; timeline={len(self.timeline)}")
                return

            # Drop the oldest entries, then prune forward-orphans. The native
            # projection is rebuilt from the entries that actually remain — by
            # occurrence, not by identity set — rather than by slicing a
            # same-length tail off ``_native_messages``, which goes wrong as soon
            # as the two lists differ in length (any prior compaction prepends a
            # summary message).
            with self._state_lock:
                new_entries = self._ensure_tool_pair_integrity(self._remove_forward_orphans(self.timeline[drop_count:]))
                mirrors = self._rebuild_native_mirrors(self.timeline, self._suffix_retention_flags(self.timeline, new_entries))
                self._install_model_state(new_entries, mirrors)

            # Re-summarize via internal primitives so failures propagate into the
            # circuit-breaker counter (compress() swallows exceptions).
            try:
                llm_call_fn = self._llm_call_fn or self._get_default_llm_call_fn()
                if llm_call_fn is None:
                    raise RuntimeError("no LLM call function available for reactive_compact")
                plan = self._plan_compression()
                n = 0
                if plan is not None:
                    entries_to_compress, prompt, plan_generation = plan
                    response = llm_call_fn(prompt)
                    summary = self._extract_summary_from_response(response)
                    if not summary:
                        raise RuntimeError("empty summary from LLM")
                    n = self._apply_compression(
                        entries_to_compress,
                        summary,
                        plan_generation=plan_generation,
                    )
                # Success — close circuit and reset counter.
                self._consecutive_compact_failures = 0
                self._compaction_disabled = False
                self._circuit_opened_at = None
                logger.info(
                    "reactive_compact_done",
                    attempt=attempt,
                    entries_dropped=drop_count,
                    entries_compressed=n,
                )
            except CompactCircuitOpenError:
                raise
            except Exception as e:
                self._consecutive_compact_failures += 1
                if self._consecutive_compact_failures >= 3:
                    self._compaction_disabled = True
                    self._circuit_opened_at = datetime.now()
                    raise CompactCircuitOpenError(f"reactive_compact failed 3x consecutive; last error: {e}") from e
                raise

    def get_entries_for_compression(self: CompressedTimeline) -> list[TimelineEntry]:
        """
        Get the entries that would be compressed.

        This method implements the TimelineProtocol interface and returns
        the entries that would be replaced by a summary during compression.
        It uses the CompressedTimeline's own partitioning logic which respects
        token limits and recent entry counts.

        Returns:
            List of old entries that would be replaced by a summary
        """
        _, entries_to_compress = self.get_entries_to_keep_and_compress()
        return entries_to_compress
