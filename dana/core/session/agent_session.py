"""
AgentSession — host-neutral orchestrator for one durable text turn.

An :class:`AgentSession` owns one isolated agent, an owner/workspace scope, a
session journal identity and version, and the active turn. It serializes
mutations: only one active turn per session; a conflicting :meth:`prompt`
raises :class:`SessionBusy`. All turn lifecycle facts are journaled before,
during, and after the model call, enforcing input durability and exactly one
terminal fact per turn.

D1 is text-only: the agent is driven through
:meth:`~dana.core.agent.star_agent_streaming.STARAgentStreamingMixin.aquery_text_stream`,
which yields immediate text deltas without buffering or emitting THINKING events.

D2 adds tool lifecycle wiring: the session can emit thought events and tool
lifecycle events (requested, started, progress, result, cancellation) as
:class:`HostEvent` values. The :class:`ToolExecutionEngine` is wired in to
execute tool calls and journal tool lifecycle facts.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import time
from typing import Any
from uuid import uuid4

import structlog

from dana.core.policy.modes import PermissionMode
from dana.core.session.journal.protocol import JournalRepository
from dana.core.session.models import FactType, NewJournalFact, OwnerScope
from dana.core.session.projections.conversation import ConversationProjector, ConversationView
from dana.core.session.projections.host_events import HostEvent, HostEventProjector, HostEventType
from dana.core.session.protected_state import ProtectedStateCodec


logger = structlog.get_logger()


# The three fact types that close a turn. Exactly one of these terminates a turn.
_TERMINAL_FACT_TYPES = frozenset({FactType.TURN_COMPLETED, FactType.TURN_CANCELLED, FactType.TURN_ERROR})


@dataclass(frozen=True, slots=True)
class TextBlock:
    """A text content block for a prompt."""

    text: str


@dataclass(frozen=True, slots=True)
class TurnTerminal:
    """The terminal outcome of a turn.

    Exactly one terminal fact is appended per turn; this value is exposed via
    :attr:`AgentSession.last_terminal` after the :meth:`AgentSession.prompt`
    generator is exhausted.
    """

    fact_type: FactType
    sequence: int
    text: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# D2: Agent stream event types — richer than text-only
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentThought:
    """A thought/thinking chunk emitted by the agent during a turn.

    The session yields a :attr:`HostEventType.THOUGHT` host event for each
    thought chunk.
    """

    text: str


@dataclass(frozen=True, slots=True)
class AgentToolCallRequest:
    """A tool call request emitted by the agent.

    The session routes this through the :class:`ToolExecutionEngine` and
    yields tool lifecycle host events.
    """

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    kind: str | None = None


@dataclass(frozen=True, slots=True)
class AgentToolResult:
    """A tool result emitted by the agent (after the engine executed it).

    The session yields a :attr:`HostEventType.TOOL_RESULT` host event.
    """

    tool_call_id: str
    result: dict[str, Any] | None = None
    error: str | None = None


# Union of all event types the agent can yield in its stream.
AgentStreamEvent = str | AgentThought | AgentToolCallRequest | AgentToolResult


class SessionBusy(Exception):
    """Raised when a prompt conflicts with an already-active turn."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session {session_id} has an active turn")


class AgentSession:
    """Host-neutral session owning one agent and one active turn.

    Serializes mutations: only one active turn per session. A conflicting
    :meth:`prompt` raises :class:`SessionBusy` immediately (non-blocking).
    All turn lifecycle facts are journaled:

    - ``TURN_STARTED`` + ``USER_CONTENT_FINAL`` are appended BEFORE the model
      call (input durability).
    - Assistant text chunks are flushed to the journal by a byte/time bound.
    - ``ASSISTANT_CONTENT_FINAL`` + exactly one terminal fact are appended in
      ONE final batch.
    - No facts are appended after the terminal (no post-terminal reflection).
    """

    # Flush buffered chunks to the journal when accumulated bytes exceed this.
    CHUNK_FLUSH_BYTES = 4096
    # Or when this many seconds pass since the last flush.
    CHUNK_FLUSH_INTERVAL = 2.0

    def __init__(
        self,
        owner_scope: OwnerScope,
        session_id: str,
        repository: JournalRepository,
        agent_factory: Callable[[], Any],
        protected_state_codec: ProtectedStateCodec | None = None,
        tool_engine: Any | None = None,
        use_legacy_executor: bool = False,
    ) -> None:
        self._owner_scope = owner_scope
        self._session_id = session_id
        self._repository = repository
        self._agent_factory = agent_factory
        self._codec = protected_state_codec
        self._conversation_projector = ConversationProjector(protected_state_codec)
        self._host_event_projector = HostEventProjector()
        self._lock = asyncio.Lock()
        self._cancel_event: asyncio.Event | None = None
        self._agent: Any = None
        self._current_version: int = 0
        self._last_terminal: TurnTerminal | None = None
        # D2: Tool execution engine (optional — None for text-only sessions)
        self._tool_engine = tool_engine
        # D2: Rollback flag — selects legacy executor for non-ACP hosts
        self._use_legacy_executor = use_legacy_executor
        # D3: Permission mode state (ADR-013: mode state in session/new + session/set_mode)
        self._permission_mode: PermissionMode = PermissionMode.DEFAULT
        # D3: Policy evaluator (optional — wired by ACP agent for permission adapter)
        self._policy_evaluator: Any = None

    @property
    def last_terminal(self) -> TurnTerminal | None:
        """The terminal outcome of the most recently completed turn, or ``None``."""
        return self._last_terminal

    # ------------------------------------------------------------------
    # D3: Permission mode (ADR-013)
    # ------------------------------------------------------------------

    @property
    def permission_mode(self) -> PermissionMode:
        """The current permission mode for this session."""
        return self._permission_mode

    def set_permission_mode(self, mode: PermissionMode) -> None:
        """Set the permission mode (ADR-013: outside an active turn).

        Args:
            mode: The ``PermissionMode`` to set.
        """
        self._permission_mode = mode
        if self._policy_evaluator is not None:
            self._policy_evaluator.set_mode(mode)

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Load the session from the journal: read facts, set up agent + version."""
        await self._prepare_agent()

    async def cancel(self) -> None:
        """Request cancellation of the active turn.

        Sets the internal cancel event; the running :meth:`prompt` loop will
        catch the resulting :class:`asyncio.CancelledError` and terminalize the
        turn as ``TURN_CANCELLED``. The terminal outcome is available via
        :attr:`last_terminal` once :meth:`prompt` completes.
        """
        if self._cancel_event is None:
            raise RuntimeError("cancel() called with no active turn; call prompt() first and consume it concurrently")
        self._cancel_event.set()

    async def replay_host_events(self, after_sequence: int = 0) -> AsyncIterator[HostEvent]:
        """Replay host-visible events from the journal in sequence order."""
        facts = await self._repository.read_facts(self._owner_scope, self._session_id, after_sequence)
        for event in self._host_event_projector.project(facts):
            yield event

    async def prompt(self, blocks: Sequence[TextBlock]) -> AsyncIterator[HostEvent]:
        """Run one text turn. Yields host events as they occur.

        After the generator is exhausted, the :class:`TurnTerminal` outcome is
        available via :attr:`last_terminal`.

        Raises :class:`SessionBusy` if a turn is already active.
        """
        # Non-blocking conflict check: do not await the lock if it is held.
        if self._lock.locked():
            raise SessionBusy(self._session_id)
        async with self._lock:
            self._cancel_event = asyncio.Event()
            correlation_id = str(uuid4())
            user_text = " ".join(b.text for b in blocks)

            await self._prepare_agent()

            # --- Input durability: persist TURN_STARTED + USER_CONTENT_FINAL
            # BEFORE invoking the model. ---
            start_facts = [
                NewJournalFact(
                    fact_type=FactType.TURN_STARTED,
                    correlation_id=correlation_id,
                    causation_id=None,
                    payload={"prompt_summary": user_text[:200]},
                ),
                NewJournalFact(
                    fact_type=FactType.USER_CONTENT_FINAL,
                    correlation_id=correlation_id,
                    causation_id=correlation_id,
                    payload={"text": user_text},
                ),
            ]
            start_result = await self._repository.append(self._owner_scope, self._session_id, self._current_version, start_facts)
            self._current_version = start_result.new_version

            logger.info(
                "turn started",
                session_id=self._session_id,
                correlation_id=correlation_id,
                version=self._current_version,
            )

            yield HostEvent(
                event_type=HostEventType.TURN_STARTED,
                sequence=start_result.appended_facts[0].sequence,
                correlation_id=correlation_id,
                timestamp=start_result.appended_facts[0].timestamp,
                metadata={"prompt_summary": user_text[:200]},
            )
            yield HostEvent(
                event_type=HostEventType.USER_MESSAGE,
                sequence=start_result.appended_facts[1].sequence,
                correlation_id=correlation_id,
                timestamp=start_result.appended_facts[1].timestamp,
                text=user_text,
            )

            self._add_user_message_to_timeline(user_text)

            accumulated: list[str] = []
            chunk_buffer: list[str] = []
            chunk_buffer_bytes = 0
            last_flush = time.monotonic()
            chunk_index = 0

            try:
                result_holder: dict[str, Any] = {}
                async for chunk in self._agent.aquery_text_stream(
                    message=user_text,
                    cancel_event=self._cancel_event,
                    result_holder=result_holder,
                ):
                    accumulated.append(chunk)
                    # Pre-persistence chunk: the fact sequence isn't known until
                    # the buffered chunks are flushed to the journal. Use 0 to
                    # signal "not yet persisted"; the host receives chunks in
                    # stream order regardless. On replay, replay_host_events
                    # returns these events with their real fact sequences.
                    yield HostEvent(
                        event_type=HostEventType.ASSISTANT_CONTENT_CHUNK,
                        sequence=0,
                        correlation_id=correlation_id,
                        timestamp=datetime.now(UTC),
                        text=chunk,
                    )
                    # Bounded flush: accumulate then persist when bound is hit.
                    chunk_buffer.append(chunk)
                    chunk_buffer_bytes += len(chunk)
                    now = time.monotonic()
                    if chunk_buffer_bytes >= self.CHUNK_FLUSH_BYTES or (now - last_flush) >= self.CHUNK_FLUSH_INTERVAL:
                        await self._flush_chunks(correlation_id, chunk_buffer, chunk_index)
                        chunk_index += len(chunk_buffer)
                        chunk_buffer.clear()
                        chunk_buffer_bytes = 0
                        last_flush = now

                # Flush any remaining buffered chunks before the terminal batch.
                if chunk_buffer:
                    await self._flush_chunks(correlation_id, chunk_buffer, chunk_index)

                full_text = result_holder.get("full_text") or "".join(accumulated)
                protected_payload = result_holder.get("protected_payload")

                # --- Terminal batch: ASSISTANT_CONTENT_FINAL + terminal in ONE append. ---
                terminal_facts = [
                    NewJournalFact(
                        fact_type=FactType.ASSISTANT_CONTENT_FINAL,
                        correlation_id=correlation_id,
                        causation_id=correlation_id,
                        payload={"text": full_text},
                        protected_payload=protected_payload,
                    ),
                    NewJournalFact(
                        fact_type=FactType.TURN_COMPLETED,
                        correlation_id=correlation_id,
                        causation_id=correlation_id,
                        payload={},
                    ),
                ]
                terminal_result = await self._repository.append(self._owner_scope, self._session_id, self._current_version, terminal_facts)
                self._current_version = terminal_result.new_version

                yield HostEvent(
                    event_type=HostEventType.ASSISTANT_CONTENT_FINAL,
                    sequence=terminal_result.appended_facts[0].sequence,
                    correlation_id=correlation_id,
                    timestamp=terminal_result.appended_facts[0].timestamp,
                    text=full_text,
                )
                yield HostEvent(
                    event_type=HostEventType.TURN_COMPLETED,
                    sequence=terminal_result.appended_facts[1].sequence,
                    correlation_id=correlation_id,
                    timestamp=terminal_result.appended_facts[1].timestamp,
                )
                self._last_terminal = TurnTerminal(
                    fact_type=FactType.TURN_COMPLETED,
                    sequence=terminal_result.appended_facts[1].sequence,
                    text=full_text,
                )
                logger.info(
                    "turn completed",
                    session_id=self._session_id,
                    correlation_id=correlation_id,
                    version=self._current_version,
                )
                return

            except asyncio.CancelledError:
                # Cancellation requested via cancel(). Persist any buffered
                # partial text, then a single TURN_CANCELLED terminal.
                try:
                    if chunk_buffer:
                        await self._flush_chunks(correlation_id, chunk_buffer, chunk_index)
                    partial = "".join(accumulated)
                    cancel_facts = [
                        NewJournalFact(
                            fact_type=FactType.TURN_CANCELLED,
                            correlation_id=correlation_id,
                            causation_id=correlation_id,
                            payload={"partial_text": partial},
                        ),
                    ]
                    cancel_result = await self._repository.append(self._owner_scope, self._session_id, self._current_version, cancel_facts)
                    self._current_version = cancel_result.new_version
                except Exception:
                    # Don't swallow the original cancellation: log the
                    # terminal-append failure and re-raise CancelledError so
                    # the caller knows cancellation was the trigger.
                    logger.warning(
                        "failed to append cancellation terminal",
                        session_id=self._session_id,
                        correlation_id=correlation_id,
                        exc_info=True,
                    )
                    raise
                yield HostEvent(
                    event_type=HostEventType.TURN_CANCELLED,
                    sequence=cancel_result.appended_facts[0].sequence,
                    correlation_id=correlation_id,
                    timestamp=cancel_result.appended_facts[0].timestamp,
                    text=partial,
                )
                self._last_terminal = TurnTerminal(
                    fact_type=FactType.TURN_CANCELLED,
                    sequence=cancel_result.appended_facts[0].sequence,
                    text=partial,
                )
                logger.warning(
                    "turn cancelled",
                    session_id=self._session_id,
                    correlation_id=correlation_id,
                    version=self._current_version,
                )
                return
            except Exception as exc:
                # Model/runtime error: terminalize as TURN_ERROR.
                try:
                    error_facts = [
                        NewJournalFact(
                            fact_type=FactType.TURN_ERROR,
                            correlation_id=correlation_id,
                            causation_id=correlation_id,
                            payload={"error": str(exc)},
                        ),
                    ]
                    error_result = await self._repository.append(self._owner_scope, self._session_id, self._current_version, error_facts)
                    self._current_version = error_result.new_version
                except Exception:
                    # Don't swallow the original error: log the terminal-append
                    # failure and re-raise the original exception.
                    logger.warning(
                        "failed to append error terminal",
                        session_id=self._session_id,
                        correlation_id=correlation_id,
                        exc_info=True,
                    )
                    raise
                yield HostEvent(
                    event_type=HostEventType.TURN_ERROR,
                    sequence=error_result.appended_facts[0].sequence,
                    correlation_id=correlation_id,
                    timestamp=error_result.appended_facts[0].timestamp,
                    metadata={"error": str(exc)},
                )
                self._last_terminal = TurnTerminal(
                    fact_type=FactType.TURN_ERROR,
                    sequence=error_result.appended_facts[0].sequence,
                    error=str(exc),
                )
                logger.warning(
                    "turn errored",
                    session_id=self._session_id,
                    correlation_id=correlation_id,
                    version=self._current_version,
                    error=str(exc),
                )
                return
            finally:
                self._cancel_event = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _prepare_agent(self) -> None:
        """Create the agent (once) and (re)build its timeline from the journal."""
        # TODO(d2): incremental conversation view update instead of full re-read.
        if self._agent is None:
            self._agent = self._agent_factory()
        facts = await self._repository.read_facts(self._owner_scope, self._session_id)
        self._current_version = max((f.sequence for f in facts), default=0)
        view = self._conversation_projector.project(facts)
        self._populate_timeline(view)

    async def _flush_chunks(self, correlation_id: str, chunk_buffer: list[str], start_index: int) -> None:
        """Persist buffered assistant text as a single ASSISTANT_CONTENT_CHUNK fact."""
        if not chunk_buffer:
            return
        chunk_text = "".join(chunk_buffer)
        chunk_facts = [
            NewJournalFact(
                fact_type=FactType.ASSISTANT_CONTENT_CHUNK,
                correlation_id=correlation_id,
                causation_id=correlation_id,
                payload={"text": chunk_text, "index": start_index},
            )
        ]
        result = await self._repository.append(self._owner_scope, self._session_id, self._current_version, chunk_facts)
        self._current_version = result.new_version

    def _populate_timeline(self, view: ConversationView) -> None:
        """Rebuild the agent's timeline from the projected conversation view.

        Defensive against agents that lack a ``_timeline`` (e.g. fakes in tests).
        """
        from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType

        timeline = getattr(self._agent, "_timeline", None)
        entries = getattr(timeline, "timeline", None)
        if entries is None:
            return
        entries.clear()
        if view.interruption_observation:
            entries.append(TimelineEntry(entry_type=TimelineEntryType.CONTEXT, content=view.interruption_observation))
        for msg in view.messages:
            if msg.role == "user":
                entries.append(TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content=str(msg.content)))
            elif msg.role == "assistant":
                entries.append(TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content=str(msg.content)))

    def _add_user_message_to_timeline(self, text: str) -> None:
        """Append the current user message to the agent's timeline (for build_prompt)."""
        from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType

        timeline = getattr(self._agent, "_timeline", None)
        entries = getattr(timeline, "timeline", None)
        if entries is None:
            return
        entries.append(TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content=text))

    # ------------------------------------------------------------------
    # D2: Tool lifecycle — emit host events and journal tool facts
    # ------------------------------------------------------------------

    async def emit_thought(self, text: str, correlation_id: str) -> HostEvent:
        """Emit a thought event (not journaled — live-only)."""
        event = HostEvent(
            event_type=HostEventType.THOUGHT,
            sequence=0,
            correlation_id=correlation_id,
            timestamp=datetime.now(UTC),
            text=text,
        )
        return event

    async def journal_tool_requested(
        self,
        tool_call_id: str,
        tool_name: str,
        correlation_id: str,
        arguments: dict[str, Any] | None = None,
        kind: str | None = None,
    ) -> HostEvent:
        """Journal a TOOL_REQUESTED fact and return the host event."""
        facts = [
            NewJournalFact(
                fact_type=FactType.TOOL_REQUESTED,
                correlation_id=correlation_id,
                causation_id=correlation_id,
                payload={
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "arguments": arguments or {},
                    "kind": kind,
                },
            )
        ]
        result = await self._repository.append(self._owner_scope, self._session_id, self._current_version, facts)
        self._current_version = result.new_version
        return HostEvent(
            event_type=HostEventType.TOOL_REQUESTED,
            sequence=result.appended_facts[0].sequence,
            correlation_id=correlation_id,
            timestamp=result.appended_facts[0].timestamp,
            metadata={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "kind": kind,
                "raw_input": arguments,
            },
        )

    async def journal_tool_authorized_or_denied(
        self,
        tool_call_id: str,
        correlation_id: str,
        authorized: bool = True,
        reason: str | None = None,
    ) -> HostEvent:
        """Journal a TOOL_AUTHORIZED_OR_DENIED fact and return the host event."""
        facts = [
            NewJournalFact(
                fact_type=FactType.TOOL_AUTHORIZED_OR_DENIED,
                correlation_id=correlation_id,
                causation_id=correlation_id,
                payload={
                    "tool_call_id": tool_call_id,
                    "authorized": authorized,
                    "reason": reason,
                },
            )
        ]
        result = await self._repository.append(self._owner_scope, self._session_id, self._current_version, facts)
        self._current_version = result.new_version
        return HostEvent(
            event_type=HostEventType.TOOL_AUTHORIZED_OR_DENIED,
            sequence=result.appended_facts[0].sequence,
            correlation_id=correlation_id,
            timestamp=result.appended_facts[0].timestamp,
            metadata={
                "tool_call_id": tool_call_id,
                "authorized": authorized,
                "reason": reason,
            },
        )

    async def journal_tool_started(
        self,
        tool_call_id: str,
        correlation_id: str,
    ) -> HostEvent:
        """Journal a TOOL_STARTED fact and return the host event."""
        facts = [
            NewJournalFact(
                fact_type=FactType.TOOL_STARTED,
                correlation_id=correlation_id,
                causation_id=correlation_id,
                payload={"tool_call_id": tool_call_id},
            )
        ]
        result = await self._repository.append(self._owner_scope, self._session_id, self._current_version, facts)
        self._current_version = result.new_version
        return HostEvent(
            event_type=HostEventType.TOOL_STARTED,
            sequence=result.appended_facts[0].sequence,
            correlation_id=correlation_id,
            timestamp=result.appended_facts[0].timestamp,
            metadata={"tool_call_id": tool_call_id},
        )

    async def journal_tool_progress(
        self,
        tool_call_id: str,
        correlation_id: str,
        progress: dict[str, Any] | None = None,
    ) -> HostEvent:
        """Journal a TOOL_PROGRESS fact and return the host event."""
        facts = [
            NewJournalFact(
                fact_type=FactType.TOOL_PROGRESS,
                correlation_id=correlation_id,
                causation_id=correlation_id,
                payload={
                    "tool_call_id": tool_call_id,
                    "progress": progress or {},
                },
            )
        ]
        result = await self._repository.append(self._owner_scope, self._session_id, self._current_version, facts)
        self._current_version = result.new_version
        return HostEvent(
            event_type=HostEventType.TOOL_PROGRESS,
            sequence=result.appended_facts[0].sequence,
            correlation_id=correlation_id,
            timestamp=result.appended_facts[0].timestamp,
            metadata={"tool_call_id": tool_call_id, "progress": progress},
        )

    async def journal_tool_cancellation_requested(
        self,
        tool_call_id: str,
        correlation_id: str,
    ) -> HostEvent:
        """Journal a TOOL_CANCELLATION_REQUESTED fact and return the host event."""
        facts = [
            NewJournalFact(
                fact_type=FactType.TOOL_CANCELLATION_REQUESTED,
                correlation_id=correlation_id,
                causation_id=correlation_id,
                payload={"tool_call_id": tool_call_id},
            )
        ]
        result = await self._repository.append(self._owner_scope, self._session_id, self._current_version, facts)
        self._current_version = result.new_version
        return HostEvent(
            event_type=HostEventType.TOOL_CANCELLATION_REQUESTED,
            sequence=result.appended_facts[0].sequence,
            correlation_id=correlation_id,
            timestamp=result.appended_facts[0].timestamp,
            metadata={"tool_call_id": tool_call_id},
        )

    async def journal_tool_terminal(
        self,
        tool_call_id: str,
        correlation_id: str,
        terminal_type: FactType,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> HostEvent:
        """Journal a terminal tool fact and return the host event.

        Args:
            terminal_type: One of TOOL_RESULT, TOOL_FAILURE, TOOL_ACKNOWLEDGED,
                TOOL_TIMED_OUT, TOOL_EFFECT_UNKNOWN.
        """
        payload: dict[str, Any] = {"tool_call_id": tool_call_id}
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = error

        facts = [
            NewJournalFact(
                fact_type=terminal_type,
                correlation_id=correlation_id,
                causation_id=correlation_id,
                payload=payload,
            )
        ]
        result_obj = await self._repository.append(self._owner_scope, self._session_id, self._current_version, facts)
        self._current_version = result_obj.new_version

        # Map terminal fact type to host event type
        event_type_map = {
            FactType.TOOL_RESULT: HostEventType.TOOL_RESULT,
            FactType.TOOL_FAILURE: HostEventType.TOOL_FAILURE,
            FactType.TOOL_ACKNOWLEDGED: HostEventType.TOOL_ACKNOWLEDGED,
            FactType.TOOL_TIMED_OUT: HostEventType.TOOL_TIMED_OUT,
            FactType.TOOL_EFFECT_UNKNOWN: HostEventType.TOOL_EFFECT_UNKNOWN,
        }
        host_type = event_type_map[terminal_type]

        meta: dict[str, Any] = {"tool_call_id": tool_call_id}
        if result is not None:
            meta["result"] = result
        if error is not None:
            meta["error"] = error

        return HostEvent(
            event_type=host_type,
            sequence=result_obj.appended_facts[0].sequence,
            correlation_id=correlation_id,
            timestamp=result_obj.appended_facts[0].timestamp,
            metadata=meta,
        )

    async def execute_tool_call(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        correlation_id: str,
        kind: str | None = None,
    ) -> AsyncIterator[HostEvent]:
        """Execute a single tool call through the engine, yielding lifecycle events.

        Yields host events for each lifecycle stage: requested, started,
        progress (if any), and terminal (result/failure/cancellation).

        If ``_use_legacy_executor`` is set, routes through the legacy executor
        (non-ACP host fallback per ADR-012).
        """
        # 1. Journal TOOL_REQUESTED
        yield await self.journal_tool_requested(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            correlation_id=correlation_id,
            arguments=arguments,
            kind=kind,
        )

        # 2. Authorize (default: authorized)
        yield await self.journal_tool_authorized_or_denied(
            tool_call_id=tool_call_id,
            correlation_id=correlation_id,
            authorized=True,
        )

        # 3. Journal TOOL_STARTED
        yield await self.journal_tool_started(
            tool_call_id=tool_call_id,
            correlation_id=correlation_id,
        )

        if self._tool_engine is None:
            # No engine — emit a result with a stub
            yield await self.journal_tool_terminal(
                tool_call_id=tool_call_id,
                correlation_id=correlation_id,
                terminal_type=FactType.TOOL_RESULT,
                result={"success": True, "message": f"Tool {tool_name} executed (no engine)"},
            )
            return

        # 4. Execute via engine
        try:
            tool_call = {
                "tool_call_id": tool_call_id,
                "function": tool_name,
                "arguments": arguments,
            }

            if self._use_legacy_executor:
                # Legacy executor path (non-ACP host fallback)
                result_dict = self._tool_engine.execute(tool_call)
            else:
                result_dict = await self._tool_engine.execute_async(tool_call)

            # 5. Journal terminal outcome
            if result_dict.get("success"):
                yield await self.journal_tool_terminal(
                    tool_call_id=tool_call_id,
                    correlation_id=correlation_id,
                    terminal_type=FactType.TOOL_RESULT,
                    result=result_dict.get("result") or result_dict,
                )
            else:
                error = result_dict.get("error", str(result_dict.get("message", "Tool execution failed")))
                yield await self.journal_tool_terminal(
                    tool_call_id=tool_call_id,
                    correlation_id=correlation_id,
                    terminal_type=FactType.TOOL_FAILURE,
                    error=error,
                )

        except asyncio.CancelledError:
            # Tool was cancelled — journal cancellation states
            yield await self.journal_tool_cancellation_requested(
                tool_call_id=tool_call_id,
                correlation_id=correlation_id,
            )
            yield await self.journal_tool_terminal(
                tool_call_id=tool_call_id,
                correlation_id=correlation_id,
                terminal_type=FactType.TOOL_ACKNOWLEDGED,
                result={"cancellation": "acknowledged"},
            )

        except Exception as exc:
            yield await self.journal_tool_terminal(
                tool_call_id=tool_call_id,
                correlation_id=correlation_id,
                terminal_type=FactType.TOOL_FAILURE,
                error=str(exc),
            )
