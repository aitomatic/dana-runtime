"""
AgentSession — host-neutral orchestrator for one durable text turn.

An :class:`AgentSession` owns one isolated agent, an owner/workspace scope, a
session journal identity and version, and the active turn. It serializes
mutations: only one active turn per session; a conflicting :meth:`prompt`
raises :class:`SessionBusy`. All turn lifecycle facts are journaled before,
during, and after the model call, enforcing input durability and exactly one
terminal fact per turn.

D1 is text-only: the agent is driven through the streaming STAR loop
(:meth:`~dana.core.agent.star_agent_streaming.STARAgentStreamingMixin.aquery_stream`)
which runs see/think/act with tool-calling + reflection and yields
:class:`~dana.core.runtime.protocols.StreamEvent` values that the session maps to
:class:`HostEvent` values.

D2 adds tool lifecycle wiring: the session can emit thought events and tool
lifecycle events (requested, started, progress, result, cancellation) as
:class:`HostEvent` values. The :class:`ToolExecutionEngine` is wired in to
execute tool calls and journal tool lifecycle facts.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
import time
from typing import Any
from uuid import uuid4

import structlog

from dana.core.policy.modes import PermissionMode
from dana.core.session.journal.protocol import JournalRepository
from dana.core.session.legacy_timeline_migration import recover_interrupted_turns
from dana.core.session.models import FactType, JournalFact, NewJournalFact, OwnerScope
from dana.core.session.projections.conversation import ConversationProjector, ConversationView
from dana.core.session.projections.host_events import HostEvent, HostEventProjector, HostEventType
from dana.core.session.protected_state import ProtectedStateCodec


logger = structlog.get_logger()


# The three fact types that close a turn. Exactly one of these terminates a turn.
_TERMINAL_FACT_TYPES = frozenset({FactType.TURN_COMPLETED, FactType.TURN_CANCELLED, FactType.TURN_ERROR})


def default_agent_factory() -> Any:
    """Build a coding-assistant agent for host adapters (text-turn streaming).

    Returns a :class:`~dana.core.agent.builtin_agents.dana_coding_agent.DanaCodingAgent`
    with the coding-assistant identity (IDENTITY system prompt) and provider/model
    read from the environment (DANA_LLM_PROVIDER / DANA_MODEL) — the same
    construction the legacy ``DanaCodeApp._initialize_legacy_agent`` uses and
    that is verified to answer real prompts correctly.

    The prior bare ``STARAgent`` (``identity_override=None``) emitted a generic
    STAR system prompt with no coding identity, so it could not answer coding
    questions — every prompt got a generic greeting (the ``d6b73d6`` fix only
    corrected the empty stream, not the non-functional agent). DanaCodingAgent
    handles an explicit ``llm_provider``/``model`` correctly (the legacy path
    proves it), so passing them does not reintroduce the ``d6b73d6``
    misconfigured-azure-client empty-stream bug.
    """
    import os

    from dana.core.agent.builtin_agents.dana_coding_agent import DanaCodingAgent

    llm_provider = os.environ.get("DANA_LLM_PROVIDER", "openai")
    model = os.environ.get("DANA_MODEL", "gpt-5")

    return DanaCodingAgent(
        agent_id="dana-code",
        agent_type="dana_coding_agent",
        llm_provider=llm_provider,
        model=model,
    )


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
        agent_factory: Callable[[], Any] | None = None,
        protected_state_codec: ProtectedStateCodec | None = None,
        tool_engine: Any | None = None,
        use_legacy_executor: bool = False,
    ) -> None:
        self._owner_scope = owner_scope
        self._session_id = session_id
        self._repository = repository
        self._agent_factory = agent_factory or default_agent_factory
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
        # D7.6 follow-up: host-provided interactive prompt callback for
        # NEEDS_PROMPT (CLI-only; None on ACP -> proceed, request_permission is
        # the ACP resolution surface). Async callable (operation) -> PermissionVerdict.
        self._permission_prompt_callback: Any = None
        # D4: Model state — current provider and model for compatibility gating
        self._current_provider: str | None = None
        self._current_model: str | None = None
        # D7.5 (AC #4): MCP single-dispatch wrapper wiring (None when MCP is
        # disabled/unconfigured). Built once when the agent is first prepared.
        self._mcp_wiring: Any = None
        # D7.6 (AC #1/AC #2-live): native-tool ToolCatalog for policy
        # classification. Built per turn from the agent's native tools; fed to
        # build_policy_operation + the TOOL_CALL permission hook. None = no
        # policy classification (text-only / not-yet-prepared).
        self._tool_catalog: Any = None
        # D7.6: EventBus TOOL_CALL permission-hook unsubscribe handle (None when
        # the hook is not registered). Built once when the agent is first
        # prepared + a policy_evaluator is wired + preflight enabled.
        self._policy_unsub: Any = None
        # D7.6: per-turn catalog version counter (AC #1 — stable identity +
        # per-turn versioned; ADR-004). Bumped each turn when the catalog is built.
        self._catalog_version: int = 0

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
    # D3: Policy evaluator accessors (ADR-006)
    # ------------------------------------------------------------------

    def set_policy_evaluator(self, evaluator: Any) -> None:
        """Wire the permission PolicyEvaluator (D3, ADR-006).

        Mirrors how ``DanaACPAgent.new_session`` attaches an evaluator. The
        evaluator owns grant precedence and ``affected_locations`` matching;
        host adapters (CLI, ACP) provide the *decision* surface, not the policy.
        """
        self._policy_evaluator = evaluator
        evaluator.set_mode(self._permission_mode)

    def set_permission_prompt_callback(self, callback: Any) -> None:
        """Wire a host-provided interactive prompt for NEEDS_PROMPT (CLI-only).

        The callback is an async callable ``(operation) -> PermissionVerdict``
        invoked by the ``TOOL_CALL`` hook when the policy reaches
        ``NEEDS_PROMPT``. ``allowed=True`` -> the tool proceeds; ``allowed=False``
        -> the tool is blocked (journal denied, not executed). When ``None``
        (the default; e.g. ACP), NEEDS_PROMPT proceeds (the host's own
        ``request_permission`` is the resolution surface).
        """
        self._permission_prompt_callback = callback

    @property
    def policy_evaluator(self) -> Any:
        """The wired PolicyEvaluator, or ``None`` when policy grants are disabled."""
        return self._policy_evaluator

    # ------------------------------------------------------------------
    # Identity (public read accessors — host adapters must not read privates)
    # ------------------------------------------------------------------

    @property
    def owner_scope(self) -> OwnerScope:
        """The OwnerScope (owner + workspace) for this session."""
        return self._owner_scope

    @property
    def session_id(self) -> str:
        """The durable session id."""
        return self._session_id

    @property
    def version(self) -> int:
        """The current journal version (sequence) for this session."""
        return self._current_version

    # ------------------------------------------------------------------
    # D4: Model state (ADR-007)
    # ------------------------------------------------------------------

    @property
    def current_provider(self) -> str | None:
        """The current provider for this session, or ``None``."""
        return self._current_provider

    @property
    def current_model(self) -> str | None:
        """The current model for this session, or ``None``."""
        return self._current_model

    def rebind_model(self, target: Any, provider: Any, runtime: Any) -> None:
        """Rebind the session to a new model (ADR-007: atomic switch).

        Called by the ModelSwitcher's ``apply_switch`` callback. Updates
        the session's provider/model identity in-memory. The caller
        (DanaACPAgent) journals the MODEL_CHANGED fact asynchronously
        after the switch completes.

        Args:
            target: The ModelTarget being switched to.
            provider: The built provider client.
            runtime: The built model runtime.
        """
        self._current_provider = target.provider
        self._current_model = target.model

    # ------------------------------------------------------------------
    # D8: Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        session_id: str | None = None,
        journal_path: str | None = None,
        owner_scope: OwnerScope | None = None,
        agent_factory: Callable[[], Any] | None = None,
    ) -> AgentSession:
        """Convenience constructor: journal + SESSION_CREATED + agent in one call (D8).

        Mirrors ``DanaACPAgent.new_session`` / ``DanaCodeApp._initialize_session``:
        ``OwnerScope`` (``$USER`` + cwd by default), SQLite journal at
        ``journal_path`` (or ``DANA_CODE_JOURNAL`` / ``DANA_ACP_JOURNAL`` /
        ``~/.dana/journal.db``; the directory is created), one SESSION_CREATED
        fact, and :func:`default_agent_factory`. Returns the session with its
        version already set — ``prompt`` can be called immediately.

        Idempotent on an existing journal: when ``session_id`` already exists,
        resumes it instead (no duplicate SESSION_CREATED; interrupted turns are
        recovered first, matching the ACP ``session/load`` path).
        """
        import os

        from dana.core.session.journal.models import SessionNotFound, SessionRecord
        from dana.core.session.journal.sqlite import SQLiteJournalRepository

        if journal_path is None:
            journal_path = os.environ.get("DANA_CODE_JOURNAL") or os.environ.get("DANA_ACP_JOURNAL") or "~/.dana/journal.db"
        journal_path = os.path.expanduser(journal_path)
        os.makedirs(os.path.dirname(journal_path) or ".", exist_ok=True)
        repo = await SQLiteJournalRepository.open(journal_path)

        scope = owner_scope or OwnerScope(owner_id=os.environ.get("USER", "local"), workspace=os.getcwd())
        sid = session_id or str(uuid4())

        try:
            await repo.load_session(scope, sid)
        except SessionNotFound:
            record = SessionRecord.new(sid, scope)
            init_facts = [
                JournalFact(
                    fact_id=str(uuid4()),
                    owner_scope=scope,
                    session_id=sid,
                    sequence=1,
                    fact_type=FactType.SESSION_CREATED,
                    timestamp=datetime.now(UTC),
                    correlation_id=str(uuid4()),
                    causation_id=None,
                    schema_version=1,
                    payload={},
                ),
            ]
            await repo.create_session(record, init_facts)
            session = cls(owner_scope=scope, session_id=sid, repository=repo, agent_factory=agent_factory)
            session._current_version = init_facts[0].sequence
            return session

        # Existing journal -> resume (idempotent: no duplicate SESSION_CREATED).
        await recover_interrupted_turns(repo, scope, sid)
        session = cls(owner_scope=scope, session_id=sid, repository=repo, agent_factory=agent_factory)
        await session.load()
        return session

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

    async def prompt(
        self,
        blocks: Sequence[TextBlock],
        content_blocks: list[dict] | None = None,
    ) -> AsyncIterator[HostEvent]:
        """Run one text turn. Yields host events as they occur.

        After the generator is exhausted, the :class:`TurnTerminal` outcome is
        available via :attr:`last_terminal`.

        D6: When ``content_blocks`` is provided (list of normalized content block
        dicts), the ``USER_CONTENT_FINAL`` fact payload includes a
        ``content_blocks`` key. The ConversationView projects these as
        ``list[ContentBlock]`` instead of a plain string.

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
            user_payload: dict[str, object] = {"text": user_text}
            if content_blocks:
                user_payload["content_blocks"] = content_blocks
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
                    payload=user_payload,
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

            # NOTE: do NOT pre-add the user message to the timeline here.
            # aquery_stream drives the full STAR loop; STARAgent._see adds the
            # caller_message to the timeline itself so build_prompt includes
            # it. Pre-adding would duplicate the user message.

            from dana.core.runtime.protocols import StreamEventType

            accumulated: list[str] = []
            chunk_buffer: list[str] = []
            chunk_buffer_bytes = 0
            last_flush = time.monotonic()
            chunk_index = 0

            async def _flush_pending() -> None:
                nonlocal chunk_index, chunk_buffer_bytes, last_flush
                if chunk_buffer:
                    await self._flush_chunks(correlation_id, chunk_buffer, chunk_index)
                    chunk_index += len(chunk_buffer)
                    chunk_buffer.clear()
                    chunk_buffer_bytes = 0
                    last_flush = time.monotonic()

            try:
                # Drive the streaming STAR loop (see -> think -> act with
                # tool-calling + reflection) instead of the text-only
                # aquery_text_stream, so the model engages with the user's
                # prompt. Map each StreamEvent to a HostEvent and journal the
                # corresponding fact, reusing the existing journal helpers.
                agen = self._agent.aquery_stream(message=user_text)
                try:
                    async for event in agen:
                        # Cooperative cancellation: aquery_stream does not check
                        # cancel_event itself, so check between events and
                        # terminalize as TURN_CANCELLED on cancel().
                        if self._cancel_event is not None and self._cancel_event.is_set():
                            raise asyncio.CancelledError
                        etype = event.event_type
                        data = event.data
                        if etype == StreamEventType.TEXT_DELTA:
                            chunk = data if isinstance(data, str) else (str(data) if data else "")
                            if not chunk:
                                continue
                            accumulated.append(chunk)
                            # Pre-persistence chunk: sequence 0 = "not yet
                            # persisted"; replay returns real sequences.
                            yield HostEvent(
                                event_type=HostEventType.ASSISTANT_CONTENT_CHUNK,
                                sequence=0,
                                correlation_id=correlation_id,
                                timestamp=datetime.now(UTC),
                                text=chunk,
                            )
                            chunk_buffer.append(chunk)
                            chunk_buffer_bytes += len(chunk)
                            now = time.monotonic()
                            if chunk_buffer_bytes >= self.CHUNK_FLUSH_BYTES or (now - last_flush) >= self.CHUNK_FLUSH_INTERVAL:
                                await _flush_pending()
                        elif etype == StreamEventType.THINKING:
                            thought = data if isinstance(data, str) else (str(data) if data else "")
                            if thought:
                                # Live-only reasoning event (not journaled).
                                yield await self.emit_thought(thought, correlation_id)
                        elif etype == StreamEventType.TOOL_CALL_START:
                            await _flush_pending()
                            for tc in data if isinstance(data, list) else ([data] if data else []):
                                if not isinstance(tc, dict):
                                    continue
                                tc_id = tc.get("id") or tc.get("tool_call_id") or str(uuid4())
                                tc_name = tc.get("name", "")
                                tc_args = tc.get("input", tc.get("arguments", {}))
                                yield await self.journal_tool_requested(
                                    tc_id,
                                    tc_name,
                                    correlation_id,
                                    tc_args,
                                )
                                yield await self.journal_tool_authorized_or_denied(
                                    tc_id,
                                    correlation_id,
                                    authorized=True,
                                )
                                yield await self.journal_tool_started(tc_id, correlation_id)
                        elif etype == StreamEventType.TOOL_RESULT:
                            await _flush_pending()
                            for tr in data if isinstance(data, list) else ([data] if data else []):
                                if not isinstance(tr, dict):
                                    continue
                                tc_id = tr.get("tool_call_id") or tr.get("id") or ""
                                yield await self.journal_tool_terminal(
                                    tc_id,
                                    correlation_id,
                                    FactType.TOOL_RESULT,
                                    result=tr.get("result"),
                                    error=tr.get("error"),
                                )
                        elif etype == StreamEventType.ERROR:
                            raise RuntimeError(str(data) if data else "stream error")
                        elif etype == StreamEventType.DONE:
                            break
                finally:
                    with contextlib.suppress(Exception):
                        await agen.aclose()

                await _flush_pending()

                full_text = "".join(accumulated)
                protected_payload = None

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
    # D4: Fact append helper (used by model switching)
    # ------------------------------------------------------------------

    async def append_fact(self, fact: NewJournalFact) -> None:
        """Append a single fact to the journal.

        Used by the ACP agent to journal model-change facts after a
        successful model switch. The fact is appended with the current
        version as the expected version (optimistic concurrency).
        """
        result = await self._repository.append(
            self._owner_scope,
            self._session_id,
            self._current_version,
            [fact],
        )
        self._current_version = result.new_version

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _prepare_agent(self) -> None:
        """Create the agent (once) and (re)build its timeline from the journal."""
        # TODO(d2): incremental conversation view update instead of full re-read.
        if self._agent is None:
            self._agent = self._agent_factory()
            # D7.5 (AC #4): wire MCP single-dispatch wrapper once, when the agent
            # is first built. Both ACP and CLI sessions share this path.
            await self._wire_mcp_tools(self._agent)
        facts = await self._repository.read_facts(self._owner_scope, self._session_id)
        self._current_version = max((f.sequence for f in facts), default=0)
        # D4: Pass current provider for compatibility gating on protected state
        view = self._conversation_projector.project(facts, provider_key=self._current_provider)
        # D4: Restore model state from projected model changes
        if view.current_provider is not None and self._current_provider is None:
            self._current_provider = view.current_provider
            self._current_model = view.current_model
        self._populate_timeline(view)
        # D7.6 (AC #1/AC #2-live): build the native-tool catalog for policy
        # classification (per-turn pinned version) + register the TOOL_CALL
        # permission hook. Idempotent: the hook is registered once.
        await self._build_tool_catalog()
        self._register_policy_hook()

    async def _wire_mcp_tools(self, agent: Any) -> None:
        """Attach the MCP single-dispatch wrapper to the agent (D7.5 AC #4).

        Gated by ``DANA_CODE_MCP_ENABLED`` (default on) and the presence of an
        MCP server config (``DANA_MCP_SERVERS``). When enabled, builds the
        per-server transports + adapters + the ``MCPDispatchResource`` and
        appends it to ``agent._resources`` so the STAR loop's native-tool
        registry discovers + dispatches ``call_mcp_tool``. No policy gating
        (catalog-coupled; deferred to the D7.6 D2 Catalog Migration story).
        """
        if self._mcp_wiring is not None:
            return  # already wired
        try:
            from dana.config.code_capabilities import mcp_enabled
        except Exception:
            return
        if not mcp_enabled():
            return
        try:
            from dana.core.mcp.config import load_mcp_config_from_env
            from dana.core.mcp.dispatch_wrapper import build_mcp_dispatch_resource
        except Exception:
            return
        config = load_mcp_config_from_env()
        if config is None:
            return  # no MCP config -> MCP disabled by absence
        try:
            wiring = await build_mcp_dispatch_resource(config)
        except Exception as exc:  # noqa: BLE001 — required-lease failure: surface, don't crash the turn
            logger.warning(f"MCP wiring failed: {exc}")
            return
        if wiring is None:
            return
        self._mcp_wiring = wiring
        resources = getattr(agent, "_resources", None)
        if isinstance(resources, list):
            resources.append(wiring.resource)
            logger.info("MCP dispatch resource attached", tool_count=len(wiring.resource.available_tools))

    async def dispose_mcp(self) -> None:
        """Release MCP transports + leases (session teardown). Best-effort."""
        if self._mcp_wiring is None:
            return
        with contextlib.suppress(Exception):
            await self._mcp_wiring.close()
        self._mcp_wiring = None

    # ------------------------------------------------------------------
    # D7.6: native-tool catalog (policy classification) + TOOL_CALL hook
    # ------------------------------------------------------------------

    @property
    def tool_catalog(self) -> Any:
        """The per-turn native-tool ToolCatalog for policy classification (D7.6).

        ``None`` until the agent is prepared or when preflight is disabled.
        Host permission adapters (CLI, ACP) feed this to
        :func:`build_policy_operation` so effect classification is consistent.
        """
        return self._tool_catalog

    async def _build_tool_catalog(self) -> None:
        """Build the native-tool catalog from the agent's native-tool schemas.

        Pins a per-turn version (AC #1). No-op if the agent has no runtime /
        native-tools (text-only). Gated by ``DANA_CODE_TOOL_CATALOG_ENABLED``
        (default on). Does NOT reroute execution through the D2 engine
        (Decision 2): the catalog feeds the policy classifier only.
        """
        try:
            from dana.config.code_capabilities import tool_catalog_enabled
        except Exception:
            tool_catalog_enabled = lambda: True  # noqa: E731
        if not tool_catalog_enabled():
            self._tool_catalog = None
            return
        runtime = getattr(self._agent, "_runtime", None) if self._agent is not None else None
        if runtime is None:
            self._tool_catalog = None
            return
        # Ensure _native_tools is built (cached/idempotent) so MCP schemas can be
        # appended (D7 follow-up 1+2: per-tool MCP UX surfaces via _native_tools).
        if hasattr(runtime, "_build_native_tools_if_supported"):
            runtime._build_native_tools_if_supported(self._agent)
        native_tools = getattr(runtime, "_native_tools", None)
        if not native_tools:
            self._tool_catalog = None
            return

        # D7 follow-up 1+2: per-tool MCP UX + per-MCP policy. Append the per-MCP
        # schemas (correct inputSchema, namespaced server:tool) to _native_tools
        # so the LLM sees each MCP tool by name; wire the MCP dispatch map on the
        # agent's ToolExecutor (additive, checked before the registry); collect
        # mcp_names so the policy classifier treats them as EXECUTE/non-sensitive.
        # Idempotent: MCP schemas are appended once (the cached _native_tools is
        # not rebuilt, so they persist across turns).
        mcp_names: frozenset[str] = frozenset()
        wiring = self._mcp_wiring
        if wiring is not None and getattr(wiring, "mcp_schemas", None):
            existing = {(t.get("function", t).get("name") if isinstance(t, dict) else None) for t in native_tools}
            for schema in wiring.mcp_schemas:
                name = schema.get("function", {}).get("name") if isinstance(schema, dict) else None
                if name and name not in existing:
                    native_tools.append(schema)
                    existing.add(name)
            mcp_names = wiring.mcp_names
            tool_executor = getattr(runtime, "_tool_executor", None)
            if tool_executor is not None and hasattr(tool_executor, "set_mcp_dispatch_getter"):
                tool_executor.set_mcp_dispatch_getter(lambda w=wiring: w.dispatch_map)

        from dana.core.tool.native_catalog import build_native_tool_catalog

        self._catalog_version += 1
        self._tool_catalog = build_native_tool_catalog(
            list(native_tools),
            version=self._catalog_version,
            mcp_names=mcp_names,
        )

    def _register_policy_hook(self) -> None:
        """Subscribe the TOOL_CALL permission hook on the agent's EventBus.

        Gated by ``DANA_CODE_PERMISSION_PREFLIGHT_ENABLED`` (default on) AND a
        wired ``policy_evaluator``. Idempotent: skips if already registered or
        if the gate is closed. The hook blocks ONLY on ``PolicyDecision.DENY``;
        ``ALLOW``/``NEEDS_PROMPT`` proceed (interactive prompting is a
        host-layer follow-up per the D7.5 adjusted ruling).
        """
        if self._policy_unsub is not None:
            return  # already registered
        if self._policy_evaluator is None:
            return  # no policy -> no gate (today's authorized=True behaviour)
        try:
            from dana.config.code_capabilities import permission_preflight_enabled
        except Exception:
            permission_preflight_enabled = lambda: True  # noqa: E731
        if not permission_preflight_enabled():
            return
        bus = getattr(self._agent, "event_bus", None)
        if bus is None or not hasattr(bus, "subscribe"):
            return
        from dana.core.ext.events import TOOL_CALL

        self._policy_unsub = bus.subscribe(TOOL_CALL, self._on_tool_call)
        logger.info(
            "permission preflight hook registered",
            session_id=self._session_id,
            catalog_version=self._catalog_version,
        )

    def _unregister_policy_hook(self) -> None:
        """Detach the TOOL_CALL permission hook (session teardown). Best-effort."""
        if self._policy_unsub is None:
            return
        with contextlib.suppress(Exception):
            self._policy_unsub()
        self._policy_unsub = None

    async def _on_tool_call(self, event: Any) -> dict[str, Any] | None:
        """EventBus ``TOOL_CALL`` handler — hard-deny enforcement (D7.6 AC #2-live).

        Reconstructs a policy ``Operation`` from the bus ``Operation`` + the
        per-turn catalog and evaluates it through the wired ``PolicyEvaluator``.
        Returns ``{"block": True, "reason": ...}`` ONLY on ``PolicyDecision.DENY``;
        ``ALLOW``/``NEEDS_PROMPT`` return ``None`` (pass-through -> the tool
        executes). The tool_executor respects ``block`` by skipping dispatch and
        surfacing a ``policy_block`` tool_result.
        """
        if self._policy_evaluator is None:
            return None
        # No catalog (disabled / text-only) -> cannot classify -> pass-through
        # (proceed). Without this guard, preflight-on + catalog-off would DENY
        # every tool (unknown/sensitive) and regress the P0 turn path.
        if self._tool_catalog is None:
            return None
        ext_op = event.payload.get("operation") if isinstance(getattr(event, "payload", None), dict) else None
        if ext_op is None:
            return None
        from dana.core.policy.evaluator import PolicyDecision
        from dana.core.policy.operations import build_policy_operation

        tool_call = {
            "function": getattr(ext_op.tool_identity, "name", ""),
            "arguments": dict(ext_op.arguments),
        }
        op = build_policy_operation(
            tool_call,
            catalog=self._tool_catalog,
            owner=self._owner_scope.owner_id,
            workspace=self._owner_scope.workspace,
        )
        result = await self._policy_evaluator.evaluate(op, self._owner_scope)
        if result.decision is PolicyDecision.DENY:
            return {"block": True, "reason": f"denied: {result.reason}"}
        if result.decision is PolicyDecision.NEEDS_PROMPT and self._permission_prompt_callback is not None:
            verdict = await self._permission_prompt_callback(op)
            if not verdict.allowed:
                return {"block": True, "reason": f"denied: {verdict.reason}"}
        return None  # ALLOW / NEEDS_PROMPT (no callback) -> proceed

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
