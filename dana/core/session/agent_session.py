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

Tool dispatch uses authoritative executor hooks: authorization and start are
durable before effects; each effect outcome is durable before the turn terminal.
The event bus remains best-effort interception, not a journal boundary.
"""

from __future__ import annotations

import asyncio
import base64
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
import contextlib
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import inspect
import json
import os
import time
from typing import Any
from uuid import uuid4

import structlog

from dana.core.policy.modes import PermissionMode
from dana.core.session.journal.models import ProjectionCheckpoint
from dana.core.session.journal.protocol import JournalRepository
from dana.core.session.legacy_timeline_migration import recover_interrupted_turns
from dana.core.session.models import FactType, JournalFact, NewJournalFact, OwnerScope
from dana.core.session.projections.conversation import ConversationProjectionState, ConversationProjector, ConversationView
from dana.core.session.projections.host_events import HostEvent, HostEventProjector, HostEventType
from dana.core.session.protected_state import EnvProtectedStateKeyProvider, ProtectedStateCodec


logger = structlog.get_logger()


# The three fact types that close a turn. Exactly one of these terminates a turn.
_TERMINAL_FACT_TYPES = frozenset({FactType.TURN_COMPLETED, FactType.TURN_CANCELLED, FactType.TURN_ERROR})
_TOOL_TERMINAL_FACT_TYPES = frozenset(
    {
        FactType.TOOL_RESULT,
        FactType.TOOL_FAILURE,
        FactType.TOOL_ACKNOWLEDGED,
        FactType.TOOL_TIMED_OUT,
        FactType.TOOL_EFFECT_UNKNOWN,
    }
)
_TOOL_FACT_TYPES = _TOOL_TERMINAL_FACT_TYPES | {
    FactType.TOOL_REQUESTED,
    FactType.TOOL_AUTHORIZED_OR_DENIED,
    FactType.TOOL_STARTED,
    FactType.TOOL_PROGRESS,
    FactType.TOOL_CANCELLATION_REQUESTED,
}

# Metadata marker carried by the single derived interruption note injected
# into the working timeline (I2). The note is derived state: it is re-derived
# from the fold state on every reconciliation and retired as soon as the
# projected observation clears, so it is never persisted in a checkpoint and
# never duplicated across reopen cycles.
_INTERRUPTION_NOTE_KEY = "dana_interruption_observation"


def _supports_agent_scoped_hooks(executor: Any) -> bool:
    """Whether ``executor.set_execution_hooks`` accepts an ``agent=`` registration.

    The session's durable hooks close over one turn (owner scope, correlation id,
    journal target), so a runtime/executor shared by several overlapping sessions
    must resolve them from the *invoking* agent instead of letting the last
    session to start a turn overwrite the previous one. That per-agent contract is
    detected on the executor's own signature; an executor without it keeps the
    legacy process-global hooks, which remain correct for the one-session-per-
    runtime shape every in-tree host uses.
    """
    installer = getattr(executor, "set_execution_hooks", None)
    if installer is None:
        return False
    try:
        return "agent" in inspect.signature(installer).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callable
        return False


def _install_execution_hooks(
    executor: Any,
    agent: Any,
    *,
    before: Any,
    after: Any,
    authorize: Any | None,
) -> None:
    """Install this session's tool-lifecycle hooks, owner-scoped when possible."""
    if _supports_agent_scoped_hooks(executor):
        executor.set_execution_hooks(before=before, after=after, authorize=authorize, agent=agent)
    else:
        executor.set_execution_hooks(before=before, after=after, authorize=authorize)


def _clear_execution_hooks(executor: Any, agent: Any) -> None:
    """Retire this session's hooks without disturbing another session's."""
    if _supports_agent_scoped_hooks(executor):
        executor.set_execution_hooks(agent=agent)
    else:
        executor.set_execution_hooks()


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
        self._owns_repository = False
        self._agent_factory = agent_factory or default_agent_factory
        self._codec = protected_state_codec
        if self._codec is None and os.environ.get("DANA_SESSION_STATE_KEY"):
            self._codec = ProtectedStateCodec(EnvProtectedStateKeyProvider())
        self._conversation_projector = ConversationProjector(self._codec)
        self._conversation_state: ConversationProjectionState | None = None
        self._working_context: list[dict[str, Any]] | None = None
        self._working_format: str | None = None
        self._hydrated_tail_count = 0
        self._needs_hydration = True
        self._host_event_projector = HostEventProjector()
        self._lock = asyncio.Lock()
        self._append_lock = asyncio.Lock()
        self._cancel_event: asyncio.Event | None = None
        # I4: set only by cancel(); distinguishes an explicit session cancel from
        # the caller's own task cancellation. Reset per turn.
        self._turn_cancel_requested = False
        # Record tool lifecycle ownership at commit, before an interrupted hook
        # can propagate cancellation and trigger pending-tool reconciliation.
        self._tool_commit_observer: Callable[[Sequence[JournalFact], bool], None] | None = None
        # I4: fact types of the most recently committed batch. Lets the
        # cancellation handler tell whether a durable batch that was interrupted
        # by the caller's cancellation already carried this turn's terminal.
        self._committed_batch_types: tuple[FactType, ...] = ()
        self._stream_step: asyncio.Task | None = None
        self._agent: Any = None
        self._closed = False
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
        # Optional host interaction for NEEDS_PROMPT. Without a callback the
        # configured policy fails closed; no execution is pre-authorized.
        self._permission_prompt_callback: Any = None
        # D4: Model state — current provider and model for compatibility gating
        self._current_provider: str | None = None
        self._current_model: str | None = None
        # D7.5 (AC #4): MCP single-dispatch wrapper wiring (None when MCP is
        # disabled/unconfigured). Built once when the agent is first prepared.
        self._mcp_wiring: Any = None
        # Classification is pinned for each turn and reused until discovery,
        # structural agent membership, runtime, or MCP wiring changes.
        self._tool_catalog: Any = None
        self._catalog_key: tuple[Any, ...] | None = None
        self._catalog_mcp_names: frozenset[str] = frozenset()
        # D7.6: EventBus TOOL_CALL permission-hook unsubscribe handle (None when
        # the hook is not registered). Built once when the agent is first
        # prepared + a policy_evaluator is wired + preflight enabled.
        self._policy_unsub: Any = None
        # Version changes only when the classified catalog changes.
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
        """Wire an async ``(operation) -> PermissionVerdict`` host callback.

        NEEDS_PROMPT requires an affirmative verdict. A missing callback or
        rejected verdict denies execution before the durable started boundary.
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
            session = cls(owner_scope=scope, session_id=sid, repository=repo, agent_factory=agent_factory)
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
                session._current_version = init_facts[0].sequence
            else:
                await recover_interrupted_turns(repo, scope, sid)
                await session.load()
            session._owns_repository = True
            return session
        except BaseException:
            await repo.close()
            raise

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
        # I4: remember that *this* cancellation was host-requested. The prompt
        # loop uses it to tell an explicit session cancel (the caller keeps
        # iterating and observes TURN_CANCELLED) from the caller's own task being
        # cancelled (asyncio.timeout / TaskGroup / teardown), which must
        # propagate after the durable terminal.
        self._turn_cancel_requested = True
        self._cancel_event.set()
        if self._last_terminal is None and self._stream_step is not None:
            self._stream_step.cancel()

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
        if self._closed:
            raise RuntimeError("Session is closed")
        if self._lock.locked():
            raise SessionBusy(self._session_id)
        async with self._lock:
            if self._closed:
                raise RuntimeError("Session is closed")
            self._cancel_event = asyncio.Event()
            self._turn_cancel_requested = False
            self._committed_batch_types = ()
            self._last_terminal = None
            correlation_id = str(uuid4())
            user_text = " ".join(b.text for b in blocks)

            try:
                await self._prepare_agent()

                # Persist input before invoking the model.
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
                start_result = await self._append_facts(start_facts)
                # The live agent may mutate its working set until a terminal
                # outcome. Failed/cancelled turns must reconcile from facts.
                self._needs_hydration = True
            except BaseException as exc:
                # I4: the caller's task can be cancelled while THIS batch is already
                # committing. The batch is durable either way, so it must not be
                # duplicated; but if it was the turn-start batch, the turn now has
                # no terminal at all, so one TURN_CANCELLED is appended to keep the
                # "exactly one terminal per turn" invariant (a failure here simply
                # leaves the turn to the reopen recovery path, never masking the
                # cancellation).
                if isinstance(exc, asyncio.CancelledError) and FactType.TURN_STARTED in self._committed_batch_types:
                    try:
                        cancel_result = await self._append_facts(
                            [
                                NewJournalFact(
                                    fact_type=FactType.TURN_CANCELLED,
                                    correlation_id=correlation_id,
                                    causation_id=correlation_id,
                                    payload={"partial_text": "", "provider_responses": []},
                                )
                            ]
                        )
                        self._current_version = cancel_result.new_version
                    except BaseException:
                        logger.warning(
                            "failed to terminalize an interrupted turn start",
                            session_id=self._session_id,
                            correlation_id=correlation_id,
                            exc_info=True,
                        )
                self._cancel_event = None
                raise

            logger.info(
                "turn started",
                session_id=self._session_id,
                correlation_id=correlation_id,
                version=self._current_version,
            )

            # NOTE: do NOT pre-add the user message to the timeline here.
            # aquery_stream drives the full STAR loop; STARAgent._see adds the
            # caller_message to the timeline itself so build_prompt includes
            # it. Pre-adding would duplicate the user message.

            from dana.core.runtime.protocols import StreamEventType

            accumulated: list[str] = []
            text_iteration: int | None = None
            chunk_buffer: list[str] = []
            chunk_buffer_bytes = 0
            last_flush = time.monotonic()
            chunk_index = 0
            provider_responses: list[dict[str, Any]] = []
            requested_tools: dict[str, None] = {}
            terminal_tools: set[str] = set()
            started_tools: set[str] = set()
            tool_events: deque[HostEvent] = deque()
            tool_events_ready = asyncio.Event()
            executor = getattr(getattr(self._agent, "_runtime", None), "_tool_executor", None)

            def _record_tool_commits(facts: Sequence[JournalFact], cancelled: bool) -> None:
                """Account for durable outcomes before cancellation leaves a hook."""
                for fact in facts:
                    if fact.correlation_id != correlation_id or fact.fact_type not in _TOOL_FACT_TYPES:
                        continue
                    call_id = fact.payload.get("tool_call_id")
                    if not call_id:
                        continue
                    if fact.fact_type is FactType.TOOL_REQUESTED:
                        requested_tools[call_id] = None
                    elif fact.fact_type is FactType.TOOL_STARTED:
                        started_tools.add(call_id)
                    elif fact.fact_type in _TOOL_TERMINAL_FACT_TYPES:
                        terminal_tools.add(call_id)
                    if cancelled:
                        # The helper will raise instead of returning its event.
                        # Keep the committed event visible to an explicit-cancel
                        # consumer, which remains alive to receive TURN_CANCELLED.
                        event = self._host_event_projector.project([fact])[0]
                        if fact.fact_type is FactType.TOOL_REQUESTED:
                            event = HostEvent(
                                event_type=event.event_type,
                                sequence=event.sequence,
                                correlation_id=event.correlation_id,
                                timestamp=event.timestamp,
                                metadata={
                                    "tool_call_id": call_id,
                                    "tool_name": fact.payload.get("tool_name", ""),
                                    "kind": fact.payload.get("kind"),
                                    "raw_input": fact.payload.get("arguments", {}),
                                },
                            )
                        tool_events.append(event)
                        tool_events_ready.set()

            self._tool_commit_observer = _record_tool_commits

            async def _request_tool(call: dict[str, Any]) -> str:
                call_id = call.get("tool_call_id") or call.get("id")
                if not call_id:
                    raise ValueError("Tool dispatch requires a stable tool_call_id")
                if call_id not in requested_tools:
                    tool_events.append(
                        await self.journal_tool_requested(
                            call_id,
                            call.get("function", call.get("name", "")),
                            correlation_id,
                            call.get("arguments", call.get("input", {})),
                        )
                    )
                    tool_events_ready.set()
                return call_id

            async def _tool_terminal(call: dict[str, Any], result: dict[str, Any]) -> None:
                call_id = await _request_tool(call)
                if call_id in terminal_tools:
                    return
                if result.get("_effect_unknown"):
                    terminal_type = FactType.TOOL_EFFECT_UNKNOWN
                elif result.get("_cancelled"):
                    terminal_type = FactType.TOOL_ACKNOWLEDGED
                elif result.get("success", not result.get("error")):
                    terminal_type = FactType.TOOL_RESULT
                else:
                    terminal_type = FactType.TOOL_FAILURE
                tool_events.append(
                    await self.journal_tool_terminal(
                        call_id, correlation_id, terminal_type, result=result.get("result", result), error=result.get("error")
                    )
                )
                tool_events_ready.set()

            async def _finish_pending_tools(cancelled: bool) -> None:
                for call_id in requested_tools:
                    if call_id in terminal_tools:
                        continue
                    result = {"success": False, "error": "Turn ended before a tool outcome was available"}
                    if call_id in started_tools:
                        result["_effect_unknown"] = True
                    elif cancelled:
                        result["_cancelled"] = True
                    await _tool_terminal({"tool_call_id": call_id}, result)

            async def _authorize_tool(call: dict[str, Any], block_reason: str | None) -> dict[str, Any] | None:
                """I5/I6: journal requested + the permission decision BEFORE admission.

                Returns a recoverable block verdict when the call must not be
                dispatched, so the executor turns it into a normal ``policy_block``
                tool result and the turn continues. Journal/effect failures still
                propagate (the durable boundary must never be swallowed). No
                admission permit is held here, so a queue of permission prompts
                cannot starve the tool budget.
                """
                call_id = await _request_tool(call)
                decision = await self._evaluate_tool_permission(call) if block_reason is None else None
                reason = block_reason or (decision.get("reason") if decision else None)
                tool_events.append(
                    await self.journal_tool_authorized_or_denied(
                        call_id,
                        correlation_id,
                        authorized=reason is None,
                        reason=reason,
                        arguments=call.get("arguments", call.get("input", {})),
                    )
                )
                tool_events_ready.set()
                if reason is None:
                    return None
                # Denied by the bus (block_reason set) or by policy. The executor
                # records the policy_block terminal through the ``after`` hook, so
                # this hook only reports the verdict; ``_tool_terminal`` stays
                # idempotent if the executor also calls it.
                return {"block": True, "reason": reason}

            async def _before_tool(call: dict[str, Any], block_reason: str | None) -> dict[str, Any] | None:
                """Admission-owned seam: journal TOOL_STARTED and re-check cancellation.

                Runs after the permit is acquired, immediately before dispatch. A
                block reason reaching this hook means the EventBus blocked the call
                (``authorize`` already journaled requested/denied); nothing started
                is recorded and the call is not dispatched.
                """
                call_id = await _request_tool(call)
                if block_reason is not None:
                    return {"block": True, "reason": block_reason}
                if self._cancel_event is not None and self._cancel_event.is_set():
                    await _tool_terminal(call, {"success": False, "_cancelled": True, "error": "Cancelled before dispatch"})
                    raise asyncio.CancelledError
                tool_events.append(await self.journal_tool_started(call_id, correlation_id))
                tool_events_ready.set()
                if self._cancel_event is not None and self._cancel_event.is_set():
                    await _tool_terminal(call, {"success": False, "_cancelled": True, "error": "Cancelled before dispatch"})
                    raise asyncio.CancelledError
                return None

            if executor is not None and hasattr(executor, "set_execution_hooks"):
                # Authoritative hooks propagate journal/policy failures. The
                # best-effort bus must not own permission or effect durability.
                # The executor owns the three-hook contract
                # (authorize -> admission -> before -> after); every hook is wired
                # so requested/authorized, started and terminal facts stay durable.
                self._unregister_policy_hook()
                _install_execution_hooks(
                    executor,
                    self._agent,
                    before=_before_tool,
                    after=_tool_terminal,
                    authorize=_authorize_tool,
                )

            async def _flush_pending() -> None:
                nonlocal chunk_index, chunk_buffer_bytes, last_flush
                if chunk_buffer:
                    await self._flush_chunks(correlation_id, chunk_buffer, chunk_index)
                    chunk_index += len(chunk_buffer)
                    chunk_buffer.clear()
                    chunk_buffer_bytes = 0
                    last_flush = time.monotonic()

            try:
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
                # Drive the streaming STAR loop (see -> think -> act with
                # tool-calling + reflection) instead of the text-only
                # aquery_text_stream, so the model engages with the user's
                # prompt. Map each StreamEvent to a HostEvent and journal the
                # corresponding fact, reusing the existing journal helpers.
                agen = self._agent.aquery_stream(message=content_blocks if content_blocks else user_text)
                try:
                    while True:
                        if self._cancel_event is not None and self._cancel_event.is_set():
                            raise asyncio.CancelledError
                        self._stream_step = asyncio.create_task(anext(agen))
                        try:
                            while not self._stream_step.done():
                                while tool_events:
                                    yield tool_events.popleft()
                                tool_events_ready.clear()
                                if self._stream_step.done():
                                    break
                                ready = asyncio.create_task(tool_events_ready.wait())
                                try:
                                    await asyncio.wait((self._stream_step, ready), return_when=asyncio.FIRST_COMPLETED)
                                finally:
                                    ready.cancel()
                                    await asyncio.gather(ready, return_exceptions=True)
                            event = await self._stream_step
                        except StopAsyncIteration:
                            break
                        finally:
                            if self._stream_step is not None and not self._stream_step.done():
                                self._stream_step.cancel()
                                await asyncio.gather(self._stream_step, return_exceptions=True)
                            self._stream_step = None
                        while tool_events:
                            yield tool_events.popleft()
                        if self._cancel_event is not None and self._cancel_event.is_set():
                            raise asyncio.CancelledError
                        etype = event.event_type
                        data = event.data
                        if etype == StreamEventType.TEXT_DELTA:
                            chunk = data if isinstance(data, str) else (str(data) if data else "")
                            if not chunk:
                                continue
                            if text_iteration != event.iteration:
                                accumulated.clear()
                                text_iteration = event.iteration
                            accumulated.append(chunk)
                            chunk_buffer.append(chunk)
                            chunk_buffer_bytes += len(chunk)
                            # Pre-persistence chunk: sequence 0 = "not yet
                            # persisted"; replay returns real sequences.
                            yield HostEvent(
                                event_type=HostEventType.ASSISTANT_CONTENT_CHUNK,
                                sequence=0,
                                correlation_id=correlation_id,
                                timestamp=datetime.now(UTC),
                                text=chunk,
                                metadata={"provisional": True, "iteration": event.iteration},
                            )
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
                            # Narration remains visible, but is not the final
                            # answer of a round that subsequently invokes tools.
                            accumulated.clear()
                            for tc in data if isinstance(data, list) else ([data] if data else []):
                                if not isinstance(tc, dict):
                                    continue
                                await _request_tool(tc)
                                while tool_events:
                                    yield tool_events.popleft()
                        elif etype == StreamEventType.TOOL_RESULT:
                            await _flush_pending()
                            for tr in data if isinstance(data, list) else ([data] if data else []):
                                if not isinstance(tr, dict):
                                    continue
                                tc_id = tr.get("tool_call_id") or tr.get("id") or ""
                                await _tool_terminal({"tool_call_id": tc_id}, tr)
                                while tool_events:
                                    yield tool_events.popleft()
                        elif etype == StreamEventType.PROVIDER_RESPONSE:
                            provider_responses.append(data)
                        elif etype == StreamEventType.ERROR:
                            if isinstance(data, Exception):
                                raise data
                            raise RuntimeError(str(data) if data else "stream error")
                        elif etype == StreamEventType.DONE:
                            break
                finally:
                    with contextlib.suppress(Exception):
                        await agen.aclose()

                await _flush_pending()
                if self._cancel_event is not None and self._cancel_event.is_set():
                    raise asyncio.CancelledError
                if not terminal_tools.issuperset(requested_tools):
                    raise RuntimeError("Tool dispatch ended without a terminal outcome")

                full_text = "".join(accumulated)
                protected_payload = None
                timeline = getattr(self._agent, "_timeline", None)
                # Internal synchronous reads do not expose mutable messages to callers.
                native = getattr(timeline, "_native_messages", None)
                if native is None:
                    native = getattr(timeline, "native_messages", None)
                if native is None and self._codec is not None and hasattr(timeline, "to_llm_messages"):
                    from dana.core.timeline.native_message import NativeMessage, NativeToolCall

                    native = [
                        NativeMessage(
                            role=message.role,
                            content=message.content,
                            tool_call_id=message.tool_call_id,
                            tool_calls=[NativeToolCall.from_dict(call) for call in message.tool_calls] if message.tool_calls else None,
                            metadata={
                                "reasoning_items": message.reasoning_items,
                                "fingerprint": message.reasoning_fingerprint,
                                "response_id": message.response_id,
                            }
                            if message.reasoning_items
                            else {},
                        )
                        for message in timeline.to_llm_messages()
                    ]
                native = native or ()
                if self._codec is not None and any(message.metadata.get("reasoning_items") for message in native):
                    start = next((index for index in range(len(native) - 1, -1, -1) if native[index].role == "user"), None)
                    if start is None:
                        raise ValueError("Cannot persist replay state without the current user turn")
                    turn = native[start:]
                    if any(message.metadata.get("reasoning_items") for message in turn):
                        protected_payload = self._codec.encrypt(
                            json.dumps({"schema": "dana.native-turn/1", "messages": [message.to_dict() for message in turn]}).encode()
                        )
                model_config = getattr(self._agent, "_llm_config", {})

                # --- Terminal batch: ASSISTANT_CONTENT_FINAL + terminal in ONE append. ---
                terminal_facts = [
                    NewJournalFact(
                        fact_type=FactType.ASSISTANT_CONTENT_FINAL,
                        correlation_id=correlation_id,
                        causation_id=correlation_id,
                        payload={
                            "text": full_text,
                            "provider": self._current_provider or model_config.get("provider"),
                            "model": self._current_model or model_config.get("model"),
                            "provider_responses": provider_responses,
                        },
                        protected_payload=protected_payload,
                    ),
                    NewJournalFact(
                        fact_type=FactType.TURN_COMPLETED,
                        correlation_id=correlation_id,
                        causation_id=correlation_id,
                        payload={},
                    ),
                ]
                terminal_result = await self._append_facts(terminal_facts)
                self._current_version = terminal_result.new_version

                self._last_terminal = TurnTerminal(
                    fact_type=FactType.TURN_COMPLETED,
                    sequence=terminal_result.appended_facts[1].sequence,
                    text=full_text,
                )
                await self._checkpoint_completed_turn()
                yield HostEvent(
                    event_type=HostEventType.ASSISTANT_CONTENT_FINAL,
                    sequence=terminal_result.appended_facts[0].sequence,
                    correlation_id=correlation_id,
                    timestamp=terminal_result.appended_facts[0].timestamp,
                    text=full_text,
                    metadata={"provider_responses": provider_responses},
                )
                yield HostEvent(
                    event_type=HostEventType.TURN_COMPLETED,
                    sequence=terminal_result.appended_facts[1].sequence,
                    correlation_id=correlation_id,
                    timestamp=terminal_result.appended_facts[1].timestamp,
                )
                logger.info(
                    "turn completed",
                    session_id=self._session_id,
                    correlation_id=correlation_id,
                    version=self._current_version,
                )
                return

            except (asyncio.CancelledError, GeneratorExit) as cancellation:
                # Only cancellation of the prompt consumer belongs to its caller.
                # cancel() cancels the owned stream step and its tool tasks, not
                # this task. The event alone cannot distinguish simultaneous
                # explicit and caller cancellation.
                consumer = asyncio.current_task()
                caller_cancelled = consumer is not None and consumer.cancelling() > 0
                if self._last_terminal is not None:
                    if caller_cancelled:
                        raise
                    return
                if caller_cancelled or isinstance(cancellation, GeneratorExit):
                    # _append_facts resolved any interrupted commit and recorded
                    # its version, so a durable terminal must not be duplicated.
                    if self._last_terminal is None and not any(
                        fact_type in _TERMINAL_FACT_TYPES for fact_type in self._committed_batch_types
                    ):
                        # The committed batch was not a terminal (start/user/tool/
                        # chunk), so the turn is still owed exactly one. Append it
                        # here; a further cancellation simply leaves the turn to the
                        # reopen recovery path (recover_interrupted_turns).
                        try:
                            if chunk_buffer:
                                await self._flush_chunks(correlation_id, chunk_buffer, chunk_index)
                            partial = "".join(accumulated)
                            await _finish_pending_tools(cancelled=True)
                            cancel_result = await self._append_facts(
                                [
                                    NewJournalFact(
                                        fact_type=FactType.TURN_CANCELLED,
                                        correlation_id=correlation_id,
                                        causation_id=correlation_id,
                                        payload={"partial_text": partial, "provider_responses": provider_responses},
                                    )
                                ]
                            )
                            self._current_version = cancel_result.new_version
                            self._last_terminal = TurnTerminal(
                                fact_type=FactType.TURN_CANCELLED,
                                sequence=cancel_result.appended_facts[0].sequence,
                                text=partial,
                            )
                        except BaseException:
                            logger.warning(
                                "failed to append cancellation terminal for a caller cancellation",
                                session_id=self._session_id,
                                correlation_id=correlation_id,
                                exc_info=True,
                            )
                    # Only cancellation of the consuming task propagates to its
                    # caller. Closing the iterator terminalizes without yielding.
                    if caller_cancelled:
                        raise
                    return
                # Internal or explicit session cancellation: persist buffered
                # partial text, then a single TURN_CANCELLED terminal.
                try:
                    if chunk_buffer:
                        await self._flush_chunks(correlation_id, chunk_buffer, chunk_index)
                    partial = "".join(accumulated)
                    await _finish_pending_tools(cancelled=True)
                    cancel_facts = [
                        NewJournalFact(
                            fact_type=FactType.TURN_CANCELLED,
                            correlation_id=correlation_id,
                            causation_id=correlation_id,
                            payload={"partial_text": partial, "provider_responses": provider_responses},
                        ),
                    ]
                    cancel_result = await self._append_facts(cancel_facts)
                    self._current_version = cancel_result.new_version
                except asyncio.CancelledError:
                    # A second, unrelated cancellation (host teardown) landed while
                    # the cancellation terminal was durable-pending. The batch is
                    # committed at most once: if it committed, this turn already has
                    # its single terminal; if the append itself was cancelled, the
                    # turn is left open and the reopen recovery path
                    # (``recover_interrupted_turns``) terminalizes it. Either way a
                    # second terminal must never be appended, and the caller's
                    # cancellation must still propagate.
                    logger.warning(
                        "cancellation terminal interrupted",
                        session_id=self._session_id,
                        correlation_id=correlation_id,
                    )
                    raise
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
                self._last_terminal = TurnTerminal(
                    fact_type=FactType.TURN_CANCELLED,
                    sequence=cancel_result.appended_facts[0].sequence,
                    text=partial,
                )
                # Internal stream cancellation and explicit cancel() both produce
                # a normal cancellation terminal. The consumer's task was not
                # cancelled, so it must not receive an unrelated CancelledError.
                while tool_events:
                    yield tool_events.popleft()
                yield HostEvent(
                    event_type=HostEventType.TURN_CANCELLED,
                    sequence=cancel_result.appended_facts[0].sequence,
                    correlation_id=correlation_id,
                    timestamp=cancel_result.appended_facts[0].timestamp,
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
                if self._last_terminal is not None:
                    raise
                # Model/runtime error: terminalize as TURN_ERROR.
                terminal_response = getattr(exc, "terminal_response", None)
                if terminal_response is not None:
                    metadata = {
                        "iteration": getattr(self._agent, "_star_loop_count", None),
                        "model": terminal_response.model,
                        "finish_reason": terminal_response.finish_reason,
                        "usage": terminal_response.usage,
                        "reasoning_tokens": terminal_response.reasoning_tokens,
                        "response_id": terminal_response.response_id,
                    }
                    if not provider_responses or provider_responses[-1] != metadata:
                        provider_responses.append(metadata)
                try:
                    await _flush_pending()
                    await _finish_pending_tools(cancelled=False)
                    error_facts = [
                        NewJournalFact(
                            fact_type=FactType.TURN_ERROR,
                            correlation_id=correlation_id,
                            causation_id=correlation_id,
                            payload={"error": str(exc), "error_type": type(exc).__name__, "provider_responses": provider_responses},
                        ),
                    ]
                    error_result = await self._append_facts(error_facts)
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
                self._last_terminal = TurnTerminal(
                    fact_type=FactType.TURN_ERROR,
                    sequence=error_result.appended_facts[0].sequence,
                    error=str(exc),
                )
                while tool_events:
                    yield tool_events.popleft()
                yield HostEvent(
                    event_type=HostEventType.TURN_ERROR,
                    sequence=error_result.appended_facts[0].sequence,
                    correlation_id=correlation_id,
                    timestamp=error_result.appended_facts[0].timestamp,
                    metadata={"error": str(exc), "error_type": type(exc).__name__, "provider_responses": provider_responses},
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
                self._tool_commit_observer = None
                if executor is not None and hasattr(executor, "set_execution_hooks"):
                    _clear_execution_hooks(executor, self._agent)
                if self._agent is not None:
                    self._agent._session_cancel_event = None
                self._cancel_event = None
                self._stream_step = None

    # ------------------------------------------------------------------
    # D4: Fact append helper (used by model switching)
    # ------------------------------------------------------------------

    async def _append_facts(self, facts: list[NewJournalFact]) -> Any:
        """Resolve a journal commit before acknowledging task cancellation.

        Cancelling an await cannot roll back a database worker that has already
        committed. Keep ownership until its outcome and version are known.
        """
        async with self._append_lock:
            operation = asyncio.create_task(self._repository.append(self._owner_scope, self._session_id, self._current_version, facts))
            cancelled = False
            while True:
                try:
                    result = await asyncio.shield(operation)
                    break
                except asyncio.CancelledError:
                    if operation.cancelled():
                        raise
                    cancelled = True
                    if self._cancel_event is not None:
                        self._cancel_event.set()
                    if operation.done():
                        result = operation.result()
                        break
            self._current_version = result.new_version
            self._committed_batch_types = tuple(fact.fact_type for fact in result.appended_facts)
            try:
                if self._tool_commit_observer is not None:
                    self._tool_commit_observer(result.appended_facts, cancelled)
                self._advance_conversation(result.appended_facts)
            except Exception:
                # The batch is already durable. Observer or derived-fold failure
                # must not turn committed success into a second terminal.
                # Rebuild the derived cache from the journal on the next request.
                self._invalidate_conversation_cache()
                logger.warning(
                    "post-commit observation failed",
                    session_id=self._session_id,
                    version=self._current_version,
                    exc_info=True,
                )
            if cancelled:
                # This may be the prompt consumer, its owned stream step, or a
                # tool task. Preserve cancellation here; prompt() classifies the
                # consumer's own cancellation separately from session.cancel().
                raise asyncio.CancelledError
            return result

    async def append_fact(self, fact: NewJournalFact) -> None:
        """Append a single fact to the journal.

        Used by the ACP agent to journal model-change facts after a
        successful model switch. The fact is appended with the current
        version as the expected version (optimistic concurrency).
        """
        result = await self._append_facts([fact])
        self._current_version = result.new_version

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _prepare_agent(self) -> None:
        """Reconcile new journal facts without rebuilding an unchanged timeline."""
        if self._agent is None:
            self._agent = self._agent_factory()
            # Opaque replay belongs in the codec-protected journal, never a
            # second plaintext legacy snapshot, including when no codec is set.
            self._agent._journal_owned_protected_replay = True
            # The journal owns durable history; standalone agents retain their
            # legacy snapshot/resume path.
            self._agent._journal_managed_session = True
            # Relabel the working timeline to the journal session before hydration.
            set_sid = getattr(self._agent, "set_session_id", None)
            if callable(set_sid):
                set_sid(self._session_id)
            # D7.5 (AC #4): wire MCP single-dispatch wrapper once, when the agent
            # is first built. Both ACP and CLI sessions share this path.
            await self._wire_mcp_tools(self._agent)
        self._agent._session_cancel_event = self._cancel_event
        await self._refresh_conversation()
        await self._build_tool_catalog()
        executor = getattr(getattr(self._agent, "_runtime", None), "_tool_executor", None)
        if executor is None or not hasattr(executor, "set_execution_hooks"):
            self._register_policy_hook()

    def _invalidate_conversation_cache(self) -> None:
        """Drop every derived projection artifact and force a journal rebuild.

        Used on a post-commit fold failure (C3) and on any other derived-state
        failure: the Session Journal is authoritative, so the next request
        reconstructs the working context from facts rather than trusting a
        half-updated state or a stale checkpoint.
        """
        self._conversation_state = None
        self._working_context = None
        self._working_format = None
        self._hydrated_tail_count = 0
        self._needs_hydration = True

    def _advance_conversation(self, facts: Sequence[JournalFact]) -> None:
        """Fold definitively committed local facts under the append lock."""
        state = self._conversation_state
        if state is None:
            return
        if state.provider_key != self._current_provider:
            self._invalidate_conversation_cache()
            return
        state.apply(facts)

    async def _refresh_conversation(self) -> None:
        """Use the journal cursor for external-writer visibility and recovery."""
        header = await self._repository.load_session(self._owner_scope, self._session_id)
        state = self._conversation_state
        if state is not None and (state.provider_key != self._current_provider or state.last_sequence > header.version):
            self._conversation_state = state = None
            self._working_context = None
            self._hydrated_tail_count = 0
            self._needs_hydration = True
        if state is None:
            await self._load_conversation_checkpoint(header.version)
            state = self._conversation_state
        if state is None:
            state = self._conversation_projector.new_state(self._current_provider)
            self._conversation_state = state
        previous_revision = state.rewrite_revision
        delta = await self._repository.read_facts(self._owner_scope, self._session_id, after_sequence=state.last_sequence)
        state.apply(delta)
        self._current_version = state.last_sequence
        if state.current_provider is not None and self._current_provider is None:
            self._current_provider = state.current_provider
            self._current_model = state.current_model
            # An initial ungated fold can span provider switches. Re-gate once
            # before exposing any opaque state to the newly restored provider.
            facts = await self._repository.read_facts(self._owner_scope, self._session_id)
            state = self._conversation_projector.new_state(self._current_provider)
            state.apply(facts)
            self._conversation_state = state
            self._working_context = None
            self._needs_hydration = True
            self._hydrated_tail_count = 0
        rewritten = state.rewrite_revision != previous_revision
        if self._needs_hydration or rewritten:
            if self._working_context is None:
                self._populate_timeline(state.view())
            else:
                self._restore_working_context()
                self._populate_timeline(
                    ConversationView(
                        messages=state.messages_after(0),
                        replay_state=None,
                        interruption_observation=state.interruption_observation,
                        last_sequence=state.last_sequence,
                    ),
                    append=True,
                )
            self._hydrated_tail_count = state.message_count
            self._needs_hydration = False
        elif delta:
            self._populate_timeline(
                ConversationView(
                    messages=state.messages_after(self._hydrated_tail_count),
                    replay_state=None,
                    interruption_observation=state.interruption_observation,
                    last_sequence=state.last_sequence,
                ),
                append=True,
            )
            self._hydrated_tail_count = state.message_count
        self._reconcile_interruption_note(state.interruption_observation)
        if self._working_context is None and not state.open_correlation_ids:
            self._capture_working_context()

    def _capture_working_context(self) -> None:
        """Retain the actual model working set, not the full journal history."""
        state = self._conversation_state
        if state is None or state.open_correlation_ids:
            return
        timeline = getattr(self._agent, "_timeline", None)
        native = getattr(timeline, "_native_messages", None)
        if native is None:
            native = getattr(timeline, "native_messages", None)
        entries = native if native is not None else getattr(timeline, "timeline", None)
        if entries is None or any(not callable(getattr(entry, "to_dict", None)) for entry in entries):
            return
        self._working_context = deepcopy(
            [
                entry.to_dict()
                for entry in entries
                if not getattr(entry, "ephemeral", False)
                and not (getattr(entry, "metadata", None) or {}).get("ephemeral")
                # I2: the interruption note is derived state, re-injected on the
                # next reconciliation when the observation is still current.
                # Persisting it would freeze a stale instruction into the cache.
                and not (getattr(entry, "metadata", None) or {}).get(_INTERRUPTION_NOTE_KEY)
            ]
        )
        self._working_format = "native" if native is not None else "entries"
        state.discard_closed_history()
        self._hydrated_tail_count = state.message_count
        self._needs_hydration = False

    def _restore_working_context(self) -> None:
        timeline = getattr(self._agent, "_timeline", None)
        loader = getattr(timeline, "load_from_entries", None)
        # Decoders borrow nested content/metadata. A failed turn must not mutate
        # the retained committed snapshot used for the next recovery.
        snapshot = deepcopy(self._working_context or [])
        if callable(loader):
            loader(snapshot)
            return
        from dana.core.timeline.timeline import TimelineEntry

        entries = getattr(timeline, "timeline", None)
        if entries is not None:
            entries.clear()
            add_entry = getattr(timeline, "add_entry", entries.append)
            for data in snapshot:
                add_entry(TimelineEntry.from_dict(data))

    def _checkpoint_aad(self, sequence: int) -> bytes:
        return json.dumps(
            ["dana.conversation-working/1", self._owner_scope.owner_id, self._owner_scope.workspace, self._session_id, sequence],
            separators=(",", ":"),
        ).encode()

    async def _load_conversation_checkpoint(self, version: int) -> None:
        checkpoint = await self._repository.load_projection_checkpoint(self._owner_scope, self._session_id, "conversation-working-v1")
        if checkpoint is None or checkpoint.last_sequence > version:
            return
        try:
            data = checkpoint.data
            if data.get("schema") != "dana.conversation-working/1":
                return
            if data.get("encoding") == "aes-gcm":
                if self._codec is None:
                    return
                raw = self._codec.decrypt(
                    base64.b64decode(data["ciphertext"], validate=True), aad=self._checkpoint_aad(checkpoint.last_sequence)
                )
                restored = json.loads(raw)
            elif data.get("encoding") == "public":
                restored = data["state"]
            else:
                return
            state = ConversationProjectionState.from_dict(restored["conversation"], protected_state_codec=self._codec)
            if state.last_sequence != checkpoint.last_sequence:
                return
            if self._current_provider is not None and state.provider_key != self._current_provider:
                return
            context = restored["working_context"]
            if not isinstance(context, list) or any(not isinstance(item, dict) for item in context):
                return
            timeline = getattr(self._agent, "_timeline", None)
            native = getattr(timeline, "_native_messages", None)
            if native is None:
                native = getattr(timeline, "native_messages", None)
            working_format = "native" if native is not None else "entries"
            if restored["working_format"] != working_format:
                return
            # Validate with the same decoder before accepting this derived cache.
            # Otherwise malformed entries fail later, outside the fallback guard,
            # and prevent an intact authoritative journal from reopening.
            if working_format == "native":
                from dana.core.timeline.native_message import NativeMessage

                for item in context:
                    NativeMessage.from_dict(item)
            else:
                from dana.core.timeline.timeline import TimelineEntry

                for item in context:
                    TimelineEntry.from_dict(item)
        except Exception:
            # Derived cache only: unknown versions, invalid keys, corruption or
            # an incompatible shape fall back to the authoritative journal.
            logger.warning("conversation checkpoint rejected", session_id=self._session_id)
            return
        self._conversation_state = state
        self._working_context = context
        self._working_format = working_format
        self._current_provider = state.provider_key
        self._current_model = state.current_model
        self._hydrated_tail_count = 0
        self._needs_hydration = True

    async def _save_conversation_checkpoint(self) -> None:
        """Persist a bounded, owner/sequence-bound derived working-set cache."""
        state = self._conversation_state
        if state is None or self._working_context is None:
            return
        restored = {
            "conversation": state.to_dict(),
            "working_context": self._working_context,
            "working_format": self._working_format,
        }
        data: dict[str, Any] = {"schema": "dana.conversation-working/1"}
        if self._codec is not None:
            raw = json.dumps(restored, separators=(",", ":"), ensure_ascii=False).encode()
            encrypted = self._codec.encrypt(raw, aad=self._checkpoint_aad(state.last_sequence))
            data.update(encoding="aes-gcm", ciphertext=base64.b64encode(encrypted).decode("ascii"))
        else:
            # Without a key, keep the existing public-conversation semantics;
            # never introduce a plaintext persistence path for provider replay.
            if any(item.get("metadata", {}).get("reasoning_items") for item in self._working_context):
                return
            data.update(encoding="public", state=restored)
        try:
            await self._repository.save_projection_checkpoint(
                self._owner_scope,
                self._session_id,
                ProjectionCheckpoint(projection_name="conversation-working-v1", last_sequence=state.last_sequence, data=data),
            )
        except Exception:
            # A failed derived-cache write cannot undo or hide a committed
            # terminal outcome. Reopen can always reconstruct from facts.
            logger.warning("conversation checkpoint write failed", session_id=self._session_id)

    def _reconcile_interruption_note(self, observation: str | None) -> None:
        """Hold exactly one model-visible interruption note while it is current.

        I2: the note is derived state, not history. It is injected once when the
        fold reports an interruption observation (so the model is told not to
        assume unfinished effects completed), and retired as soon as that
        observation clears - which a later completed/errored/cancelled turn, or
        the recovery turn's own terminal fact, does. Reopening a session
        therefore never re-injects or duplicates it, and it is never persisted in
        the derived working-context checkpoint.

        The scan is authoritative rather than flag-based, so a projection that
        dropped the note on its own (``Timeline.set_context`` replaces every
        CONTEXT entry on each prompt build) is repaired on the next request while
        the observation is still current.
        """
        timeline = getattr(self._agent, "_timeline", None)
        if timeline is None:
            return
        native = getattr(timeline, "_native_messages", None)
        if native is None:
            native = getattr(timeline, "native_messages", None)
        entries = getattr(timeline, "timeline", None)
        if entries is None and native is None:
            return

        def _is_note(item: Any) -> bool:
            return bool((getattr(item, "metadata", None) or {}).get(_INTERRUPTION_NOTE_KEY))

        present = (entries is not None and any(_is_note(entry) for entry in entries)) or (
            native is not None and any(_is_note(message) for message in native)
        )
        if present and observation:
            return
        if present:
            if entries is not None:
                entries[:] = [entry for entry in entries if not _is_note(entry)]
            if native is not None:
                native[:] = [message for message in native if not _is_note(message)]
            return
        if not observation:
            return

        from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType

        note = TimelineEntry(
            entry_type=TimelineEntryType.CONTEXT,
            content=observation,
            metadata={_INTERRUPTION_NOTE_KEY: True},
        )
        add_entry = getattr(timeline, "add_entry", None)
        if callable(add_entry):
            # CompressedTimeline.add_entry keeps the entries list and the native
            # mirror in step for one entry.
            add_entry(note)
        elif entries is not None:
            entries.append(note)
        else:
            native.append(note)

    async def _checkpoint_completed_turn(self) -> None:
        """Publish derived context only after the final journal batch commits."""
        try:
            state = self._conversation_state
            if state is None:
                return
            if not getattr(self._agent, "_stream_updates_timeline", False):
                # Public custom factories need not maintain STAR's timeline.
                # Reconcile their committed messages before retaining context.
                if self._working_context is None:
                    self._populate_timeline(state.view())
                else:
                    self._restore_working_context()
                    self._populate_timeline(
                        ConversationView(
                            messages=state.messages_after(0),
                            replay_state=None,
                            interruption_observation=state.interruption_observation,
                            last_sequence=state.last_sequence,
                        ),
                        append=True,
                    )
            self._reconcile_interruption_note(state.interruption_observation)
            self._capture_working_context()
            await self._save_conversation_checkpoint()
        except Exception:
            self._invalidate_conversation_cache()
            logger.warning("committed turn context cache unavailable", session_id=self._session_id)

    async def _wire_mcp_tools(self, agent: Any) -> None:
        """Attach the MCP single-dispatch wrapper to the agent (D7.5 AC #4).

        Gated by ``DANA_CODE_MCP_ENABLED`` (default on) and the presence of an
        MCP server config (``DANA_MCP_SERVERS``). When enabled, builds the
        per-server transports + adapters + the ``MCPDispatchResource`` and
        appends it to ``agent._resources`` so the STAR loop's native-tool
        registry discovers and dispatches ``call_mcp_tool``. The native catalog
        classifies individual MCP tools for the same execution permission gate.
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

    async def aclose(self) -> None:
        """Release this session's operations and clients before its repository.

        Hosts must exhaust or close their prompt iterator before teardown. An
        active consumer is cancelled and allowed to finish its durable terminal
        before resources used by that turn are released.
        """
        if self._cancel_event is not None:
            await self.cancel()
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                close_agent = getattr(self._agent, "aclose", None)
                if close_agent is not None:
                    await close_agent()
            finally:
                try:
                    await self.dispose_mcp()
                finally:
                    self._unregister_policy_hook()
                    # I9: clear this session's owner-scoped scheduling artifacts
                    # (effect-order catalog + MCP effect map) from the agent that
                    # carries them. Setting is the same operation that published
                    # them, so it is a no-op for an agent that was never prepared.
                    if self._agent is not None:
                        self._agent._session_effect_catalog = None
                        self._agent._session_mcp_effects = None
                    if self._owns_repository:
                        await self._repository.close()

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

        Reuse the current version until runtime discovery or MCP wiring changes.
        Gated by ``DANA_CODE_TOOL_CATALOG_ENABLED`` (default on). Classification
        does not itself execute tools.
        """
        try:
            from dana.config.code_capabilities import tool_catalog_enabled
        except Exception:
            tool_catalog_enabled = lambda: True  # noqa: E731
        if not tool_catalog_enabled():
            self._tool_catalog = None
            self._catalog_key = None
            return
        runtime = getattr(self._agent, "_runtime", None) if self._agent is not None else None
        if runtime is None:
            self._tool_catalog = None
            self._catalog_key = None
            return
        # Ensure _native_tools is built (cached/idempotent) so MCP schemas can be
        # appended (D7 follow-up 1+2: per-tool MCP UX surfaces via _native_tools).
        if hasattr(runtime, "_build_native_tools_if_supported"):
            runtime._build_native_tools_if_supported(self._agent)
        wiring = self._mcp_wiring
        revision = getattr(runtime, "tool_catalog_revision", None)
        key = (runtime, revision(), wiring) if callable(revision) else None
        if key is not None and key == self._catalog_key:
            return
        native_tools = getattr(runtime, "_native_tools", None)
        if native_tools is None:
            self._tool_catalog = None
            self._catalog_key = None
            return
        mcp_names = wiring.mcp_names if wiring is not None else frozenset()
        if self._catalog_mcp_names:
            native_tools[:] = [
                schema for schema in native_tools if schema.get("function", schema).get("name") not in self._catalog_mcp_names
            ]
        if wiring is not None:
            native_tools.extend(wiring.mcp_schemas)
        tool_executor = getattr(runtime, "_tool_executor", None)
        if tool_executor is not None and hasattr(tool_executor, "set_mcp_dispatch_getter"):
            tool_executor.set_mcp_dispatch_getter((lambda w=wiring: w.dispatch_map) if wiring is not None else None)
            tool_executor.invalidate_tool_metadata()
        self._catalog_mcp_names = mcp_names

        from dana.core.tool.native_catalog import build_native_tool_catalog

        self._catalog_version += 1
        self._tool_catalog = build_native_tool_catalog(
            list(native_tools),
            version=self._catalog_version,
            mcp_names=mcp_names,
            mcp_effects=getattr(wiring, "mcp_effects", None) if wiring is not None else None,
        )
        self._catalog_key = key
        if tool_executor is not None:
            # I9: publish this session's scheduling artifacts ON THE AGENT that
            # owns them, and wire one owner-aware getter per executor. A
            # runtime/executor can be shared by several overlapping sessions, so
            # both the effect catalog and the MCP effect map are resolved from the
            # *invoking* agent (the same rule the tool-name registry uses) - a
            # session-scoped mapping would let the last writer win and schedule
            # another session's calls against the wrong metadata. A getter that
            # resolves to None degrades to the executor's own conservative
            # fallbacks (native table / per-server EXECUTE ordering).
            self._agent._session_effect_catalog = self._tool_catalog
            # ``mcp_effects`` is optional on the wiring contract: a test double or
            # an older wiring object simply has none, which leaves the executor's
            # per-server ordering in charge.
            self._agent._session_mcp_effects = dict(getattr(wiring, "mcp_effects", None) or {}) if wiring is not None else None
            if hasattr(tool_executor, "set_effect_catalog_getter"):
                tool_executor.set_effect_catalog_getter(lambda agent: getattr(agent, "_session_effect_catalog", None))
            if hasattr(tool_executor, "set_mcp_effect_resolver"):
                tool_executor.set_mcp_effect_resolver(
                    lambda agent=None: getattr(agent, "_session_mcp_effects", None) if agent is not None else None
                )

    def _register_policy_hook(self) -> None:
        """Subscribe the TOOL_CALL permission hook on the agent's EventBus.

        Gated by ``DANA_CODE_PERMISSION_PREFLIGHT_ENABLED`` (default on) AND a
        wired ``policy_evaluator``. Idempotent: skips if already registered or
        if the gate is closed. Denied or unresolved interactive decisions block
        execution. The live executor uses awaited authoritative hooks instead.
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
        """Apply policy to event-bus dispatches, failing closed on missing consent."""
        if self._policy_evaluator is None:
            return None
        ext_op = event.payload.get("operation") if isinstance(getattr(event, "payload", None), dict) else None
        if ext_op is None:
            return {"block": True, "reason": "denied: missing operation for permission evaluation"}
        return await self._evaluate_tool_permission(
            {"function": getattr(ext_op.tool_identity, "name", ""), "arguments": dict(ext_op.arguments)}
        )

    async def _evaluate_tool_permission(self, tool_call: dict[str, Any]) -> dict[str, Any] | None:
        if self._policy_evaluator is None:
            return None
        from dana.core.policy.evaluator import PolicyDecision
        from dana.core.policy.operations import build_policy_operation

        operation = build_policy_operation(
            tool_call,
            catalog=self._tool_catalog,
            owner=self._owner_scope.owner_id,
            workspace=self._owner_scope.workspace,
        )
        result = await self._policy_evaluator.evaluate(operation, self._owner_scope)
        if result.decision is PolicyDecision.DENY:
            return {"block": True, "reason": f"denied: {result.reason}"}
        if result.decision is PolicyDecision.NEEDS_PROMPT:
            if self._permission_prompt_callback is None:
                return {"block": True, "reason": "denied: interactive permission is unavailable"}
            verdict = await self._permission_prompt_callback(operation)
            if not verdict.allowed:
                return {"block": True, "reason": f"denied: {verdict.reason}"}
        return None

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
        result = await self._append_facts(chunk_facts)
        self._current_version = result.new_version

    def _populate_timeline(self, view: ConversationView, *, append: bool = False) -> None:
        """Hydrate or append projected messages through the timeline abstraction.

        Defensive against agents that lack a ``_timeline`` (e.g. fakes in tests).
        """
        from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType

        timeline = getattr(self._agent, "_timeline", None)
        entries = getattr(timeline, "timeline", None)
        if entries is None:
            return
        restored = []
        for msg in view.messages:
            if msg.role == "user":
                entry_type = TimelineEntryType.USER_MESSAGE
            elif msg.role == "tool":
                entry_type = TimelineEntryType.RESOURCE_RESULT
            elif msg.role == "system":
                entry_type = TimelineEntryType.CONTEXT
            elif msg.tool_calls:
                entry_type = TimelineEntryType.TOOL_CALL
            else:
                entry_type = TimelineEntryType.AGENT_RESPONSE
            metadata = {}
            if msg.reasoning_items:
                metadata = {
                    "reasoning_items": msg.reasoning_items,
                    "fingerprint": msg.reasoning_fingerprint,
                    "response_id": msg.response_id,
                }
            restored.append(
                TimelineEntry(
                    entry_type=entry_type,
                    content=msg.content,
                    tool_calls=msg.tool_calls,
                    tool_call_id=msg.tool_call_id,
                    metadata=metadata,
                )
            )
        loader = getattr(timeline, "load_from_entries", None)
        if loader is not None and not append:
            loader(restored)
        else:
            if not append:
                entries.clear()
            add_entry = getattr(timeline, "add_entry", entries.append)
            for entry in restored:
                add_entry(entry)

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
        result = await self._append_facts(facts)
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
        *,
        arguments: dict[str, Any] | None = None,
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
                    **({"arguments": arguments} if arguments is not None else {}),
                },
            )
        ]
        result = await self._append_facts(facts)
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
                **({"arguments": arguments} if arguments is not None else {}),
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
        result = await self._append_facts(facts)
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
        result = await self._append_facts(facts)
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
        result = await self._append_facts(facts)
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
        result_obj = await self._append_facts(facts)
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

        decision = await self._evaluate_tool_permission({"function": tool_name, "arguments": arguments})
        denied = bool(decision and decision.get("block"))
        yield await self.journal_tool_authorized_or_denied(
            tool_call_id=tool_call_id,
            correlation_id=correlation_id,
            authorized=not denied,
            reason=decision.get("reason") if decision else None,
            arguments=arguments,
        )
        if denied:
            yield await self.journal_tool_terminal(
                tool_call_id=tool_call_id,
                correlation_id=correlation_id,
                terminal_type=FactType.TOOL_FAILURE,
                error=decision.get("reason", "Permission denied"),
            )
            return

        if self._tool_engine is None:
            yield await self.journal_tool_terminal(
                tool_call_id=tool_call_id,
                correlation_id=correlation_id,
                terminal_type=FactType.TOOL_FAILURE,
                error=f"No execution engine configured for tool {tool_name}",
            )
            return
        # 3. Journal TOOL_STARTED
        yield await self.journal_tool_started(
            tool_call_id=tool_call_id,
            correlation_id=correlation_id,
        )

        # 4. Execute via engine
        try:
            tool_call = {
                "tool_call_id": tool_call_id,
                "function": tool_name,
                "arguments": arguments,
            }

            if self._use_legacy_executor:
                # Legacy executor path (non-ACP host fallback)
                from dana.core.tool.offload import run_blocking

                result_dict = await run_blocking(self._tool_engine.execute, tool_call)
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
                terminal_type=FactType.TOOL_EFFECT_UNKNOWN,
                result={"cancellation": "requested", "effect": "unknown"},
            )

        except Exception as exc:
            yield await self.journal_tool_terminal(
                tool_call_id=tool_call_id,
                correlation_id=correlation_id,
                terminal_type=FactType.TOOL_FAILURE,
                error=str(exc),
            )
