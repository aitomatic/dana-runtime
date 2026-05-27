"""
STARAgent implementation using composition-based architecture.

This is the main STARAgent implementation using composition instead of mixin inheritance.
It provides a cleaner, more maintainable architecture for the STAR (See-Think-Act-Reflect) pattern
and conversational agent functionality using composable components.
"""

from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
import inspect
from typing import Any
from uuid import uuid4

import structlog

from dana.common.config import config_manager
from dana.common.llm import LLM
from dana.common.llm.types import LLMMessage
from dana.common.observable import observable
from dana.common.protocols import AgentProtocol, DictParams, Notifiable, ResourceProtocol, WorkflowProtocol
from dana.common.protocols.types import LearningPhase
from dana.core.timeline.compressed_timeline import CompressedTimeline
from dana.core.timeline.timeline import Timeline, TimelineEntry, TimelineEntryType
from dana.repositories.repository_factory import DEFAULT_REPOSITORY_FACTORY, RepositoryFactory

from ..knowledge.prompts.codecs import AbstractCodec, NativeToolsCodec
from ..prompt.prompt_api import PromptAPIProtocol
from ..runtime import AgentRuntime
from .base_star_agent import BaseSTARAgent
from .components import Communicator, LearnerProtocol, State
from .components.observer import ObserverProtocol
from .star_agent_streaming import STARAgentStreamingMixin


logger = structlog.get_logger()


# from dana.apps.dana.thought_logger import ThoughtLogger  # Moved to avoid circular import


class STARAgent(STARAgentStreamingMixin, BaseSTARAgent):
    """STARAgent implementation using composition-based architecture."""

    # Configuration constants
    MAX_THINK_RETRIES = 3  # Maximum retries when output format or done/response rules are invalid

    def __init__(
        self,
        agent_type: str | None = None,
        agent_id: str | None = None,
        llm_provider: str | None = None,
        model: str | None = None,
        config: dict[str, Any] | None = None,
        max_context_tokens: int = 4000,
        auto_register: bool = True,
        registry=None,
        codec: type[AbstractCodec] = NativeToolsCodec,
        runtime: AgentRuntime | None = None,
        repository_factory: RepositoryFactory = DEFAULT_REPOSITORY_FACTORY,
        prompt_api: PromptAPIProtocol | None = None,
        observer: ObserverProtocol | None = None,
        learner: LearnerProtocol | None = None,
        ltmemory_path: str | None = None,
        enable_skills: bool = True,
        skills_output_dir: str = "./skill_output",
        enable_web_search: bool = False,
        enable_code_execution: bool = False,
        enable_assistant: bool = True,
        identity_override: str | None = None,
        compress_timeline: bool = True,
        compress_trigger_tokens: int | None = None,
        **kwargs,
    ):
        """
        Initialize the STARAgent with composition-based architecture.

        Args:
            agent_type: Type of agent (e.g., 'coding', 'financial_analyst').
            agent_id: ID of the agent (defaults to None)
            llm_provider: LLM provider name (e.g., 'anthropic', 'openai')
            model: Model name to use (defaults to provider's default)
            config: Optional configuration dictionary
            max_context_tokens: Maximum tokens for timeline context
            auto_register: Whether to automatically register with the global registry
            registry: Specific registry to use (defaults to global registry)
            enable_web_search: Whether to enable web search resource (default: False).
                Provides search() and fetch_url() methods without requiring API keys.
            enable_code_execution: Whether to enable code execution resource (default: False).
                Provides secure Python code execution in a sandbox.
            enable_assistant: Whether to enable the built-in AssistantAgent (default: True).
                The assistant can search the web and execute code, returning concise results.
                Useful for delegating subtasks that require research or computation.
            runtime: Runtime implementation that encapsulates prompt building, LLM calls,
                response parsing, and tool execution. Defaults to DefaultRuntime.
            codec: Codec class for tool call encoding/decoding (default: NativeToolsCodec).
            ltmemory_path: Optional path for long-term memory storage (enables cross-session learning)
            enable_skills: Whether to enable Claude Code skills resource discovery
            skills_output_dir: Directory to use for skill output files
            identity_override: Optional identity string that overrides the class docstring.
                Used by fork subagents to inject skill content as their identity.
            compress_timeline: Whether to enable timeline compression (default True).
                Set to False to use plain Timeline without LLM-based compression.
            **kwargs: Additional arguments passed to components
        """
        # Initialize base class first (handles registration)
        kwargs |= {
            "agent_type": agent_type,
            "agent_id": agent_id,
            "auto_register": auto_register,
            "registry": registry,
        }
        super().__init__(**kwargs)

        # Determine effective LLM provider: explicit > first available > anthropic fallback
        if llm_provider is None:
            llm_provider = config_manager.get_first_available_provider() or "anthropic"

        # Initialize LLM (lazy - only created when first accessed)
        self._llm_client = None  # Explicit init to avoid __getattr__ interception
        self._llm_config = {
            "provider": llm_provider,
            "model": model,
        }

        self._session_id = str(uuid4())
        self._repository_factory = repository_factory
        self._codec = codec
        self._identity_override = identity_override

        if runtime is None:
            from dana.core.runtime import RuntimeRegistry

            runtime = RuntimeRegistry.select_codec_runtime(
                provider=llm_provider,
                model=model,
                codec=codec,
                use_native_tools=None,
            )

        self._runtime = runtime

        # Initialize other components
        self._communicator = Communicator(self)
        self._state = State(self)
        # self._learner = learner or Learner(self, repository_factory=self._repository_factory)
        self._learner = learner
        if self._learner is not None:
            self._learner._agent = self

        # Initialize long-term memory if path provided
        if ltmemory_path:
            from dana.core.memory import LTMemory

            self._ltmemory = LTMemory(
                path=ltmemory_path,
                llm_provider=llm_provider,
                llm_model=model or config_manager.get_provider_default_model(llm_provider),
            )
        else:
            self._ltmemory = None

        # Determine storage_config for timeline and event_log

        # Initialize timeline: use CompressedTimeline by default unless explicitly injected.
        # compress_timeline=False disables LLM-based compression (behaves like plain Timeline).
        # system/tools callbacks fold system-prompt + tools-schema size into needs_compression()
        # estimate. Both use the existing len(str)//4 heuristic.
        #
        # Two independent knobs are threaded here:
        #   - max_context_tokens → LLM context-window BUDGET for to_llm_messages()
        #   - compress_trigger_tokens → compression TRIGGER (None → DANA_COMPACT_TRIGGER_TOKENS
        #     env var wins, so ops can retune without code changes).
        # Historically these were aliased to the same value; the split lets ops set
        # the trigger via env while agent authors still pick an appropriate context budget.
        # Persisted so set_session_id can rebuild an identically-configured
        # timeline when the session boundary changes (see _build_timeline).
        self._max_context_tokens = max_context_tokens
        self._compress_trigger_tokens = compress_trigger_tokens
        self._compression_enabled = compress_timeline
        self._timeline = self._build_timeline()

        # Initialize EventLog API (only if observer AND codec provided)
        # Events ONLY come from Observer - no observer = no EventLog
        if observer is not None:
            from dana.core.agent.components.event_log_api import EventLogAPI

            self._event_log = EventLogAPI(
                agent=self,
                observer=observer,  # REQUIRED - EventLog only works with Observer
                repository_factory=self._repository_factory,
            )
        else:
            # No observer or codec = no EventLog (events only come from Observer)
            self._event_log = None

        # CRITICAL-2 companion — auto-wire a resource that lets the LLM read
        # back tool_results that were dumped to disk at ingest time. Opt out
        # via env (used in tests that don't need a filesystem repository).
        import os as _os

        if _os.getenv("DANA_DISABLE_TOOL_RESULT_DUMP_RESOURCE") != "1":
            try:
                from dana.core.resource.tool_result_dump_resource import ToolResultDumpResource

                self.with_resources(ToolResultDumpResource(agent=self, auto_register=False))
            except Exception as _e:  # pragma: no cover — don't block agent boot on resource wiring
                logger.warning("tool_result_dump_resource_wire_failed", error=str(_e))

        if enable_web_search:
            try:
                from dana.core.resource.simple_search import SimpleWebSearch

                self.with_resources(SimpleWebSearch(resource_id="web-search"))
            except ImportError:
                pass  # Web search not available

        if enable_skills:
            from dana.core.skills import ClaudeCodeSkills

            skills = ClaudeCodeSkills(output_dir=skills_output_dir, notifier=self)
            if skills.enabled:
                self.with_resources(skills)

        if enable_code_execution:
            from dana.common.resource import CodeExecutionResource

            self.with_resources(CodeExecutionResource(resource_id="code-execution"))

        if enable_assistant:
            from dana.core.agent.assistant_agent import AssistantAgent

            # Create assistant with web search and code execution
            # The assistant runs in its own context, so it needs its own resources
            assistant = AssistantAgent(
                agent_id=f"{self.agent_id}_assistant",
                auto_register=False,  # Don't register sub-agent globally
            )
            self.with_agents(assistant)

        # Initialize reminder system for system-reminder injection.
        # Reminders check validity lazily during evaluate(), so order doesn't matter.
        from dana.core.reminder import ReminderManager

        self._reminder_manager = ReminderManager()
        self._star_loop_count = 0  # Tracks iterations within current query

    def _build_timeline(self) -> CompressedTimeline:
        """Construct a fresh, fully-configured CompressedTimeline.

        Single source of truth for timeline construction — used by ``__init__``
        and by ``set_session_id``. Building a new instance (rather than mutating
        the existing one) resets ALL session-scoped state, including the
        compaction-tracking fields (``_last_compression_at``,
        ``_active_compact_session_id``, ``_active_compact_compression_at``) that
        a bare ``rehydrate()`` would otherwise leak across sessions.
        """
        return CompressedTimeline(
            max_context_tokens=self._max_context_tokens,
            max_tokens_until_compression=self._compress_trigger_tokens,
            agent=self,
            repository_factory=self._repository_factory,
            compression_enabled=self._compression_enabled,
            system_tokens_fn=self._estimate_system_prompt_tokens,
            tools_tokens_fn=self._estimate_tools_tokens,
        )

    def set_session_id(self, session_id: str, reload_timeline: bool = False) -> None:
        """Switch the agent to a different session.

        With ``reload_timeline=False`` (default) the call is a pure relabel:
        the current in-memory timeline is kept and carried into the new session
        id (it is persisted under the new id on the next ``save``). This is the
        default because most callers — including subclasses that seed their own
        timeline before the STAR loop — manage their own context.

        With ``reload_timeline=True`` ``session_id`` becomes a real context
        boundary:
          1. Flushes the outgoing session's timeline to disk (no data loss).
          2. Rebuilds the timeline from scratch — resets entries AND all
             compaction-tracking state.
          3. Rehydrates from the new session's persisted entries (compaction-
             snapshot aware). An unknown session id yields an empty timeline.

        This is the internal primitive; ``reload_timeline`` is not exposed on the
        query methods. Prefer the public ``resume(session_id)`` wrapper for the
        reload path (``TaskResource`` calls it per sub-agent spawn). Note:
        skipping the rehydrate alone is not enough — the rebuild in step 2 would
        still discard the caller's timeline — so the flag gates the whole reload.

        Re-setting the current id is a no-op (no repository hit). Ordering is
        load-bearing: ``_session_id`` is assigned before ``rehydrate()`` because
        ``read_since`` reads the session id off the agent.

        Args:
            session_id: The session id to switch to.
            reload_timeline: When True, rebuild + rehydrate the timeline from
                the new session. When False (default), keep the current
                timeline and only relabel.
        """
        if session_id == self._session_id:
            return

        if not reload_timeline:
            # Pure relabel — caller owns the timeline; do not flush/rebuild.
            self._session_id = session_id
            self._invalidate_system_prompt_cache()
            return

        timeline = getattr(self, "_timeline", None)
        if timeline is not None and getattr(timeline, "_repository", None) is not None and timeline.timeline:
            timeline.save(self._session_id)

        self._session_id = session_id
        self._invalidate_system_prompt_cache()
        self._timeline = self._build_timeline()
        self._timeline.rehydrate()

    def _invalidate_system_prompt_cache(self) -> None:
        runtime = getattr(self, "_runtime", None)
        if runtime is not None and hasattr(runtime, "invalidate_system_prompt_cache"):
            runtime.invalidate_system_prompt_cache()

    def resume(self, session_id: str) -> None:
        """Resume a persisted session by id, reloading its timeline from disk.

        The single public entry point for a session reload. Thin wrapper over
        ``set_session_id(session_id, reload_timeline=True)``: flushes the
        current session, rebuilds the timeline (resetting all session-scoped
        and compaction-tracking state), then rehydrates from ``session_id``'s
        persisted entries. An unknown id yields an empty timeline.

        Mutates instance state (timeline + session id) — must NOT be interleaved
        with an in-flight query on a shared agent. ``TaskResource`` calls this on
        a freshly built per-spawn instance, so isolation is guaranteed there.

        Fork semantics: ``resume(A)`` followed by ``aquery(session_id=B)`` reads
        from A and writes to B — A's history is branched into B and A on disk is
        left untouched. Resuming and continuing in place means omitting
        ``session_id`` on ``aquery`` (or passing the same id).

        See ``resume_from_timeline`` to adopt an in-memory ``Timeline`` object
        instead of loading by id.

        Args:
            session_id: The persisted session to load and continue.
        """
        self.set_session_id(session_id, reload_timeline=True)

    def resume_from_timeline(self, timeline: Timeline, session_id: str | None = None) -> None:
        """
        Resume the STAR loop from a previously saved Timeline.

        Loads the given timeline as the agent's current timeline and derives
        ``_star_loop_count`` from its entries so the next query continues from
        where the previous session left off.

        Args:
            timeline:   A Timeline instance loaded from disk (or any source).
            session_id: Optional session identifier to adopt.  If None, the
                        agent's current session id is kept.
        """
        self._timeline = timeline

        if session_id is not None:
            self._session_id = session_id

        self._star_loop_count = timeline.count_iterations()

    def register_reminder(self, reminder) -> None:
        """
        Register a custom reminder for system-reminder injection.

        Reminders are evaluated during each STAR loop iteration and injected
        into the LLM prompt when their trigger conditions are met.

        Args:
            reminder: Any object matching Reminder protocol.
                Must have: name attribute and evaluate(agent, messages) method.

        Example:
            >>> class MyReminder:
            ...     name = "domain_context"
            ...
            ...     def evaluate(self, agent, messages) -> None:
            ...         if getattr(agent, "_star_loop_count", 0) == 1:
            ...             messages.append(LLMMessage(
            ...                 role="user",
            ...                 content="<system-reminder>\\nFinancial analysis context.\\n</system-reminder>"
            ...             ))
            >>>
            >>> agent.register_reminder(MyReminder())
        """
        if self._reminder_manager is not None:
            self._reminder_manager.register(reminder)

    @property
    def llm_client(self) -> LLM:
        """Get the LLM client."""
        if self._llm_client is None:
            self._llm_client = LLM(provider=self._llm_config["provider"], model=self._llm_config["model"])
        return self._llm_client

    @llm_client.setter
    def llm_client(self, value: LLM):
        """Set the LLM client."""
        self._llm_client = value
        if hasattr(self._runtime, "set_llm"):
            self._runtime.set_llm(value)

    # ============================================================================
    # PUBLIC API - AGENT IDENTITY & PROMPTS
    # ============================================================================

    def with_agents(self, *agents: AgentProtocol) -> BaseSTARAgent:
        """Add agents to the agent."""
        if hasattr(self._runtime, "reset"):
            self._runtime.reset()
        super().with_agents(*agents)
        return self

    def with_resources(self, *resources: ResourceProtocol) -> BaseSTARAgent:
        """Add resources to the agent."""
        if hasattr(self._runtime, "reset"):
            self._runtime.reset()
        super().with_resources(*resources)
        return self

    def with_workflows(self, *workflows: WorkflowProtocol) -> BaseSTARAgent:
        """Add workflows to the agent."""
        if hasattr(self._runtime, "reset"):
            self._runtime.reset()
        super().with_workflows(*workflows)
        return self

    def with_notifiable(self, *notifiables: Notifiable) -> BaseSTARAgent:
        """Add notifiables to the agent."""
        for agent in self._agents:
            agent.with_notifiable(*notifiables)
        for resource in self._resources:
            resource.with_notifiable(*notifiables)
        for workflow in self._workflows:
            workflow.with_notifiable(*notifiables)
        super().with_notifiable(*notifiables)
        return self

    @property
    def public_description(self) -> str:
        """Get the public description of the agent."""
        if hasattr(self._runtime, "public_description"):
            return self._runtime.public_description(self)
        return inspect.getdoc(self.__class__) or f"{self.agent_type} agent."

    @property
    def private_identity(self) -> str:
        """Get the private identity of the agent."""
        if hasattr(self._runtime, "private_identity"):
            return self._runtime.private_identity(self)
        return super().private_identity

    @property
    def system_prompt(self) -> str:
        """Get the system prompt of the agent."""
        if hasattr(self._runtime, "system_prompt"):
            return self._runtime.system_prompt(self)
        return super().system_prompt

    # ============================================================================
    # PUBLIC API - STATE & CONTEXT MANAGEMENT
    # ============================================================================

    def get_state(self) -> dict[str, Any]:
        """Get current agent state as dictionary."""
        return self._state.get_state()

    # ============================================================================
    # PUBLIC API - TIMELINE & CONVERSATION
    # ============================================================================

    def get_timeline_summary(self) -> str:
        """Get a summary of the agent's timeline."""
        return self._timeline.get_timeline_summary()

    def query(self, **kwargs) -> DictParams:
        # Generate session_id if not provided
        new_session_id = kwargs.get("session_id")
        if new_session_id is not None:
            self.set_session_id(new_session_id)
        session_id = self._session_id

        # Reset STAR loop counter for new query
        self._star_loop_count = 0

        # Set session_id for EventLog if it exists
        if hasattr(self, "_event_log") and self._event_log is not None:
            self._event_log._current_session_id = session_id

        try:
            result = super().query(**kwargs)
            return result
        finally:
            # Save events if EventLog exists
            if hasattr(self, "_event_log") and self._event_log is not None:
                self._event_log.save(session_id)

            # Save timeline (agent, codec, storage_config already set in __init__)
            if hasattr(self, "_timeline") and self._timeline is not None:
                self._timeline.save(session_id)

    async def aquery(self, **kwargs) -> DictParams:
        # session_id relabels the in-memory session / write target (see
        # set_session_id). To reload a persisted session from disk first, call
        # resume(session_id) before aquery() — relabel never reloads.
        new_session_id = kwargs.get("session_id")
        if new_session_id is not None:
            self.set_session_id(new_session_id)
        session_id = self._session_id

        # Reset STAR loop counter for new query
        self._star_loop_count = 0

        # Set session_id for EventLog if it exists
        if hasattr(self, "_event_log") and self._event_log is not None:
            self._event_log._current_session_id = session_id

        try:
            result = await super().aquery(**kwargs)
            return result
        finally:
            # Save events if EventLog exists
            if hasattr(self, "_event_log") and self._event_log is not None:
                self._event_log.save(session_id)

            # Save timeline (agent, codec, storage_config already set in __init__)
            if hasattr(self, "_timeline") and self._timeline is not None:
                self._timeline.save(session_id)

    def converse(self, initial_message: str | None = None, session_id: str | None = None) -> None:
        """Interactive conversation loop with a human user.

        Args:
            initial_message: Optional initial message to start the conversation
            session_id: Optional session identifier. If None, generates UUID.
        """
        self._communicator.converse(initial_message=initial_message, session_id=session_id)

    async def aconverse(
        self,
        initial_message: str | None = None,
        session_id: str | None = None,
        input_handler: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        """Async interactive conversation loop with pluggable input handler.

        The in-memory timeline is kept across turns (each turn relabels, never
        reloads). To continue a persisted conversation, call ``resume(session_id)``
        before ``aconverse``.

        Args:
            initial_message: Optional initial message to start the conversation
            session_id: Optional session identifier. If None, generates UUID.
            input_handler: Async callable that returns user input string.
                          If None, uses default blocking input() wrapped in executor.
        """
        await self._communicator.aconverse(
            initial_message=initial_message,
            session_id=session_id,
            input_handler=input_handler,
        )

    def __getattr__(self, name: str):
        """
        Magic function: Convert unknown method calls to natural language and call converse.

        Examples:
            agent.hi_how_are_you() -> converse("hi how are you")
            agent.research_coffee_companies() -> converse("research coffee companies")
            agent.find_exporters_in_dak_lak() -> converse("find exporters in dak lak")
        """

        def magic_method(*args, **kwargs):
            # Convert method name to natural language
            # Replace underscores with spaces and clean up
            natural_language = name.replace("_", " ").strip()

            # Add any positional arguments as additional context
            if args:
                args_str = " ".join(str(arg) for arg in args)
                natural_language += f" {args_str}"

            # Add any keyword arguments as additional context
            if kwargs:
                kwargs_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
                natural_language += f" {kwargs_str}"

            # Call converse with the natural language message (starts interactive conversation)
            return self.converse(initial_message=natural_language)

        return magic_method

    # ============================================================================
    # TIMELINE COMPRESSION
    # ============================================================================

    def _estimate_system_prompt_tokens(self) -> int:
        """Return char/4 estimate of system prompt size for compression trigger."""
        try:
            prompt = self.system_prompt
        except Exception:
            return 0
        if not prompt:
            return 0
        return len(str(prompt)) // 4

    def _estimate_tools_tokens(self) -> int:
        """Return char/4 estimate of tools-schema size for compression trigger."""
        try:
            tools = None
            runtime = getattr(self, "_runtime", None)
            if runtime is not None and hasattr(runtime, "get_tools"):
                tools = runtime.get_tools(self)
            if not tools:
                return 0
            import json as _json

            return len(_json.dumps(tools, default=str)) // 4
        except Exception:
            return 0

    def _maybe_compress_timeline(self, timeline: Timeline) -> None:
        """
        Compress timeline if it exceeds the configured threshold.

        Uses the LLM to summarize old entries, preserving recent context.
        This prevents unbounded context growth in long conversations.
        """
        if not timeline.needs_compression():
            return

        compression_prompt = timeline.build_compression_prompt()
        if not compression_prompt:
            return

        try:
            # Use a quick LLM call to summarize old entries
            from dana.common.llm.types import LLMMessage

            summary_response = self.llm_client.chat_response_sync(
                [LLMMessage(role="user", content=compression_prompt)],
                agent_id=self.object_id,
                agent_type=self.agent_type,
                temperature=0,
            )

            raw_content = summary_response.content.strip() if summary_response.content else ""

            # Parse JSON response to extract summary
            summary = self._extract_compression_summary(raw_content)

            if summary:
                compressed_count = timeline.compress_old_entries(summary)
                logger.info(
                    "Timeline compressed",
                    agent_id=self.object_id,
                    compressed_entries=compressed_count,
                    summary_length=len(summary),
                )
        except Exception as e:
            # PTL must propagate so the caller-layer (llm_caller.py) can trigger
            # reactive_compact + retry — don't let this summary path swallow it.
            from dana.common.llm.types import PromptTooLongError

            if isinstance(e, PromptTooLongError):
                raise
            # Don't fail the main operation if compression fails
            logger.warning("Timeline compression failed", error=str(e))

    async def _maybe_compress_timeline_async(self, timeline: Timeline) -> None:
        """Async version of _maybe_compress_timeline."""
        if not timeline.needs_compression():
            return

        compression_prompt = timeline.build_compression_prompt()
        if not compression_prompt:
            return

        try:
            from dana.common.llm.types import LLMMessage

            summary_response = await self.llm_client.chat_response(
                [LLMMessage(role="user", content=compression_prompt)],
                agent_id=self.object_id,
                agent_type=self.agent_type,
                temperature=0,
            )

            raw_content = summary_response.content.strip() if summary_response.content else ""

            # Parse JSON response to extract summary
            summary = self._extract_compression_summary(raw_content)

            if summary:
                compressed_count = timeline.compress_old_entries(summary)
                logger.info(
                    "Timeline compressed",
                    agent_id=self.object_id,
                    compressed_entries=compressed_count,
                    summary_length=len(summary),
                )
        except Exception as e:
            from dana.common.llm.types import PromptTooLongError

            if isinstance(e, PromptTooLongError):
                raise
            logger.warning("Timeline compression failed", error=str(e))

    def _extract_compression_summary(self, content: str) -> str:
        """Extract summary from JSON compression response.

        Args:
            content: Raw LLM response (expected JSON with 'summary' field)

        Returns:
            Extracted summary text, or the raw content as fallback
        """
        import json

        if not content:
            return ""

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "summary" in parsed:
                return parsed["summary"]
        except json.JSONDecodeError:
            pass

        # Fallback: return raw content if not valid JSON
        return content

    # ============================================================================
    # STAR PATTERN IMPLEMENTATION (BaseSTARAgent abstract methods)
    # ============================================================================

    @observable
    def _see(self, trace_inputs: DictParams) -> DictParams:
        """
        SEE: See the user/caller inputs and produce percepts.

        Args:
            trace_inputs (DictParams): any new user/agent inputs, plus trace_outputs from the previous loop (if any)
              - caller_message (str): Caller message (may be user or another agent)
              - caller_type (str): Type of caller (agent or human)
              - caller_id (str): ID of the caller (agent.object_id or user) for conversation tracking.
              - response (str): Response from the previous loop (if any)
              - tool_calls (list[DictParams]): Tool calls from the previous loop (if any)
              - tool_results (list[DictParams]): Tool results from the previous loop (if any)

        Returns:
            - trace_percepts (DictParams): the percepts produced by this SEE phase.
              - timeline (Timeline): Timeline of the agent, appending any new entries from our perceptions
              - caller_message (str): Caller message (may be user or another agent)
              - caller_type (str): Type of caller (agent or human)
              - caller_id (str): ID of the caller (agent.object_id or user) for conversation tracking.
        """

        # Increment STAR loop counter at the start of each iteration
        self._star_loop_count += 1

        # Input parameter checking
        trace_inputs = trace_inputs or {}
        if self._do_exit_star_loop(trace_inputs):
            return {"trace_percepts": self._mark_star_loop_exit(trace_inputs)}

        previous_tool_calls: list[DictParams] = trace_inputs.get("tool_calls", None)
        if previous_tool_calls:
            # This is a subsequent loop - perceiving tool results
            tool_results = trace_inputs.get("tool_results", [])
            num_results = len(tool_results) if isinstance(tool_results, list) else 0

            # Add perception message for notification visibility
            trace_inputs["perception"] = f"Perceived {num_results} tool result(s)"

            del trace_inputs["response"]
            del trace_inputs["tool_calls"]
            del trace_inputs["tool_results"]
        else:
            # This is the first loop
            caller_message: str = trace_inputs.get("caller_message", trace_inputs.get("message", None))
            if not caller_message:
                return {"trace_percepts": self._mark_star_loop_exit(trace_inputs)}

            # Add caller_message to timeline with caller tracking
            if isinstance(caller_message, str):
                # Create new entry and mark it as latest
                new_entry = TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content=caller_message, is_latest_user_message=True)
                self._timeline.add_entry(new_entry)

            # Preserve caller_message for notifications but remove original keys
            trace_inputs.pop("message", None)  # Remove 'message' alias
            # Keep caller_message in trace_inputs for notification
            if "caller_message" not in trace_inputs:
                trace_inputs["caller_message"] = caller_message

        trace_inputs |= {"timeline": self._timeline}

        return super()._see(trace_inputs)

    def _format_tool_call_as_xml(self, tool_call: DictParams) -> str:
        """Format a tool call dict as XML for consistent LLM context.

        This ensures the LLM sees tool calls in the same XML format it should
        produce, rather than Python dict format which it might mimic.
        """
        func = tool_call.get("function", "unknown:unknown")
        args = tool_call.get("arguments", {})

        # Build XML parameters
        params_xml = ""
        for key, value in args.items():
            params_xml += f'<parameter name="{key}">{value}</parameter>'

        return f'<function_call><invoke name="{func}">{params_xml}</invoke></function_call>'

    # ============================================================================
    # SHARED HELPERS (used by both sync and async STAR methods)
    # ============================================================================

    def _provider_fingerprint(self) -> str | None:
        """Active provider's replay fingerprint, or None if unavailable.

        Used to gate cross-turn reasoning replay — items captured from provider X
        are only replayed when the same X handles the next call. Defensive lookup
        so non-OpenAI providers without the property don't break agent flow.
        """
        try:
            client = self._llm_client
            if client is None:
                return None
            provider = getattr(client, "provider", None)
            return getattr(provider, "fingerprint", None) if provider is not None else None
        except Exception:
            return None

    def _build_thinking_metadata(self, reasoning_items: list[dict] | None, response_id: str | None) -> dict:
        """Metadata payload attached to AGENT_THOUGHTS entries for replay.

        Empty dict when there's nothing to replay — keeps existing entries clean.
        """
        if not reasoning_items:
            return {}
        return {
            "reasoning_items": reasoning_items,
            "fingerprint": self._provider_fingerprint(),
            "response_id": response_id,
        }

    def _record_think_results(
        self,
        timeline: Timeline,
        trace_percepts: DictParams,
        response: str | None,
        reasoning: str | None,
        tool_calls: list,
        done: bool | None,
        todo_list: list | None,
        output_state: str,
        reasoning_items: list[dict] | None = None,
        response_id: str | None = None,
    ) -> DictParams:
        """Record think results to timeline and build output trace.

        Shared by _think() and _think_async() to eliminate duplication.
        Handles: retry fallback, timeline entry recording, output assembly.
        """
        if output_state == "retry":
            response = "No response generated"
            tool_calls = []
            done = True
            output_state = "exit"

        thinking_metadata = self._build_thinking_metadata(reasoning_items, response_id)

        if not tool_calls or len(tool_calls) == 0:
            # Persist reasoning even on direct-answer turns. Without this, the
            # model's internal reasoning (LLMResponse.reasoning_content for
            # gpt-5/o3/o4, or <thinking> tags / JSON reasoning fields for other
            # codecs) is silently dropped whenever the agent answers without
            # invoking a tool. Same emit pattern as the tool-calls branch below.
            # Gate on reasoning_items, not summary text: GPT-5/o3/o4 low-summary
            # turns return a reasoning item (rs_… + encrypted_content) with an
            # empty summary. Skipping the entry there drops the encrypted item
            # and the turn replays without reasoning state.
            if (reasoning and len(reasoning) > 0) or thinking_metadata.get("reasoning_items"):
                timeline.add_entry(
                    TimelineEntry(
                        entry_type=TimelineEntryType.AGENT_THOUGHTS,
                        content=reasoning or "",
                        metadata=dict(thinking_metadata),
                    )
                )
            response = response if (response and len(response) > 0) else "No response generated"
            timeline.add_entry(
                TimelineEntry(
                    entry_type=TimelineEntryType.AGENT_RESPONSE,
                    content=response,
                )
            )
        else:
            # See gate rationale above — reasoning_items must persist even when
            # the summary text is empty, or cross-turn replay loses the item.
            if (reasoning and len(reasoning) > 0) or thinking_metadata.get("reasoning_items"):
                timeline.add_entry(
                    TimelineEntry(
                        entry_type=TimelineEntryType.AGENT_THOUGHTS,
                        content=reasoning or "",
                        metadata=dict(thinking_metadata),
                    )
                )

            if response and len(response) > 0:
                timeline.add_entry(
                    TimelineEntry(
                        entry_type=TimelineEntryType.AGENT_THOUGHTS,
                        content=response,
                    )
                )

            # Add todo_list BEFORE tool calls so it doesn't break the tool_call → tool_result sequence
            # (OpenAI requires tool results immediately after tool calls)
            if todo_list:
                in_progress = [t for t in todo_list if t.status == "in_progress"]
                pending = [t for t in todo_list if t.status == "pending"]
                completed = [t for t in todo_list if t.status == "completed"]
                logger.info(
                    "Todo list updated",
                    in_progress=len(in_progress),
                    pending=len(pending),
                    completed=len(completed),
                )
                todo_lines = []
                for item in todo_list:
                    status_marker = {"in_progress": "[IN PROGRESS]", "pending": "[PENDING]", "completed": "[COMPLETED]"}.get(
                        item.status, "[?]"
                    )
                    todo_lines.append(f"{status_marker} {item.content}")
                # Remove any existing TODO_LIST entries to avoid accumulation
                timeline.timeline = [e for e in timeline.timeline if e.entry_type != TimelineEntryType.TODO_LIST]
                timeline.add_entry(
                    TimelineEntry(
                        entry_type=TimelineEntryType.TODO_LIST,
                        content="\n".join(todo_lines),
                    )
                )

            # Check if these are native tool calls (have tool_call_id)
            has_native_tool_calls = any(tc.get("tool_call_id") for tc in tool_calls)

            if has_native_tool_calls:
                timeline.add_entry(
                    TimelineEntry(
                        entry_type=TimelineEntryType.TOOL_CALL,
                        content="",
                        tool_calls=tool_calls,
                    )
                )
            else:
                # Legacy XML format - store each tool call separately
                for tool_call in tool_calls:
                    timeline.add_entry(
                        TimelineEntry(
                            entry_type=TimelineEntryType.TOOL_CALL,
                            content=self._format_tool_call_as_xml(tool_call),
                        )
                    )

        # Build output trace
        response = response or ""
        assert isinstance(response, str)
        assert isinstance(tool_calls, list)
        trace_percepts |= {
            "response": response,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "done": done,
            "todo_list": todo_list,
        }

        if output_state == "exit":
            trace_percepts = self._mark_star_loop_exit(trace_percepts)

        result = {"trace_thoughts": trace_percepts}
        self.broadcast(result)
        return result

    def _record_tool_results(self, tool_results: list) -> None:
        """Record tool execution results to timeline.

        Shared by _act() and _act_async() to eliminate duplication.
        Handles: entry type detection, content extraction, deferred user injections.
        """
        deferred_injections = []
        for tool_result in tool_results:
            if isinstance(tool_result, dict):
                # Determine entry type based on tool type
                tool_type = tool_result.get("type")
                if tool_type == "agent":
                    entry_type = TimelineEntryType.SUB_AGENT_RESPONSE
                elif tool_type == "resource":
                    entry_type = TimelineEntryType.RESOURCE_RESULT
                elif tool_type == "workflow":
                    entry_type = TimelineEntryType.WORKFLOW_RESULT
                else:
                    entry_type = TimelineEntryType.UNKNOWN_TOOL_CALL

                result_content = tool_result.get("result", "Unknown tool result")

                # Extract inject_as_user before serializing (deferred to avoid
                # breaking OpenAI's tool_calls → tool results message ordering)
                if isinstance(result_content, dict):
                    inject_content = result_content.pop("inject_as_user", None)
                    if inject_content:
                        deferred_injections.append(inject_content)
                    if "message" in result_content:
                        result_content = result_content["message"]

                if not isinstance(result_content, str):
                    import json

                    result_content = json.dumps(result_content)

                # CRITICAL-2: dump oversized content to a session-scoped file
                # and leave a compact marker in the timeline. Prevents the
                # "huge recent tool_result is unreclaimable" wedge where
                # reactive_compact can't shed the offending entry because it's
                # within the keep-recent window.
                from dana.core.agent.tool_result_dump import maybe_dump_oversized_content, resolve_session_folder_for_agent

                result_content = maybe_dump_oversized_content(
                    result_content,
                    tool_result.get("tool_call_id"),
                    resolve_session_folder_for_agent(self),
                )

                self._timeline.add_entry(
                    TimelineEntry(
                        entry_type=entry_type,
                        content=result_content,
                        tool_call_id=tool_result.get("tool_call_id"),
                    )
                )

        # Inject deferred user messages AFTER all tool results are in timeline
        for content in deferred_injections:
            self._timeline.add_entry(
                TimelineEntry(
                    entry_type=TimelineEntryType.USER_MESSAGE,
                    content=content,
                )
            )

    def _maybe_add_multistep_reminder(self) -> None:
        """Add multi-step XML format reminder for codec-based runtimes.

        Only applies to XML-based codec runtimes (not native tool calling).
        Estimates remaining steps from user request and adds a continuation prompt.
        """
        from dana.core.runtime.codec.codec_base import CodecRuntimeBase

        if not isinstance(self._runtime, CodecRuntimeBase) or self._runtime._use_native_tools:
            return

        # Find original user request (skip multimodal content blocks)
        original_request = ""
        for entry in self._timeline.timeline:
            if entry.entry_type == TimelineEntryType.USER_MESSAGE and isinstance(entry.content, str):
                original_request = entry.content
                break

        tool_call_count = sum(1 for entry in self._timeline.timeline if entry.entry_type == TimelineEntryType.TOOL_CALL)

        # Estimate expected steps from sequential action patterns
        import re

        request_lower = original_request.lower()
        step_separators = re.findall(r"\bthen\b|\bafter that\b|\bnext\b|\bfinally\b", request_lower)
        expected_steps = min(1 + len(step_separators), 5)

        if tool_call_count < expected_steps:
            remaining = expected_steps - tool_call_count
            self._timeline.add_entry(
                TimelineEntry(
                    entry_type=TimelineEntryType.AGENT_THOUGHTS,
                    content=f"[MULTI-STEP TASK - {remaining} STEP(S) REMAINING] "
                    f"Completed {tool_call_count}/{expected_steps} steps. "
                    "I MUST call the next tool using this EXACT XML format: "
                    '<function_call><invoke name="resource-id:method"><parameter name="param">value</parameter></invoke></function_call>',
                )
            )

    @observable
    def _think(self, trace_percepts: DictParams) -> DictParams:
        """THINK: Think about the percepts and produce thoughts via sync LLM call."""
        trace_percepts = trace_percepts or {}
        if self._do_exit_star_loop(trace_percepts) or not trace_percepts:
            return {"trace_thoughts": self._mark_star_loop_exit(trace_percepts)}

        timeline: Timeline = trace_percepts.get("timeline", self._timeline)
        trace_percepts.pop("timeline", None)

        self._maybe_compress_timeline(timeline)

        # Factory used by LLMCaller's PTL retry loop to rebuild messages after
        # each reactive_compact so the retry observes the compacted timeline
        # (CRITICAL-1 fix). Closes over self + timeline, not the messages list.
        def _rebuild_llm_messages() -> list[LLMMessage]:
            return self._runtime.build_prompt(self, timeline)

        llm_messages = _rebuild_llm_messages()

        response, reasoning, tool_calls, done, todo_list = None, None, [], None, None
        reasoning_items: list[dict] | None = None
        response_id: str | None = None
        output_state = "retry"
        for attempt in range(self.MAX_THINK_RETRIES):
            raw = self._runtime.call_llm(llm_messages, messages_fn=_rebuild_llm_messages)
            parsed = self._runtime.parse_response(raw)
            response, reasoning, tool_calls, done, todo_list = (
                parsed.response,
                parsed.reasoning,
                parsed.tool_calls,
                parsed.done,
                parsed.todo_list,
            )
            reasoning_items = parsed.reasoning_items
            response_id = parsed.response_id

            has_tool_calls = bool(tool_calls)
            has_response = bool(response and response.strip())
            output_state = self._runtime.validate_done_output(done, has_tool_calls, has_response)

            if output_state != "retry":
                break

            if attempt < self.MAX_THINK_RETRIES - 1:
                llm_messages.append(self._runtime.build_output_format_correction())
                logger.warning(
                    "Invalid output format, retrying",
                    attempt=attempt + 1,
                    done=done,
                    has_tool_calls=has_tool_calls,
                    has_response=has_response,
                )

        return self._record_think_results(
            timeline,
            trace_percepts,
            response,
            reasoning,
            tool_calls,
            done,
            todo_list,
            output_state,
            reasoning_items=reasoning_items,
            response_id=response_id,
        )

    @observable
    def _act(self, trace_thoughts: DictParams) -> DictParams:
        """ACT: Execute tool calls and return results (sync)."""
        trace_thoughts = trace_thoughts or {}
        if not trace_thoughts or self._do_exit_star_loop(trace_thoughts):
            return {"trace_outputs": self._mark_star_loop_exit(trace_thoughts)}

        tool_calls: list[DictParams] = trace_thoughts.get("tool_calls")
        tool_results = self._runtime.execute_tools(self, tool_calls)

        if isinstance(tool_results, list):
            self._record_tool_results(tool_results)

            # Multi-step XML reminder for codec-based runtimes only
            # (async path uses native tool calling, so this is sync-only)
            self._maybe_add_multistep_reminder()

        assert isinstance(tool_results, list)
        trace_thoughts |= {"tool_results": tool_results}

        return super()._act(trace_thoughts)

    # @observable
    def _reflect(self, trace_outputs: DictParams) -> DictParams:
        """
        REFLECT: Reflect on the actions or episode, depending on the reflection phase.

        Args:
            trace_outputs (DictParams): the outputs produced by this ACT phase.
              - phase (LearningPhase): specifies which learning phase we are in
              - response (str): Response from the THINK phase.
              - tool_calls (list[DictParams]): Tool calls from the THINK phase.
              - tool_results (list[DictParams]): Tool results from the ACT phase.
              - caller_message (str): Caller message (may be user or another agent)
              - caller_type (str): Type of caller (agent or human)
              - caller_id (str): ID of the caller (agent.object_id or user) for conversation tracking.

        Returns:
            - trace_learning (DictParams): the learning produced by this REFLECT phase.
        """

        # Input parameter checking
        trace_outputs = trace_outputs or {}
        if not trace_outputs or self._do_exit_star_loop(trace_outputs):
            return {"trace_learning": self._mark_star_loop_exit(trace_outputs)}
        phase: LearningPhase = trace_outputs.get("phase") or LearningPhase.ACQUISITIVE

        trace_learning = {}
        if self._learner is not None:
            match phase:
                case LearningPhase.ACQUISITIVE:
                    trace_learning |= self._learner._reflect_acquisitive(trace_outputs)
                    trace_learning["learning_note"] = "Initial learning and trial-level plasticity"

                case LearningPhase.EPISODIC:
                    trace_learning |= self._learner._reflect_episodic(trace_outputs)
                    trace_learning["learning_note"] = "Episodic binding of information"

                case LearningPhase.INTEGRATIVE:
                    trace_learning |= self._learner._reflect_integrative(trace_outputs)
                    trace_learning["learning_note"] = "Offline replay and integration"

                case LearningPhase.RETENTIVE:
                    trace_learning |= self._learner._reflect_retentive(trace_outputs)
                    trace_learning["learning_note"] = "Long-term maintenance and habit formation"

                case _:
                    raise ValueError(f"Unknown learning phase {phase}")

            trace_learning |= {
                "timestamp": datetime.now().isoformat(),
                "phase": phase.value,
            }

            # Add to timeline for persistence
            self._timeline.add_entry(
                TimelineEntry(
                    entry_type=TimelineEntryType.AGENT_LEARNING,
                    content=f"Learning ({phase.value}): {trace_learning.get('learning_note', 'No learning note')}",
                )
            )

        return super()._reflect(trace_learning)

    # ============================================================================
    # ASYNC STAR METHODS
    # ============================================================================

    @observable
    async def _think_async(self, trace_percepts: DictParams) -> DictParams:
        """THINK (async): Async version of _think with native async LLM calls."""
        trace_percepts = trace_percepts or {}
        if self._do_exit_star_loop(trace_percepts) or not trace_percepts:
            return {"trace_thoughts": self._mark_star_loop_exit(trace_percepts)}

        timeline: Timeline = trace_percepts.get("timeline", self._timeline)
        trace_percepts.pop("timeline", None)

        await self._maybe_compress_timeline_async(timeline)

        # Factory used by LLMCaller's PTL retry loop to rebuild messages after
        # each reactive_compact so the retry observes the compacted timeline
        # (CRITICAL-1 fix).
        def _rebuild_llm_messages_async() -> list[LLMMessage]:
            return self._runtime.build_prompt(self, timeline)

        llm_messages = _rebuild_llm_messages_async()

        response, reasoning, tool_calls, done, todo_list = None, None, [], None, None
        reasoning_items: list[dict] | None = None
        response_id: str | None = None
        output_state = "retry"
        for attempt in range(self.MAX_THINK_RETRIES):
            if hasattr(self._runtime, "call_llm_async"):
                raw = await self._runtime.call_llm_async(llm_messages, messages_fn=_rebuild_llm_messages_async)
            else:
                import asyncio

                raw = await asyncio.to_thread(
                    self._runtime.call_llm,
                    llm_messages,
                    messages_fn=_rebuild_llm_messages_async,
                )
            parsed = self._runtime.parse_response(raw)
            response, reasoning, tool_calls, done, todo_list = (
                parsed.response,
                parsed.reasoning,
                parsed.tool_calls,
                parsed.done,
                parsed.todo_list,
            )
            reasoning_items = parsed.reasoning_items
            response_id = parsed.response_id

            has_tool_calls = bool(tool_calls)
            has_response = bool(response and response.strip())
            output_state = self._runtime.validate_done_output(done, has_tool_calls, has_response)

            if output_state != "retry":
                break

            if attempt < self.MAX_THINK_RETRIES - 1:
                llm_messages.append(self._runtime.build_output_format_correction())
                logger.warning(
                    "Invalid output format, retrying",
                    attempt=attempt + 1,
                    done=done,
                    has_tool_calls=has_tool_calls,
                    has_response=has_response,
                )

        return self._record_think_results(
            timeline,
            trace_percepts,
            response,
            reasoning,
            tool_calls,
            done,
            todo_list,
            output_state,
            reasoning_items=reasoning_items,
            response_id=response_id,
        )

    @observable
    async def _act_async(self, trace_thoughts: DictParams) -> DictParams:
        """ACT (async): Async version of _act with native async tool execution."""
        trace_thoughts = trace_thoughts or {}
        if not trace_thoughts or self._do_exit_star_loop(trace_thoughts):
            return {"trace_outputs": self._mark_star_loop_exit(trace_thoughts)}

        tool_calls: list[DictParams] = trace_thoughts.get("tool_calls")

        if hasattr(self._runtime, "execute_tools_async"):
            tool_results = await self._runtime.execute_tools_async(self, tool_calls)
        else:
            import asyncio

            tool_results = await asyncio.to_thread(self._runtime.execute_tools, self, tool_calls)

        if isinstance(tool_results, list):
            self._record_tool_results(tool_results)

        assert isinstance(tool_results, list)
        trace_thoughts |= {"tool_results": tool_results}

        result = {"trace_outputs": trace_thoughts}
        self.broadcast(result)
        return result

    # ============================================================================
    # DISCOVERY INTERFACE (Override from BaseSTARAgent)
    # ============================================================================

    @property
    def _registry_available_agents(self) -> Sequence[AgentProtocol]:
        """List available agents (excluding self)."""
        if self._registry:
            all_agents = self._registry.list_agents()
            # Exclude self
            return [agent for agent in all_agents if agent.object_id != self.object_id]
        else:
            return []
