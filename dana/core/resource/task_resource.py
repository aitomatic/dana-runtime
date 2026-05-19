"""TaskResource for dispatching tasks to sub-agents with dynamic tool descriptions.

Agents are registered as **factories** — callables that construct a fresh
agent per ``task()`` invocation. Each spawn gets a disjoint object graph
(``_timeline``, ``_star_loop_count``, ``_session_id``, EventLog cursor), which
eliminates both the concurrent-same-type state corruption and the sequential
timeline-accumulation bug that a single shared instance suffered.

A legacy instance registration is still accepted (wrapped as a constant
factory) but is unsafe under concurrency and emits a ``DeprecationWarning``.
"""

from collections.abc import Callable
from typing import Any
import uuid
import warnings

from dana.common.protocols import Notifiable
from dana.common.protocols.war import named_tool
from dana.core.resource.base_resource import BaseResource


class TaskResource(BaseResource):
    """Resource for launching tasks to sub-agents.

    The TaskResource dispatches tasks to specialized sub-agents, with tool
    descriptions dynamically generated from the registered agent factories.
    """

    def __init__(self, resource_id: str, agents: dict[str, Any] | None = None, **kwargs):
        """Initialize the TaskResource.

        Args:
            resource_id: Unique identifier for this resource instance.
            agents: Mapping of agent type name -> agent factory (preferred) or
                agent instance (legacy, deprecated). Factories must satisfy the
                contract documented on ``register_agent``.
            **kwargs: Additional arguments passed to the base resource.
        """
        super().__init__(resource_id=resource_id, **kwargs)
        # Factory per agent type; constructs a fresh agent per task() call.
        self._agents: dict[str, Callable[[], Any]] = {}
        # TASK_TOOL_DESCRIPTION cached per type — read off the agent class so
        # no live instance is needed to render the tool docstring.
        self._descriptions: dict[str, str] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        # Notifiables (inherited self._notifiables list) are applied to each
        # agent in task() right after the factory spawns it — agents do not
        # exist until a task() call constructs one.

        for name, agent_or_factory in (agents or {}).items():
            self.register_agent(name, agent_or_factory)
        self._update_task_docstring()

    def with_notifiable(self, *notifiables: Notifiable) -> "TaskResource":
        """Record notifiables; applied to every agent spawned by ``task()``.

        Also forwarded to any already-live session agents so in-flight or
        resumable sub-agents stay observable.
        """
        for session in self._sessions.values():
            agent = session.get("agent")
            if agent is not None and hasattr(agent, "with_notifiable"):
                agent.with_notifiable(*notifiables)
        super().with_notifiable(*notifiables)
        return self

    def register_agent(self, name: str, agent_or_factory: Any) -> None:
        """Register an agent factory for task dispatch.

        Factory contract (preferred path):
            1. MUST be a callable that constructs a fresh agent with no
               arguments — e.g. ``functools.partial(AgentCls, ...)``.
            2. MUST expose its target class via ``.func`` so the agent's
               ``TASK_TOOL_DESCRIPTION`` is reachable without instantiation.
               ``functools.partial`` satisfies this; a bare ``lambda`` does NOT.
            3. SHOULD pin a stable ``agent_id`` — timeline storage is keyed by
               it; an unpinned id breaks session resume.
            4. SHOULD pass ``auto_register=False`` — per-spawn agents must not
               pollute the global registry.

        Legacy path: passing a ``BaseAgent`` instance is still accepted but
        deprecated. A shared instance corrupts state across concurrent and
        sequential ``Task`` calls; it is wrapped as a constant factory and a
        ``DeprecationWarning`` is emitted.

        Args:
            name: The name to use for this agent type.
            agent_or_factory: An agent factory (preferred) or instance (legacy).
        """
        from dana.core.agent.base_agent import BaseAgent

        if isinstance(agent_or_factory, BaseAgent):
            warnings.warn(
                f"register_agent('{name}', <instance>) is deprecated and unsafe "
                "under concurrency: a shared agent instance interleaves mutable "
                "state across Task calls. Pass a factory instead, e.g. "
                "functools.partial(AgentCls, agent_id=..., auto_register=False).",
                DeprecationWarning,
                stacklevel=2,
            )
            instance = agent_or_factory
            cls: type = type(instance)
            # Constant factory — intentionally has no `.func`; the description
            # is cached in _descriptions below, so nothing re-reads the class
            # off this callable.
            factory: Callable[[], Any] = lambda inst=instance: inst  # noqa: E731
        else:
            factory = agent_or_factory
            if not callable(factory):
                raise TypeError(f"Agent factory for '{name}' must be callable, got {type(factory)!r}.")
            if not hasattr(factory, "func"):
                raise TypeError(
                    f"Agent factory for '{name}' must expose its target class via '.func' "
                    "— use functools.partial(AgentCls, ...), not a bare lambda."
                )
            cls = getattr(factory, "func")  # noqa: B009 - Pyright cannot narrow Any here

        self._agents[name] = factory
        self._descriptions[name] = getattr(cls, "TASK_TOOL_DESCRIPTION", "No description available")
        self._update_task_docstring()

    def unregister_agent(self, name: str) -> bool:
        """Unregister an agent from task dispatch.

        Args:
            name: The agent type name to remove.

        Returns:
            True if agent was removed, False if not found.
        """
        if name in self._agents:
            del self._agents[name]
            self._descriptions.pop(name, None)
            self._update_task_docstring()
            return True
        return False

    def _update_task_docstring(self) -> None:
        """Update the task method docstring with current agent information."""
        self.task.__func__.__doc__ = self._build_task_description()

    def _build_task_description(self) -> str:
        """Build description matching the expected format for dynamic agent documentation."""
        parts = [
            "Launch a new agent to handle complex, multi-step tasks autonomously.",
            "",
            "The Task tool launches specialized agents (subprocesses) that autonomously handle complex tasks. Each agent type has specific capabilities and tools available to it.",
            "",
            "Available agent types:",
        ]

        for name, desc in self._descriptions.items():
            parts.append(f"- {name}: {desc}")

        parts.extend(
            [
                "",
                "When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.",
                "",
                "When NOT to use the Task tool:",
                "- If you want to read a specific file path, use the Read or Glob tool instead of the Task tool, to find the match more quickly",
                '- If you are searching for a specific class definition like "class Foo", use the Glob tool instead, to find the match more quickly',
                "- If you are searching for code within a specific file or set of 2-3 files, use the Read tool instead of the Task tool, to find the match more quickly",
                "- Other tasks that are not related to the agent descriptions above",
                "",
                "Usage notes:",
                "- Always include a short description (3-5 words) summarizing what the agent will do",
                "- Provide clear, detailed prompts so the agent can work autonomously and return exactly the information you need",
                "- When the agent is done, it will return a single message back to you along with its session_id. You can use this ID to resume the agent later if needed for follow-up work.",
                "- Agents can be resumed using the `resume` parameter by passing the session_id from a previous invocation. When resumed, the agent continues with its full previous context preserved.",
                "- You can optionally run agents in the background using the run_in_background parameter.",
            ]
        )

        return "\n".join(parts)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID.

        Returns:
            An 8-character unique session identifier.
        """
        return str(uuid.uuid4())[:8]

    @named_tool(name="Task")
    async def task(
        self,
        description: str,  # noqa: ARG002 - Used for logging/display purposes
        prompt: str,
        subagent_type: str,
        model: str | None = None,  # noqa: ARG002 - Reserved for future model override
        resume: str | None = None,
        run_in_background: bool = False,  # noqa: ARG002 - Reserved for background execution
        max_turns: int = 50,  # noqa: ARG002 - Reserved for turn limiting
    ) -> str:
        """Placeholder docstring - replaced at runtime by _update_task_docstring().

        Args:
            description: A short (3-5 word) description of the task.
            prompt: The task for the agent to perform.
            subagent_type: The type of specialized agent to use for this task.
            model: Optional model to use for this agent.
            resume: Optional session_id to resume from a previous invocation.
            run_in_background: Set to true to run this agent in the background.
            max_turns: Maximum number of agentic turns before stopping.

        Returns:
            The agent's response along with a session_id for resumption.
        """
        _ = (description, model, run_in_background, max_turns)  # Reserved for future use

        # Validate agent type
        if subagent_type not in self._agents:
            available = ", ".join(self._agents.keys()) if self._agents else "none"
            return f"Error: Unknown agent type '{subagent_type}'. Available agents: {available}"

        session_id = resume or self._generate_session_id()

        # Resume reuses the live instance if it is still in memory; otherwise
        # (and for every fresh task) the factory builds a disjoint agent so no
        # mutable state is shared across spawns. Disk-based resume for an
        # evicted session is handled by the agent's own set_session_id reload.
        existing = self._sessions.get(session_id, {}).get("agent") if resume else None
        agent = existing if existing is not None else self._agents[subagent_type]()

        # Propagate recorded notifiables to the freshly constructed agent.
        if self._notifiables and hasattr(agent, "with_notifiable"):
            agent.with_notifiable(*self._notifiables)

        # Store session state
        self._sessions[session_id] = {
            "agent": agent,
            "subagent_type": subagent_type,
            "status": "running",
        }

        # Execute the agent query. On failure, mark the session "failed" (so
        # task_output does not report it "running" forever) and re-raise so
        # the caller still observes the error.
        #
        # reload_timeline=True: a sub-agent's session_id IS a hard context
        # boundary — each spawn gets a disjoint, disk-accurate timeline (fresh
        # for a new session, rehydrated for a resumed one). This overrides the
        # default False, which keeps the caller's in-memory timeline.
        try:
            result = await agent.aquery(message=prompt, session_id=session_id, reload_timeline=True)
        except Exception as e:
            self._sessions[session_id]["status"] = "failed"
            self._sessions[session_id]["error"] = str(e)
            raise

        # Update session state
        self._sessions[session_id]["status"] = "completed"
        self._sessions[session_id]["result"] = result

        # Extract response from result
        response = result.get("response", str(result)) if isinstance(result, dict) else str(result)
        return f"{response}\n\n[session_id: {session_id}]"

    @named_tool(name="TaskOutput")
    async def task_output(self, task_id: str, block: bool = True, timeout: int = 30000) -> str:
        """Retrieve output from a running or completed task.

        Args:
            task_id: The session ID to get output from.
            block: Whether to wait for completion (default: True).
            timeout: Max wait time in milliseconds (default: 30000).

        Returns:
            The task status and output.
        """
        _ = (block, timeout)  # Reserved for async/blocking behavior

        if task_id not in self._sessions:
            return f"Error: Session '{task_id}' not found."

        session = self._sessions[task_id]
        status = session.get("status", "unknown")

        if status == "completed":
            result = session.get("result", {})
            response = result.get("response", str(result)) if isinstance(result, dict) else str(result)
            return f"Status: completed\n\n{response}"

        if status == "failed":
            return f"Status: failed\n\n{session.get('error', 'Unknown error')}"

        return f"Status: {status}"
