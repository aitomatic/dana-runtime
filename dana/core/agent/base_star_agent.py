"""
Clean BaseSTARAgent implementation with minimal STAR pattern contract.

This module provides the core STAR (See-Think-Act-Reflect) pattern contract
without implementation details like LLM integration or rich state management.
"""

from abc import abstractmethod
import asyncio
from collections.abc import AsyncIterator, Callable
import logging
import threading
from typing import TYPE_CHECKING, Any

from dana.common.observable import observable
from dana.common.protocols import DictParams, STARAgentProtocol
from dana.common.protocols.types import LearningPhase
from dana.core.agent.base_agent import BaseAgent
from dana.core.ext.event_bus import Event, EventBus
from dana.core.ext.events import ACT_END, REFLECT_END, SEE_END, THINK_END
from dana.core.llm.llm_caller import is_transient_llm_error
from dana.core.runtime.protocols import StreamEvent, StreamEventType


if TYPE_CHECKING:
    from dana.core.ext.extensions import ExtensionManager


logger = logging.getLogger(__name__)


EXIT_STAR_LOOP_FLAG = "EXIT_STAR_LOOP_FLAG"

# STAR loop retry budget for transient LLM errors (per iteration).
# This sits ON TOP of LLMCaller's own retry — it covers cases where the network
# recovers between iterations or the failure surfaces outside the LLM call itself.
# Kept small to avoid pathological wait times when stacked with LLMCaller retries.
_STAR_TRANSIENT_RETRIES = 1
_STAR_TRANSIENT_BASE_DELAY = 1.0


class BaseSTARAgent(BaseAgent, STARAgentProtocol):
    """
    Minimal base class defining the STAR (See-Think-Act-Reflect) pattern contract.

    Provides core STAR pattern orchestration and basic agent identity without
    implementation details like LLM integration or rich state management.

    The STAR loop executes: SEE -> THINK -> ACT -> REFLECT

    See docs/architecture/star-pattern-schema.md for the historical XML prompt schema.
    """

    MAX_ITERATIONS = 20

    # ============================================================================
    # CORE STAR PATTERN CONTRACT (Abstract Methods)
    # ============================================================================

    @abstractmethod
    def _see(self, trace_inputs: DictParams) -> DictParams:
        """
        SEE: See the inputs and produce percepts.

        Args:
            trace_inputs (DictParams): any new user/agent inputs

        Returns:
            - bool: True if the agent should continue the loop, False otherwise.
            - trace_percepts (DictParams): the percepts produced by this SEE phase.
        """
        result = {"trace_percepts": trace_inputs}
        self.broadcast(result)
        return result

    @abstractmethod
    def _think(self, trace_percepts: DictParams) -> DictParams:
        """
        THINK: Think about the percepts and produce thoughts.

        Args:
            trace_percepts (DictParams): the percepts produced by this SEE phase.

        Returns:
            - bool: True if the agent should continue the loop, False otherwise.
            - trace_thoughts (DictParams): the thoughts produced by this THINK phase.
        """
        result = {"trace_thoughts": trace_percepts}
        self.broadcast(result)
        return result

    @abstractmethod
    def _act(self, trace_thoughts: DictParams) -> DictParams:
        """
        ACT: Act on the thoughts and produce outputs.

        TODO: this is a good place to send feedback to the user if we are about to make tool calls

        Args:
            trace_thoughts (DictParams): the thoughts produced by this THINK phase.

        Returns:
            - bool: True if the agent should continue the loop, False otherwise.
            - trace_outputs (DictParams): the outputs produced by this ACT phase.
        """
        result = {"trace_outputs": trace_thoughts}
        self.broadcast(result)
        return result

    @abstractmethod
    def _reflect(self, trace_outputs: DictParams) -> DictParams:
        """
        REFLECT: Reflect on the outputs for learning.

        Args:
            trace_outputs (DictParams): the outputs produced by this ACT phase.

        Returns:
            - bool: True if the agent should continue the loop, False otherwise.
            - trace_learning (DictParams): the learning produced by this REFLECT phase.
        """
        result = {"trace_learning": trace_outputs}
        self.broadcast(result)
        return result

    # ============================================================================
    # ASYNC STAR METHODS
    # ============================================================================

    @abstractmethod
    async def _think_async(self, trace_percepts: DictParams) -> DictParams:
        """
        THINK (async): Async version of _think with native async LLM calls.

        Args:
            trace_percepts (DictParams): the percepts produced by this SEE phase.

        Returns:
            - trace_thoughts (DictParams): the thoughts produced by this THINK phase.
        """
        result = {"trace_thoughts": trace_percepts}
        self.broadcast(result)
        return result

    @abstractmethod
    async def _act_async(self, trace_thoughts: DictParams) -> DictParams:
        """
        ACT (async): Async version of _act with native async tool execution.

        Args:
            trace_thoughts (DictParams): the thoughts produced by this THINK phase.

        Returns:
            - trace_outputs (DictParams): the outputs produced by this ACT phase.
        """
        result = {"trace_outputs": trace_thoughts}
        self.broadcast(result)
        return result

    # ============================================================================
    # EXIT STAR LOOP FLAG
    # ============================================================================

    def _mark_star_loop_exit(self, trace: DictParams | None = None) -> DictParams:
        if not trace:
            trace = {}

        trace[EXIT_STAR_LOOP_FLAG] = True
        return trace

    def _do_exit_star_loop(self, trace: DictParams) -> bool:
        return trace.get(EXIT_STAR_LOOP_FLAG, False) if trace else True

    # ============================================================================
    # PHASE EVENT EMIT (M2 — STAR loop wire EventBus)
    # ============================================================================

    def _emit_phase(self, event_type: str, result: DictParams) -> DictParams:
        """Emit a phase_end event (intercept-capable); apply modify/block. M2.

        - handler ``{"modify": new}``      -> result = new
        - handler ``{"block": True, ...}`` -> set ``EXIT_STAR_LOOP_FLAG`` at the
          result's top level so the orchestrator exits before the next phase
        - handler raise / None             -> pass-through (bus S1 catches raises)

        Returns the (possibly modified) result. Never raises. Does NOT call
        ``broadcast`` — phase bodies + ``star_agent`` think/act_async still
        broadcast for instrumentation; this method only adds the 2-way emit.
        """
        handler = self.event_bus.emit_sync(Event(event_type, {"result": result}))
        return self._apply_phase_handler(handler, result)

    async def _emit_phase_async(self, event_type: str, result: DictParams) -> DictParams:
        """Async counterpart of ``_emit_phase`` (awaits handlers on the same loop)."""
        handler = await self.event_bus.emit(Event(event_type, {"result": result}))
        return self._apply_phase_handler(handler, result)

    @staticmethod
    def _apply_phase_handler(handler: DictParams | None, result: DictParams) -> DictParams:
        """Shared block/modify interpretation for sync + async emit helpers."""
        if not isinstance(handler, dict):
            return result
        if handler.get("block") is True:
            out = dict(result) if isinstance(result, dict) else {"payload": result}
            out[EXIT_STAR_LOOP_FLAG] = True
            return out
        modified = handler.get("modify")
        if isinstance(modified, dict):
            return modified
        if modified is not None:
            logger.warning("phase handler returned non-dict modify; ignored")
        return result

    # ============================================================================
    # STAR LOOP ORCHESTRATION
    # ============================================================================

    def query(self, **kwargs) -> DictParams:
        """Main entry point - orchestrates the STAR loop.

        Args:
            **kwargs: Additional arguments passed to STAR loop.
        """

        @observable(name=f"Dana {self.agent_type}-agent-query")
        def _do_query(trace_inputs: DictParams) -> DictParams:
            import time

            trace_outputs: DictParams = {}

            for iteration in range(self.MAX_ITERATIONS):
                # Inner loop retries the See/Think/Act cycle on transient LLM errors.
                # LLMCaller already retries inside its own scope; this is a second-line
                # defense for transient failures that escape (or whose retry budget
                # was exhausted) before we mark the whole session as failed.
                attempt = 0
                star_failed = False
                phase_blocked = False
                while True:
                    try:
                        trace_percepts = self._see(trace_inputs.get("trace_inputs", {}))
                        trace_percepts = self._emit_phase(SEE_END, trace_percepts)
                        if trace_percepts.get(EXIT_STAR_LOOP_FLAG) is True:
                            trace_outputs = trace_percepts
                            phase_blocked = True
                            break
                        trace_thoughts = self._think(trace_percepts.get("trace_percepts", {}))
                        trace_thoughts = self._emit_phase(THINK_END, trace_thoughts)
                        if trace_thoughts.get(EXIT_STAR_LOOP_FLAG) is True:
                            trace_outputs = trace_thoughts
                            phase_blocked = True
                            break
                        trace_outputs = self._act(trace_thoughts.get("trace_thoughts", {}))
                        trace_outputs = self._emit_phase(ACT_END, trace_outputs)
                        if trace_outputs.get(EXIT_STAR_LOOP_FLAG) is True:
                            phase_blocked = True
                        break
                    except Exception as e:
                        if is_transient_llm_error(e) and attempt < _STAR_TRANSIENT_RETRIES:
                            delay = _STAR_TRANSIENT_BASE_DELAY * (2**attempt)
                            logger.warning(
                                "STAR iteration transient error, retrying (iteration=%d, attempt=%d/%d, delay=%.1fs): %s",
                                iteration,
                                attempt + 1,
                                _STAR_TRANSIENT_RETRIES,
                                delay,
                                e,
                                exc_info=True,
                            )
                            time.sleep(delay)
                            attempt += 1
                            continue
                        logger.error(
                            "Error in query (iteration=%d, transient=%s): %s",
                            iteration,
                            is_transient_llm_error(e),
                            e,
                            exc_info=True,
                        )
                        trace_outputs = {"trace_outputs": {"error": e}}
                        star_failed = True
                        break

                if star_failed or phase_blocked:
                    break

                # Trigger acquisitive learning asynchronously at end of each STAR loop
                if not self._do_exit_star_loop(trace_outputs.get("trace_outputs", {})):
                    acquisitive_input = trace_outputs.get("trace_outputs", {}).copy()
                    acquisitive_input["phase"] = LearningPhase.ACQUISITIVE

                    # Sync path: use thread (no event loop available)
                    def run_reflect(acq_input):
                        try:
                            learning = self._reflect(acq_input)
                            self._emit_phase(REFLECT_END, learning)
                        except Exception as reflect_err:
                            logger.error("Reflection failed: %s", reflect_err, exc_info=True)

                    threading.Thread(target=run_reflect, args=(acquisitive_input,), daemon=True).start()

                if self._do_exit_star_loop(trace_outputs.get("trace_outputs", {})):
                    break

            return trace_outputs

        try:
            result = _do_query(trace_inputs={"trace_inputs": kwargs})
            result = result.get("trace_outputs", {}) if result else {}

        except Exception as e:
            logger.error("Error in query: %s", e, exc_info=True)
            result = {"error": e}

        return result

    async def aquery(self, **kwargs) -> DictParams:
        """Async version of query that uses async STAR methods.

        Args:
            **kwargs: Additional arguments passed to STAR loop.
        """

        @observable(name=f"Dana {self.agent_type}-agent-aquery")
        async def _do_aquery(trace_inputs: DictParams) -> DictParams:
            trace_outputs: DictParams = {}

            for iteration in range(self.MAX_ITERATIONS):
                # Inner loop retries the See/Think/Act cycle on transient LLM errors.
                # See _do_query for rationale.
                attempt = 0
                star_failed = False
                phase_blocked = False
                while True:
                    try:
                        # _see is sync (no async ops needed)
                        trace_percepts = self._see(trace_inputs.get("trace_inputs", {}))
                        trace_percepts = await self._emit_phase_async(SEE_END, trace_percepts)
                        if trace_percepts.get(EXIT_STAR_LOOP_FLAG) is True:
                            trace_outputs = trace_percepts
                            phase_blocked = True
                            break
                        # _think_async uses native async LLM call
                        trace_thoughts = await self._think_async(trace_percepts.get("trace_percepts", {}))
                        trace_thoughts = await self._emit_phase_async(THINK_END, trace_thoughts)
                        if trace_thoughts.get(EXIT_STAR_LOOP_FLAG) is True:
                            trace_outputs = trace_thoughts
                            phase_blocked = True
                            break
                        # _act_async uses native async tool execution
                        trace_outputs = await self._act_async(trace_thoughts.get("trace_thoughts", {}))
                        trace_outputs = await self._emit_phase_async(ACT_END, trace_outputs)
                        if trace_outputs.get(EXIT_STAR_LOOP_FLAG) is True:
                            phase_blocked = True
                        break
                    except Exception as e:
                        if is_transient_llm_error(e) and attempt < _STAR_TRANSIENT_RETRIES:
                            delay = _STAR_TRANSIENT_BASE_DELAY * (2**attempt)
                            logger.warning(
                                "STAR iteration transient error, retrying (iteration=%d, attempt=%d/%d, delay=%.1fs): %s",
                                iteration,
                                attempt + 1,
                                _STAR_TRANSIENT_RETRIES,
                                delay,
                                e,
                                exc_info=True,
                            )
                            await asyncio.sleep(delay)
                            attempt += 1
                            continue
                        logger.error(
                            "Error in aquery (iteration=%d, transient=%s): %s",
                            iteration,
                            is_transient_llm_error(e),
                            e,
                            exc_info=True,
                        )
                        trace_outputs = {"trace_outputs": {"error": e}}
                        star_failed = True
                        break

                if star_failed or phase_blocked:
                    break

                # Trigger acquisitive learning asynchronously at end of each STAR loop
                if not self._do_exit_star_loop(trace_outputs.get("trace_outputs", {})):
                    acquisitive_input = trace_outputs.get("trace_outputs", {}).copy()
                    acquisitive_input["phase"] = LearningPhase.ACQUISITIVE

                    # Async path: use asyncio.create_task (proper async, not threads)
                    async def _async_reflect(acq_input):
                        try:
                            learning = self._reflect(acq_input)
                            await self._emit_phase_async(REFLECT_END, learning)
                        except Exception as reflect_err:
                            logger.error("Async reflection failed: %s", reflect_err, exc_info=True)

                    asyncio.create_task(_async_reflect(acquisitive_input))

                if self._do_exit_star_loop(trace_outputs.get("trace_outputs", {})):
                    break

            return trace_outputs

        try:
            result = await _do_aquery(trace_inputs={"trace_inputs": kwargs})
            result = result.get("trace_outputs", {}) if result else {}

        except Exception as e:
            logger.error("Error in aquery: %s", e, exc_info=True)
            result = {"error": e}

        return result

    async def aquery_stream(self, **kwargs) -> AsyncIterator[StreamEvent]:
        """Streaming version of aquery. Yields StreamEvent objects.

        Subclasses override _think_stream() to stream text deltas during the
        think phase. The default implementation falls back to aquery() and
        emits a single DONE event — subclasses provide richer streaming.

        Yields:
            StreamEvent: Events with types TEXT_DELTA, TOOL_CALL_START,
                         TOOL_RESULT, ERROR, or DONE.
        """
        # Default: no streaming support — subclasses override
        try:
            result = await self.aquery(**kwargs)
            response = result.get("response", "") if result else ""
            if response:
                yield StreamEvent(
                    event_type=StreamEventType.TEXT_DELTA,
                    data=response,
                    iteration=0,
                )
        except Exception as exc:
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data=str(exc),
                iteration=0,
            )
            return
        yield StreamEvent(event_type=StreamEventType.DONE, data=None, iteration=0)

    # ============================================================================
    # EXTENSIBILITY (S1)
    # ============================================================================

    @property
    def event_bus(self) -> EventBus:
        """Per-agent intercept-capable event bus.

        Lazily created on first access so the mount point adds zero cost to
        agent construction and no ``__init__`` coupling. Each agent owns its own
        bus (correct session scope; never a global).

        Implementation note: MUST read/write ``self.__dict__`` directly — NOT
        ``getattr(self, "_event_bus", None)``. ``STARAgent.__getattr__`` returns
        a "magic method" stub for ANY unknown attribute (natural-language
        converse), so ``getattr`` would return that stub instead of None and the
        bus would never be created. ``__dict__`` access bypasses ``__getattr__``.
        """
        bus = self.__dict__.get("_event_bus")
        if not isinstance(bus, EventBus):
            bus = EventBus()
            self.__dict__["_event_bus"] = bus
        return bus

    def on(self, event_type: str, handler: Callable[..., Any]) -> Callable[[], None]:
        """Subscribe ``handler`` to ``event_type`` on this agent's bus. M4.

        Thin alias for ``self.event_bus.subscribe(event_type, handler)``, exposed
        as the extension-facing registration API (``setup(agent): agent.on(...)``).
        Returns the unsubscribe callable.
        """
        return self.event_bus.subscribe(event_type, handler)

    @property
    def extensions(self) -> "ExtensionManager":
        """Per-agent extension manager (lazy, ``self.__dict__`` storage). M4.

        Like ``event_bus``, MUST use ``__dict__`` (not ``getattr``) to avoid
        ``STARAgent.__getattr__`` returning a magic-method stub.
        """
        from dana.core.ext.extensions import ExtensionManager

        mgr = self.__dict__.get("_extensions")
        if not isinstance(mgr, ExtensionManager):
            mgr = ExtensionManager(self)
            self.__dict__["_extensions"] = mgr
        return mgr

    def load_extensions(self) -> Any:
        """Discover + load drop-in extensions (global always, project if trusted). M4.

        NOT auto-called at construction (hosts call this after creating an agent).
        Returns a ``LoadReport``.
        """
        return self.extensions.load_all()

    def reload_extensions(self) -> Any:
        """Hot-reload extensions: unsubscribe old, re-discover, re-load. M4.

        MUST be called at idle (not concurrent with a turn). Emits
        ``session_reload``. Returns a ``LoadReport``.
        """
        return self.extensions.reload_all()

    # ============================================================================
    # UTILITIES
    # ============================================================================

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"BaseSTARAgent(type={self.agent_type}, id={self.object_id})"

    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return f"BaseSTARAgent(agent_type='{self.agent_type}', object_id='{self.object_id}')"
