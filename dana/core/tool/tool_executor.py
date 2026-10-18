"""
ToolExecutor — implements ToolExecutorProtocol.

Orchestrates batch tool execution (sync and async) and delegates single-call
dispatch to internal helpers. Tool interception is handled by the EventBus
(milestone M3): ``tool_call``/``tool_result`` are emitted around dispatch in
both single-call paths (see sprint/plans/S3-tool-execution-engine.md).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import inspect
import traceback
from typing import Any

import structlog

from dana.common.observable import observable
from dana.common.utils.misc import Misc
from dana.core.ext.event_bus import Event, EventBus
from dana.core.ext.events import TOOL_CALL, TOOL_RESULT
from dana.core.ext.operation import Operation, build_operation
from dana.core.policy.effects import EffectMetadata
from dana.core.tool.admission import AdmissionController, get_default_admission_controller
from dana.core.tool.catalog import ToolCatalog
from dana.core.tool.effect_ordering import plan_batch
from dana.core.tool.offload import (
    BlockingIOPool,
    bound_blocking_io_pool,
    is_loop_affine,
    run_blocking,
    use_blocking_io_pool,
)
from dana.core.tool.tool_executor_helpers import (
    create_tool_error,
    create_tool_success,
    find_object_by_class_name,
    find_object_by_id,
    get_available_class_names,
    parse_function_name,
    validate_and_cast_method_arguments,
)


logger = structlog.get_logger()


@dataclass
class _PreparedCall:
    call: dict[str, Any]
    operation: Operation | None = None
    block_reason: str | None = None
    error: dict[str, Any] | None = None


class ToolExecutor:
    """Executes tool calls on behalf of an agent (sync and async).

    Implements ToolExecutorProtocol.

    Constructor args:
        agent_getter:             Callable[[], agent] — returns the current agent
                                  (used only when caller does not pass agent directly).
        tool_name_registry_getter: ``Callable[[agent], dict]`` for an agent-scoped
                                  registry (the runtime wires one so a shared
                                  runtime resolves the registry for the
                                  *dispatching* agent), or ``Callable[[], dict]``
                                  for a static caller-owned registry (frozen
                                  benchmark drivers and standalone executors).
                                  Both contracts are detected and supported.

    Tool interception is handled by the EventBus (M3): emit ``tool_call``/
    ``tool_result`` around dispatch in both single-call paths. The previous
    ``hooks``/``approval`` scaffold (never wired) was removed in M3 — route
    intercept through a ``tool_call`` handler (e.g. ``PermissionPolicy``) instead.
    """

    def __init__(
        self,
        agent_getter: Callable[[], Any] | None = None,
        tool_name_registry_getter: Callable[[], dict[str, tuple[Any, str]]] | None = None,
        max_workers: int | None = None,
        tool_catalog: ToolCatalog | None = None,
        mcp_dispatch_getter: Callable[[], dict[str, Any]] | None = None,
        admission: AdmissionController | None = None,
        effect_catalog_getter: Callable[[], Any] | None = None,
        lifecycle_observer: Callable[[str, dict[str, Any]], None] | None = None,
        catalog_revision_getter: Callable[[], int] | None = None,
        offload_pool: BlockingIOPool | None = None,
        before_hook: Callable[[dict[str, Any], str | None], Any] | None = None,
        after_hook: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None,
        *,
        agent_tool_name_registry_getter: Callable[[Any], dict[str, tuple[Any, str]]] | None = None,
    ) -> None:
        self._agent_getter = agent_getter
        self._tool_name_registry_getter = tool_name_registry_getter
        self._agent_tool_name_registry_getter = agent_tool_name_registry_getter
        self._max_workers = max_workers
        self._tool_catalog = tool_catalog
        # D7 follow-up 1+2: per-tool MCP dispatch map (tool_name -> async callable).
        self._mcp_dispatch_getter = mcp_dispatch_getter
        # P1 admission: bounds concurrent leaf tool work and nested sub-agent spawns
        # without nesting permits across a parent/child await. The process-wide
        # default controller is SHARED (so nested subtrees inherit one budget), so
        # this executor records whether it owns the controller: closing an executor
        # releases only its own, never the shared one (that would zero another live
        # session's admission bookkeeping).
        self._owns_admission = admission is not None
        self._admission = admission if admission is not None else get_default_admission_controller()
        # Effect-ordering source: the session-owned catalog used for scheduling. A
        # separate getter (not ``tool_catalog``) so the scheduling catalog can be
        # pinned per turn without rewriting the executor's dispatch catalog.
        self._effect_catalog_getter = effect_catalog_getter
        # Lifecycle observer: called with (event_name, payload) for terminal
        # disposition facts. Best-effort; never allowed to break dispatch.
        self._lifecycle_observer = lifecycle_observer
        # Catalog revision: an incrementing integer owned by the runtime. Cached
        # schema/signature/registry metadata is rebuilt only when it changes, so a
        # prompt turn never re-derives metadata that discovery did not invalidate.
        self._catalog_revision_getter = catalog_revision_getter
        self._catalog_revision: int | None = None
        self._tool_metadata_revision = 0
        # Blocking-I/O pool owned by THIS executor/session: closing this executor
        # closes only its own threads, so concurrent and future sessions keep
        # working (the process-wide default pool is never killed here).
        self._offload_pool = offload_pool if offload_pool is not None else BlockingIOPool()
        # Pre/post-execution hooks (durable-write seam). ``before`` runs right
        # before dispatch (after interception + admission); ``after`` runs for
        # every result, including policy blocks. Their failures PROPAGATE — a
        # durable append that fails must fail the turn, never be swallowed the
        # way the best-effort lifecycle observer is.
        self._before_hook = before_hook
        self._after_hook = after_hook
        self._closed = False

    def set_catalog_revision_getter(self, getter: Callable[[], int] | None) -> None:
        """Wire the runtime-owned catalog revision getter (stable-metadata cache).

        The getter returns an integer that the runtime bumps whenever tool
        discovery/schema/wiring changes. Cached metadata is invalidated when the
        integer changes; passing ``None`` disables revision-based invalidation.
        """
        self._catalog_revision_getter = getter

    def invalidate_tool_metadata(self) -> None:
        """Force a metadata rebuild on the next dispatch (explicit invalidation)."""
        self._catalog_revision = None
        self._tool_metadata_revision += 1

    @property
    def tool_metadata_revision(self) -> int:
        """How many times cached tool metadata has been rebuilt."""
        return self._tool_metadata_revision

    def set_execution_hooks(
        self,
        before: Callable[[dict[str, Any], str | None], Any] | None = None,
        after: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None,
    ) -> None:
        """Wire the authoritative tool-lifecycle hooks (durable-write seam).

        These are the session's durable-journal boundary — the *only* place a
        session can make "requested/authorized/started" and "terminal" durable
        around a tool call. Unlike :meth:`set_lifecycle_observer`, hook failures
        are NOT swallowed by the tool executor: they propagate so a failed
        durable append can never be silently lost.

        Async executor (``execute_tools_async`` / ``_execute_single_call_async``):

        * ``before(tool_call, policy_block_reason)`` is awaited **immediately before
          dispatch**, *after* the EventBus ``tool_call`` interception has run and
          after the admission permit was acquired, and *outside* the never-raise
          dispatch guard. ``tool_call`` carries the model-visible call shape with
          the post-interception (possibly modified) ``arguments`` — the actual
          arguments about to be dispatched. ``policy_block_reason`` is ``None`` when
          the call will be dispatched, or the block reason string when the EventBus
          already decided to block it; for a blocked call the hook still runs (so
          the session can persist requested/authorized) and the call is *not*
          dispatched — the session persists ``started`` only when the reason is
          ``None``. If ``before`` raises, no dispatch happens and the exception
          propagates (no side effect).
        * ``after(tool_call, result)`` is awaited for **every** completed, failed,
          cancelled or policy-blocked call, *outside* the best-effort observer
          guard, *after* the EventBus ``tool_result`` interception has applied its
          modification, and before the executor returns — so the journal terminal
          fact is durable before the turn's terminal fact. A cancellation result
          carries a truthful marker: ``_cancelled`` (acknowledged by the tool) or
          ``_effect_unknown`` (non-cooperative work whose terminal state cannot be
          confirmed).

        Each hook may be sync or async (an awaitable return is awaited). The
        legacy synchronous executor does not invoke these hooks: a synchronous
        caller has no session-owned journal to make durable, and no loop to await
        on. A session-bound sync caller journals through the EventBus ``tool_call``
        handler instead.
        """
        self._before_hook = before
        self._after_hook = after

    async def _run_hook(self, hook: Callable[..., Any] | None, *args: Any) -> None:
        """Run a lifecycle hook, propagating any failure (durable-append seam)."""
        if hook is None:
            return
        outcome = hook(*args)
        if inspect.isawaitable(outcome):
            await outcome

    @property
    def offload_pool(self) -> BlockingIOPool:
        """The blocking-I/O pool owned by this executor."""
        return self._offload_pool

    def _sync_catalog_revision(self) -> None:
        """Rebuild cached metadata when the runtime reports a new catalog revision."""
        if self._catalog_revision_getter is None:
            return
        try:
            revision = self._catalog_revision_getter()
        except Exception:  # noqa: BLE001 — a broken getter must not break dispatch
            logger.warning("catalog_revision_getter_failed")
            return
        if revision != self._catalog_revision:
            self._catalog_revision = revision
            self._tool_metadata_revision += 1

    def set_mcp_dispatch_getter(self, getter: Callable[[], dict[str, Any]] | None) -> None:
        """Wire the MCP dispatch map getter (per-tool MCP UX, A2).

        The getter returns ``{tool_name: async callable(arguments) -> str}``;
        ``None`` disables MCP per-tool dispatch (fall through to the registry).
        """
        self._mcp_dispatch_getter = getter

    def set_effect_catalog_getter(self, getter: Callable[[], Any] | None) -> None:
        """Wire the catalog getter used for effect-ordered batch scheduling.

        Distinct from the dispatch catalog so a session can pin a per-turn
        catalog for ordering without changing where calls are dispatched.
        """
        self._effect_catalog_getter = getter

    def set_lifecycle_observer(self, observer: Callable[[str, dict[str, Any]], None] | None) -> None:
        """Wire the lifecycle observer (terminal disposition + cleanup facts)."""
        self._lifecycle_observer = observer

    # ------------------------------------------------------------------
    # Lifecycle (parent session teardown)
    # ------------------------------------------------------------------

    def close(self, *, wait_for_offload: bool = False) -> None:
        """Release executor-owned resources (sync; lifecycle cleanup API).

        Performs, in order:
        1. Releases admission bookkeeping — only when this executor owns its
           controller; the shared process-wide default is never cleared, and no
           in-flight work is ever cancelled.
        2. Closes the blocking-I/O pool **owned by this executor** — only its own
           worker threads. The process-wide default pool is untouched, so
           concurrent sessions keep running and later sessions still work.
        3. Emits a ``tool_executor_closed`` lifecycle observation.

        Idempotent. Never reports an abandoned future as a confirmed cancellation:
        a thread cannot be interrupted, so in-flight blocking work is *not*
        rolled back — it is allowed to finish (``wait_for_offload=False`` drops
        queued-but-unstarted work only).

        Args:
            wait_for_offload: When True, block until queued blocking work
                finishes. Call from a non-loop thread (it joins worker threads).
        """
        if self._closed:
            return
        self._closed = True
        if self._owns_admission:
            self._admission.close()
        self._offload_pool.shutdown(wait=wait_for_offload)
        self._observe("tool_executor_closed", {"waited_for_offload": wait_for_offload})

    async def aclose(self, *, wait_for_offload: bool = False) -> None:
        """Async lifecycle cleanup (alias of :meth:`close`).

        Exists so a host can call ``await executor.aclose()`` inside its own
        ``aclose()`` without blocking the loop, and ``executor.close()`` from
        sync teardown. MCP transports and provider clients are NOT owned here;
        the session releases those.
        """
        self.close(wait_for_offload=wait_for_offload)

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run."""
        return self._closed

    def _observe(self, event: str, payload: dict[str, Any]) -> None:
        """Emit a lifecycle observation; a broken observer never breaks dispatch."""
        if self._lifecycle_observer is None:
            return
        try:
            self._lifecycle_observer(event, payload)
        except Exception:  # noqa: BLE001
            logger.warning("lifecycle_observer_failed", event=event)

    # ------------------------------------------------------------------
    # Public API — ToolExecutorProtocol
    # ------------------------------------------------------------------

    @observable
    def execute_tools(
        self,
        agent: Any,
        tool_calls: list[dict[str, Any]],
        parallel: bool = False,
    ) -> list[dict[str, Any]]:
        """Sync batch executor — sequential by default, order-safe parallel when asked.

        When ``parallel=True``, calls are grouped into conflict-ordered waves
        (:mod:`dana.core.tool.effect_ordering`): calls that provably cannot interfere
        run concurrently on a bounded thread pool, while conflicting calls keep the
        model's relative order. ``parallel=False`` remains strictly sequential.
        """
        if parallel:
            from concurrent.futures import ThreadPoolExecutor

            self._sync_catalog_revision()
            prepared = [self._prepare_single_call(agent, call) for call in tool_calls]
            plan = plan_batch([item.call for item in prepared], self._effect_catalog(), metadata_resolver=self._effect_metadata_for)
            results: list[dict[str, Any] | None] = [None] * len(tool_calls)
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                for wave in plan.waves:
                    futures = [(i, executor.submit(self._execute_single_call, agent, tool_calls[i], _prepared=prepared[i])) for i in wave]
                    for index, future in futures:
                        results[index] = future.result()
        else:
            self._sync_catalog_revision()
            results = [self._execute_single_call(agent, call) for call in tool_calls]

        for result, call in zip(results, tool_calls, strict=False):
            if result is None:
                continue
            if "tool_call_id" in call:
                result["tool_call_id"] = call["tool_call_id"]
        return [r for r in results if r is not None]

    @observable
    async def execute_tools_async(self, agent: Any, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Async batch executor — effect-ordered waves, bounded by admission.

        Scheduling (P1 — "make tool concurrency real and bounded"):

        1. Conflict-ordered waves. Calls are classified from catalog effect
           metadata and packed into waves such that a call only overlaps calls it
           provably cannot interfere with. The model's call order is preserved for
           every conflicting pair (see :mod:`dana.core.tool.effect_ordering`).
        2. Bounded admission. Each dispatch takes one permit from either the leaf
           pool or the nested (sub-agent) pool, so total concurrency is bounded and
           nested spawns cannot multiply resource usage.
        3. An outcome for every id. Each call is individually guarded — one call's
           failure, error or cancellation never releases, cancels, or skips a
           sibling — so the next provider turn never sees an unanswered tool call.
        """
        if not tool_calls:
            return []

        self._sync_catalog_revision()
        prepared = [await self._prepare_single_call_async(agent, call) for call in tool_calls]
        plan = plan_batch([item.call for item in prepared], self._effect_catalog(), metadata_resolver=self._effect_metadata_for)
        results: list[dict[str, Any]] = [None] * len(tool_calls)  # type: ignore[list-item]

        # Only two kinds of exception can escape a call now: the session's own hook
        # failures (durable-append / refusal) and cancellation. Neither may be
        # converted into a tool result, so the first one is remembered and re-raised
        # after every call in its wave has been accounted for.
        fatal: BaseException | None = None
        with use_blocking_io_pool(bound_blocking_io_pool() or self._offload_pool):
            for wave in plan.waves:
                wave_results = await asyncio.gather(
                    *[self._execute_single_call_async(agent, tool_calls[i], _prepared=prepared[i]) for i in wave],
                    return_exceptions=True,
                )
                for index, result in zip(wave, wave_results, strict=False):
                    call = tool_calls[index]
                    if isinstance(result, BaseException):
                        if fatal is None:
                            fatal = result
                        if isinstance(result, asyncio.CancelledError):
                            # Cancelled mid-dispatch: the call had begun, so its
                            # effect is not confirmed. The truthful terminal marker
                            # was already written by the ``after`` hook; this keeps
                            # the returned shape truthful too.
                            result = {
                                "success": False,
                                "type": "cancelled",
                                "target": call.get("function", ""),
                                "result": "Error: Cancelled during execution; effect not confirmed",
                            }
                            result["_effect_unknown"] = True
                        else:
                            # A hook failure: the session owns that call's terminal
                            # fact (it may not be durable). Record nothing here.
                            result = None
                    if isinstance(result, dict) and "tool_call_id" in call:
                        result["tool_call_id"] = call["tool_call_id"]
                    results[index] = result
                if fatal is not None:
                    # Every call in this wave has been accounted for; the turn's
                    # cancellation / journal failure is not swallowed. Calls in
                    # waves not reached are reconciled by the session's pending-tool
                    # fallback (requested -> cancelled, started -> effect-unknown).
                    raise fatal

        return results

    def _effect_catalog(self) -> Any | None:
        """The catalog used for effect-ordered scheduling (may be None)."""
        if self._effect_catalog_getter is not None:
            return self._effect_catalog_getter()
        return self._tool_catalog

    def _tag_admission(self, result: dict[str, Any], admission_kind: str) -> None:
        """Record the admission pool a result was produced under (nested only).

        Nested (sub-agent) work is tagged so the exclusive-handoff budget is
        observable at the result boundary. Leaf results stay untagged to keep the
        common payload byte-identical.
        """
        if admission_kind == AdmissionController.NESTED and isinstance(result, dict):
            result["_admission"] = "nested"

    @staticmethod
    async def _invoke_mcp_dispatcher(dispatcher: Any, arguments: dict[str, Any], tool_call_id: Any) -> Any:
        """Invoke an MCP dispatch callable, forwarding the harness tool_call_id.

        The built dispatchers accept ``(arguments, tool_call_id)``. A pre-existing
        single-argument dispatcher (an injected test double or a caller-registered
        map) is still supported: it is called with ``arguments`` only, so the
        harness id is simply not forwarded rather than raising TypeError.
        """
        signature = inspect.signature(dispatcher)
        try:
            signature.bind(arguments, tool_call_id)
        except TypeError:
            signature.bind(arguments)
            return await dispatcher(arguments)
        return await dispatcher(arguments, tool_call_id)

    def _effect_metadata_for(self, function_name: str) -> EffectMetadata:
        """Effect metadata for a call: catalog entry first, native table second."""
        catalog = self._effect_catalog()
        if catalog is not None:
            entry = catalog.get(function_name)
            if entry is not None:
                return getattr(entry, "effects", None) or EffectMetadata.unknown()
        from dana.core.tool.native_catalog import native_effect_metadata

        return native_effect_metadata(function_name)

    @staticmethod
    def _apply_interception(tool_call: dict[str, Any], operation: Operation, pre: dict[str, Any] | None) -> _PreparedCall:
        arguments = dict(operation.arguments)
        block_reason = None
        if isinstance(pre, dict) and pre.get("block") is True:
            block_reason = str(pre.get("reason", "blocked"))
        elif isinstance(pre, dict) and isinstance(pre.get("modify"), dict):
            new_arguments = pre["modify"].get("arguments")
            if isinstance(new_arguments, dict):
                arguments = dict(new_arguments)
        effective_call = {**tool_call, "arguments": arguments}
        operation = Operation(tool_identity=operation.tool_identity, arguments=arguments)
        return _PreparedCall(effective_call, operation, block_reason)

    def _preparation_error(self, tool_call: dict[str, Any], exc: Exception) -> _PreparedCall:
        name = tool_call.get("function", "")
        result = create_tool_error("execution_error", name, f"Error preparing call {name}: {exc}\n{traceback.format_exc()}")
        return _PreparedCall({**tool_call, "arguments": {}}, error=result)

    def _prepare_single_call(self, agent: Any, tool_call: dict[str, Any]) -> _PreparedCall:
        try:
            operation = build_operation(tool_call, self._registry_for(agent))
            pre = self._emit_tool_call(agent, operation, tool_call.get("tool_call_id"))
            return self._apply_interception(tool_call, operation, pre)
        except Exception as exc:
            return self._preparation_error(tool_call, exc)

    async def _prepare_single_call_async(self, agent: Any, tool_call: dict[str, Any]) -> _PreparedCall:
        try:
            operation = build_operation(tool_call, self._registry_for(agent))
        except Exception as exc:
            return self._preparation_error(tool_call, exc)
        pre = await self._emit_tool_call_async(agent, operation, tool_call.get("tool_call_id"))
        return self._apply_interception(tool_call, operation, pre)

    # ------------------------------------------------------------------
    # Single-call dispatch (sync)
    # ------------------------------------------------------------------

    @observable
    def _execute_single_call(self, agent: Any, tool_call: dict[str, Any], *, _prepared: _PreparedCall | None = None) -> dict[str, Any]:
        """Dispatch one tool call synchronously, with EventBus interception (M3).

        Emits ``tool_call`` (before) and ``tool_result`` (after). A ``tool_call``
        handler may block (``{"block": True, "reason": ...}``) or modify arguments
        (``{"modify": {"arguments": ...}}``); a ``tool_result`` handler may modify
        the result (``{"modify": new_result}``). See
        sprint/plans/S3-tool-execution-engine.md.

        Never raises: the bus never raises (S1) and dispatch is covered by the
        outer guard. An escaping exception would abort the surrounding batch loop.
        """
        function_name = tool_call.get("function", "")
        try:
            prepared = _prepared if _prepared is not None else self._prepare_single_call(agent, tool_call)
            if prepared.error is not None:
                return prepared.error
            arguments = prepared.call["arguments"]
            operation = prepared.operation
            tool_call_id = prepared.call.get("tool_call_id")
            if prepared.block_reason is not None:
                return create_tool_error("policy_block", function_name, prepared.block_reason)

            result = self._dispatch_single_call(agent, function_name, arguments)

            # --- tool_result event (after dispatch) ---
            post = self._emit_tool_result(agent, operation, tool_call_id, result)
            if isinstance(post, dict) and isinstance(post.get("modify"), dict):
                result = post["modify"]
            elif isinstance(post, dict) and "modify" in post:
                logger.warning(
                    "tool_result handler returned non-dict modify (tool=%s); ignored",
                    function_name,
                )
            return result
        except Exception as exc:
            return create_tool_error(
                "execution_error",
                function_name,
                f"Error executing call {function_name}: {exc}\n{traceback.format_exc()}",
            )

    def _dispatch_single_call(self, agent: Any, function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Pure dispatch of one resolved tool call (sync). Never emits.

        Extracted from ``_execute_single_call`` so the emit orchestration can wrap
        it. Raises propagate to the caller's never-raise guard. Dispatch logic is
        intentionally NOT merged with the async variant (only emit is shared).

        Dispatch order:
        1. ToolCatalog (if wired) — primary path per ADR-004.
        2. @named_tool registry — legacy fast path.
        3. Standard name parsing fallback.

        This is the legacy synchronous entry point: blocking work stays on the
        caller's own thread by design (a synchronous caller has no loop to stall).
        """
        # --- ToolCatalog fast path (ADR-004) ---
        if self._tool_catalog is not None:
            entry = self._tool_catalog.get(function_name)
            if entry is not None:
                if not callable(entry.adapter):
                    # Classification-only entries are never executable (the native
                    # catalog's adapter is a no-op that must not become a tool result).
                    return create_tool_error(
                        "catalog_entry_not_executable",
                        function_name,
                        f"Catalog entry '{function_name}' has no callable adapter; it is registered for classification only.",
                    )
                result = entry.adapter(arguments)
                if isinstance(result, dict) and "success" in result:
                    return result
                return create_tool_success("resource", function_name, result)

        registry = self._registry_for(agent)

        # --- @named_tool registry fast path ---
        if function_name in registry:
            obj, method_name = registry[function_name]
            method = getattr(obj, method_name)
            arguments = validate_and_cast_method_arguments(method, arguments)
            if asyncio.iscoroutinefunction(method):
                result = Misc.safe_asyncio_run(method, **arguments)
            else:
                result = method(**arguments)
            return create_tool_success("resource", function_name, result)

        # --- Standard name parsing fallback ---
        parsed = parse_function_name(function_name)
        if not parsed:
            return create_tool_error("format_error", function_name, "Expected ClassName:methodName or object_id__method format")

        identifier, method_name = parsed
        obj_info = find_object_by_id(agent, identifier) or find_object_by_class_name(agent, identifier)
        if not obj_info:
            available = get_available_class_names(agent)
            return create_tool_error(
                "class_not_found",
                identifier,
                "Object not found by object_id or class_name. Available classes: "
                + ", ".join(available[:10])
                + ("..." if len(available) > 10 else ""),
            )

        if hasattr(obj_info["object"], method_name):
            method = getattr(obj_info["object"], method_name)
            arguments = validate_and_cast_method_arguments(method, arguments)
            # Inject session_id for agent calls
            if obj_info["type"] == "agent":
                arguments = self._inject_session_id(agent, arguments)
            if asyncio.iscoroutinefunction(method):
                result = Misc.safe_asyncio_run(method, **arguments)
            else:
                result = method(**arguments)
            return create_tool_success(obj_info["type"], f"{identifier}.{method_name}", result)

        return create_tool_error(
            "method_not_found",
            f"{identifier}.{method_name}",
            f"Method '{method_name}' not found in object '{identifier}'",
        )

    # ------------------------------------------------------------------
    # Single-call dispatch (async)
    # ------------------------------------------------------------------

    @observable
    async def _execute_single_call_async(
        self, agent: Any, tool_call: dict[str, Any], *, _prepared: _PreparedCall | None = None
    ) -> dict[str, Any]:
        """Dispatch one tool call asynchronously, with EventBus interception (M3).

        Async counterpart of ``_execute_single_call``: same event flow, but uses
        ``await bus.emit`` (via the async emit helpers) so handlers run on the
        same loop as dispatch.

        The EventBus never raises (S1), so it cannot enforce durability. The
        session's durable boundary is :meth:`set_execution_hooks`: ``before`` is
        awaited after interception + admission and immediately before dispatch
        (outside the never-raise dispatch guard), and ``after`` for every
        terminal outcome including policy blocks. A hook failure propagates —
        unlike the best-effort observer.
        """
        prepared = _prepared if _prepared is not None else await self._prepare_single_call_async(agent, tool_call)
        if prepared.error is not None:
            return prepared.error
        effective_call = prepared.call
        function_name = effective_call.get("function", "")
        arguments = effective_call["arguments"]
        tool_call_id = effective_call.get("tool_call_id")
        operation = prepared.operation
        block_reason = prepared.block_reason

        if block_reason is not None:
            # Interception already refused the call: the "before" hook still runs
            # (so the session records requested/authorized) but the call is NOT
            # dispatched, so no admission permit is taken and no "started" fact is
            # truthfully recordable.
            await self._run_hook(self._before_hook, effective_call, block_reason)
            result = create_tool_error("policy_block", function_name, block_reason)
            await self._run_hook(self._after_hook, effective_call, result)
            return result

        # Outcome accounting for every terminal path. ``propagate`` re-raises the
        # original exception AFTER the terminal fact is durable, so a cancellation
        # is neither swallowed nor reported without a truthful terminal marker.
        result: dict[str, Any] | None = None
        propagate: BaseException | None = None

        def _cancel_marker(effect_unknown: bool) -> dict[str, Any]:
            """Truthful terminal shape for an interrupted call."""
            marker = {
                "success": False,
                "type": "cancelled",
                "target": function_name,
                "result": (
                    "Error: Cancelled during execution; effect not confirmed"
                    if effect_unknown
                    else "Error: Cancelled before execution; the tool had no effect"
                ),
            }
            marker["_effect_unknown" if effect_unknown else "_cancelled"] = True
            return marker

        kind = self._admission.classify(function_name, arguments)
        try:
            permit = await self._admission.acquire_kind(kind)
        except asyncio.CancelledError as exc:
            # Cancelled while queued for a permit: never dispatched, so it had no
            # effect. The truthful marker is the acknowledged/no-effect form.
            result = _cancel_marker(effect_unknown=False)
            propagate = exc
        except Exception as exc:
            result = create_tool_error(
                "execution_error",
                function_name,
                f"Error awaiting admission for {function_name}: {exc}\n{traceback.format_exc()}",
            )
        else:
            try:
                # "Before" runs AFTER the permit is acquired, immediately before
                # dispatch, so the session's TOOL_STARTED fact is truthful (the call
                # is admitted and executing) rather than only queued. It sits
                # OUTSIDE the dispatch error guard: a failed durable append must
                # propagate rather than become a tool result.
                #
                # It is also the single decision point for queue cancellation: the
                # session re-checks its cancel event after appending TOOL_STARTED and
                # raises ``CancelledError`` (having journaled the acknowledgement)
                # when the turn was cancelled during the durable commit. The executor
                # therefore never short-circuits the hook, which would lose that
                # call's requested/authorized facts.
                await self._run_hook(self._before_hook, effective_call, None)
            except asyncio.CancelledError as exc:
                # Queue cancellation decided by the session's own hook (e.g. the
                # turn was cancelled after this call was admitted but before it
                # ran): no dispatch, no effect. The terminal marker is still written
                # through ``after`` so no caller sees a cancellation without a fact.
                result = _cancel_marker(effect_unknown=False)
                propagate = exc
            except Exception:
                # The session's before-hook refused or failed (policy denial, failed
                # durable append). The hook is the session's authoritative journal
                # boundary and owns the terminal fact in that case, so the executor
                # records nothing further and propagates the failure unchanged — a
                # durable append that failed must never be swallowed.
                raise
            else:
                if propagate is None:
                    try:
                        result = await self._dispatch_single_call_async(
                            agent, function_name, arguments, tool_call_id=tool_call_id, admission_kind=kind
                        )
                    except asyncio.CancelledError as exc:
                        # Cancelled *during* dispatch. The tool may already have
                        # committed part of its effect and no acknowledgement was
                        # observed, so the truthful disposition is effect-unknown (never
                        # auto-retry, ADR-011) — not a fabricated "cancelled".
                        result = _cancel_marker(effect_unknown=True)
                        propagate = exc
                    except Exception as exc:
                        result = create_tool_error(
                            "execution_error",
                            function_name,
                            f"Error executing call {function_name}: {exc}\n{traceback.format_exc()}",
                        )
            finally:
                permit.release()

        # --- tool_result event (after dispatch) ---
        post = await self._emit_tool_result_async(agent, operation, tool_call_id, result)
        if isinstance(post, dict) and isinstance(post.get("modify"), dict):
            result = post["modify"]
        elif isinstance(post, dict) and "modify" in post:
            logger.warning(
                "tool_result handler returned non-dict modify (tool=%s); ignored",
                function_name,
            )

        # Terminal fact is durable before the executor returns (and before the
        # original cancellation is re-raised), so the journal terminal precedes the
        # turn's terminal.
        await self._run_hook(self._after_hook, effective_call, result)
        if propagate is not None:
            raise propagate
        return result

    async def _dispatch_single_call_async(
        self,
        agent: Any,
        function_name: str,
        arguments: dict[str, Any],
        tool_call_id: Any = None,
        admission_kind: str = "leaf",
    ) -> dict[str, Any]:
        """Pure dispatch of one resolved tool call (async). Never emits.

        Dispatch logic is intentionally NOT merged with the sync variant (only
        emit is shared). Raises propagate to the caller's never-raise guard.

        Dispatch order:
        0. MCP per-tool dispatch map (D7 follow-up 1+2, A2) — namespaced MCP tools.
        1. ToolCatalog (if wired) — primary path per ADR-004.
        2. @named_tool registry — legacy fast path.
        3. Standard name parsing fallback.

        Args:
            tool_call_id: The harness tool_call_id, forwarded to MCP dispatch so
                the real id (not a generated stand-in) reaches per-call tracking
                and the transport request-id record.
            admission_kind: The pool holding this call's permit (``"leaf"`` or
                ``"nested"``); the nested value tags the result so nested work is
                observable without inspecting the scheduler.
        """
        # --- MCP per-tool dispatch (D7 follow-up 1+2, A2) ---
        # Namespaced MCP tool names (server:tool) do not collide with native
        # @named_tool names, so checking here is additive + safe.
        if self._mcp_dispatch_getter is not None:
            mcp_map = self._mcp_dispatch_getter()
            if mcp_map and function_name in mcp_map:
                dispatcher = mcp_map[function_name]
                dispatched = await self._invoke_mcp_dispatcher(dispatcher, arguments, tool_call_id)
                result = (
                    dispatched
                    if isinstance(dispatched, dict) and "success" in dispatched
                    else create_tool_success("mcp", function_name, dispatched)
                )
                self._tag_admission(result, admission_kind)
                return result

        # --- ToolCatalog fast path (ADR-004) ---
        if self._tool_catalog is not None:
            entry = self._tool_catalog.get(function_name)
            if entry is not None:
                if not callable(entry.adapter):
                    # Classification-only entries (native catalog, and MCP entries
                    # whose adapter the session has not wired) are NOT executable.
                    # Refuse explicitly instead of dispatching a no-op.
                    return create_tool_error(
                        "catalog_entry_not_executable",
                        function_name,
                        f"Catalog entry '{function_name}' has no callable adapter; it is registered for classification only.",
                    )
                if asyncio.iscoroutinefunction(entry.adapter):
                    result = await entry.adapter(arguments)
                elif is_loop_affine(entry.adapter):
                    result = entry.adapter(arguments)
                else:
                    # Synchronous adapter: keep its blocking work off the loop.
                    result = await run_blocking(entry.adapter, arguments)
                if isinstance(result, dict) and "success" in result:
                    final = result
                else:
                    final = create_tool_success("resource", function_name, result)
                self._tag_admission(final, admission_kind)
                return final

        registry = self._registry_for(agent)

        # --- @named_tool registry fast path ---
        if function_name in registry:
            obj, method_name = registry[function_name]
            method = getattr(obj, method_name)
            arguments = validate_and_cast_method_arguments(method, arguments)
            if asyncio.iscoroutinefunction(method):
                result = await method(**arguments)
            elif is_loop_affine(method):
                result = method(**arguments)  # declared thread-affine: keep on the loop
            else:
                # Synchronous method containing blocking/filesystem work: run it on
                # the bounded offload pool so it cannot stall the event loop. The
                # registry's synchronous API keeps working unchanged.
                result = await run_blocking(method, **arguments)
            final = create_tool_success("resource", function_name, result)
            self._tag_admission(final, admission_kind)
            return final

        # --- Standard name parsing fallback ---
        parsed = parse_function_name(function_name)
        if not parsed:
            return create_tool_error("format_error", function_name, "Expected ClassName:methodName or object_id__method format")

        identifier, method_name = parsed
        obj_info = find_object_by_id(agent, identifier) or find_object_by_class_name(agent, identifier)
        if not obj_info:
            available = get_available_class_names(agent)
            return create_tool_error(
                "class_not_found",
                identifier,
                "Object not found by object_id or class_name. Available classes: "
                + ", ".join(available[:10])
                + ("..." if len(available) > 10 else ""),
            )

        # For async agent calls, prefer aquery over query
        actual_method_name = method_name
        if obj_info["type"] == "agent" and method_name == "query":
            actual_method_name = "aquery"

        if hasattr(obj_info["object"], actual_method_name):
            method = getattr(obj_info["object"], actual_method_name)
            arguments = validate_and_cast_method_arguments(method, arguments)
            if obj_info["type"] == "agent":
                arguments = self._inject_session_id(agent, arguments)
            if asyncio.iscoroutinefunction(method):
                result = await method(**arguments)
            elif is_loop_affine(method):
                result = method(**arguments)
            else:
                result = await run_blocking(method, **arguments)
            final = create_tool_success(obj_info["type"], f"{identifier}.{actual_method_name}", result)
            self._tag_admission(final, admission_kind)
            return final

        return create_tool_error(
            "method_not_found",
            f"{identifier}.{actual_method_name}",
            f"Method '{actual_method_name}' not found in object '{identifier}'",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _registry_for(self, agent: Any) -> dict[str, tuple[Any, str]]:
        """Return the @named_tool registry valid for the *invoking* ``agent``.

        Never reads executor- or runtime-wide "current agent" state: on a shared
        runtime an overlapping invocation for another agent may have replaced it
        at any await. The agent travels with the call, so it is the only correct
        selector.
        """
        if self._agent_tool_name_registry_getter is not None:
            return self._agent_tool_name_registry_getter(agent)
        if self._tool_name_registry_getter is not None:
            return self._tool_name_registry_getter()
        return {}

    def _inject_session_id(self, agent: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        """Inject session_id into arguments when the calling agent has an active event log."""
        if hasattr(agent, "_event_log") and agent._event_log is not None:
            session_id = agent._event_log._current_session_id
            if session_id is not None:
                arguments = dict(arguments)
                arguments["session_id"] = session_id
        return arguments

    # ------------------------------------------------------------------
    # EventBus emit helpers (M3). Guard on a REAL EventBus instance: bare
    # MagicMock agents (existing tests) auto-create a non-awaitable
    # ``event_bus`` attribute, so an ``is None`` check would route through the
    # mock and crash the async path. ``isinstance`` treats those as no-bus.
    # ------------------------------------------------------------------

    def _emit_tool_call(self, agent: Any, operation: Any, tool_call_id: Any) -> dict[str, Any] | None:
        bus = getattr(agent, "event_bus", None)
        if not isinstance(bus, EventBus):
            return None
        return bus.emit_sync(Event(TOOL_CALL, {"tool_call_id": tool_call_id, "operation": operation}))

    async def _emit_tool_call_async(self, agent: Any, operation: Any, tool_call_id: Any) -> dict[str, Any] | None:
        bus = getattr(agent, "event_bus", None)
        if not isinstance(bus, EventBus):
            return None
        return await bus.emit(Event(TOOL_CALL, {"tool_call_id": tool_call_id, "operation": operation}))

    def _emit_tool_result(self, agent: Any, operation: Any, tool_call_id: Any, result: dict[str, Any]) -> dict[str, Any] | None:
        bus = getattr(agent, "event_bus", None)
        if not isinstance(bus, EventBus):
            return None
        return bus.emit_sync(Event(TOOL_RESULT, {"tool_call_id": tool_call_id, "operation": operation, "result": result}))

    async def _emit_tool_result_async(self, agent: Any, operation: Any, tool_call_id: Any, result: dict[str, Any]) -> dict[str, Any] | None:
        bus = getattr(agent, "event_bus", None)
        if not isinstance(bus, EventBus):
            return None
        return await bus.emit(Event(TOOL_RESULT, {"tool_call_id": tool_call_id, "operation": operation, "result": result}))
