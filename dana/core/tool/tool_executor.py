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
import traceback
from typing import Any

import structlog

from dana.common.observable import observable
from dana.common.utils.misc import Misc
from dana.core.ext.event_bus import Event, EventBus
from dana.core.ext.events import TOOL_CALL, TOOL_RESULT
from dana.core.ext.operation import build_operation
from dana.core.tool.catalog import ToolCatalog
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


class ToolExecutor:
    """Executes tool calls on behalf of an agent (sync and async).

    Implements ToolExecutorProtocol.

    Constructor args:
        agent_getter:             Callable[[], agent] — returns the current agent
                                  (used only when caller does not pass agent directly).
        tool_name_registry_getter: Callable[[], dict] — returns the @named_tool registry
                                  maintained on AgentRuntime.

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
    ) -> None:
        self._agent_getter = agent_getter
        self._tool_name_registry_getter = tool_name_registry_getter
        self._max_workers = max_workers
        self._tool_catalog = tool_catalog
        # D7 follow-up 1+2: per-tool MCP dispatch map (tool_name -> async callable).
        self._mcp_dispatch_getter = mcp_dispatch_getter

    def set_mcp_dispatch_getter(self, getter: Callable[[], dict[str, Any]] | None) -> None:
        """Wire the MCP dispatch map getter (per-tool MCP UX, A2).

        The getter returns ``{tool_name: async callable(arguments) -> str}``;
        ``None`` disables MCP per-tool dispatch (fall through to the registry).
        """
        self._mcp_dispatch_getter = getter

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
        """Sync batch executor — sequential by default, parallel via ThreadPoolExecutor when parallel=True."""
        if parallel:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = [executor.submit(self._execute_single_call, agent, call) for call in tool_calls]
                results = [f.result() for f in futures]
        else:
            results = []
            for call in tool_calls:
                results.append(self._execute_single_call(agent, call))

        for result, call in zip(results, tool_calls, strict=False):
            if "tool_call_id" in call:
                result["tool_call_id"] = call["tool_call_id"]
        return results

    @observable
    async def execute_tools_async(self, agent: Any, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Async batch executor — runs all tool calls concurrently via asyncio.gather.

        return_exceptions=True so one call's failure cannot abort the batch:
        every tool_call_id must get a result, or the next OpenAI turn 400s on
        an unanswered tool call. _execute_single_call_async is already
        non-raising; this is defense-in-depth for anything it misses.
        """
        raw = await asyncio.gather(
            *[self._execute_single_call_async(agent, call) for call in tool_calls],
            return_exceptions=True,
        )
        results: list[dict[str, Any]] = []
        for result, call in zip(raw, tool_calls, strict=False):
            if isinstance(result, BaseException):
                result = create_tool_error(
                    "execution_error",
                    call.get("function", ""),
                    f"Unhandled error executing call: {result}",
                )
            if "tool_call_id" in call:
                result["tool_call_id"] = call["tool_call_id"]
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Single-call dispatch (sync)
    # ------------------------------------------------------------------

    @observable
    def _execute_single_call(self, agent: Any, tool_call: dict[str, Any]) -> dict[str, Any]:
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
        arguments = tool_call.get("arguments", {})
        try:
            registry = self._get_registry()
            operation = build_operation(tool_call, registry)
            tool_call_id = tool_call.get("tool_call_id")

            # --- tool_call event (before dispatch) ---
            pre = self._emit_tool_call(agent, operation, tool_call_id)
            if isinstance(pre, dict) and pre.get("block") is True:
                return create_tool_error("policy_block", function_name, str(pre.get("reason", "blocked")))
            if isinstance(pre, dict) and isinstance(pre.get("modify"), dict):
                new_arguments = pre["modify"].get("arguments")
                if isinstance(new_arguments, dict):
                    arguments = new_arguments

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
        """
        # --- ToolCatalog fast path (ADR-004) ---
        if self._tool_catalog is not None:
            entry = self._tool_catalog.get(function_name)
            if entry is not None:
                result = entry.adapter(arguments)
                if isinstance(result, dict) and "success" in result:
                    return result
                return create_tool_success("resource", function_name, result)

        registry = self._get_registry()

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
    async def _execute_single_call_async(self, agent: Any, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one tool call asynchronously, with EventBus interception (M3).

        Async counterpart of ``_execute_single_call``: same event flow, but uses
        ``await bus.emit`` (via the async emit helpers) so handlers run on the
        same loop as dispatch. Never raises (bus S1 + outer guard).
        """
        function_name = tool_call.get("function", "")
        arguments = tool_call.get("arguments", {})
        try:
            registry = self._get_registry()
            operation = build_operation(tool_call, registry)
            tool_call_id = tool_call.get("tool_call_id")

            # --- tool_call event (before dispatch) ---
            pre = await self._emit_tool_call_async(agent, operation, tool_call_id)
            if isinstance(pre, dict) and pre.get("block") is True:
                return create_tool_error("policy_block", function_name, str(pre.get("reason", "blocked")))
            if isinstance(pre, dict) and isinstance(pre.get("modify"), dict):
                new_arguments = pre["modify"].get("arguments")
                if isinstance(new_arguments, dict):
                    arguments = new_arguments

            result = await self._dispatch_single_call_async(agent, function_name, arguments)

            # --- tool_result event (after dispatch) ---
            post = await self._emit_tool_result_async(agent, operation, tool_call_id, result)
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

    async def _dispatch_single_call_async(self, agent: Any, function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Pure dispatch of one resolved tool call (async). Never emits.

        Dispatch logic is intentionally NOT merged with the sync variant (only
        emit is shared). Raises propagate to the caller's never-raise guard.

        Dispatch order:
        0. MCP per-tool dispatch map (D7 follow-up 1+2, A2) — namespaced MCP tools.
        1. ToolCatalog (if wired) — primary path per ADR-004.
        2. @named_tool registry — legacy fast path.
        3. Standard name parsing fallback.
        """
        # --- MCP per-tool dispatch (D7 follow-up 1+2, A2) ---
        # Namespaced MCP tool names (server:tool) do not collide with native
        # @named_tool names, so checking here is additive + safe.
        if self._mcp_dispatch_getter is not None:
            mcp_map = self._mcp_dispatch_getter()
            if mcp_map and function_name in mcp_map:
                formatted = await mcp_map[function_name](arguments)
                return create_tool_success("mcp", function_name, formatted)

        # --- ToolCatalog fast path (ADR-004) ---
        if self._tool_catalog is not None:
            entry = self._tool_catalog.get(function_name)
            if entry is not None:
                result = entry.adapter(arguments)
                if isinstance(result, dict) and "success" in result:
                    return result
                return create_tool_success("resource", function_name, result)

        registry = self._get_registry()

        # --- @named_tool registry fast path ---
        if function_name in registry:
            obj, method_name = registry[function_name]
            method = getattr(obj, method_name)
            arguments = validate_and_cast_method_arguments(method, arguments)
            if asyncio.iscoroutinefunction(method):
                result = await method(**arguments)
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
            else:
                result = method(**arguments)
            return create_tool_success(obj_info["type"], f"{identifier}.{actual_method_name}", result)

        return create_tool_error(
            "method_not_found",
            f"{identifier}.{actual_method_name}",
            f"Method '{actual_method_name}' not found in object '{identifier}'",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_registry(self) -> dict[str, tuple[Any, str]]:
        """Return the @named_tool registry, or empty dict if not wired."""
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
