"""Tool Execution Engine — cancellation-first, cooperative latency contract (D2).

Per ADR-005 (Cancellation-First Tool Execution Engine):
- Every tool goes through the engine.
- Never report ``cancelled`` because a future was abandoned.
- Cooperative tools declare max cancellation latency (no hidden default).
- Non-cooperative tools use isolated worker (process groups/jobs/reap).
- Child ownership defaults to ``cascade``; ``detach`` requires Durable Job handoff.

Per ADR-004 (Stable Tool Identity and Versioned Catalog):
- Catalog entries declare cancellation capability + max latency at registration.
- Engine reads this from the catalog, not from ad hoc config.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any
from uuid import uuid4

import structlog

from dana.core.mcp.cancellation import MCPCancellationTracker
from dana.core.mcp.execution import MCPExecutionAdapter
from dana.core.tool.catalog import ToolCatalog, ToolCatalogEntry
from dana.core.tool.tool_executor_helpers import create_tool_error, create_tool_success
from dana.core.tool.worker import (
    WorkerProcessManager,
    WorkerRequest,
    WorkerStatus,
)


logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# In-flight tool tracking
# ---------------------------------------------------------------------------


class _InFlight:
    """Tracks a tool call that is currently executing.

    ``tool_call_id`` — the tool_call_id from the request.
    ``entry``        — the catalog entry for the tool.
    ``started_at``   — monotonic timestamp when execution began.
    ``cancelled``    — whether cancellation was requested.
    ``result``       — the result once execution completes (or None).
    """

    __slots__ = ("tool_call_id", "entry", "started_at", "_cancelled", "_result")

    def __init__(self, tool_call_id: str, entry: ToolCatalogEntry) -> None:
        self.tool_call_id = tool_call_id
        self.entry = entry
        self.started_at = time.monotonic()
        self._cancelled = False
        self._result: dict[str, Any] | None = None

    def cancel(self) -> None:
        """Request cancellation of this in-flight tool."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def set_result(self, result: dict[str, Any]) -> None:
        self._result = result

    @property
    def result(self) -> dict[str, Any] | None:
        return self._result


# ---------------------------------------------------------------------------
# Tool Execution Engine
# ---------------------------------------------------------------------------


class ToolExecutionEngine:
    """Cancellation-first tool execution engine.

    Routes every tool call through the engine. Cooperative tools run in-process
    with a cancellation contract. Non-cooperative (isolated) tools run in a
    separate worker process with process-group cleanup.

    Args:
        tool_catalog: The session-owned ToolCatalog (ADR-004).
        max_workers: Max concurrent cooperative tool calls (default: no limit).
    """

    def __init__(
        self,
        tool_catalog: ToolCatalog,
        max_workers: int | None = None,
    ) -> None:
        self._catalog = tool_catalog
        self._max_workers = max_workers

        # Worker managers keyed by worker module (shared across calls to same module)
        self._worker_managers: dict[str, WorkerProcessManager] = {}

        # In-flight tracking
        self._in_flight: dict[str, _InFlight] = {}
        # Tracked MCP cancellation notifications: retained until completion so a
        # cancellation task is never abandoned (and can be awaited at teardown).
        self._cancel_tasks: set[asyncio.Task[Any]] = set()

        # Remote adapters (e.g. MCP) keyed by server name
        self._remote_adapters: dict[str, MCPExecutionAdapter] = {}

        # Cancellation tracker shared across remote adapters
        self._mcp_cancellation_tracker = MCPCancellationTracker()

    # ------------------------------------------------------------------
    # Remote adapter registration (D5 — MCP integration)
    # ------------------------------------------------------------------

    def register_remote_adapter(
        self,
        server_name: str,
        adapter: MCPExecutionAdapter,
    ) -> None:
        """Register a remote execution adapter (e.g. MCP).

        Remote adapters provide callable tool execution for tools that
        run on external servers. The engine routes tool calls to the
        appropriate adapter based on the catalog entry's source.

        Args:
            server_name: The server name (e.g. MCP server name).
            adapter: The ``MCPExecutionAdapter`` instance.
        """
        self._remote_adapters[server_name] = adapter
        logger.info("remote_adapter_registered", server_name=server_name)

    def unregister_remote_adapter(self, server_name: str) -> None:
        """Unregister a remote execution adapter.

        Args:
            server_name: The server name to unregister.
        """
        self._remote_adapters.pop(server_name, None)
        logger.info("remote_adapter_unregistered", server_name=server_name)

    @property
    def remote_adapters(self) -> dict[str, MCPExecutionAdapter]:
        """Registered remote adapters (copy)."""
        return dict(self._remote_adapters)

    @property
    def mcp_cancellation_tracker(self) -> MCPCancellationTracker:
        """The MCP cancellation tracker."""
        return self._mcp_cancellation_tracker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        tool_call: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single tool call synchronously.

        Routes to the cooperative or isolated path based on the catalog entry.
        Never raises: all errors are returned as tool error dicts.
        """
        function_name = tool_call.get("function", "")
        tool_call_id = tool_call.get("tool_call_id", str(uuid4()))

        entry = self._catalog.get(function_name)
        if entry is None:
            return create_tool_error(
                "not_found",
                function_name,
                f"Tool '{function_name}' not found in catalog",
            )

        try:
            if entry.isolated:
                in_flight = _InFlight(tool_call_id, entry)
                self._in_flight[tool_call_id] = in_flight
                return self._execute_isolated_sync(entry, tool_call, tool_call_id, in_flight)
            # D5: Route MCP tools to remote adapter
            if entry.identity.source and entry.identity.source.startswith("mcp:"):
                server_name = entry.identity.source[len("mcp:") :]
                adapter = self._remote_adapters.get(server_name)
                if adapter is not None:
                    # The adapter API takes the tool_call dict + the harness id;
                    # passing the dict as the *name* (and the id as arguments) would
                    # call a tool literally named "{...}" and lose per-call tracking.
                    from dana.common.utils.misc import Misc

                    in_flight = _InFlight(tool_call_id, entry)
                    self._in_flight[tool_call_id] = in_flight
                    try:
                        result = Misc.safe_asyncio_run(adapter.call_tool_from_dict, tool_call, tool_call_id)
                        in_flight.set_result(result)
                        return result
                    finally:
                        self._in_flight.pop(tool_call_id, None)
                return create_tool_error(
                    "remote_adapter_not_found",
                    function_name,
                    f"No remote adapter registered for MCP server '{server_name}'",
                )
            return self._execute_cooperative_sync(entry, tool_call, tool_call_id)
        except Exception as exc:
            return create_tool_error(
                "execution_error",
                function_name,
                f"Error executing {function_name}: {exc}\n{traceback.format_exc()}",
            )

    async def execute_async(
        self,
        tool_call: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single tool call asynchronously.

        Routes to the cooperative or isolated path based on the catalog entry.
        Never raises: all errors are returned as tool error dicts.
        """
        function_name = tool_call.get("function", "")
        tool_call_id = tool_call.get("tool_call_id", str(uuid4()))

        entry = self._catalog.get(function_name)
        if entry is None:
            return create_tool_error(
                "not_found",
                function_name,
                f"Tool '{function_name}' not found in catalog",
            )

        try:
            if entry.isolated:
                in_flight = _InFlight(tool_call_id, entry)
                self._in_flight[tool_call_id] = in_flight
                return await self._execute_isolated_async(entry, tool_call, tool_call_id, in_flight)
            # D5: Route MCP tools to remote adapter
            if entry.identity.source and entry.identity.source.startswith("mcp:"):
                server_name = entry.identity.source[len("mcp:") :]
                adapter = self._remote_adapters.get(server_name)
                if adapter is not None:
                    # Route the whole tool_call through the adapter's dict entry
                    # point so the harness tool_call_id (not a stand-in) keys
                    # per-call tracking and the recorded transport request id.
                    in_flight = _InFlight(tool_call_id, entry)
                    self._in_flight[tool_call_id] = in_flight
                    try:
                        result = await adapter.call_tool_from_dict(tool_call, tool_call_id)
                        in_flight.set_result(result)
                        return result
                    finally:
                        self._in_flight.pop(tool_call_id, None)
                return create_tool_error(
                    "remote_adapter_not_found",
                    function_name,
                    f"No remote adapter registered for MCP server '{server_name}'",
                )
            return await self._execute_cooperative_async(entry, tool_call, tool_call_id)
        except Exception as exc:
            return create_tool_error(
                "execution_error",
                function_name,
                f"Error executing {function_name}: {exc}\n{traceback.format_exc()}",
            )

    def execute_batch(
        self,
        tool_calls: list[dict[str, Any]],
        parallel: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute a batch of tool calls synchronously.

        When ``parallel=True``, uses a ThreadPoolExecutor for cooperative tools.
        Isolated tools always run in their own worker process regardless.
        """
        if parallel:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = [executor.submit(self.execute, call) for call in tool_calls]
                results = [f.result() for f in futures]
        else:
            results = [self.execute(call) for call in tool_calls]

        for result, call in zip(results, tool_calls, strict=False):
            if "tool_call_id" in call:
                result["tool_call_id"] = call["tool_call_id"]
        return results

    async def execute_batch_async(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute a batch of tool calls asynchronously.

        Runs all calls concurrently via asyncio.gather with return_exceptions=True
        so one failure cannot abort the batch.
        """
        raw = await asyncio.gather(
            *[self.execute_async(call) for call in tool_calls],
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
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self, tool_call_id: str) -> None:
        """Request cancellation of an in-flight tool.

        For cooperative tools: sets the cancellation flag. The tool is expected
        to check this flag and stop within ``max_latency_ms``.

        For isolated tools: kills the worker process group.

        For remote (MCP) tools: sends cancellation notification to the remote
        server. Per ADR-005, cancellation is terminal only after the remote
        system acknowledges it.

        Never reports ``cancelled`` because a future was abandoned — only when
        cancellation was actually requested and confirmed.
        """
        in_flight = self._in_flight.get(tool_call_id)
        if in_flight is None:
            logger.warning("cancel_ignored", tool_call_id=tool_call_id, reason="not_in_flight")
            return

        in_flight.cancel()
        entry = in_flight.entry

        if entry.isolated:
            # Kill the worker process group
            worker_id = self._worker_id_for(entry)
            manager = self._worker_managers.get(worker_id)
            if manager is not None:
                manager.kill_process_group()
                logger.info("cancel_isolated", tool_call_id=tool_call_id, worker_id=worker_id)
        else:
            # Check if this is a remote (MCP) tool
            source = entry.identity.source or ""
            if source.startswith("mcp:"):
                server_name = source[4:]  # Strip "mcp:" prefix
                adapter = self._remote_adapters.get(server_name)
                if adapter is not None:
                    # Send the cancellation notification without blocking the caller
                    # (the remote's acknowledgement arrives on the tool response,
                    # which the executing call observes). The task is retained so it
                    # is never abandoned: a strong reference keeps it alive until
                    # completion, its exception is consumed, and :meth:`aclose` can
                    # await it during orderly teardown.
                    self._spawn_cancel_notification(adapter, server_name, tool_call_id)
                    logger.info(
                        "cancel_mcp",
                        tool_call_id=tool_call_id,
                        server_name=server_name,
                    )
                else:
                    logger.warning(
                        "cancel_mcp_no_adapter",
                        tool_call_id=tool_call_id,
                        server_name=server_name,
                    )
            else:
                logger.info(
                    "cancel_cooperative",
                    tool_call_id=tool_call_id,
                    max_latency_ms=entry.max_latency_ms,
                )

    # ------------------------------------------------------------------
    # Cooperative execution
    # ------------------------------------------------------------------

    def _execute_cooperative_sync(
        self,
        entry: ToolCatalogEntry,
        tool_call: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any]:
        """Execute a cooperative tool synchronously.

        Cooperative tools run in-process via the catalog adapter. The engine
        enforces the max cancellation latency contract: if cancellation is
        requested, the engine waits up to ``max_latency_ms`` for the tool to
        notice and stop. If the tool exceeds its declared latency, the engine
        logs a contract violation but still returns the result (cooperative
        tools cannot be force-killed in-process).
        """
        in_flight = _InFlight(tool_call_id, entry)
        self._in_flight[tool_call_id] = in_flight

        try:
            result = entry.adapter(tool_call.get("arguments", {}))
            if isinstance(result, dict) and "success" in result:
                final = result
            else:
                final = create_tool_success("resource", entry.identity.name, result)

            # A latency budget is diagnostic, never proof of cancellation.
            if in_flight.is_cancelled:
                elapsed = time.monotonic() - in_flight.started_at
                max_latency = (entry.max_latency_ms or 0) / 1000.0
                if elapsed > max_latency:
                    logger.warning(
                        "cooperative_latency_violation",
                        tool=entry.identity.name,
                        tool_call_id=tool_call_id,
                        elapsed_ms=round(elapsed * 1000),
                        max_latency_ms=entry.max_latency_ms,
                    )

            in_flight.set_result(final)
            return final
        except Exception as exc:
            if not in_flight.is_cancelled:
                raise
            final = create_tool_error(
                "execution_error",
                entry.identity.name,
                f"Error executing {entry.identity.name}: {exc}\n{traceback.format_exc()}",
            )
            final["_effect_unknown"] = True
            in_flight.set_result(final)
            return final
        finally:
            self._in_flight.pop(tool_call_id, None)

    async def _execute_cooperative_async(
        self,
        entry: ToolCatalogEntry,
        tool_call: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any]:
        """Execute a cooperative tool asynchronously.

        Same contract as the sync path but uses the async adapter if available.
        """
        in_flight = _InFlight(tool_call_id, entry)
        self._in_flight[tool_call_id] = in_flight

        try:
            adapter = entry.adapter
            arguments = tool_call.get("arguments", {})

            if asyncio.iscoroutinefunction(adapter):
                result = await adapter(arguments)
            else:
                result = adapter(arguments)

            if isinstance(result, dict) and "success" in result:
                final = result
            else:
                final = create_tool_success("resource", entry.identity.name, result)

            # A completed result remains authoritative, even after cancellation.
            if in_flight.is_cancelled:
                elapsed = time.monotonic() - in_flight.started_at
                max_latency = (entry.max_latency_ms or 0) / 1000.0
                if elapsed > max_latency:
                    logger.warning(
                        "cooperative_latency_violation",
                        tool=entry.identity.name,
                        tool_call_id=tool_call_id,
                        elapsed_ms=round(elapsed * 1000),
                        max_latency_ms=entry.max_latency_ms,
                    )

            in_flight.set_result(final)
            return final
        except Exception as exc:
            if not in_flight.is_cancelled:
                raise
            final = create_tool_error(
                "execution_error",
                entry.identity.name,
                f"Error executing {entry.identity.name}: {exc}\n{traceback.format_exc()}",
            )
            final["_effect_unknown"] = True
            in_flight.set_result(final)
            return final
        finally:
            self._in_flight.pop(tool_call_id, None)

    # ------------------------------------------------------------------
    # Isolated (worker) execution
    # ------------------------------------------------------------------

    @staticmethod
    def _worker_id_for(entry: ToolCatalogEntry) -> str:
        """Derive a worker manager key from a catalog entry."""
        return f"{entry.worker_module}:{entry.worker_callable}"

    def _get_or_create_worker(self, entry: ToolCatalogEntry) -> WorkerProcessManager:
        """Get or create a worker process manager for the given entry."""
        worker_id = self._worker_id_for(entry)
        if worker_id not in self._worker_managers:
            manager = WorkerProcessManager(worker_id=worker_id)
            existing = self._worker_managers.setdefault(worker_id, manager)
            if existing is not manager:
                return existing
        return self._worker_managers[worker_id]

    def _spawn_cancel_notification(
        self,
        adapter: MCPExecutionAdapter,
        server_name: str,
        tool_call_id: str,
    ) -> None:
        """Send an MCP cancellation notification on a tracked, never-abandoned task.

        Requires a running loop (``cancel`` is called from the turn's loop). Without
        one the notification cannot be sent, so the fact is logged rather than
        silently dropped — the call's disposition stays effect-unknown until the
        remote is heard from.
        """
        in_flight = self._in_flight.get(tool_call_id)
        if in_flight is None:
            return
        # Publish cancellation before yielding to the notification task. The
        # transport can fail in this same turn, before that task ever runs.
        adapter.cancellation_tracker.request_cancellation(tool_call_id)

        async def send_if_active() -> None:
            # The call may already have completed and forgotten its tracker state.
            # Do not recreate that state, or cancel a new call reusing the same ID.
            if self._in_flight.get(tool_call_id) is in_flight:
                await adapter.send_cancellation_notification(tool_call_id)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "cancel_mcp_no_loop",
                tool_call_id=tool_call_id,
                server_name=server_name,
            )
            return

        notification = send_if_active()
        task = loop.create_task(notification)
        if not isinstance(task, asyncio.Future):
            # A loop whose ``create_task`` does not return a task cannot have the
            # notification tracked or awaited. Never leak the coroutine: close both
            # the returned object and the coroutine that was handed over, and record
            # that the notification was not sent, so the disposition stays
            # effect-unknown rather than silently assumed cancelled.
            for candidate in (task, notification):
                close = getattr(candidate, "close", None)
                if close is not None:
                    close()
            logger.warning(
                "cancel_notification_unschedulable",
                tool_call_id=tool_call_id,
                server_name=server_name,
            )
            return
        self._cancel_tasks.add(task)

        def _done(finished: asyncio.Task) -> None:
            self._cancel_tasks.discard(finished)
            if finished.cancelled():
                return
            exc = finished.exception()
            if exc is not None:
                logger.warning(
                    "cancel_notification_failed",
                    tool_call_id=tool_call_id,
                    server_name=server_name,
                    error=str(exc),
                )

        task.add_done_callback(_done)

    async def aclose(self) -> None:
        """Await in-flight cancellation notifications, then close the engine.

        Orderly teardown: outstanding MCP cancellation notifications are awaited so
        no task is abandoned and the remote is told about every cancelled call
        before the transports are released.
        """
        if self._cancel_tasks:
            await asyncio.gather(*list(self._cancel_tasks), return_exceptions=True)
            self._cancel_tasks.clear()
        self.close()

    def _execute_isolated_sync(
        self,
        entry: ToolCatalogEntry,
        tool_call: dict[str, Any],
        tool_call_id: str,
        in_flight: _InFlight,
    ) -> dict[str, Any]:
        """Execute an isolated (non-cooperative) tool synchronously.

        The tool runs in a separate worker process with process-group isolation.
        If the worker crashes or times out, the entire process group is killed.
        """
        if in_flight.is_cancelled:
            return create_tool_error(
                "cancelled",
                entry.identity.name,
                "Cancelled before worker started",
            )

        try:
            manager = self._get_or_create_worker(entry)
            arguments = tool_call.get("arguments", {})

            request = WorkerRequest(
                request_id=tool_call_id,
                module=entry.worker_module or "",
                callable=entry.worker_callable or "",
                kwargs=arguments,
                timeout_ms=entry.max_latency_ms or 30000,
            )

            response = manager.send_request(request)

            if in_flight.is_cancelled:
                # The worker process group was killed mid-work for a
                # non-cooperative tool: whether its side effects committed before
                # the kill cannot be confirmed. Report the truthful disposition.
                final = create_tool_error(
                    "cancelled",
                    entry.identity.name,
                    "Cancelled during execution",
                )
                final["_effect_unknown"] = True
            elif response.status == WorkerStatus.SUCCESS:
                final = create_tool_success("resource", entry.identity.name, response.result)
            elif response.status == WorkerStatus.CANCELLED:
                final = create_tool_error(
                    "cancelled",
                    entry.identity.name,
                    str(response.result or "Cancelled"),
                )
            elif response.status == WorkerStatus.CRASHED:
                final = create_tool_error(
                    "worker_crash",
                    entry.identity.name,
                    str(response.result or "Worker crashed"),
                )
            else:
                final = create_tool_error(
                    "execution_error",
                    entry.identity.name,
                    str(response.result or "Unknown error"),
                )

            in_flight.set_result(final)
            return final
        finally:
            self._in_flight.pop(tool_call_id, None)

    async def _execute_isolated_async(
        self,
        entry: ToolCatalogEntry,
        tool_call: dict[str, Any],
        tool_call_id: str,
        in_flight: _InFlight,
    ) -> dict[str, Any]:
        """Execute an isolated tool asynchronously.

        Runs the synchronous isolated path in a thread pool to avoid blocking
        the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._execute_isolated_sync,
            entry,
            tool_call,
            tool_call_id,
            in_flight,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down all worker processes and clean up.

        Must be called when the engine is no longer needed to prevent
        subprocess leaks.
        """
        for worker_id, manager in list(self._worker_managers.items()):
            try:
                manager.close()
            except Exception:
                logger.exception("worker_close_error", worker_id=worker_id)
        self._worker_managers.clear()
        self._in_flight.clear()
        self._remote_adapters.clear()

    def assert_no_leaks(self) -> None:
        """Assert that no owned subprocesses are still alive.

        Raises:
            RuntimeError: if any owned PID is still running.
        """
        for worker_id, manager in list(self._worker_managers.items()):
            try:
                manager.assert_no_leaks()
            except RuntimeError:
                raise
            except Exception:
                logger.exception("leak_check_error", worker_id=worker_id)

    def __enter__(self) -> ToolExecutionEngine:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
