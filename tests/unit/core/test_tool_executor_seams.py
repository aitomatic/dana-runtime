"""Executor seams owned by the harness optimization work: offload pool ownership,
admission, effect-ordered scheduling, lifecycle hooks and truthful cancellation.

These tests exercise behavior (not wiring pins): what the executor *does* when
blocking resources exist, when a call is cancelled at each stage, when effect
metadata conflicts, and how lifecycle ownership behaves across sessions.
"""

from __future__ import annotations

import asyncio
import pathlib
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from dana.core.ext.event_bus import EventBus
from dana.core.ext.events import TOOL_CALL, TOOL_RESULT
from dana.core.policy.effects import Effect, EffectKind, EffectMetadata
from dana.core.resource.file_edit_resource import FileEditResource
from dana.core.resource.file_io_resource import FileIOResource
from dana.core.resource.search_resource import SearchResource
from dana.core.tool.admission import AdmissionController
from dana.core.tool.catalog import ToolCatalog, ToolCatalogEntry, ToolIdentity
from dana.core.tool.effect_ordering import plan_batch
from dana.core.tool.offload import (
    BlockingIOPool,
    BlockingIOShutdown,
    blocking_io,
    bound_blocking_io_pool,
    current_blocking_io_pool,
    current_physical_work,
    get_default_blocking_io_pool,
    use_blocking_io_pool,
)
from dana.core.tool.tool_executor import ToolExecutor


async def _run(executor: ToolExecutor, calls: list[dict]) -> list[dict]:
    return await executor.execute_tools_async(object(), calls)


def _io_executor() -> tuple[ToolExecutor, pathlib.Path]:
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "a.txt").write_text("alpha\nneedle here\nomega\n")
    (d / "sub").mkdir()
    (d / "sub" / "b.py").write_text("def needle():\n    return 1\n")
    io_res = FileIOResource(resource_id="io", base_path=str(d))
    edit_res = FileEditResource(resource_id="edit", base_path=str(d))
    search_res = SearchResource(resource_id="search", base_path=str(d))
    executor = ToolExecutor(
        tool_name_registry_getter=lambda: {
            "Read": (io_res, "read"),
            "Edit": (edit_res, "edit"),
            "Write": (edit_res, "write"),
            "Grep": (search_res, "grep"),
            "Glob": (search_res, "glob"),
        }
    )
    return executor, d


# ---------------------------------------------------------------------------
# Offload: blocking I/O must not run on the event loop
# ---------------------------------------------------------------------------


class TestBlockingOffload:
    def test_pool_is_per_owner_and_default_survives_close(self):
        first = ToolExecutor()
        second = ToolExecutor()
        assert first.offload_pool is not second.offload_pool
        default = get_default_blocking_io_pool()
        first.close()
        assert first.offload_pool.closed
        assert not second.offload_pool.closed
        assert not default.closed  # a session close never poisons the process
        second.close()

    @pytest.mark.asyncio
    async def test_resource_read_offloads_without_stalling_loop(self):
        executor, d = _io_executor()
        ticked = asyncio.Event()

        async def spinner():
            for _ in range(20):
                await asyncio.sleep(0)
            ticked.set()

        results, _ = await asyncio.gather(
            _run(executor, [{"function": "Read", "arguments": {"file_path": str(d / "a.txt")}, "tool_call_id": "t1"}]),
            spinner(),
        )
        assert ticked.is_set(), "event loop kept running while the read was offloaded"
        assert results[0]["success"] is True
        assert "needle here" in results[0]["result"]
        executor.close()

    @pytest.mark.asyncio
    async def test_grep_and_glob_offload_and_return_real_matches(self):
        executor, d = _io_executor()
        grep, glob = await asyncio.gather(
            _run(
                executor,
                [
                    {
                        "function": "Grep",
                        "arguments": {"pattern": "needle", "path": str(d), "output_mode": "content"},
                        "tool_call_id": "g",
                    }
                ],
            ),
            _run(executor, [{"function": "Glob", "arguments": {"pattern": "**/*.py", "path": str(d)}, "tool_call_id": "l"}]),
        )
        assert grep[0]["success"] is True
        assert "needle" in grep[0]["result"]
        assert glob[0]["success"] is True
        assert "b.py" in glob[0]["result"]
        executor.close()

    @pytest.mark.asyncio
    async def test_concurrent_edits_same_file_do_not_interleave(self):
        executor, d = _io_executor()
        target = str(d / "a.txt")
        calls = [
            {
                "function": "Edit",
                "arguments": {"file_path": target, "old_string": "alpha", "new_string": "ALPHA"},
                "tool_call_id": f"e{i}",
            }
            for i in range(4)
        ]
        results = await _run(executor, calls)
        # Exactly one edit succeeds; the others report the string is gone. The
        # per-path lock is what makes that deterministic rather than lossy.
        succeeded = [r for r in results if r["success"] and "ALPHA" in str(r["result"])]
        assert len(succeeded) == 1
        assert (d / "a.txt").read_text().count("ALPHA") == 1
        executor.close()

    @pytest.mark.asyncio
    async def test_write_refuses_overwrite_under_per_path_lock(self):
        executor, d = _io_executor()
        target = str(d / "new.txt")
        results = await _run(
            executor,
            [
                {"function": "Write", "arguments": {"file_path": target, "content": "one"}, "tool_call_id": "w1"},
                {"function": "Write", "arguments": {"file_path": target, "content": "two"}, "tool_call_id": "w2"},
            ],
        )
        # The tool returns its own error string (not success=False) when the file
        # exists; the per-path lock is what makes the outcome deterministic: the
        # first create wins, the second is refused, and the file is never clobbered.
        wrote = [r for r in results if "Successfully wrote" in str(r["result"])]
        refused = [r for r in results if "already exists" in str(r["result"])]
        assert len(wrote) == 1 and len(refused) == 1
        assert (d / "new.txt").read_text() == "one"
        executor.close()

    @pytest.mark.asyncio
    async def test_shutdown_pool_refuses_new_offloads(self):
        pool = BlockingIOPool()
        pool.shutdown()
        with pytest.raises(BlockingIOShutdown):
            await pool.run(lambda: 1)


# ---------------------------------------------------------------------------
# Nested pool inheritance
# ---------------------------------------------------------------------------


class TestNestedPoolInheritance:
    @pytest.mark.asyncio
    async def test_nested_dispatch_shares_root_pool(self):
        root, d = _io_executor()
        inner = ToolExecutor()
        observed: list[bool] = []
        original = inner._dispatch_single_call_async

        async def spy(*args, **kwargs):
            observed.append(current_blocking_io_pool() is root.offload_pool)
            return await original(*args, **kwargs)

        inner._dispatch_single_call_async = spy  # type: ignore[method-assign]
        inner._tool_name_registry_getter = lambda: {"Read": (FileIOResource(resource_id="i2", base_path=str(d)), "read")}
        with use_blocking_io_pool(root.offload_pool):
            await _run(inner, [{"function": "Read", "arguments": {"file_path": str(d / "a.txt")}, "tool_call_id": "n1"}])
        assert observed == [True], "nested dispatch inherited the root session pool"
        assert bound_blocking_io_pool() is None, "binding is scoped, not leaked"
        root.close()
        inner.close()


class TestLifecycleOwnership:
    def test_executor_close_never_clears_shared_admission(self):
        from dana.core.tool.admission import get_default_admission_controller

        shared = get_default_admission_controller()
        first = ToolExecutor()
        second = ToolExecutor()
        first.close()
        # The shared controller and the sibling executor keep working.
        assert not second.offload_pool.closed
        assert shared is get_default_admission_controller()
        second.close()

    def test_owned_admission_controller_is_released(self):
        from dana.core.tool.admission import AdmissionController

        controller = AdmissionController()
        executor = ToolExecutor(admission=controller)
        executor.close()
        assert executor.closed is True

    @pytest.mark.asyncio
    async def test_closing_one_session_does_not_kill_another(self):
        a, da = _io_executor()
        b, db = _io_executor()
        a.close()
        results = await _run(b, [{"function": "Read", "arguments": {"file_path": str(db / "a.txt")}, "tool_call_id": "s2"}])
        assert results[0]["success"] is True
        b.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent_and_blocks_new_offloads(self):
        executor, d = _io_executor()
        executor.close()
        executor.close()
        assert executor.closed
        pool = executor.offload_pool
        with pytest.raises(BlockingIOShutdown):
            await pool.run(lambda: 1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("nested_limit", [1, 3])
    async def test_nested_and_leaf_limits_are_separate(self, nested_limit):
        controller = AdmissionController(leaf_limit=2, nested_limit=nested_limit)
        active = {"nested": 0, "leaf": 0}
        peak = {"nested": 0, "leaf": 0}
        completed = {"nested": 0, "leaf": 0}

        async def work(kind):
            active[kind] += 1
            peak[kind] = max(peak[kind], active[kind])
            try:
                await asyncio.sleep(0.01)
                completed[kind] += 1
                return "finished"
            finally:
                active[kind] -= 1

        async def nested(arguments):
            if arguments.get("root"):
                return await asyncio.gather(
                    _run(executor, [{"function": "Task", "arguments": {"depth": 6}, "tool_call_id": f"child-{i}"} for i in range(10)]),
                    _run(executor, [{"function": "Read", "arguments": {}, "tool_call_id": f"leaf-{i}"} for i in range(6)]),
                )
            if arguments["depth"]:
                return await _run(
                    executor,
                    [
                        {
                            "function": "Task",
                            "arguments": {"depth": arguments["depth"] - 1},
                            "tool_call_id": f"depth-{arguments['depth']}",
                        }
                    ],
                )
            return await work("nested")

        async def leaf(arguments):
            return await work("leaf")

        executor = ToolExecutor(
            admission=controller,
            tool_catalog=ToolCatalog(
                [
                    ToolCatalogEntry(ToolIdentity("Task"), {}, nested, effects=EffectMetadata((Effect(EffectKind.EXECUTE),))),
                    ToolCatalogEntry(ToolIdentity("Read"), {}, leaf, effects=EffectMetadata((Effect(EffectKind.READ),))),
                ]
            ),
        )
        try:
            results = await asyncio.wait_for(
                _run(
                    executor,
                    [
                        {
                            "function": "Task",
                            "arguments": {"root": True},
                            "tool_call_id": "root",
                        }
                    ],
                ),
                3,
            )
            assert results[0]["success"] is True
            assert peak == {"nested": nested_limit, "leaf": 2}
            assert completed == {"nested": 10, "leaf": 6}
            snapshot = controller.snapshot()
            assert snapshot["nested"]["peak_in_flight"] == nested_limit
            assert snapshot["leaf"]["peak_in_flight"] == 2
            assert snapshot["nested"]["in_flight"] == snapshot["leaf"]["in_flight"] == 0
        finally:
            executor.close()

    @pytest.mark.asyncio
    async def test_nested_loan_survives_released_ancestor(self):
        controller = AdmissionController(nested_limit=1)
        parent = await controller.acquire_kind(controller.NESTED)
        borrowed = asyncio.Event()
        finish = asyncio.Event()
        sibling_started = asyncio.Event()

        async def child():
            async with controller.nested_slot():
                borrowed.set()
                await finish.wait()

        async def sibling():
            async with controller.nested_slot():
                sibling_started.set()

        child_task = asyncio.create_task(child())
        await borrowed.wait()
        parent.release()
        sibling_task = asyncio.create_task(sibling())
        try:
            await asyncio.sleep(0)
            assert not sibling_started.is_set(), "the descendant still owns the ancestor's slot"
        finally:
            finish.set()
            await asyncio.wait_for(asyncio.gather(child_task, sibling_task), 1)
        assert sibling_started.is_set()
        assert controller.snapshot()["nested"]["in_flight"] == 0

    @pytest.mark.asyncio
    async def test_cancelled_nested_sibling_does_not_steal_loan(self):
        controller = AdmissionController(nested_limit=1)
        borrowed = asyncio.Event()
        finish = asyncio.Event()

        async def child():
            async with controller.nested_slot():
                borrowed.set()
                await finish.wait()

        async with controller.nested_slot():
            child_task = asyncio.create_task(child())
            await borrowed.wait()
            queued = asyncio.create_task(controller.acquire_kind(controller.NESTED))
            await asyncio.sleep(0)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert controller.snapshot()["nested"]["in_flight"] == 1
            finish.set()
            await child_task
            async with controller.nested_slot():
                assert controller.snapshot()["nested"]["in_flight"] == 1
        assert controller.snapshot()["nested"]["in_flight"] == 0


# ---------------------------------------------------------------------------
# Lifecycle hooks: durable-write boundary
# ---------------------------------------------------------------------------


class TestExecutionHooks:
    @pytest.mark.asyncio
    async def test_hooks_receive_actual_arguments_and_terminal_result(self):
        executor, d = _io_executor()
        seen: list[tuple] = []
        executor.set_execution_hooks(
            before=lambda call, reason: seen.append(("before", call["arguments"]["file_path"], reason)),
            after=lambda call, result: seen.append(("after", result["success"])),
        )
        await _run(executor, [{"function": "Read", "arguments": {"file_path": str(d / "a.txt")}, "tool_call_id": "h1"}])
        assert seen[0] == ("before", str(d / "a.txt"), None)
        assert seen[-1] == ("after", True)
        executor.close()

    @pytest.mark.asyncio
    async def test_blocked_call_invokes_hooks_and_never_dispatches(self):
        executor = ToolExecutor()
        seen: list[tuple] = []
        executor.set_execution_hooks(
            before=lambda call, reason: seen.append(("before", reason)),
            after=lambda call, result: seen.append(("after", result["success"])),
        )
        dispatched = False
        original = executor._dispatch_single_call_async

        async def spy(*args, **kwargs):
            nonlocal dispatched
            dispatched = True
            return await original(*args, **kwargs)

        executor._dispatch_single_call_async = spy  # type: ignore[method-assign]

        async def block(agent, operation, tool_call_id):
            return {"block": True, "reason": "denied by policy"}

        executor._emit_tool_call_async = block  # type: ignore[method-assign]
        results = await _run(executor, [{"function": "Read", "arguments": {}, "tool_call_id": "b1"}])
        assert dispatched is False
        assert results[0]["success"] is False
        assert seen == [("before", "denied by policy"), ("after", False)]
        executor.close()

    @pytest.mark.asyncio
    async def test_hook_failure_propagates_and_does_not_fabricate_result(self):
        executor = ToolExecutor()

        def failing(call, reason):
            raise RuntimeError("journal append failed")

        executor.set_execution_hooks(before=failing)
        with pytest.raises(RuntimeError, match="journal append failed"):
            await _run(executor, [{"function": "Read", "arguments": {}, "tool_call_id": "f1"}])
        executor.close()

    @pytest.mark.asyncio
    async def test_after_hook_runs_when_dispatch_raises(self):
        executor, d = _io_executor()
        seen: list[tuple] = []
        executor.set_execution_hooks(after=lambda call, result: seen.append((result["success"], result["type"])))

        async def boom(*args, **kwargs):
            raise ValueError("dispatch exploded")

        executor._dispatch_single_call_async = boom  # type: ignore[method-assign]
        await _run(executor, [{"function": "Read", "arguments": {}, "tool_call_id": "e1"}])
        assert seen and seen[0][0] is False
        executor.close()


# ---------------------------------------------------------------------------
# Cancellation: truthful terminal disposition
# ---------------------------------------------------------------------------


class TestCancellationDisposition:
    @pytest.mark.asyncio
    async def test_cancel_before_dispatch_marks_no_effect_and_propagates(self):
        executor = ToolExecutor()
        seen: list[tuple] = []
        executor.set_execution_hooks(after=lambda call, result: seen.append((result.get("_cancelled"), result.get("_effect_unknown"))))

        def cancel_queued(call, reason):
            raise asyncio.CancelledError()

        executor.set_execution_hooks(
            before=cancel_queued, after=lambda call, r: seen.append((r.get("_cancelled"), r.get("_effect_unknown")))
        )
        with pytest.raises(asyncio.CancelledError):
            await _run(executor, [{"function": "Read", "arguments": {}, "tool_call_id": "c1"}])
        assert seen == [(True, None)]
        executor.close()

    @pytest.mark.asyncio
    async def test_cancel_during_dispatch_marks_effect_unknown_and_propagates(self):
        executor, d = _io_executor()
        seen: list[tuple] = []
        executor.set_execution_hooks(after=lambda call, r: seen.append((r.get("_cancelled"), r.get("_effect_unknown"))))

        async def cancel_midway(*args, **kwargs):
            raise asyncio.CancelledError()

        executor._dispatch_single_call_async = cancel_midway  # type: ignore[method-assign]
        with pytest.raises(asyncio.CancelledError):
            await _run(executor, [{"function": "Read", "arguments": {}, "tool_call_id": "c2"}])
        assert seen == [(None, True)]
        executor.close()

    @pytest.mark.asyncio
    async def test_sibling_calls_still_terminal_when_one_is_cancelled(self):
        executor, d = _io_executor()
        terminals: list[str] = []
        executor.set_execution_hooks(after=lambda call, r: terminals.append(call["tool_call_id"]))

        original = executor._dispatch_single_call_async

        async def mixed(agent, function_name, arguments, tool_call_id=None, admission_kind="leaf"):
            if tool_call_id == "cancel-me":
                raise asyncio.CancelledError()
            return await original(agent, function_name, arguments, tool_call_id=tool_call_id, admission_kind=admission_kind)

        executor._dispatch_single_call_async = mixed  # type: ignore[method-assign]
        with pytest.raises(asyncio.CancelledError):
            await _run(
                executor,
                [
                    {"function": "Read", "arguments": {"file_path": str(d / "a.txt")}, "tool_call_id": "ok"},
                    {"function": "Read", "arguments": {"file_path": str(d / "a.txt")}, "tool_call_id": "cancel-me"},
                ],
            )
        assert set(terminals) == {"ok", "cancel-me"}, "every call in the wave got a terminal"
        executor.close()


# ---------------------------------------------------------------------------
# Effect-ordered scheduling
# ---------------------------------------------------------------------------


class TestEffectOrderedScheduling:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("sync_parallel", [False, True])
    async def test_read_edit_read_preserves_real_effect_order(self, sync_parallel):
        executor, directory = _io_executor()
        path = str(directory / "a.txt")
        calls = [
            {"function": "Read", "arguments": {"file_path": path}, "tool_call_id": "before"},
            {"function": "Edit", "arguments": {"file_path": path, "old_string": "alpha", "new_string": "changed"}, "tool_call_id": "edit"},
            {"function": "Read", "arguments": {"file_path": path}, "tool_call_id": "after"},
        ]
        try:
            results = executor.execute_tools(object(), calls, parallel=True) if sync_parallel else await _run(executor, calls)
            assert "alpha" in results[0]["result"]
            assert "changed" in results[2]["result"]
            assert "alpha" not in results[2]["result"]
            assert [result["tool_call_id"] for result in results] == ["before", "edit", "after"]
        finally:
            executor.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sync_parallel", [False, True])
    async def test_intercepted_targets_are_planned_once_and_reach_hooks(self, sync_parallel):
        executor, directory = _io_executor()
        path = str(directory / "a.txt")
        other_path = str(directory / "sub" / "b.py")
        edit = FileEditResource(resource_id="delayed", base_path=str(directory))

        class DelayedEdit:
            async def edit(self, file_path: str, old_string: str, new_string: str):
                if old_string == "alpha":
                    await asyncio.sleep(0.02)
                return await edit.edit(file_path=file_path, old_string=old_string, new_string=new_string)

        resource = DelayedEdit()
        executor._tool_name_registry_getter = lambda: {"Edit": (resource, "edit")}
        agent = SimpleNamespace(event_bus=EventBus())
        emitted, authorized, terminal_paths = [], [], []

        def intercept(event):
            emitted.append(event.payload["tool_call_id"])
            if event.payload["tool_call_id"] == "second":
                return {"modify": {"arguments": {**event.payload["operation"].arguments, "file_path": path}}}

        agent.event_bus.subscribe(TOOL_CALL, intercept)
        agent.event_bus.subscribe(TOOL_RESULT, lambda event: terminal_paths.append(event.payload["operation"].arguments["file_path"]))
        executor.set_execution_hooks(before=lambda call, reason: authorized.append((call["arguments"]["file_path"], reason)))
        calls = [
            {"function": "Edit", "arguments": {"file_path": path, "old_string": "alpha", "new_string": "middle"}, "tool_call_id": "first"},
            {
                "function": "Edit",
                "arguments": {"file_path": other_path, "old_string": "middle", "new_string": "final"},
                "tool_call_id": "second",
            },
        ]
        try:
            if sync_parallel:
                results = executor.execute_tools(agent, calls, parallel=True)
            else:
                results = await executor.execute_tools_async(agent, calls)
                assert authorized == [(path, None), (path, None)]
            assert "final" in (directory / "a.txt").read_text()
            assert "def needle" in (directory / "sub" / "b.py").read_text()
            assert all(result["success"] for result in results)
            assert emitted == ["first", "second"]
            assert terminal_paths == [path, path]
        finally:
            executor.close()

    @pytest.mark.asyncio
    async def test_noninterfering_reads_overlap(self):
        executor, d = _io_executor()
        overlap = {"max": 0}

        in_flight = {"n": 0}

        async def tracked(agent, function_name, arguments, tool_call_id=None, admission_kind="leaf"):
            in_flight["n"] += 1
            overlap["max"] = max(overlap["max"], in_flight["n"])
            await asyncio.sleep(0.01)
            in_flight["n"] -= 1
            return {"success": True, "type": "x", "target": function_name, "result": "ok"}

        executor._dispatch_single_call_async = tracked  # type: ignore[method-assign]
        await _run(
            executor,
            [
                {"function": "Read", "arguments": {"file_path": str(d / "a.txt")}, "tool_call_id": "r1"},
                {"function": "Read", "arguments": {"file_path": str(d / "a.txt")}, "tool_call_id": "r2"},
                {"function": "Read", "arguments": {"file_path": str(d / "sub" / "b.py")}, "tool_call_id": "r3"},
            ],
        )
        assert overlap["max"] >= 2, "non-interfering reads ran concurrently"
        executor.close()

    @pytest.mark.asyncio
    async def test_unknown_effects_are_conservative(self):
        executor = ToolExecutor()
        seen: list[str] = []

        async def tracked(agent, function_name, arguments, tool_call_id=None, admission_kind="leaf"):
            seen.append(tool_call_id)
            return {"success": True, "type": "x", "target": function_name, "result": "ok"}

        executor._dispatch_single_call_async = tracked  # type: ignore[method-assign]
        await _run(
            executor,
            [
                {"function": "mystery_one", "arguments": {}, "tool_call_id": "u1"},
                {"function": "mystery_two", "arguments": {}, "tool_call_id": "u2"},
            ],
        )
        assert seen == ["u1", "u2"], "unknown-effect calls kept the model's order"
        executor.close()

    @pytest.mark.asyncio
    async def test_mcp_effects_overlap_reads_and_order_unknown_mutations_per_server(self):
        """I9: MCP scheduling uses the server's own declaration, not a guess.

        A ``readOnlyHint=True`` tool may overlap another read-only tool of the same
        server. Two *unannotated* tools of one server share an unknown mutation blast
        radius and stay ordered; two different servers are independent and overlap.
        A genuinely unclassified tool still keeps the model's order.
        """
        from dana.core.mcp.dispatch_wrapper import _mcp_effect_metadata
        from dana.core.policy.effects import EffectKind

        def tool(name: str, read_only: bool | None):
            annotations = None if read_only is None else SimpleNamespace(readOnlyHint=read_only)
            return SimpleNamespace(name=name, annotations=annotations)

        # The server's declaration is what the executor consumes.
        assert [e.kind for e in _mcp_effect_metadata(tool("peek", True), "srv").effects] == [EffectKind.READ]
        same_server = _mcp_effect_metadata(tool("poke", None), "srv")
        assert [e.target for e in same_server.effects] == ["srv"], "unannotated tool is not provably disjoint from its server"
        assert same_server.is_sensitive is False, "an unannotated MCP tool is promptable, just ordered"

        effects = {
            "srv:peek": _mcp_effect_metadata(tool("peek", True), "srv"),
            "srv:poke": _mcp_effect_metadata(tool("poke", None), "srv"),
            "srv:shove": _mcp_effect_metadata(tool("shove", None), "srv"),
            "other:poke": _mcp_effect_metadata(tool("poke", None), "other"),
        }
        executor = ToolExecutor(mcp_dispatch_getter=lambda: dict.fromkeys(effects, object()))
        executor.set_mcp_effect_resolver(lambda: effects)

        def waves(calls):
            return plan_batch(
                [{**c, "arguments": c.get("arguments", {})} for c in calls], None, metadata_resolver=executor._effect_metadata_for
            ).waves

        # Read-only tools of one server overlap.
        assert waves(
            [
                {"function": "srv:peek", "arguments": {"token": "a"}},
                {"function": "srv:peek", "arguments": {"token": "b"}},
            ]
        ) == [[0, 1]]
        # Unannotated mutations of the SAME server stay ordered, even when their
        # arguments happen to name different paths (the server may ignore them).
        assert waves(
            [
                {"function": "srv:poke", "arguments": {"path": "/a"}},
                {"function": "srv:shove", "arguments": {"path": "/b"}},
            ]
        ) == [[0], [1]]
        # Different servers are independent.
        assert waves(
            [
                {"function": "srv:poke", "arguments": {}},
                {"function": "other:poke", "arguments": {}},
            ]
        ) == [[0, 1]]
        # An unclassified tool is still sensitive and keeps the model's order.
        assert waves(
            [
                {"function": "srv:peek", "arguments": {}},
                {"function": "mystery", "arguments": {}},
            ]
        ) == [[0], [1]]
        executor.close()

    @pytest.mark.asyncio
    async def test_independent_mcp_reads_run_concurrently_while_conflicting_writes_stay_ordered(self):
        """I9 + acceptance: real dispatch overlaps independent MCP reads, orders
        conflicting mutations of one server, and keeps harness ids distinct.

        Scheduling is only honest if it is the *same* metadata the executor
        schedules with (``_effect_metadata_for``), so this drives real dispatches
        through the wired dispatch map and observes actual overlap.
        """
        from dana.core.mcp.dispatch_wrapper import _mcp_effect_metadata

        def tool(name: str, read_only: bool | None):
            annotations = None if read_only is None else SimpleNamespace(readOnlyHint=read_only)
            return SimpleNamespace(name=name, annotations=annotations)

        effects = {
            "srv:peek": _mcp_effect_metadata(tool("peek", True), "srv"),
            "srv:poke": _mcp_effect_metadata(tool("poke", None), "srv"),
            "other:peek": _mcp_effect_metadata(tool("peek", True), "other"),
        }
        running: list[str] = []
        overlap: list[str] = []
        order: list[str] = []

        def _dispatcher(name: str):
            async def dispatch(arguments: dict[str, Any], tool_call_id: Any = None) -> dict[str, Any]:
                running.append(name)
                if len(running) > 1:
                    overlap.append("+".join(sorted(running)))
                try:
                    await asyncio.sleep(0.02)
                finally:
                    running.remove(name)
                order.append(name)
                return {"success": True, "result": tool_call_id, "tool_call_id": tool_call_id}

            return dispatch

        executor = ToolExecutor(mcp_dispatch_getter=lambda: {name: _dispatcher(name) for name in effects})
        executor.set_mcp_effect_resolver(lambda: effects)
        try:
            results = await executor.execute_tools_async(
                object(),
                [
                    {"function": "srv:peek", "arguments": {"token": "a"}, "tool_call_id": "tc-a"},
                    {"function": "other:peek", "arguments": {"token": "b"}, "tool_call_id": "tc-b"},
                    {"function": "srv:poke", "arguments": {}, "tool_call_id": "tc-c"},
                    {"function": "srv:poke", "arguments": {}, "tool_call_id": "tc-d"},
                ],
            )
        finally:
            executor.close()

        assert overlap == ["other:peek+srv:peek"], "independent read-only MCP calls did not overlap"
        # Conflicting mutations of one server are serialized in model order, and
        # the ordered writes come after the reads they may conflict with.
        assert order.index("srv:poke") > order.index("srv:peek")
        assert [results[i]["tool_call_id"] for i in range(4)] == ["tc-a", "tc-b", "tc-c", "tc-d"]
        assert [r["result"] for r in results] == ["tc-a", "tc-b", "tc-c", "tc-d"], "harness ids drifted between dispatch and result"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_name", ["lookup__query", "lookup:query", "lookup:aquery"])
    async def test_query_shaped_resource_alias_obeys_the_leaf_limit(self, tool_name):
        from dana.common.protocols.war import named_tool
        from dana.core.agent.base_agent import BaseAgent
        from dana.core.resource.base_resource import BaseResource
        from dana.core.runtime.default import DefaultRuntime

        class LookupResource(BaseResource):
            def __init__(self):
                super().__init__(resource_id="lookup", auto_register=False)
                self.active = 0
                self.peak = 0

            @named_tool(tool_name)
            async def query(self, value: str) -> str:
                """Read one lookup value."""
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    await asyncio.sleep(0.01)
                    return value
                finally:
                    self.active -= 1

        resource = LookupResource()
        agent = BaseAgent(auto_register=False).with_resources(resource)
        runtime = DefaultRuntime()
        controller = AdmissionController(leaf_limit=1, nested_limit=4)
        catalog = ToolCatalog([ToolCatalogEntry(ToolIdentity(tool_name), {}, None, effects=EffectMetadata((Effect(EffectKind.READ),)))])
        executor = ToolExecutor(
            admission=controller,
            agent_tool_name_registry_getter=runtime._tool_name_registry_for,
            effect_catalog_getter=lambda: catalog,
        )
        try:
            results = await asyncio.wait_for(
                executor.execute_tools_async(
                    agent,
                    [{"function": tool_name, "arguments": {"value": str(i)}, "tool_call_id": f"lookup-{i}"} for i in range(4)],
                ),
                3,
            )
            assert [result["result"] for result in results] == ["0", "1", "2", "3"]
            assert [result["tool_call_id"] for result in results] == [f"lookup-{i}" for i in range(4)]
            assert resource.peak == 1, "an ordinary resource query bypassed the leaf limit"
            assert controller.snapshot()["leaf"]["peak_in_flight"] == 1
            assert controller.snapshot()["nested"]["peak_in_flight"] == 0
        finally:
            executor.close()
            runtime._tool_executor.close()

    @pytest.mark.asyncio
    async def test_registered_agent_chain_borrows_nested_capacity_and_admits_its_leaf(self):
        from dana.common.protocols.war import named_tool
        from dana.core.agent.base_agent import BaseAgent
        from dana.core.resource.base_resource import BaseResource
        from dana.core.runtime.default import DefaultRuntime

        observed: list[str] = []

        class LeafResource(BaseResource):
            @named_tool("LeafRead")
            async def read(self) -> str:
                """Perform the descendant's actual leaf tool."""
                observed.append("leaf effect")
                return "leaf result"

        class DelegatingAgent(BaseAgent):
            @named_tool("delegate__query")
            async def query(self, message: str = "") -> str:
                """Delegate to the next agent or run the leaf resource."""
                name = "delegate__query" if self.available_agents else "LeafRead"
                results = await executor.execute_tools_async(self, [{"function": name, "arguments": {}, "tool_call_id": self.agent_type}])
                assert results[0]["success"] is True
                return results[0]["result"]

        leaf = DelegatingAgent(agent_type="leaf", auto_register=False).with_resources(
            LeafResource(resource_id="leaf-reader", auto_register=False)
        )
        middle = DelegatingAgent(agent_type="middle", auto_register=False).with_agents(leaf)
        root = BaseAgent(agent_type="root", auto_register=False).with_agents(middle)
        runtime = DefaultRuntime()
        controller = AdmissionController(leaf_limit=1, nested_limit=1)
        executor = ToolExecutor(admission=controller, agent_tool_name_registry_getter=runtime._tool_name_registry_for)
        try:
            results = await asyncio.wait_for(
                executor.execute_tools_async(root, [{"function": "delegate__query", "arguments": {}, "tool_call_id": "root-middle"}]),
                3,
            )
            assert results[0]["result"] == "leaf result"
            assert observed == ["leaf effect"]
            snapshot = controller.snapshot()
            assert snapshot["leaf"]["peak_in_flight"] == 1
            assert snapshot["nested"]["peak_in_flight"] == 1
            assert all(pool["in_flight"] == 0 for pool in snapshot.values())
        finally:
            executor.close()
            runtime._tool_executor.close()

    @pytest.mark.asyncio
    async def test_sub_agent_query_is_nested_admission_and_does_not_deadlock(self):
        """C1: a parent's sub-agent dispatch must not hold a leaf slot.

        A sub-agent tool is named ``<agent_type>__query`` (with ``-``/``.`` becoming
        ``_``), the shape the runtime's own schema builder emits. Treated as *leaf*
        work, the parent awaiting its child holds a leaf slot the child's own tool
        calls need, so a chain saturates ``leaf_limit`` and never completes.
        """

        class SubAgent:
            agent_type = "math-agent"

            async def query(self, message: str = "") -> str:
                return "child"

        class Agent:
            def __init__(self):
                self.available_agents = [SubAgent()]
                self.available_resources = []
                self.available_workflows = []

        agent = Agent()
        controller = AdmissionController(leaf_limit=2, nested_limit=2)
        executor = ToolExecutor(admission=controller)

        async def leaf_work(arguments):
            await asyncio.sleep(0.01)
            return "leaf-done"

        async def child_query(arguments):
            # The child runs the parent's executor: its tool call needs a leaf slot
            # while the parent is still awaiting this dispatch.
            return await executor._dispatch_single_call_async(agent, "Read", {"file_path": "/tmp/x"}, tool_call_id="child-leaf")

        executor._tool_catalog = ToolCatalog(
            [
                ToolCatalogEntry(ToolIdentity("math_agent__query"), {}, child_query),
                ToolCatalogEntry(ToolIdentity("Read"), {}, leaf_work, effects=EffectMetadata((Effect(EffectKind.READ),))),
            ]
        )
        try:
            results = await asyncio.wait_for(
                executor.execute_tools_async(agent, [{"function": "math_agent__query", "arguments": {}, "tool_call_id": "root"}]),
                3,
            )
            assert results[0]["success"] is True
            assert results[0].get("_admission") == "nested", "sub-agent dispatch used the nested pool"
            assert controller.snapshot()["leaf"]["in_flight"] == 0
            assert controller.snapshot()["nested"]["in_flight"] == 0
        finally:
            executor.close()


class TestExecutionPolicyHooks:
    """I5/I6: authorization resolves outside admission; a returned denial is recoverable.

    A ``blocked``/authorized decision must be reached *before* a permit is taken, so
    a queue of interactive prompts can never starve the tool budget, and a refusal
    the session *returns* must leave the turn alive for the model to adapt.
    """

    @staticmethod
    def _executor(controller: AdmissionController) -> ToolExecutor:
        async def body(arguments):
            return "ran"

        executor = ToolExecutor(admission=controller)
        executor._tool_catalog = ToolCatalog(
            [
                ToolCatalogEntry(ToolIdentity("Safe"), {}, body, effects=EffectMetadata((Effect(EffectKind.READ),))),
                ToolCatalogEntry(ToolIdentity("Risky"), {}, body, effects=EffectMetadata((Effect(EffectKind.EXECUTE),))),
            ]
        )
        return executor

    @pytest.mark.asyncio
    async def test_authorize_runs_before_admission_and_returned_denial_is_recoverable(self):
        controller = AdmissionController(leaf_limit=1, nested_limit=1)
        executor = self._executor(controller)
        seen: list[tuple[str, int]] = []

        async def authorize(call, block_reason):
            # The hook awaits while no permit may be held: with a limit of one, a
            # hook that held it would deadlock the very next call's authorization.
            seen.append((call["function"], controller.snapshot()["leaf"]["in_flight"]))
            if call["function"] == "Risky":
                return {"block": True, "reason": "denied by the operator"}
            return None

        executor.set_execution_hooks(authorize=authorize)
        results = await asyncio.wait_for(
            executor.execute_tools_async(
                object(),
                [
                    {"function": "Safe", "arguments": {}, "tool_call_id": "s"},
                    {"function": "Risky", "arguments": {}, "tool_call_id": "r"},
                ],
            ),
            5,
        )
        assert seen == [("Safe", 0), ("Risky", 0)], "authorization ran while a permit was held"
        assert results[0]["success"] is True
        assert results[1]["type"] == "policy_block", "a returned denial is an adapt-able tool result"
        assert results[1]["success"] is False
        assert "denied by the operator" in results[1]["result"]
        assert controller.snapshot()["leaf"]["in_flight"] == 0
        executor.close()

    @pytest.mark.asyncio
    async def test_returned_denial_from_the_before_hook_is_recoverable_too(self):
        controller = AdmissionController(leaf_limit=1, nested_limit=1)
        executor = self._executor(controller)

        async def before(call, block_reason):
            return {"block": True, "reason": "queue policy refused"} if call["function"] == "Risky" else None

        executor.set_execution_hooks(before=before)
        results = await executor.execute_tools_async(object(), [{"function": "Risky", "arguments": {}, "tool_call_id": "r"}])
        assert results[0]["type"] == "policy_block"
        assert "queue policy refused" in results[0]["result"]
        executor.close()

    @pytest.mark.asyncio
    async def test_cancellation_during_authorization_records_a_no_effect_terminal(self):
        """I6: a cancelled decision still yields exactly one truthful terminal fact."""
        executor = self._executor(AdmissionController(leaf_limit=1))
        terminals: list[dict] = []

        async def authorize(call, block_reason):
            raise asyncio.CancelledError

        async def after(call, result):
            terminals.append(result)

        executor.set_execution_hooks(authorize=authorize, after=after)
        with pytest.raises(asyncio.CancelledError):
            await executor.execute_tools_async(object(), [{"function": "Safe", "arguments": {}, "tool_call_id": "s"}])
        assert len(terminals) == 1, "a cancelled authorization must record its terminal fact once"
        assert terminals[0]["_cancelled"] is True
        assert "_effect_unknown" not in terminals[0], "nothing was dispatched, so the effect is known to be none"
        executor.close()

    @pytest.mark.asyncio
    async def test_raised_authorization_failure_stays_fatal(self):
        executor = self._executor(AdmissionController(leaf_limit=1))

        async def authorize(call, block_reason):
            raise RuntimeError("durable append failed")

        executor.set_execution_hooks(authorize=authorize)
        with pytest.raises(RuntimeError, match="durable append failed"):
            await executor.execute_tools_async(object(), [{"function": "Safe", "arguments": {}, "tool_call_id": "s"}])
        executor.close()


class TestOwnerScopedExecutionHooks:
    """A runtime/executor shared by two live sessions must resolve each call's
    durable-journal hooks from the *invoking* agent.

    The hooks are a session's journal boundary: wiring them globally lets the
    second session's ``prompt`` overwrite the first's, so the first session's
    in-flight tool journals requested/authorized/started/terminal facts into the
    wrong owner's turn even though the effect catalog is already owner-aware.
    """

    @pytest.mark.asyncio
    async def test_two_agents_on_one_executor_keep_their_own_hooks(self):
        executor = self._executor()

        class Owner:
            pass

        first, second = Owner(), Owner()
        seen: list[tuple[str, str]] = []

        async def authorize_first(call, block_reason):
            seen.append(("first", call["function"]))
            return {"block": True, "reason": "first denied"} if call["function"] == "Risky" else None

        async def authorize_second(call, block_reason):
            seen.append(("second", call["function"]))

        executor.set_execution_hooks(authorize=authorize_first, agent=first)
        executor.set_execution_hooks(authorize=authorize_second, agent=second)

        # Each owner's call is decided by its own hook, concurrently.
        results = await asyncio.gather(
            executor.execute_tools_async(first, [{"function": "Risky", "arguments": {}, "tool_call_id": "a"}]),
            executor.execute_tools_async(second, [{"function": "Risky", "arguments": {}, "tool_call_id": "b"}]),
        )
        assert sorted(seen) == [("first", "Risky"), ("second", "Risky")], "hooks did not resolve per invoking agent"
        assert results[0][0]["type"] == "policy_block", "the first owner's denial was lost"
        assert "first denied" in results[0][0]["result"]

        # Clearing one owner leaves the other's hooks (and the global path) intact.
        executor.set_execution_hooks(agent=first)
        global_seen: list[str] = []
        executor.set_execution_hooks(authorize=lambda call, reason: global_seen.append("global"))
        results = await asyncio.gather(
            executor.execute_tools_async(first, [{"function": "Safe", "arguments": {}, "tool_call_id": "c"}]),
            executor.execute_tools_async(second, [{"function": "Safe", "arguments": {}, "tool_call_id": "d"}]),
        )
        assert [r[0]["success"] for r in results] == [True, True]
        assert global_seen == ["global"], "the cleared owner fell back to the global hooks, not the sibling's"
        executor.close()

    @pytest.mark.asyncio
    async def test_unknown_agent_uses_the_global_legacy_hooks(self):
        """Every pre-existing caller (no ``agent=``) keeps today's behavior."""
        executor = self._executor()
        seen: list[str] = []

        async def after(call, result):
            seen.append(call["tool_call_id"])

        executor.set_execution_hooks(after=after)
        await executor.execute_tools_async(object(), [{"function": "Safe", "arguments": {}, "tool_call_id": "g"}])
        assert seen == ["g"]
        executor.close()

    @staticmethod
    def _executor() -> ToolExecutor:
        async def body(arguments):
            return "ran"

        executor = ToolExecutor()
        executor._tool_catalog = ToolCatalog(
            [
                ToolCatalogEntry(ToolIdentity("Safe"), {}, body, effects=EffectMetadata((Effect(EffectKind.READ),))),
                ToolCatalogEntry(ToolIdentity("Risky"), {}, body, effects=EffectMetadata((Effect(EffectKind.EXECUTE),))),
            ]
        )
        return executor


class TestOffloadEligibilityAndOwnership:
    """I7/I8: only explicitly eligible sync work is offloaded, and a cancelled
    dispatch keeps ownership of the physical work until it really stopped."""

    @pytest.mark.asyncio
    async def test_only_explicitly_marked_sync_callables_leave_the_loop(self):
        executor = ToolExecutor()
        loop_thread = threading.get_ident()
        seen: dict[str, bool] = {}

        @blocking_io
        def marked(arguments):
            seen["marked"] = threading.get_ident() != loop_thread
            return "marked"

        def unmarked(arguments):
            seen["unmarked"] = threading.get_ident() != loop_thread
            return "unmarked"

        executor._tool_catalog = ToolCatalog(
            [ToolCatalogEntry(ToolIdentity("Marked"), {}, marked), ToolCatalogEntry(ToolIdentity("Unmarked"), {}, unmarked)]
        )
        results = await executor.execute_tools_async(
            object(),
            [
                {"function": "Marked", "arguments": {}, "tool_call_id": "m"},
                {"function": "Unmarked", "arguments": {}, "tool_call_id": "u"},
            ],
        )
        assert [r["success"] for r in results] == [True, True]
        assert seen == {"marked": True, "unmarked": False}, "unmarked sync work must stay on the calling thread"
        executor.close()

    @pytest.mark.asyncio
    async def test_cancelled_dispatch_waits_for_the_physical_work_to_stop(self):
        controller = AdmissionController(leaf_limit=1)
        executor = ToolExecutor(admission=controller)
        entered, exited = threading.Event(), threading.Event()

        @blocking_io
        def blocking_body(arguments):
            entered.set()
            claim = current_physical_work()
            assert claim is not None, "an eligible offload must publish its ownership claim"
            for _ in range(300):  # cooperatively cancellable work
                if claim.cancel.is_set():
                    break
                time.sleep(0.01)
            exited.set()
            return "returned"

        executor._tool_catalog = ToolCatalog([ToolCatalogEntry(ToolIdentity("Slow"), {}, blocking_body)])
        call = asyncio.create_task(executor.execute_tools_async(object(), [{"function": "Slow", "arguments": {}, "tool_call_id": "s"}]))
        assert await asyncio.to_thread(entered.wait, 5), "the offload never entered a worker"
        assert controller.snapshot()["leaf"]["in_flight"] == 1, "the dispatch ran without holding its permit"
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(call), 5)
        assert exited.is_set(), "the dispatch returned before the physical work stopped"
        assert controller.snapshot()["leaf"]["in_flight"] == 0, "the permit outlived the work it covered"
        executor.close(wait_for_offload=True)

    @pytest.mark.asyncio
    async def test_repeated_cancellation_retains_capacity_and_terminal_ownership_until_worker_exit(self):
        """An uncooperative worker can outlive a status timeout, not its owner."""
        controller = AdmissionController(leaf_limit=1)
        entered, release, exited = threading.Event(), threading.Event(), threading.Event()
        following_authorized = asyncio.Event()
        following_started = asyncio.Event()
        terminal: list[tuple[str, bool, bool]] = []

        @blocking_io
        def uncooperative(arguments):
            entered.set()
            try:
                assert release.wait(15), "the test never released its physical worker"
                return "physical result"
            finally:
                exited.set()

        async def following(arguments):
            assert exited.is_set(), "a following effect overlapped an abandoned worker"
            following_started.set()
            return "following result"

        executor = ToolExecutor(
            admission=controller,
            tool_catalog=ToolCatalog(
                [
                    ToolCatalogEntry(ToolIdentity("Physical"), {}, uncooperative),
                    ToolCatalogEntry(ToolIdentity("Following"), {}, following),
                ]
            ),
        )

        def authorize(call, reason):
            if call["tool_call_id"] == "following":
                following_authorized.set()

        def after(call, result):
            terminal.append((call["tool_call_id"], exited.is_set(), result.get("_effect_unknown", False)))

        executor.set_execution_hooks(authorize=authorize, after=after)
        physical = asyncio.create_task(
            executor.execute_tools_async(object(), [{"function": "Physical", "arguments": {}, "tool_call_id": "physical"}])
        )
        tasks = [physical]
        try:
            assert await asyncio.to_thread(entered.wait, 3), "the physical worker never started"
            follower = asyncio.create_task(
                executor.execute_tools_async(object(), [{"function": "Following", "arguments": {}, "tool_call_id": "following"}])
            )
            tasks.append(follower)
            await asyncio.wait_for(following_authorized.wait(), 3)
            physical.cancel()
            await asyncio.sleep(0.05)
            physical.cancel()
            # Cross the former five-second ownership cutoff while the worker is
            # explicitly gated, rather than mocking a timeout implementation.
            await asyncio.sleep(5.2)
            assert not physical.done(), "cancellation was acknowledged while the physical effect was still active"
            assert not exited.is_set()
            assert not following_started.is_set()
            assert terminal == [], "a terminal hook committed while its worker was still running"
            assert controller.snapshot()["leaf"]["in_flight"] == 1

            physical.cancel()
            await asyncio.sleep(0.05)
            assert not physical.done(), "re-cancellation abandoned the physical claim"
            assert not following_started.is_set()
            assert terminal == []
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(physical), 3)
            results = await asyncio.wait_for(follower, 3)
            assert results[0]["result"] == "following result"
            assert set(terminal) == {("physical", True, True), ("following", True, False)}
            assert controller.snapshot()["leaf"]["in_flight"] == 0
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
            await executor.aclose(wait_for_offload=True)

    @pytest.mark.asyncio
    async def test_unmarked_sync_callable_records_no_offload_claim(self):
        """A call the executor cannot offload truthfully reports no physical claim."""
        executor = ToolExecutor()
        observed: list[object] = []

        def inline(arguments):
            observed.append(current_physical_work())
            return "inline"

        executor._tool_catalog = ToolCatalog([ToolCatalogEntry(ToolIdentity("Inline"), {}, inline)])
        await executor.execute_tools_async(object(), [{"function": "Inline", "arguments": {}, "tool_call_id": "i"}])
        assert observed == [None], "an inline call must not see a claim it does not own"
        executor.close()


class TestAdmissionLoopRetention:
    """S8: a closed loop's admission state must not be retained forever."""

    @pytest.mark.asyncio
    async def test_closed_loop_pools_are_reclaimed(self):
        controller = AdmissionController(leaf_limit=2)

        async def touch():
            permit = await controller.acquire_kind(controller.LEAF)
            permit.release()

        await touch()
        assert controller.tracked_loops == 1
        # Repeated acquisitions (e.g. one per test) must not accumulate entries:
        # closed loops are reclaimed, so exactly one live entry is retained.
        await touch()
        await touch()
        snapshot = controller.snapshot()
        assert controller.tracked_loops == 1, "closed loop entries accumulated"
        assert snapshot["leaf"]["in_flight"] == 0
        assert controller.snapshot()["nested"]["in_flight"] == 0
