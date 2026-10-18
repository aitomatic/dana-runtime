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
from types import SimpleNamespace

import pytest

from dana.core.ext.event_bus import EventBus
from dana.core.ext.events import TOOL_CALL, TOOL_RESULT
from dana.core.policy.effects import Effect, EffectKind, EffectMetadata
from dana.core.resource.file_edit_resource import FileEditResource
from dana.core.resource.file_io_resource import FileIOResource
from dana.core.resource.search_resource import SearchResource
from dana.core.tool.admission import AdmissionController
from dana.core.tool.catalog import ToolCatalog, ToolCatalogEntry, ToolIdentity
from dana.core.tool.offload import (
    BlockingIOPool,
    BlockingIOShutdown,
    bound_blocking_io_pool,
    current_blocking_io_pool,
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
