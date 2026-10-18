"""Bash resource: public sync API, cooperative wait cancellation, state safety.

The wait is the one genuinely blocking tool body in the harness: it blocks on a
subprocess for up to the ten-minute maximum. These tests pin the behavior the
executor relies on (I7/I8):

* the public synchronous API is unchanged, so the offload pool can run it;
* a cancelled wait returns promptly and leaves the *command* running (the wait is
  abandoned, not the model's process);
* concurrent poll/kill/list calls cannot tear the shared task state.
"""

from __future__ import annotations

import asyncio
from functools import wraps
import inspect
import shlex
import threading
import time
from typing import Any

import pytest

from dana.core.resource.bash import BashResource
from dana.core.tool.admission import AdmissionController
from dana.core.tool.offload import blocking_io, is_blocking_io
from dana.core.tool.tool_executor import ToolExecutor


def _start_sleeper(resource: BashResource, seconds: int = 30) -> str:
    task = resource._execute_background(f"exec sleep {seconds}", "long-running probe")
    assert task["success"], task
    return task["task_id"]


@pytest.fixture
def resource(tmp_path):
    bash = BashResource(working_directory=str(tmp_path))
    yield bash
    for task_id in list(bash._background_tasks):
        bash.kill_task(task_id)


class TestPublicContract:
    def test_synchronous_wait_api_is_preserved_and_offload_eligible(self, resource):
        """The executor offloads this method, so it must stay sync and marked."""
        assert not inspect.iscoroutinefunction(resource.get_task_output)
        assert is_blocking_io(resource.get_task_output)
        assert not inspect.iscoroutinefunction(resource.kill_task)
        assert not inspect.iscoroutinefunction(resource.list_tasks)
        # The SIGTERM/SIGKILL wait block is offload-eligible; list_tasks is cheap
        # registry reads that must stay on the loop.
        assert is_blocking_io(resource.kill_task)
        assert not is_blocking_io(resource.list_tasks)

    def test_unknown_task_is_not_found(self, resource):
        result = resource.get_task_output("does-not-exist")
        assert result["success"] is False
        assert result["status"] == "not_found"


class TestCooperativeWait:
    def test_wait_without_a_claim_blocks_until_the_command_finishes(self, resource):
        """The documented non-cancellable path (a synchronous caller) still waits."""
        started = resource._execute_background("exec sleep 0.3", "short probe")
        result = resource.get_task_output(started["task_id"], timeout=10000)
        assert result["status"] == "completed"
        assert result["exit_code"] == 0

    def test_cancelled_wait_returns_promptly_and_leaves_the_command_running(self, resource):
        """A cancelled wait stops *waiting*, never the model's command."""
        from dana.core.tool.offload import PhysicalWorkClaim, _physical_claim

        task_id = _start_sleeper(resource)
        claim = PhysicalWorkClaim()
        box: dict[str, Any] = {}

        def waiter() -> None:
            # Stand in for the offload pool's worker: publish the dispatch's claim
            # so the wait can observe the cancellation request.
            _physical_claim.set(claim)
            box["result"] = resource.get_task_output(task_id, wait=True, timeout=600000)

        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and "result" not in box:
            claim.request_cancel()
            time.sleep(0.02)

        assert "result" in box, "the cancelled wait never returned"
        thread.join(timeout=5)
        assert not thread.is_alive(), "the wait thread outlived its cancellation"
        assert box["result"]["cancelled"] is True
        assert box["result"]["status"] == "running", "abandoning the wait must not report the command as finished"
        still_running = resource.get_task_output(task_id, wait=False)
        assert still_running["status"] == "running"

    def test_wait_timeout_reports_running_without_killing(self, resource):
        task_id = _start_sleeper(resource)
        started = time.monotonic()
        result = resource.get_task_output(task_id, wait=True, timeout=200)
        assert result["status"] == "running"
        assert "still running" in result["stderr"]
        assert time.monotonic() - started < 5, "the bounded wait overran its timeout"

    @pytest.mark.asyncio
    async def test_queued_waiters_timeout_or_cancel_without_abandoning_output(self, resource, monkeypatch, tmp_path):
        gate = tmp_path / "finish-command"
        command = (
            "printf 'first-'; printf 'err-' >&2; "
            f"while [ ! -f {shlex.quote(str(gate))} ]; do sleep 0.02; done; "
            "printf 'last'; printf 'end' >&2"
        )
        started = resource._execute_background(command, "queued waiter output probe")
        assert started["success"], started
        task_id = started["task_id"]
        process = resource._background_tasks[task_id].process
        communicating = threading.Event()
        communicate = process.communicate

        def observe_communicate(*args, **kwargs):
            communicating.set()
            return communicate(*args, **kwargs)

        monkeypatch.setattr(process, "communicate", observe_communicate)
        entered = {timeout: threading.Event() for timeout in (600000, 120, 599999)}
        exited = {timeout: threading.Event() for timeout in entered}
        original = resource.get_task_output

        @blocking_io
        @wraps(original)
        def observe_wait(*args, **kwargs):
            timeout = kwargs.get("timeout")
            if timeout in entered:
                entered[timeout].set()
            try:
                return original(*args, **kwargs)
            finally:
                if timeout in exited:
                    exited[timeout].set()

        monkeypatch.setattr(resource, "get_task_output", observe_wait)
        executor = ToolExecutor(
            admission=AdmissionController(leaf_limit=3),
            tool_name_registry_getter=lambda: {"BashWait": (resource, "get_task_output")},
        )
        tasks = []

        def dispatch(timeout, call_id):
            task = asyncio.create_task(
                executor.execute_tools_async(
                    object(),
                    [{"function": "BashWait", "arguments": {"task_id": task_id, "timeout": timeout}, "tool_call_id": call_id}],
                )
            )
            tasks.append(task)
            return task

        try:
            first = dispatch(600000, "first")
            assert await asyncio.to_thread(communicating.wait, 3), "the first caller never started its real process wait"
            short = dispatch(120, "short")
            short_results = await asyncio.wait_for(asyncio.shield(short), 1)
            assert short_results[0]["result"]["status"] == "running"
            assert "still running after 0.12s wait" in short_results[0]["result"]["stderr"]
            assert exited[120].is_set()
            assert not first.done(), "the short caller waited for the long caller to finish"

            cancelled = dispatch(599999, "cancelled")
            assert await asyncio.to_thread(entered[599999].wait, 3)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(cancelled), 1)
            assert exited[599999].is_set(), "cancellation acknowledged before the queued worker stopped"
            assert not exited[600000].is_set()
            assert not first.done()
            assert resource.get_task_output(task_id, wait=False)["status"] == "running"

            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(first), 1)
            assert exited[600000].is_set()
            assert process.poll() is None, "abandoning a wait killed the command"
            gate.write_text("")
            results = await asyncio.wait_for(dispatch(5000, "finish"), 3)
            final = results[0]["result"]
            assert final["status"] == "completed"
            assert final["exit_code"] == 0
            assert final["stdout"] == "first-last"
            assert final["stderr"] == "err-end"
            assert resource.get_task_output(task_id, wait=False) == final
            assert resource.get_task_output(task_id, timeout=100) == final
        finally:
            gate.write_text("")
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
            await executor.aclose(wait_for_offload=True)

    @pytest.mark.asyncio
    async def test_queue_time_and_process_wait_share_one_timeout(self, resource, monkeypatch):
        task_id = _start_sleeper(resource)
        process = resource._background_tasks[task_id].process
        communicating = threading.Event()
        communicate = process.communicate

        def observe_communicate(*args, **kwargs):
            communicating.set()
            return communicate(*args, **kwargs)

        monkeypatch.setattr(process, "communicate", observe_communicate)
        queued = threading.Event()
        original = resource.get_task_output
        started_at = []

        @blocking_io
        @wraps(original)
        def observe_wait(*args, **kwargs):
            if kwargs.get("timeout") == 2000:
                started_at.append(time.monotonic())
                queued.set()
            return original(*args, **kwargs)

        monkeypatch.setattr(resource, "get_task_output", observe_wait)
        executor = ToolExecutor(
            admission=AdmissionController(leaf_limit=2),
            tool_name_registry_getter=lambda: {"BashWait": (resource, "get_task_output")},
        )
        tasks = []

        def dispatch(timeout, call_id):
            task = asyncio.create_task(
                executor.execute_tools_async(
                    object(),
                    [{"function": "BashWait", "arguments": {"task_id": task_id, "timeout": timeout}, "tool_call_id": call_id}],
                )
            )
            tasks.append(task)
            return task

        try:
            first = dispatch(600000, "first")
            assert await asyncio.to_thread(communicating.wait, 3)
            second = dispatch(2000, "second")
            assert await asyncio.to_thread(queued.wait, 3)
            await asyncio.sleep(1.5)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(first), 1)
            remaining = max(0.05, 2.7 - (time.monotonic() - started_at[0]))
            results = await asyncio.wait_for(asyncio.shield(second), remaining)
            assert results[0]["result"]["status"] == "running"
            assert time.monotonic() - started_at[0] < 2.7, "the process wait restarted the timeout after leaving the queue"
            assert process.poll() is None
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
            await executor.aclose(wait_for_offload=True)


class TestSharedStateSafety:
    def test_concurrent_poll_kill_and_list_do_not_tear_task_state(self, resource):
        """Task state is shared with offload workers, so every reader is locked."""
        task_ids = [_start_sleeper(resource, 30) for _ in range(4)]
        errors: list[BaseException] = []

        def hammer() -> None:
            try:
                for _ in range(60):
                    resource.list_tasks()
                    for task_id in task_ids:
                        resource.get_task_output(task_id, wait=False)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=hammer, daemon=True) for _ in range(4)]
        for thread in threads:
            thread.start()
        for task_id in task_ids[:2]:
            resource.kill_task(task_id)
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()

        assert errors == [], errors
        listed = {task["task_id"]: task for task in resource.list_tasks()["tasks"]}
        assert set(task_ids) <= set(listed)
        killed = [listed[task_id]["status"] for task_id in task_ids[:2]]
        assert killed == ["completed", "completed"], "a killed task must be recorded as finished"

    def test_two_concurrent_waiters_do_not_split_the_output(self, resource):
        """Only one waiter may drain the pipe; the other reads the recorded buffers."""
        started = resource._execute_background("printf 'shared-output'", "output probe")
        task_id = started["task_id"]
        results: list[dict[str, Any]] = []
        lock = threading.Lock()

        def waiter() -> None:
            outcome = resource.get_task_output(task_id, wait=True, timeout=5000)
            with lock:
                results.append(outcome)

        threads = [threading.Thread(target=waiter, daemon=True) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()

        assert len(results) == 3
        assert {row["status"] for row in results} == {"completed"}
        assert {row["stdout"] for row in results} == {"shared-output"}, "a waiter lost or duplicated output"

    def test_kill_task_unblocks_a_wait_in_progress(self, resource):
        task_id = _start_sleeper(resource)
        outcome: dict[str, Any] = {}

        def waiter() -> None:
            outcome.update(resource.get_task_output(task_id, wait=True, timeout=30000))

        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        time.sleep(0.3)
        resource.kill_task(task_id)
        thread.join(timeout=10)
        assert not thread.is_alive(), "a kill must end a wait in progress"
        assert outcome["status"] == "completed"
        assert outcome["exit_code"] in (-15, -9)
