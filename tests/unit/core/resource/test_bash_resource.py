"""Bash resource: public sync API, cooperative wait cancellation, state safety.

The wait is the one genuinely blocking tool body in the harness: it blocks on a
subprocess for up to the ten-minute maximum. These tests pin the behavior the
executor relies on (I7/I8):

* the public synchronous API is unchanged, so the offload pool can run it;
* a cancelled wait returns promptly and leaves the *command* running (the wait is
  abandoned, not the model's process);
* a shell the OS has reaped can still have its pipes held open by a *descendant*
  it spawned (``sleep 300 & echo hi``): a poll must stay bounded, a listing must
  stay responsive, and the output must still be deliverable later;
* concurrent poll/kill/list calls cannot tear the shared task state.
"""

from __future__ import annotations

import asyncio
from functools import wraps
import inspect
from pathlib import Path
import shlex
import subprocess
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


def _reap_shell(resource: BashResource, task_id: str, timeout: float = 5.0) -> None:
    """Wait until the *shell* is reaped, leaving any descendant process untouched.

    Reaping does not close the pipes: a backgrounded grandchild that inherited them
    keeps the stream open, which is exactly the state under test. The class-level
    task state stays ``completed=False`` because only the drain sets that.
    """
    process = resource._background_tasks[task_id].process
    process.wait(timeout=timeout)
    assert process.poll() is not None, "the shell was not reaped"


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


class TestInheritedPipeDrain:
    """A reaped shell whose pipes a descendant still holds must not stall the loop.

    ``sleep 300 & echo hi`` is the reported shape: the shell exits, the grandchild
    keeps the inherited pipes open, so the stream never reaches end-of-file. Draining
    that stream used to run ``communicate()`` unbounded while holding ``task.lock``;
    because :meth:`BashResource.list_tasks` takes that lock on the event loop, the
    loop stopped until the descendant died. The gate is a rendezvous: the descendant
    holds the pipes until the test creates the file, so no assertion below depends on
    how long a process takes, only on which stream a process holds open.
    """

    @staticmethod
    def _start_pipe_holder(resource: BashResource, gate: Path) -> tuple[str, subprocess.Popen[str]]:
        """Shell prints and exits; a backgrounded descendant holds the pipes until ``gate``.

        The descendant also gives up after roughly thirty seconds, so a test that
        fails before opening the gate cannot leave a looping subshell behind. The
        assertions below all run far inside that ceiling.
        """
        command = (
            f"( n=1500; while [ ! -f {shlex.quote(str(gate))} ] && [ $n -gt 0 ]; do"
            " n=$((n-1)); sleep 0.02; done; printf 'descendant-out' ) & printf 'shell-out'"
        )
        started = resource._execute_background(command, "inherited pipe holder")
        assert started["success"], started
        return started["task_id"], resource._background_tasks[started["task_id"]].process

    @staticmethod
    def _call_with_deadline(call, timeout: float = 10.0):
        """Run ``call`` on a worker thread and report whether it finished.

        A call that never returns is the failure under test, so it must surface as a
        failed assertion instead of hanging the suite. The worker is a daemon, and
        every caller opens its gate in a ``finally``, so a hung call is also released.
        """
        box: dict[str, Any] = {}

        def run() -> None:
            try:
                box["value"] = call()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                box["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            return False, None
        if "error" in box:
            raise box["error"]
        return True, box["value"]

    @staticmethod
    def _observe_drain_entered(process: subprocess.Popen[str], monkeypatch: pytest.MonkeyPatch) -> threading.Event:
        """Flag the moment the class actually enters a process drain.

        The concurrency window under test is "a caller is inside ``communicate``", so
        the tests must observe that state rather than assume it after a sleep. The
        wrapper only signals and delegates: the real blocking call is untouched.
        """
        entered = threading.Event()
        communicate = process.communicate

        def observe(*args, **kwargs):
            entered.set()
            return communicate(*args, **kwargs)

        monkeypatch.setattr(process, "communicate", observe)
        return entered

    def test_poll_of_a_reaped_shell_returns_while_a_descendant_holds_the_pipes(self, resource, tmp_path):
        gate = tmp_path / "pipe-gate"
        task_id, _ = self._start_pipe_holder(resource, gate)
        _reap_shell(resource, task_id)

        try:
            finished, outcome = self._call_with_deadline(lambda: resource.get_task_output(task_id, wait=False))
            assert finished, "a poll must not wait for a descendant to close the pipes"
            assert outcome["status"] == "running", outcome
            # The shell itself is gone, so its own exit status is already known even
            # though the stream it left open is not drained yet.
            assert outcome["exit_code"] == 0
        finally:
            gate.write_text("")

    def test_list_tasks_stays_responsive_while_a_drain_holds_a_reaped_shells_pipe(self, resource, monkeypatch, tmp_path):
        """The event-loop path must never queue behind a caller that owns ``task.lock``."""
        gate = tmp_path / "pipe-gate"
        task_id, process = self._start_pipe_holder(resource, gate)
        _reap_shell(resource, task_id)
        entered = self._observe_drain_entered(process, monkeypatch)

        poll: dict[str, Any] = {}

        def drain() -> None:
            poll.update(resource.get_task_output(task_id, wait=False))

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        try:
            assert entered.wait(5), "the poll never reached its process drain"
            for _ in range(5):
                finished, listed = self._call_with_deadline(resource.list_tasks, timeout=2.0)
                assert finished, "list_tasks blocked behind a drain that owns a task lock"
                row = next(row for row in listed["tasks"] if row["task_id"] == task_id)
                assert row["status"] == "running", "a listing reported a task finished while its pipes were still held"
            thread.join(timeout=2)
            assert not thread.is_alive(), "a poll did not return while a descendant held the pipes"
            assert poll["status"] == "running", poll
        finally:
            gate.write_text("")
        thread.join(timeout=10)
        assert not thread.is_alive()

    def test_list_tasks_stays_responsive_while_a_wait_drains_a_held_pipe(self, resource, monkeypatch, tmp_path):
        """A wait in flight holds the drain lock, never ``task.lock``, for its whole life."""
        gate = tmp_path / "pipe-gate"
        task_id, process = self._start_pipe_holder(resource, gate)
        _reap_shell(resource, task_id)
        entered = self._observe_drain_entered(process, monkeypatch)

        waiter: dict[str, Any] = {}

        def wait() -> None:
            waiter.update(resource.get_task_output(task_id, wait=True, timeout=600000))

        thread = threading.Thread(target=wait, daemon=True)
        thread.start()
        try:
            assert entered.wait(5), "the wait never reached its process drain"
            for _ in range(5):
                finished, listed = self._call_with_deadline(resource.list_tasks, timeout=2.0)
                assert finished, "list_tasks blocked behind a wait in progress"
                row = next(row for row in listed["tasks"] if row["task_id"] == task_id)
                assert row["status"] == "running", row
                finished, polled = self._call_with_deadline(lambda: resource.get_task_output(task_id, wait=False), timeout=2.0)
                assert finished, "a poll queued behind a wait in progress"
                assert polled["status"] == "running", polled
                assert not polled.get("cancelled"), "a poll must not report a cancellation it never received"
        finally:
            gate.write_text("")
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert waiter["status"] == "completed", waiter
        assert waiter["stdout"] == "shell-outdescendant-out", "the drain lost output that had already been written"

    def test_repeated_polls_are_stable_and_lose_no_output_when_the_pipe_closes(self, resource, tmp_path):
        gate = tmp_path / "pipe-gate"
        task_id, _ = self._start_pipe_holder(resource, gate)
        _reap_shell(resource, task_id)

        try:
            for _ in range(20):
                finished, polled = self._call_with_deadline(lambda: resource.get_task_output(task_id, wait=False), timeout=2.0)
                assert finished, "a poll of a held pipe blocked"
                assert polled["status"] == "running", polled
        finally:
            gate.write_text("")

        final = resource.get_task_output(task_id, timeout=5000)
        assert final["status"] == "completed", final
        assert final["exit_code"] == 0
        assert final["stdout"] == "shell-outdescendant-out", "polls must not lose or duplicate output"
        assert resource.get_task_output(task_id, wait=False) == final

    def test_list_tasks_does_not_strand_a_reaped_shells_output(self, resource):
        started = resource._execute_background("printf 'ready'", "quick probe")
        task_id = started["task_id"]
        _reap_shell(resource, task_id)
        assert not resource._background_tasks[task_id].completed

        finished, listed = self._call_with_deadline(resource.list_tasks, timeout=5.0)
        assert finished
        row = next(row for row in listed["tasks"] if row["task_id"] == task_id)
        assert row["status"] == "completed", "a listing reported a finished command as still running"
        outcome = resource.get_task_output(task_id, wait=False)
        assert outcome["status"] == "completed"
        assert outcome["stdout"] == "ready", "the listing drained the pipes without recording the output"

    def test_kill_with_a_holding_descendant_stays_bounded_and_finishes_the_task(self, resource, tmp_path):
        gate = tmp_path / "pipe-gate"
        task_id, process = self._start_pipe_holder(resource, gate)
        _reap_shell(resource, task_id)

        try:
            finished, outcome = self._call_with_deadline(lambda: resource.kill_task(task_id))
            assert finished, "a kill must not wait for a descendant to close the pipes"
            assert outcome["success"] is True
            assert process.poll() is not None

            finished, listed = self._call_with_deadline(resource.list_tasks, timeout=2.0)
            assert finished, "the loop blocked behind a kill's drain"
            row = next(row for row in listed["tasks"] if row["task_id"] == task_id)
            # The shell was already gone, so no signal was delivered and its own code
            # stands; the descendant keeps the stream open, so the drain is unfinished.
            assert row["status"] == "running", "a listing reported a drain that never completed"
            assert row["exit_code"] == 0, "a reaped process must report its own exit status"
        finally:
            gate.write_text("")
        final = resource.get_task_output(task_id, timeout=5000)
        assert final["status"] == "completed", final
        assert final["exit_code"] == 0, "the real exit status was overwritten by a signal that was never sent"
        assert final["stdout"] == "shell-outdescendant-out", final

    def test_kill_records_output_a_held_command_had_already_written(self, resource, tmp_path):
        marker = tmp_path / "command-ready"
        # The command writes, then blocks forever: the signal is delivered while the
        # output is sitting in the pipe, so losing it is the failure under test.
        command = f"printf 'kill-output'; : > {shlex.quote(str(marker))}; exec sleep 300"
        started = resource._execute_background(command, "kill output probe")
        assert started["success"], started
        task_id = started["task_id"]

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.005)
        assert marker.exists(), "the command never reached its blocking point"

        finished, outcome = self._call_with_deadline(lambda: resource.kill_task(task_id))
        assert finished
        assert outcome["success"] is True
        polled = resource.get_task_output(task_id, wait=False)
        assert polled["status"] == "completed"
        assert polled["stdout"] == "kill-output", "killing a task dropped output it had already written"
        assert polled["exit_code"] == -15

    def test_kill_under_a_busy_drain_lock_can_be_finished_by_a_later_poll(self, resource, tmp_path):
        """A kill that cannot drain must not claim the task is drained.

        The shell writes, backgrounds a descendant that holds the pipes, then waits.
        The kill's bounded drain is blocked by a caller that already owns the drain
        lock, so the final output cannot be recorded at kill time. Marking the task
        drained there stranded that output for good because nothing would ever read the
        open stream again. The kill must instead record the terminal signal status and
        leave the drain open for a later call to finish.
        """
        gate = tmp_path / "pipe-gate"
        ready = tmp_path / "descendant-ready"
        command = (
            f"printf 'killed-out'; ( : > {shlex.quote(str(ready))}; n=1500;"
            f" while [ ! -f {shlex.quote(str(gate))} ] && [ $n -gt 0 ]; do n=$((n-1)); sleep 0.02; done;"
            " printf 'descendant-out' ) & wait"
        )
        started = resource._execute_background(command, "killed pipe holder")
        assert started["success"], started
        task_id = started["task_id"]
        task = resource._background_tasks[task_id]

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.005)
        assert ready.exists(), "the descendant never started"

        # Stand in for a waiter that owns the drain lock for the whole kill.
        assert task.wait_lock.acquire(timeout=5)
        try:
            finished, outcome = self._call_with_deadline(lambda: resource.kill_task(task_id))
            assert finished, "the kill queued behind the busy drain lock instead of staying bounded"
            assert outcome["success"] is True
            assert task.completed is False, "the kill claimed a drain it never performed"
            assert task.exit_code == -15, "the terminal signal status was not retained"

            row = next(row for row in resource.list_tasks()["tasks"] if row["task_id"] == task_id)
            assert row["status"] == "running", "a listing reported a drain that never completed"
            assert row["exit_code"] == -15, "the recorded terminal signal status was not reported"

            polled = resource.get_task_output(task_id, wait=False)
            assert polled["exit_code"] == -15, "the recorded terminal code was reported as an unknown status"
            for _ in range(3):
                assert resource.get_task_output(task_id, wait=False)["stdout"] == "", "a poll reported output it had not read"
        finally:
            task.wait_lock.release()
            gate.write_text("")

        final = resource.get_task_output(task_id, timeout=10000)
        assert final["status"] == "completed", final
        assert final["stdout"] == "killed-outdescendant-out", "output already written was stranded by the kill"
        assert final["exit_code"] == -15
        assert resource.get_task_output(task_id, wait=False) == final, "the recovered output was not recorded exactly once"


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
