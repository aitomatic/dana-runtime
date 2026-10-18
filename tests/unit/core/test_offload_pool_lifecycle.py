"""Offload-pool lifecycle boundaries (P1 offload ownership, PR32 re-review).

The blocking-I/O offload pool is the one lifecycle resource the harness owns per
root session. These tests pin the *boundaries* rather than the wiring:

* a session close releases only its own pool - a sibling session and any future
  session keep working (never a process-wide kill);
* owned worker threads really end when the owner closes;
* a concurrent close cannot start a worker behind a closed pool;
* an awaited close waits for in-flight work without parking the event loop;
* the fallback (default) pool, which has no session owner, is closed at *process*
  lifecycle (interpreter exit) and never by a session.

They use the real :class:`ToolExecutor` / :class:`BlockingIOPool` and a real
:class:`FileIOResource` (no network; the LLM client is never called).
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import subprocess
import sys
import threading
import time

import pytest

from dana.core.agent.star_agent import STARAgent
from dana.core.resource.file_io_resource import FileIOResource
from dana.core.tool.offload import (
    BlockingIOPool,
    BlockingIOShutdown,
    PhysicalWorkClaim,
    blocking_io,
    get_default_blocking_io_pool,
    run_blocking_sync,
)
from dana.core.tool.tool_executor import ToolExecutor


_AGENT_KW: dict = dict(
    agent_type="offload-lifecycle",
    auto_register=False,
    enable_skills=False,
    enable_web_search=False,
    enable_code_execution=False,
    enable_assistant=False,
    compress_timeline=False,
    llm_provider="openai",
    model="gpt-4",
)


def _make_star(tmp: pathlib.Path) -> STARAgent:
    """A real STARAgent owning its own runtime/executor and one file resource."""
    tmp.mkdir()
    (tmp / "a.txt").write_text("needle here\n")
    agent = STARAgent(**_AGENT_KW)
    agent.with_resources(FileIOResource(resource_id=f"io-{tmp.name}", base_path=str(tmp)))
    return agent


async def _read(agent: STARAgent, path: pathlib.Path) -> dict:
    results = await agent._runtime._tool_executor.execute_tools_async(
        agent,
        [{"function": "Read", "arguments": {"file_path": str(path)}, "tool_call_id": "t1"}],
    )
    return results[0]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class TestSessionPoolBoundaries:
    """Closing one session must never poison another, or a future one."""

    @pytest.mark.asyncio
    async def test_two_sessions_one_close_leaves_the_other_serving(self, tmp_path):
        a_dir, b_dir = tmp_path / "a", tmp_path / "b"
        a, b = _make_star(a_dir), _make_star(b_dir)
        ex_a, ex_b = a._runtime._tool_executor, b._runtime._tool_executor

        assert ex_a.offload_pool is not ex_b.offload_pool, "sessions must not share one pool"
        assert (await _read(a, a_dir / "a.txt"))["success"] is True
        assert (await _read(b, b_dir / "a.txt"))["success"] is True
        owned_threads = set(ex_a.offload_pool._pool._threads) | set(ex_b.offload_pool._pool._threads)

        await a.aclose()
        assert ex_a.offload_pool.closed is True
        assert ex_b.offload_pool.closed is False, "a session close killed a sibling's pool"

        served = await _read(b, b_dir / "a.txt")
        assert served["success"] is True, "the surviving session could no longer run a real tool"
        assert get_default_blocking_io_pool().closed is False, "a session close killed the process default"

        await b.aclose()
        await asyncio.sleep(0.1)
        assert not any(thread.is_alive() for thread in owned_threads), "owned workers outlived both sessions"

    @pytest.mark.asyncio
    async def test_future_session_after_close_works(self, tmp_path):
        first_dir = tmp_path / "first"
        first = _make_star(first_dir)
        assert (await _read(first, first_dir / "a.txt"))["success"] is True
        owned_threads = set(first._runtime._tool_executor.offload_pool._pool._threads)
        await first.aclose()

        second = STARAgent(**_AGENT_KW)  # a *new* session after the first closed
        second_dir = tmp_path / "second"
        second_dir.mkdir()
        (second_dir / "a.txt").write_text("needle here\n")
        second.with_resources(FileIOResource(resource_id="io-second", base_path=str(second_dir)))
        try:
            ex = second._runtime._tool_executor
            assert ex.offload_pool.closed is False
            assert (await _read(second, second_dir / "a.txt"))["success"] is True
            owned_threads.update(ex.offload_pool._pool._threads)
        finally:
            await second.aclose()
        await asyncio.sleep(0.1)
        assert not any(thread.is_alive() for thread in owned_threads)


class TestConcurrentCloseRace:
    """A close racing an offload must never leave an unjoinable worker behind."""

    def test_close_claim_under_lock_refuses_instead_of_starting_a_thread(self):
        pool = BlockingIOPool(max_workers=1, name="race-probe")
        outcome: dict = {}

        def caller() -> None:
            try:
                outcome["value"] = pool.run_sync(lambda: "ran")
            except BlockingIOShutdown:
                outcome["refused"] = True

        # Force the exact window the old code lost: the caller passes the closed
        # check, then a concurrent shutdown lands before it creates a worker.
        pool._lock.acquire()
        thread = threading.Thread(target=caller)
        thread.start()
        time.sleep(0.1)  # caller is now blocked acquiring the lock inside submit
        pool._closed = True
        pool._pool = None
        pool._lock.release()
        thread.join(5)

        assert not thread.is_alive()
        assert outcome == {"refused": True}, f"a closed pool started work: {outcome}"
        assert pool.started is False, "a worker was created behind a closed pool"

    def test_contended_submit_and_shutdown_leave_no_orphan_workers(self):
        pool = BlockingIOPool(max_workers=4, name="race-load")
        stop = threading.Event()
        failures: list[BaseException] = []

        def hammer() -> None:
            while not stop.is_set():
                try:
                    pool.run_sync(lambda: 1)
                except BlockingIOShutdown:
                    return  # documented refusal, not a leak
                except BaseException as exc:  # noqa: BLE001 - surface, never hide
                    failures.append(exc)
                    return

        workers = [threading.Thread(target=hammer) for _ in range(8)]
        for w in workers:
            w.start()
        time.sleep(0.05)
        pool.shutdown(wait=True)
        stop.set()
        for w in workers:
            w.join(5)

        assert failures == [], f"a non-lifecycle error escaped a closed pool: {failures}"
        assert all(not w.is_alive() for w in workers)
        assert pool.started is False
        assert not any(thread.name.startswith("race-load") for thread in threading.enumerate())


class TestAsyncCloseNeverParksTheLoop:
    """``await executor.aclose()`` waits for physical work without stalling it."""

    @pytest.mark.asyncio
    async def test_awaited_close_keeps_the_loop_live_while_joining(self):
        executor = ToolExecutor()
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        @blocking_io
        def slow() -> int:
            entered.set()
            try:
                assert release.wait(5), "the loop could not release the worker"
                return 1
            finally:
                finished.set()

        work = asyncio.create_task(executor.offload_pool.run(slow))
        close = None
        try:
            assert await asyncio.to_thread(entered.wait, 5), "the offload never started"
            close = asyncio.create_task(executor.aclose(wait_for_offload=True))
            # This continuation must run while shutdown is joining the blocked worker.
            await asyncio.sleep(0.05)
            assert not finished.is_set(), "shutdown blocked the loop until worker timeout"
            assert not close.done(), "shutdown returned before the physical work finished"
            release.set()
            await close
            assert finished.is_set()
            assert executor.closed is True
            assert await work == 1
        finally:
            release.set()
            await asyncio.gather(work, *([close] if close is not None else []), return_exceptions=True)
            await executor.aclose(wait_for_offload=True)


class TestDefaultPoolProcessBoundary:
    """The ownerless fallback pool is closed at process lifecycle, not per session."""

    def test_session_close_leaves_the_default_pool_usable(self):
        default = get_default_blocking_io_pool()
        assert run_blocking_sync(lambda: 41 + 1) == 42  # starts the default pool

        executor = ToolExecutor()
        executor.close()
        assert default.closed is False, "a session close reached the process-wide default pool"
        assert run_blocking_sync(lambda: 2 + 2) == 4, "the default pool was poisoned by a session close"

    def test_default_pool_is_closed_at_interpreter_exit(self):
        """A program that only uses the default pool still closes it at exit.

        Without a process-lifecycle close the started worker thread stays alive
        at interpreter shutdown, which makes exit wait on the thread join. The
        probe registers its observer *before* importing the pool module, so that
        ``atexit``'s LIFO order runs the observer *after* the pool's own shutdown
        handler and it can report the true post-shutdown state.
        """
        program = (
            "import atexit\n"
            "holder = {}\n"
            "def report():\n"
            "    print('closed-at-exit:', holder['pool'].closed)\n"
            "atexit.register(report)\n"
            "from dana.core.tool.offload import get_default_blocking_io_pool, run_blocking_sync\n"
            "holder['pool'] = get_default_blocking_io_pool()\n"
            "print('value:', run_blocking_sync(lambda: 41 + 1))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=str(pathlib.Path(__file__).resolve().parents[3]),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "OPENAI_API_KEY": "test-key"},
        )
        assert completed.returncode == 0, completed.stderr[-800:]
        assert "value: 42" in completed.stdout
        assert "closed-at-exit: True" in (completed.stdout + completed.stderr), "the default pool was not closed at process exit"


@pytest.mark.asyncio
async def test_rejected_submission_owes_no_physical_work():
    pool = BlockingIOPool()
    pool.shutdown()
    claim = PhysicalWorkClaim()
    with pytest.raises(BlockingIOShutdown):
        await pool.run(lambda: None, _claim=claim)
    assert claim.finished
    assert await claim.drain(timeout=0)
