"""Offload-pool lifecycle boundaries (P1 offload ownership, PR32 re-review).

The blocking-I/O offload pool is the one lifecycle resource the harness owns per
root session. These tests pin the *boundaries* rather than the wiring:

* a session close releases only its own pool - a sibling session and any future
  session keep working (never a process-wide kill);
* owned worker threads really end when the owner closes;
* a concurrent close cannot start a worker behind a closed pool;
* an awaited close waits for in-flight work without parking the event loop;
* shutdown owns every live claim - async and legacy sync submissions alike -
  signals them before joining, and never reports non-cooperative work as stopped;
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
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from dana.core.agent.star_agent import STARAgent
from dana.core.resource.file_io_resource import FileIOResource
from dana.core.tool import offload
from dana.core.tool.offload import (
    BlockingIOPool,
    BlockingIOShutdown,
    PhysicalWorkClaim,
    blocking_io,
    current_physical_work,
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


async def _await_settled(predicate, message: str, timeout: float = 10.0) -> None:
    """Yield to the loop until ``predicate`` holds (a bounded state gate)."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail(message)
        await asyncio.sleep(0.005)


def _live_claims(pool: BlockingIOPool) -> frozenset[PhysicalWorkClaim]:
    """Snapshot the pool's live-work registry under its registry lock."""
    with pool._claims_lock:
        return frozenset(pool._claims)


class _CooperativeCall:
    """A module-level cooperative blocking callable with a stable qualified name.

    Defined at module scope so its ``__qualname__`` is exactly
    ``_CooperativeCall.run``: shutdown diagnostics must name the callable, which
    a test-local closure cannot prove. It polls the claim published by the pool
    in short slices and returns as soon as a cancellation is requested, so the
    tests can gate on events instead of elapsed time.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.observed_cancel = threading.Event()
        #: Test-only escape hatch so a failed assertion can never strand a worker.
        self.stop = threading.Event()

    def run(self) -> str:
        claim = current_physical_work()
        assert claim is not None, "an offloaded callable must own a physical claim"
        self.entered.set()
        while not (claim.cancel.is_set() or self.stop.is_set()):
            time.sleep(0.002)
        if claim.cancel.is_set():
            self.observed_cancel.set()
        return "cancelled"


class _NonCooperativeCall:
    """A module-level blocking callable that ignores cancellation requests.

    Its body waits on a test-owned event only, so it keeps owning physical work
    after a close until the test releases it.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def run(self) -> str:
        self.entered.set()
        assert self.release.wait(20), "the test never released the non-cooperative call"
        return "finished"


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


class TestShutdownOwnsLiveClaims:
    """Shutdown must account for *all* live physical work, not just dispatch claims."""

    @pytest.mark.asyncio
    async def test_async_cooperative_claim_stops_waiting_on_shutdown(self):
        pool = BlockingIOPool(max_workers=1, name="shutdown-async")
        call = _CooperativeCall()
        work = asyncio.create_task(pool.run(call.run))
        try:
            assert await asyncio.to_thread(call.entered.wait, 5), "the async offload never started"
            assert len(_live_claims(pool)) == 1, "a submitted call was not registered with its pool"

            await asyncio.to_thread(pool.shutdown, True)

            assert call.observed_cancel.is_set(), "shutdown never asked the cooperative call to stop"
            assert await work == "cancelled"
            await _await_settled(lambda: not _live_claims(pool), "the settled async claim stayed registered")
        finally:
            call.stop.set()
            await asyncio.gather(work, return_exceptions=True)
            pool.shutdown(wait=True)

    def test_sync_cooperative_claim_stops_waiting_on_shutdown(self):
        pool = BlockingIOPool(max_workers=1, name="shutdown-sync")
        call = _CooperativeCall()
        outcome: dict = {}

        def caller() -> None:
            try:
                outcome["value"] = pool.run_sync(call.run)
            except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion
                outcome["error"] = exc

        thread = threading.Thread(target=caller)
        thread.start()
        try:
            assert call.entered.wait(5), "the legacy sync offload never started"
            assert len(_live_claims(pool)) == 1, "run_sync work was untracked by its pool"
            pool.shutdown(wait=True)
        finally:
            call.stop.set()
            thread.join(10)
            pool.shutdown(wait=True)

        assert not thread.is_alive(), "shutdown did not join the sync caller"
        assert "error" not in outcome, f"the sync caller failed: {outcome.get('error')!r}"
        assert call.observed_cancel.is_set(), "shutdown never asked the sync call to stop"
        assert outcome.get("value") == "cancelled"
        assert not _live_claims(pool), "the settled sync claim stayed registered"

    @pytest.mark.asyncio
    async def test_queued_cooperative_claim_is_signalled_before_the_join(self):
        """A queued body starts *after* the signal, so it must already see it.

        This is what makes signalling before the join observable: the queued call
        never had a chance to run when shutdown requested cancellation, yet it
        must return immediately rather than wait for its own work to finish.
        """
        pool = BlockingIOPool(max_workers=1, name="shutdown-queued")
        first, second = _CooperativeCall(), _CooperativeCall()
        holder = asyncio.create_task(pool.run(first.run))
        queued = None
        try:
            assert await asyncio.to_thread(first.entered.wait, 5), "the first offload never started"
            queued = asyncio.create_task(pool.run(second.run))
            await _await_settled(lambda: len(_live_claims(pool)) == 2, "the queued call was not registered")

            await asyncio.to_thread(pool.shutdown, True)

            assert await holder == "cancelled"
            assert await queued == "cancelled"
            assert second.observed_cancel.is_set()
            await _await_settled(lambda: not _live_claims(pool), "queued work left a claim behind")
        finally:
            first.stop.set()
            second.stop.set()
            await asyncio.gather(holder, *([queued] if queued is not None else []), return_exceptions=True)
            pool.shutdown(wait=True)

    def test_submission_racing_shutdown_never_escapes_the_registry(self, monkeypatch):
        pool = BlockingIOPool(max_workers=1, name="shutdown-submit-race")
        pool.run_sync(lambda: None)
        executor = pool._pool
        submit = executor.submit
        submitted = threading.Event()
        release_submission = threading.Event()
        close_entered = threading.Event()
        close_done = threading.Event()
        call = _CooperativeCall()
        outcome: dict = {}

        def gated_submit(*args, **kwargs):
            future = submit(*args, **kwargs)
            submitted.set()
            assert release_submission.wait(10), "the test never released submission"
            return future

        def caller():
            try:
                outcome["value"] = pool.run_sync(call.run)
            except BaseException as exc:
                outcome["error"] = exc

        def closer():
            close_entered.set()
            pool.shutdown(wait=True)
            close_done.set()

        monkeypatch.setattr(executor, "submit", gated_submit)
        caller_thread = threading.Thread(target=caller)
        close_thread = threading.Thread(target=closer)
        caller_thread.start()
        try:
            assert submitted.wait(5), "submission never entered the critical section"
            assert call.entered.wait(5), "accepted physical work never started"
            (claim,) = _live_claims(pool)
            assert not claim.finished
            close_thread.start()
            assert close_entered.wait(5), "the concurrent close never started"
            assert not close_done.is_set(), "close passed an in-progress submission"
            release_submission.set()
            close_thread.join(10)
            caller_thread.join(10)
            assert close_done.is_set()
            assert outcome == {"value": "cancelled"}
            assert claim.finished and not _live_claims(pool)
        finally:
            release_submission.set()
            call.stop.set()
            caller_thread.join(10)
            if close_thread.ident is not None:
                close_thread.join(10)
            pool.shutdown(wait=True)
        assert not caller_thread.is_alive() and not close_thread.is_alive()

    @pytest.mark.asyncio
    async def test_queued_claim_settles_when_a_close_drops_it(self):
        pool = BlockingIOPool(max_workers=1, name="shutdown-drop")
        running_call = _NonCooperativeCall()
        body_ran: list[str] = []

        running = asyncio.create_task(pool.run(running_call.run))
        assert await asyncio.to_thread(running_call.entered.wait, 5), "the blocking call never started"
        queued = asyncio.create_task(pool.run(lambda: body_ran.append("ran")))
        await _await_settled(lambda: len(_live_claims(pool)) == 2, "the queued call was not registered")
        queued_claim = next(claim for claim in _live_claims(pool) if claim.started_at is None)
        running_claim = next(claim for claim in _live_claims(pool) if claim.started_at is not None)

        try:
            await asyncio.to_thread(pool.shutdown, False)
            await _await_settled(lambda: queued_claim.finished, "a dropped queued claim never settled")
            await _await_settled(lambda: queued_claim not in _live_claims(pool), "a dropped claim stayed registered")

            assert queued_claim.started_at is None, "a dropped queued claim was reported as having run"
            assert body_ran == [], "a queued body ran after the close dropped it"
            assert not running_claim.finished, "a non-cooperative running claim was reported as stopped"
            assert running_claim in _live_claims(pool), "a running claim left the registry before returning"
            with pytest.raises(asyncio.CancelledError):
                await queued
        finally:
            running_call.release.set()
            await asyncio.gather(running, queued, return_exceptions=True)

        await _await_settled(lambda: running_claim.finished, "the running claim never settled")
        await _await_settled(lambda: not _live_claims(pool), "the registry never settled")

    @pytest.mark.asyncio
    async def test_cancelled_waiter_never_releases_running_ownership(self):
        pool = BlockingIOPool(max_workers=1, name="shutdown-waiter")
        call = _NonCooperativeCall()
        claim = PhysicalWorkClaim()
        work = asyncio.create_task(pool.run(call.run, _claim=claim))
        assert await asyncio.to_thread(call.entered.wait, 5), "the blocking call never started"

        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work

        assert not claim.finished, "a cancelled waiter reported physical work as stopped"
        assert claim in _live_claims(pool), "a cancelled waiter dropped the pool's ownership record"

        call.release.set()
        await _await_settled(lambda: claim.finished, "the physical claim never settled")
        await _await_settled(lambda: not _live_claims(pool), "the registry never settled")
        await asyncio.to_thread(pool.shutdown, True)

    @pytest.mark.asyncio
    async def test_shutdown_logs_each_still_running_claim_by_name_and_physical_runtime(self, monkeypatch):
        clock = [100.0]
        monkeypatch.setattr(offload, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        pool = BlockingIOPool(max_workers=1, name="shutdown-diag")
        predecessor, call = _NonCooperativeCall(), _NonCooperativeCall()
        preceding = asyncio.create_task(pool.run(predecessor.run))
        running = queued = None
        try:
            assert await asyncio.to_thread(predecessor.entered.wait, 5)
            clock[0] = 110.0
            running = asyncio.create_task(pool.run(call.run))
            await _await_settled(lambda: len(_live_claims(pool)) == 2, "the waiting call was not registered")
            clock[0] = 200.0
            predecessor.release.set()
            assert await preceding == "finished"
            assert await asyncio.to_thread(call.entered.wait, 5)
            queued = asyncio.create_task(pool.run(lambda: None))
            await _await_settled(lambda: len(_live_claims(pool)) == 2, "the queued call was not registered")
            running_claim = next(claim for claim in _live_claims(pool) if claim.started_at is not None)
            clock[0] = 207.125

            with capture_logs() as logs:
                await asyncio.to_thread(pool.shutdown, False)
            await _await_settled(
                lambda: _live_claims(pool) == frozenset({running_claim}),
                "queued work left a claim behind",
            )

            still_running = [entry for entry in logs if entry["event"] == "blocking_io_pool_claim_still_running"]
            assert len(still_running) == 1, f"expected exactly the running claim to be reported: {still_running}"
            entry = still_running[0]
            assert entry["callable"] == "_NonCooperativeCall.run", "the diagnostic did not name the stuck callable"
            assert entry["pool"] == "shutdown-diag"
            assert entry["running_s"] == 7.125, "physical runtime must exclude the 90 seconds spent queued"
            assert set(entry) == {"event", "log_level", "pool", "callable", "running_s"}, (
                f"the diagnostic reported more than the bounded name and runtime: {sorted(entry)}"
            )
            assert not running_claim.finished, "a non-cooperative claim was reported as stopped"
            assert _live_claims(pool) == frozenset({running_claim}), "the non-cooperative claim left the registry"
        finally:
            predecessor.release.set()
            call.release.set()
            await asyncio.gather(preceding, *[task for task in (running, queued) if task is not None], return_exceptions=True)
            await pool.ashutdown(wait=True)

        assert await running == "finished"
        await _await_settled(lambda: not _live_claims(pool), "a claim survived its physical completion")

    @pytest.mark.asyncio
    async def test_shutdown_diagnoses_queued_work_that_starts_during_join(self, monkeypatch):
        pool = BlockingIOPool(max_workers=1, name="shutdown-late-start")
        first_entered = threading.Event()
        release_first = threading.Event()
        snapshot_logged = threading.Event()
        blocker = _NonCooperativeCall()
        log_running = pool._log_running_claims

        def first():
            first_entered.set()
            assert release_first.wait(10), "the test never released the first worker"

        def observe_diagnosis(*args):
            log_running(*args)
            if not args:
                snapshot_logged.set()

        monkeypatch.setattr(pool, "_log_running_claims", observe_diagnosis)
        preceding = asyncio.create_task(pool.run(first))
        running = closing = None
        try:
            assert await asyncio.to_thread(first_entered.wait, 5)
            running = asyncio.create_task(pool.run(blocker.run))
            await _await_settled(lambda: len(_live_claims(pool)) == 2, "the second call was not queued")
            with capture_logs() as logs:
                closing = asyncio.create_task(pool.ashutdown(wait=True))
                assert await asyncio.to_thread(snapshot_logged.wait, 5)
                assert not blocker.entered.is_set()
                release_first.set()
                assert await asyncio.to_thread(blocker.entered.wait, 5)
                assert not closing.done(), "shutdown must still own the non-cooperative worker"
                assert any(
                    entry["event"] == "blocking_io_pool_claim_still_running" and entry["callable"] == "_NonCooperativeCall.run"
                    for entry in logs
                ), "the callable blocking the join was omitted after it started from the queue"
        finally:
            release_first.set()
            blocker.release.set()
            await asyncio.gather(preceding, *[task for task in (running, closing) if task is not None], return_exceptions=True)
            await pool.ashutdown(wait=True)
