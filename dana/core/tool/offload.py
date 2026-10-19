"""Bounded, lifecycle-owned offload for eligible blocking I/O (P1 responsiveness).

Python's event loop stalls on every blocking call made from a coroutine. This
module moves *eligible* blocking work (filesystem I/O, subprocess-free probes)
onto a bounded thread pool so the loop keeps servicing other coroutines, without
turning the harness into a thread-framework.

Ownership model (important)
---------------------------
There are two kinds of pools:

* **Session/executor-owned pool** — created by (and owned by) one root
  :class:`~dana.core.tool.tool_executor.ToolExecutor` / agent session. Closing
  that session closes *only* its own pool: other sessions keep running, and new
  sessions create fresh pools. Nested agents dispatched by that executor share
  the same pool because the pool is bound to the dispatch context
  (:func:`bind_blocking_io_pool`) rather than passed around by hand.
* **Default pool** — the fallback for direct/legacy callers that never bound a
  pool (a bare ``FileIOResource`` used outside a session). It is process-wide, so
  it has no session owner: it is closed at **process lifecycle** (an ``atexit``
  handler registered by this module, which is why a session never has to close
  it) and can also be closed explicitly with
  :func:`shutdown_default_blocking_io_executor`; **a session close never kills
  it** (that would poison concurrent and future sessions).

``run_blocking`` resolves the pool at call time: bound pool first, default pool
otherwise. Exceptions from the callable propagate unchanged — offloading never
swallows an error and never reports abandoned work as cancelled (a thread cannot
be interrupted; queued-but-unstarted work is dropped on close, running work is
allowed to finish).

Offload eligibility is **explicit opt-in** (I7)
----------------------------------------------
A synchronous callable is moved to the pool only when it declares itself
blocking I/O with :func:`blocking_io`. Everything else stays on the event loop:
a synchronous callable that touches in-loop mutable state must not race a worker
thread, so dispatch keeps it inline unless its author says otherwise. Marked
callables receive a :class:`PhysicalWorkClaim` (see below) so they can stop
waiting cooperatively.

Physical work ownership (I8)
---------------------------
A thread cannot be interrupted, so a cancelled dispatch must not release its
admission permit while its offloaded call is still running, and must not park a
worker thread for a long timeout either. A dispatch that offloads binds a
:class:`PhysicalWorkClaim`: the pool marks ``done`` when the call returns, and
dispatch asks cooperative callables to stop waiting by setting ``cancel``.

The pool itself also has to account for every physical call, not only the ones a
dispatch happens to watch. Each call started through the async
:meth:`BlockingIOPool.run` or the legacy :meth:`BlockingIOPool.run_sync` is
registered with its pool for as long as its physical work is live. Registration,
the closed check and the shutdown snapshot all happen under the pool's lock, so
a submission and a concurrent close cannot interleave into an unowned worker.
Shutdown asks every live claim to stop waiting *before* it joins the workers,
without a join time box, and then reports each claim that is still executing,
naming the callable and how long its body has actually run.
"""

from __future__ import annotations

import asyncio
import atexit
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
import contextlib
import contextvars
from dataclasses import dataclass, field
import functools
import os
import threading
import time
from typing import Any

import structlog


logger = structlog.get_logger()


LOOP_AFFINE_ATTR = "__dana_loop_affine__"

#: Marker attribute for an *eligible* blocking-I/O callable (see :func:`blocking_io`).
BLOCKING_IO_ATTR = "__dana_blocking_io__"

#: Default bound for an explicit physical-work status observation. Dispatch
#: ownership requires an unbounded drain, not release when this timeout expires.
PHYSICAL_DRAIN_TIMEOUT_S = 5.0

#: Poll interval used while waiting for offloaded work to settle.
PHYSICAL_DRAIN_POLL_INTERVAL_S = 0.02

#: Default worker bound for offloaded blocking I/O. Mirrors the CPython
#: ``ThreadPoolExecutor`` guidance for I/O-bound work without unbounded fan-out.
MAX_BLOCKING_WORKERS = min(32, (os.cpu_count() or 1) + 4)

#: Bound on the callable name carried by a claim. Diagnostics name *what* is
#: stuck, so the name must stay short and carry no arguments or output.
MAX_CLAIM_NAME_CHARS = 120


class BlockingIOShutdown(RuntimeError):
    """Raised when an offload is requested on a pool that was already closed."""


#: ContextVar carrying the claim owned by the *current* offloaded call. A
#: cooperative callable reads it (via :func:`current_physical_work`) to observe a
#: cancellation request for the work it is performing, without any pool-wide
#: state that concurrent calls would race on.
_physical_claim: contextvars.ContextVar[PhysicalWorkClaim | None] = contextvars.ContextVar("dana_physical_claim", default=None)


@dataclass(eq=False)
class PhysicalWorkClaim:
    """Ownership of one offloaded call: is its physical work still running?

    A thread cannot be interrupted, so a cancelled dispatch must not pretend the
    work stopped. The claim lets dispatch (a) ask a *cooperative* callable to stop
    waiting early by setting :attr:`cancel`, and (b) observe :attr:`done`, which the
    pool sets when the call actually returned. Both are thread-safe: ``cancel`` is
    a :class:`threading.Event` set from the loop thread and read from the worker.

    :attr:`started_at` is the monotonic instant at which the *body* began
    executing, published by the worker itself (never the submission instant): a
    call that waited in the queue must not be reported as having run. It stays
    ``None`` until then. :attr:`name` is a bounded, non-sensitive callable label
    used only by shutdown diagnostics.
    """

    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    submitted: bool = False
    name: str = ""
    started_at: float | None = None
    #: Installed by the owning pool under its lock at registration; discards the
    #: claim from that pool's live-work registry. Cleared on settle so a late
    #: duplicate notification cannot reach a pool that already dropped the claim
    #: (the discard itself is idempotent, so an interleaved race is harmless).
    _release: Callable[[PhysicalWorkClaim], None] | None = field(default=None, repr=False, compare=False)

    @property
    def finished(self) -> bool:
        """Whether the physical call has returned (or was dropped unstarted)."""
        return self.done.is_set()

    def mark_started(self) -> None:
        """Record the monotonic instant the physical body began (worker side)."""
        self.started_at = time.monotonic()

    def settle(self) -> None:
        """Record that no physical work is owed and leave the pool's registry.

        Called on every honest outcome: the worker body returned (or raised), the
        queued work was cancelled before a worker picked it up, or submission was
        refused. A cancelled *waiter* must not call this while its work runs.
        The release hook is cleared on the first call, so a worker completion and
        a cancellation notification can race without keeping the claim registered;
        the registry discard is idempotent either way.
        """
        self.done.set()
        release = self._release
        self._release = None
        if release is not None:
            release(self)

    def request_cancel(self) -> None:
        """Ask the callable to stop waiting (cooperative callables only)."""
        self.cancel.set()

    def running_for(self) -> float | None:
        """Seconds the body has actually run, or ``None`` while it has not begun."""
        started_at = self.started_at
        if started_at is None:
            return None
        return max(0.0, time.monotonic() - started_at)

    async def drain(self, timeout: float | None = PHYSICAL_DRAIN_TIMEOUT_S) -> bool:
        """Wait (never blocking the loop) for the physical call to finish.

        Returns True when no physical work is still owed within ``timeout``. That
        is the case when the call was never submitted to a worker (an inline
        dispatch), when it was dropped before starting, and when it actually
        returned. ``None`` waits until completion. A finite timeout returning
        False is only an observation, never permission to release ownership.
        """
        if not self.submitted:
            return True
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while not self.done.is_set():
            interval = PHYSICAL_DRAIN_POLL_INTERVAL_S
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                interval = min(interval, remaining)
            await asyncio.sleep(interval)
        return True


async def drain_physical_work(claim: PhysicalWorkClaim | None, timeout: float | None = PHYSICAL_DRAIN_TIMEOUT_S) -> bool:
    """Observe physical completion without abandoning cleanup on re-cancellation.

    A dispatch must use ``timeout=None`` to retain its admission permit and effect
    scope until the worker really returns. Explicit finite timeouts remain
    available for status observations, but False never acknowledges cancellation
    or permits releasing ownership. Repeated ``cancel()`` calls cannot cut either
    wait short or extend a finite observation's deadline.
    """
    if claim is None or not claim.submitted:
        return True
    deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
    while True:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return claim.finished
        try:
            return await claim.drain(remaining)
        except asyncio.CancelledError:
            continue


def current_physical_work() -> PhysicalWorkClaim | None:
    """The claim owned by the offloaded call in this thread, or ``None``.

    A cooperative blocking callable calls this inside its worker body and checks
    ``claim.cancel`` to return early when the dispatch that started it was
    cancelled. The value is context-local, so concurrent calls never see each
    other's claim.
    """
    return _physical_claim.get()


def _claim_name(func: Callable[..., Any]) -> str:
    """A bounded, non-sensitive label for ``func`` (never its arguments/output).

    A qualified name is used for methods and module-level callables so a stuck
    shutdown says which class or module owns the blocking work; a closure falls
    back to its plain name, since ``<locals>`` adds noise without identity. The
    label passes through :data:`MAX_CLAIM_NAME_CHARS` because a generated
    callable can have an arbitrarily long name, and this string ends up in log
    records.
    """
    target = getattr(func, "__func__", func)
    qualname = getattr(target, "__qualname__", "") or ""
    name = qualname if qualname and "<locals>" not in qualname else getattr(target, "__name__", "")
    if not name:
        name = type(target).__name__
    return str(name)[:MAX_CLAIM_NAME_CHARS]


def _run_claimed(
    ctx: contextvars.Context,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    claim: PhysicalWorkClaim,
    pool: BlockingIOPool,
) -> Any:
    """Worker body: publish ``claim`` for this call, then run it in ``ctx``.

    ``claim`` must be published *inside* ``ctx`` (the context the callable
    observes), not in the worker thread's own context — ``ctx.run`` enters the
    copied context, so a value set outside it would be invisible to the callable.

    Always marks the claim done — including when the callable raises — so a
    cancelled dispatch stops holding an admission permit for work that is over.
    The start instant is published here, not at submission: a claim that waited
    in the queue has not physically run and must not be reported as if it had.
    """

    def _body() -> Any:
        _physical_claim.set(claim)
        claim.mark_started()
        try:
            # A queued call can start after close's diagnostic snapshot and then
            # become the join blocker. Diagnose that transition without polling.
            if pool.closed and claim.cancel.is_set():
                pool._log_running_claims(claim)
            return func(*args, **kwargs)
        finally:
            claim.settle()

    return ctx.run(_body)


def blocking_io(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a synchronous callable as *eligible* for offloading (explicit opt-in).

    Use for blocking work that cannot become a coroutine (filesystem I/O,
    ``subprocess`` waits, blocking HTTP). Dispatch moves a marked callable to the
    bounded offload pool; an unmarked synchronous callable stays on the event loop.

    Do NOT mark a callable that touches in-loop mutable state: a worker thread
    would race the loop's own use of that state. Such callables stay inline, and
    their own locking (if two threads really can call them) is their author's
    concern.
    """
    setattr(func, BLOCKING_IO_ATTR, True)
    return func


def is_blocking_io(func: Callable[..., Any]) -> bool:
    """Whether ``func`` declared itself eligible for offload (:func:`blocking_io`)."""
    target = getattr(func, "__func__", func)
    return bool(getattr(target, BLOCKING_IO_ATTR, False))


class BlockingIOPool:
    """A bounded, explicitly-owned pool of blocking-I/O worker threads.

    Own one per root executor/session; nest via :func:`bind_blocking_io_pool`.
    The pool is created lazily on first use so an executor that never performs
    blocking I/O never starts threads.
    """

    def __init__(self, max_workers: int | None = None, *, name: str = "dana-io") -> None:
        self._max_workers = max_workers if max_workers is not None else MAX_BLOCKING_WORKERS
        self._name = name
        self._pool: ThreadPoolExecutor | None = None
        self._closed = False
        self._lock = threading.Lock()
        #: Guards ``_claims`` only. Kept separate from ``_lock`` on purpose: a
        #: cancelled concurrent future settles its claim from inside
        #: ``shutdown(cancel_futures=True)``, which already holds the underlying
        #: executor's shutdown lock. If settlement needed ``_lock`` (held across
        #: ``executor.submit``), a submission and a close could deadlock. The only
        #: lock order ever taken is ``_lock`` then ``_claims_lock``.
        self._claims_lock = threading.Lock()
        #: Every claim whose physical work may still be owed to this pool. Added
        #: under the submission critical section, removed when the work settles.
        #: Not cleared by shutdown: a non-cooperative worker that outlives the
        #: close keeps its ownership visible until it really returns.
        self._claims: set[PhysicalWorkClaim] = set()

    @property
    def max_workers(self) -> int:
        """The configured worker bound."""
        return self._max_workers

    @property
    def closed(self) -> bool:
        """Whether :meth:`shutdown` has run."""
        return self._closed

    @property
    def started(self) -> bool:
        """Whether a worker thread has been requested yet."""
        return self._pool is not None

    @staticmethod
    def _watch_dropped(claim: PhysicalWorkClaim, concurrent: Any) -> None:
        """Settle ``claim`` if the concurrent future is cancelled before starting.

        A cancelled concurrent future is the only honest proof that the worker body
        never began. When the work is already running, ``cancel()`` fails, so the
        future is never cancelled and the claim stays open for ``_run_claimed``'s
        own completion to settle. The callback runs in whichever thread cancelled
        the future (a cancelled waiter, or shutdown dropping queued work), so
        cleanup never depends on the event loop still running.
        """
        concurrent.add_done_callback(lambda settled: claim.settle() if settled.cancelled() else None)

    def _started_executor_locked(self) -> ThreadPoolExecutor:
        """Caller must hold ``self._lock``. Refuse when closed, else create once.

        The closed check is made *under* the lock: ``shutdown`` takes the same
        lock to set ``_closed``, so a concurrent close cannot land between the
        check and the create. Without that, a close running in that window let a
        caller build a brand-new executor after ``shutdown`` had already run -
        the pool reported itself closed while a fresh worker thread (which
        nothing would ever join) was started behind it.
        """
        if self._closed:
            raise BlockingIOShutdown("blocking I/O pool was shut down; no new offloads are accepted")
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix=self._name)
            logger.info("blocking_io_pool_started", max_workers=self._max_workers, name=self._name)
        return self._pool

    def _settle_claim(self, claim: PhysicalWorkClaim) -> None:
        """Discard ``claim`` from the live registry (idempotent).

        Takes only ``_claims_lock``: a worker or a dropped-future callback can
        settle while a submission holds ``_lock`` or a close holds the executor's
        own lock, and neither may be blocked on the other through this path.
        """
        with self._claims_lock:
            self._claims.discard(claim)

    def _submit_claimed(self, claim: PhysicalWorkClaim, call: Callable[[], Any], *, name: str) -> Any:
        """Register ``claim`` and submit ``call``, all under the pool lock.

        Registration, the closed check and the executor's own submit share one
        critical section, so a concurrent :meth:`shutdown` either sees this claim
        in its snapshot (and signals/joins its work) or refuses the submission
        outright. There is no window in which a worker starts behind a close with
        nothing owning it, and no window in which shutdown's snapshot misses work
        that was accepted. A failed submit removes the claim again: refused work
        owes nothing.
        """
        with self._lock:
            executor = self._started_executor_locked()
            claim.name = claim.name or name
            claim.submitted = True
            claim._release = self._settle_claim
            with self._claims_lock:
                self._claims.add(claim)
            try:
                return executor.submit(call)
            except BaseException:
                with self._claims_lock:
                    self._claims.discard(claim)
                claim._release = None
                claim.submitted = False
                raise

    async def run(self, func: Callable[..., Any], *args: Any, _claim: PhysicalWorkClaim | None = None, **kwargs: Any) -> Any:
        """Run ``func(*args, **kwargs)`` on this pool and await the result.

        The callable's exceptions propagate unchanged. The caller's contextvars
        (including a bound offload pool, so nested synchronous offloads share this
        pool) are adopted inside the worker thread via
        ``contextvars.copy_context().run`` — ``run_in_executor`` alone would leave
        the worker in an empty context.

        ``_claim`` (I8) is the caller's ownership handle for this physical call.
        The worker publishes it through :func:`current_physical_work` so a
        cooperative callable can observe a cancellation request, and marks it done
        when the call returns. A *queued* call that is cancelled before a worker
        picks it up owes no physical work and settles the claim immediately; a call
        already running can be neither cancelled nor reported as stopped, so its
        claim stays unsettled until the callable really returns. Callers that do not
        pass a claim get an internal one, so a marked callable can always read its
        cancellation request.

        The claim is registered with this pool in the same critical section as the
        acceptance check, so a concurrent :meth:`shutdown` snapshots it and asks it
        to stop waiting before it joins. A cancelled *waiter* does not settle the
        claim while the worker still owns physical work: only a call whose future
        never started does.
        """
        loop = asyncio.get_running_loop()
        handle = _claim if _claim is not None else PhysicalWorkClaim()
        # Copy the caller's context on the loop side: ``run_in_executor`` alone
        # leaves the worker in an empty context, so the bound offload pool (and a
        # nested offload's cancellation request) must travel with the call.
        ctx = contextvars.copy_context()
        call = functools.partial(_run_claimed, ctx, func, args, kwargs, handle, self)
        # Submit directly (not via ``run_in_executor``) so the underlying concurrent
        # future is in hand: its ``cancel()`` result is the only honest signal that
        # the work never started.
        try:
            concurrent = self._submit_claimed(handle, call, name=_claim_name(func))
        except BaseException:
            handle.settle()
            raise
        self._watch_dropped(handle, concurrent)
        return await asyncio.wrap_future(concurrent, loop=loop)

    def run_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Submit ``func`` from a synchronous caller and wait for the result.

        Used by the legacy synchronous dispatch path, which has no loop to stall
        but still wants bounded, lifecycle-owned execution. Exceptions propagate.

        The call is registered with this pool exactly like an async one, so a
        concurrent :meth:`shutdown` can signal it too. A queued call that the
        close dropped settles its claim; a running call keeps its claim until the
        worker body returns, and the shutdown log names it if it survives.
        """
        claim = PhysicalWorkClaim()
        ctx = contextvars.copy_context()
        call = functools.partial(_run_claimed, ctx, func, args, kwargs, claim, self)
        try:
            concurrent = self._submit_claimed(claim, call, name=_claim_name(func))
        except BaseException:
            claim.settle()
            raise
        self._watch_dropped(claim, concurrent)
        return concurrent.result()

    def shutdown(self, wait: bool = False) -> None:
        """Close this pool (idempotent). Affects only this pool's own threads.

        Args:
            wait: When True, block until queued work finishes. When False,
                queued-but-unstarted work is dropped and already-running work is
                allowed to finish on its threads. Shutdown never pretends
                in-flight non-cooperative work was rolled back.

        Every live claim is asked to stop waiting *before* any join, so a
        cooperative body - including queued work that has not started yet - exits
        instead of making the join wait out its own timeout. The join itself is
        not time-boxed: a non-cooperative worker is joined for as long as it runs,
        and its ownership stays visible rather than being reported as cancelled.
        Each claim still executing at that point is logged with its bounded
        callable name and how long its body has actually run, so a join that does
        not return is diagnosable from the log.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pool = self._pool
            self._pool = None
            with self._claims_lock:
                claims = tuple(self._claims)
        # Signal before joining: a cooperative callable only returns early once it
        # observes the request, and joining first would wait for work that could
        # have stopped immediately. Done outside the lock so a settling claim never
        # blocks on a shutdown that is joining its own worker.
        for claim in claims:
            claim.request_cancel()
        self._log_running_claims()
        if pool is not None:
            pool.shutdown(wait=wait, cancel_futures=not wait)
        logger.info("blocking_io_pool_stopped", waited=wait, name=self._name)

    def _log_running_claims(self, claim: PhysicalWorkClaim | None = None) -> None:
        """Log every claim whose physical body is still executing.

        The registry snapshot, the finished check and the elapsed measurement are
        taken under the registry lock so the report is consistent: a claim that had
        already settled is not described as running, and a claim that settles
        during the join was genuinely running at diagnosis time. Queued claims
        (no body start yet) and finished ones are skipped - neither is a stuck
        join. Only the bounded callable name and the elapsed physical runtime are
        logged, never arguments or output.
        A worker starting after its close signal also reports just its own claim;
        it may overlap the close snapshot, but neither path repeats or polls.
        """
        with self._claims_lock:
            claims = self._claims if claim is None else (claim,)
            running = [(claim.name, claim.running_for()) for claim in claims if not claim.done.is_set() and claim.started_at is not None]
        for name, running_for in running:
            logger.warning(
                "blocking_io_pool_claim_still_running",
                pool=self._name,
                callable=name,
                running_s=round(running_for or 0.0, 3),
            )

    async def ashutdown(self, wait: bool = False) -> None:
        """Await :meth:`shutdown` without parking the event loop.

        ``ThreadPoolExecutor.shutdown(wait=True)`` *joins* the worker threads,
        so calling it from a coroutine stalls every other task on that loop for
        as long as the longest piece of in-flight blocking work takes. The join
        is run on a separate thread instead, so an awaited teardown still waits
        for the physical work to finish while the loop keeps servicing other
        coroutines. Behaviour (including the ``wait`` semantics) is otherwise
        identical to the synchronous call.
        """
        await asyncio.to_thread(self.shutdown, wait)


# ----------------------------------------------------------------------
# Pool resolution: bound (session) pool -> default pool
# ----------------------------------------------------------------------

#: The pool bound to the current dispatch context by an owning executor. Read by
#: ``run_blocking`` so nested agent/resource work shares its root session's pool.
_bound_pool: contextvars.ContextVar[BlockingIOPool | None] = contextvars.ContextVar("dana_blocking_io_pool", default=None)

_default_pool: BlockingIOPool | None = None
_default_pool_lock = threading.Lock()


def get_default_blocking_io_pool() -> BlockingIOPool:
    """The process-wide fallback pool for direct/legacy callers (lazily created)."""
    global _default_pool
    with _default_pool_lock:
        if _default_pool is None:
            _default_pool = BlockingIOPool(name="dana-io-default")
        return _default_pool


def current_blocking_io_pool() -> BlockingIOPool:
    """The pool a synchronous :func:`run_blocking` call would use right now."""
    bound = _bound_pool.get()
    return bound if bound is not None else get_default_blocking_io_pool()


def bound_blocking_io_pool() -> BlockingIOPool | None:
    """The pool bound to this context by an owning executor, or ``None``.

    Used by an executor to *inherit* its parent's pool instead of overwriting the
    binding: a nested sub-agent's resource I/O shares the root session's bounded
    pool rather than multiplying thread pools per nesting level. A top-level
    executor (nothing bound) binds its own pool.
    """
    return _bound_pool.get()


def bind_blocking_io_pool(pool: BlockingIOPool | None) -> contextvars.Token:
    """Bind ``pool`` as the current context's offload pool (returns a reset token).

    An owning executor binds its session pool for the duration of a dispatch so
    that *offloading* work - a nested :func:`run_blocking` call, including one
    made by a nested agent, or an explicitly ``@blocking_io``-marked synchronous
    callable that dispatch moves to a worker - resolves to the session's own
    bounded pool instead of the process default. Binding does not itself offload
    anything: an unmarked synchronous resource method stays on the event-loop
    thread (I7), exactly as it would with no binding. Unbinding restores the
    previous binding, so a closed session's callers fall back to the default pool
    rather than failing.
    """
    return _bound_pool.set(pool)


@contextlib.contextmanager
def use_blocking_io_pool(pool: BlockingIOPool | None) -> Iterator[BlockingIOPool]:
    """Context manager form of :func:`bind_blocking_io_pool` (scoped binding)."""
    token = _bound_pool.set(pool)
    try:
        yield pool if pool is not None else get_default_blocking_io_pool()
    finally:
        _bound_pool.reset(token)


async def run_blocking(
    func: Callable[..., Any],
    *args: Any,
    _claim: PhysicalWorkClaim | None = None,
    **kwargs: Any,
) -> Any:
    """Run ``func(*args, **kwargs)`` on the current offload pool and await it.

    Resolves the session pool bound by the owning executor, falling back to the
    process default pool for direct/legacy callers. Raises
    :class:`BlockingIOShutdown` if that pool was already closed, and propagates
    the callable's own exceptions unchanged (no error swallowing).

    ``_claim`` (I8) is the *dispatch's* ownership handle for this physical call,
    passed only by the executor's own offload of a synchronous tool body. It lets
    a cancelled dispatch ask a cooperative callable to stop waiting
    (:meth:`PhysicalWorkClaim.request_cancel`) and observe when the physical work
    actually returned (:meth:`PhysicalWorkClaim.drain`) before releasing the
    admission permit. A resource's own nested ``run_blocking`` call passes no
    claim, so it is never conflated with the dispatch permit's lifetime.
    """
    pool = _bound_pool.get()
    if pool is None:
        pool = get_default_blocking_io_pool()
    return await pool.run(func, *args, _claim=_claim, **kwargs)


def run_blocking_sync(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Synchronous counterpart of :func:`run_blocking` (legacy sync dispatch).

    Uses the pool bound to the calling context, else the default pool. A pool
    bound on the event-loop side is not visible here (thread-local context), so a
    synchronous caller passes ``scope`` through :func:`bind_blocking_io_pool`
    when it needs to share a session pool explicitly.
    """
    pool = _bound_pool.get()
    if pool is None:
        pool = get_default_blocking_io_pool()
    return pool.run_sync(func, *args, **kwargs)


def loop_affine(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a callable as loop-affine: dispatch keeps it on the event loop.

    Use for cooperative, non-blocking coroutines that must not be moved to a
    worker thread (they would lose loop affinity — e.g. anything awaiting other
    coroutines). Blocking *synchronous* methods must NOT be marked.
    """
    setattr(func, LOOP_AFFINE_ATTR, True)
    return func


def is_loop_affine(func: Callable[..., Any]) -> bool:
    """Whether ``func`` declared itself loop-affine (kept inline by dispatch)."""
    target = getattr(func, "__func__", func)
    return bool(getattr(target, LOOP_AFFINE_ATTR, False))


def blocking_io_max_workers() -> int | None:
    """The worker bound of the started pool, or ``None`` if not started yet."""
    bound = _bound_pool.get()
    if bound is not None:
        return bound.max_workers
    if _default_pool is not None and _default_pool.started:
        return _default_pool.max_workers
    return None


def is_shutdown() -> bool:
    """Whether the current offload pool has been shut down."""
    bound = _bound_pool.get()
    return bound.closed if bound is not None else (_default_pool.closed if _default_pool is not None else False)


def shutdown_default_blocking_io_executor(wait: bool = False) -> None:
    """Close the process-wide default pool (application-level cleanup; idempotent).

    Only for application shutdown. Session/executor teardown must call
    :meth:`BlockingIOPool.shutdown` on its *own* pool instead — killing the
    default pool from a session would poison concurrent and future sessions.
    """
    global _default_pool
    with _default_pool_lock:
        pool = _default_pool
        _default_pool = None
    if pool is not None:
        pool.shutdown(wait=wait)


# The default pool has no session owner, so nothing else would ever close it:
# a bare ``FileIOResource`` (or any direct/legacy ``run_blocking`` caller) starts
# worker threads that would otherwise still be alive when the interpreter begins
# waiting to join non-daemon threads at exit. This is *process* lifecycle, the one
# boundary that legitimately closes a pool shared by every session - never a
# session teardown, which must stay scoped to its own pool. Closing here only
# joins threads at a point where no further offload can be issued, so it can
# never poison the sessions the module documentation warns about. Idempotent, and
# a no-op when the default pool was never started.
atexit.register(shutdown_default_blocking_io_executor, True)
