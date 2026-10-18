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
  pool (a bare ``FileIOResource`` used outside a session). It is process-wide and
  is closed only by an explicit application-level
  :func:`shutdown_default_blocking_io_executor` call; **a session close never
  kills it** (that would poison concurrent and future sessions).

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
"""

from __future__ import annotations

import asyncio
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


class BlockingIOShutdown(RuntimeError):
    """Raised when an offload is requested on a pool that was already closed."""


#: ContextVar carrying the claim owned by the *current* offloaded call. A
#: cooperative callable reads it (via :func:`current_physical_work`) to observe a
#: cancellation request for the work it is performing, without any pool-wide
#: state that concurrent calls would race on.
_physical_claim: contextvars.ContextVar[PhysicalWorkClaim | None] = contextvars.ContextVar("dana_physical_claim", default=None)


@dataclass
class PhysicalWorkClaim:
    """Ownership of one offloaded call: is its physical work still running?

    A thread cannot be interrupted, so a cancelled dispatch must not pretend the
    work stopped. The claim lets dispatch (a) ask a *cooperative* callable to stop
    waiting early by setting :attr:`cancel`, and (b) observe :attr:`done`, which the
    pool sets when the call actually returned. Both are thread-safe: ``cancel`` is
    a :class:`threading.Event` set from the loop thread and read from the worker.
    """

    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    submitted: bool = False

    @property
    def finished(self) -> bool:
        """Whether the physical call has returned (or was dropped unstarted)."""
        return self.done.is_set()

    def request_cancel(self) -> None:
        """Ask the callable to stop waiting (cooperative callables only)."""
        self.cancel.set()

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


def _run_claimed(
    ctx: contextvars.Context,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    claim: PhysicalWorkClaim,
) -> Any:
    """Worker body: publish ``claim`` for this call, then run it in ``ctx``.

    ``claim`` must be published *inside* ``ctx`` (the context the callable
    observes), not in the worker thread's own context — ``ctx.run`` enters the
    copied context, so a value set outside it would be invisible to the callable.

    Always marks the claim done — including when the callable raises — so a
    cancelled dispatch stops holding an admission permit for work that is over.
    """

    def _body() -> Any:
        _physical_claim.set(claim)
        try:
            return func(*args, **kwargs)
        finally:
            claim.done.set()

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

    def _executor(self) -> ThreadPoolExecutor:
        """Lazily create (once) and return the underlying executor."""
        if self._closed:
            raise BlockingIOShutdown("blocking I/O pool was shut down; no new offloads are accepted")
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix=self._name)
                logger.info("blocking_io_pool_started", max_workers=self._max_workers, name=self._name)
            return self._pool

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
        """
        loop = asyncio.get_running_loop()
        handle = _claim if _claim is not None else PhysicalWorkClaim()
        handle.submitted = True
        # Copy the caller's context on the loop side: ``run_in_executor`` alone
        # leaves the worker in an empty context, so the bound offload pool (and a
        # nested offload's cancellation request) must travel with the call.
        ctx = contextvars.copy_context()
        call = functools.partial(_run_claimed, ctx, func, args, kwargs, handle)
        # Submit directly (not via ``run_in_executor``) so the underlying concurrent
        # future is in hand: its ``cancel()`` result is the only honest signal that
        # the work never started.
        concurrent = self._executor().submit(call)
        future = asyncio.wrap_future(concurrent, loop=loop)

        def _settle_if_dropped(settled: asyncio.Future) -> None:
            # Only a call whose worker body never began owes nothing. When the work
            # is already running, ``cancel()`` fails and the claim is left open for
            # ``_run_claimed``'s own completion to settle.
            if settled.cancelled() and concurrent.cancel():
                handle.done.set()

        future.add_done_callback(_settle_if_dropped)
        return await future

    def run_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Submit ``func`` from a synchronous caller and wait for the result.

        Used by the legacy synchronous dispatch path, which has no loop to stall
        but still wants bounded, lifecycle-owned execution. Exceptions propagate.
        """
        return self._executor().submit(_run_claimed, contextvars.copy_context(), func, args, kwargs, PhysicalWorkClaim()).result()

    def shutdown(self, wait: bool = False) -> None:
        """Close this pool (idempotent). Affects only this pool's own threads.

        Args:
            wait: When True, block until queued work finishes. When False,
                queued-but-unstarted work is dropped and already-running work is
                allowed to finish on its threads. Shutdown never pretends
                in-flight non-cooperative work was rolled back.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pool = self._pool
            self._pool = None
        if pool is not None:
            pool.shutdown(wait=wait, cancel_futures=not wait)
        logger.info("blocking_io_pool_stopped", waited=wait, name=self._name)


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
    every resource call inside (including nested agents) offloads to the session's
    own bounded pool. Unbinding restores the previous binding, so a closed
    session's callers fall back to the default pool rather than failing.
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
