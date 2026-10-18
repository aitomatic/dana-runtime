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
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
import contextlib
import contextvars
import functools
import os
import threading
from typing import Any

import structlog


logger = structlog.get_logger()


LOOP_AFFINE_ATTR = "__dana_loop_affine__"

#: Default worker bound for offloaded blocking I/O. Mirrors the CPython
#: ``ThreadPoolExecutor`` guidance for I/O-bound work without unbounded fan-out.
MAX_BLOCKING_WORKERS = min(32, (os.cpu_count() or 1) + 4)


class BlockingIOShutdown(RuntimeError):
    """Raised when an offload is requested on a pool that was already closed."""


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

    async def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``func(*args, **kwargs)`` on this pool and await the result.

        The callable's exceptions propagate unchanged. The caller's contextvars
        (including a bound offload pool, so nested synchronous offloads share this
        pool) are adopted inside the worker thread via
        ``contextvars.copy_context().run`` — ``run_in_executor`` alone would leave
        the worker in an empty context.
        """
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        call = functools.partial(ctx.run, functools.partial(func, *args, **kwargs))
        return await loop.run_in_executor(self._executor(), call)

    def run_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Submit ``func`` from a synchronous caller and wait for the result.

        Used by the legacy synchronous dispatch path, which has no loop to stall
        but still wants bounded, lifecycle-owned execution. Exceptions propagate.
        """
        ctx = contextvars.copy_context()
        return self._executor().submit(ctx.run, functools.partial(func, *args, **kwargs)).result()

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
    **kwargs: Any,
) -> Any:
    """Run ``func(*args, **kwargs)`` on the current offload pool and await it.

    Resolves the session pool bound by the owning executor, falling back to the
    process default pool for direct/legacy callers. Raises
    :class:`BlockingIOShutdown` if that pool was already closed, and propagates
    the callable's own exceptions unchanged (no error swallowing).
    """
    pool = _bound_pool.get()
    if pool is None:
        pool = get_default_blocking_io_pool()
    return await pool.run(func, *args, **kwargs)


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
