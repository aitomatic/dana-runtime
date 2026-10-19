"""Bounded admission for tool work and nested sub-agents (P1 — real, bounded concurrency).

Requirement: bound tool and nested-subagent admission **without holding all permits
while awaiting child work**, with no deadlock, and no unintended sibling cancellation.

Design: two independent permit pools + exclusive nested handoff
-------------------------------------------------------------
* **leaf pool** (``leaf_limit``) — ordinary tool dispatches.
* **nested pool** (``nested_limit``) — sub-agent (``Task``) branches.

A nested parent awaiting children can lend its slot to exactly one descendant
branch at a time. Other siblings must acquire a separate slot or wait. Each child
gets its own context frame, so arbitrarily deep chains can hand off that same slot
without letting copied task contexts multiply the budget. Returning a borrowed
slot wakes queued siblings. A released ancestor retains its slot until its last
borrowing descendant finishes, including during cancellation.

Leaves use a separate pool and never wait behind a nested parent. One call's
failure or cancellation never releases, cancels, or skips a sibling.

Loop affinity: semaphores are loop-bound, so pools are created lazily **per running
event loop**. One controller instance can therefore serve sessions, child runtimes,
and short-lived ``asyncio.run`` cycles without loop-binding bugs.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import structlog


logger = structlog.get_logger()


#: Default concurrent tool dispatches (leaf work).
DEFAULT_LEAF_LIMIT = 8

#: Default concurrent nested sub-agent branches.
DEFAULT_NESTED_LIMIT = 4

#: Tool names dispatched as nested sub-agent work by default. ``Task`` is the
#: model-facing sub-agent tool name; callers may override the predicate.
DEFAULT_NESTED_TOOL_NAMES = frozenset({"Task"})


def default_nested_predicate(function_name: str, arguments: dict[str, Any]) -> bool:
    """Default classifier: is this call nested sub-agent work?"""
    _ = arguments
    return function_name in DEFAULT_NESTED_TOOL_NAMES


@dataclass
class Permit:
    """One admission permit. :meth:`release` is idempotent."""

    kind: str
    pool: _Pool | None = None
    inherited: bool = False
    token: Any = None
    depth_var: ContextVar[_NestedFrame | None] | None = None
    _frame: _NestedFrame | None = None
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self.token is not None and self.depth_var is not None:
            self.depth_var.reset(self.token)
        if self._frame is not None:
            self._frame.release()
        elif self.pool is not None:
            self.pool.release()


@dataclass
class _Pool:
    """One budget pool bound to one event loop."""

    limit: int
    semaphore: asyncio.Semaphore
    in_flight: int = 0
    peak_in_flight: int = 0
    waiters: int = 0
    peak_waiters: int = 0
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    async def acquire(self) -> None:
        if self.in_flight >= self.limit:
            self.waiters += 1
            self.peak_waiters = max(self.peak_waiters, self.waiters)
            try:
                await self.semaphore.acquire()
            finally:
                self.waiters -= 1
        else:
            await self.semaphore.acquire()
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)

    def release(self) -> None:
        self.in_flight -= 1
        self.semaphore.release()
        self.wake_waiters()

    def wake_waiters(self) -> None:
        if not self.waiters:
            return
        changed = self.changed
        self.changed = asyncio.Event()
        changed.set()


@dataclass
class _NestedFrame:
    """One branch's exclusive claim on a pool slot or an ancestor's loan."""

    pool: _Pool
    parent: _NestedFrame | None = None
    child: _NestedFrame | None = None
    released: bool = False

    def release(self) -> None:
        self.released = True
        frame = self
        while frame.released and frame.child is None:
            if frame.parent is None:
                frame.pool.release()
                return
            parent = frame.parent
            parent.child = None
            frame = parent
        self.pool.wake_waiters()


@dataclass
class _LoopPools:
    """Pool pair for one running event loop, kept with its loop for identity."""

    loop: Any
    pools: dict[str, _Pool] = field(default_factory=dict)


class AdmissionController:
    """Bounded, deadlock-free admission for tool leaves and nested sub-agents.

    Args:
        leaf_limit: Max concurrent non-nested tool dispatches (>= 1).
        nested_limit: Max concurrent nested sub-agent branches (>= 1).
        nested_predicate: ``(function_name, arguments) -> bool`` classifying a
            call as nested sub-agent work. Defaults to
            :func:`default_nested_predicate`.
    """

    LEAF = "leaf"
    NESTED = "nested"

    def __init__(
        self,
        leaf_limit: int = DEFAULT_LEAF_LIMIT,
        nested_limit: int = DEFAULT_NESTED_LIMIT,
        nested_predicate: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> None:
        if leaf_limit < 1:
            raise ValueError("leaf_limit must be >= 1")
        if nested_limit < 1:
            raise ValueError("nested_limit must be >= 1")
        self.leaf_limit = leaf_limit
        self.nested_limit = nested_limit
        self._nested_predicate = nested_predicate or default_nested_predicate
        self._by_loop: dict[int, _LoopPools] = {}
        self._nested_frame: ContextVar[_NestedFrame | None] = ContextVar("dana_nested_admission_frame", default=None)

    # ------------------------------------------------------------------
    # Pool plumbing
    # ------------------------------------------------------------------

    def _pools_for_running_loop(self) -> dict[str, _Pool]:
        loop = asyncio.get_running_loop()
        self._drop_closed_loops()
        entry = self._by_loop.get(id(loop))
        if entry is None or entry.loop is not loop or entry.loop.is_closed():
            entry = _LoopPools(
                loop=loop,
                pools={
                    self.LEAF: _Pool(limit=self.leaf_limit, semaphore=asyncio.Semaphore(self.leaf_limit)),
                    self.NESTED: _Pool(limit=self.nested_limit, semaphore=asyncio.Semaphore(self.nested_limit)),
                },
            )
            self._by_loop[id(loop)] = entry
        return entry.pools

    def _drop_closed_loops(self) -> None:
        """Drop pool state for loops that have closed (S8).

        Admission state is keyed by ``id(loop)``, so a finished loop's entry would
        otherwise be retained forever: with a fresh loop in every test the mapping
        (and each entry's semaphores) grows without bound. Cleanup runs on the next
        acquisition *and* on :meth:`snapshot`, so a caller that only observes
        (``snapshot``) also reclaims. Entry removal is safe precisely because the
        loop is closed: no task on it can still be waiting for a permit, and no new
        acquisition can arrive for it.
        """
        if not self._by_loop:
            return
        for key, entry in list(self._by_loop.items()):
            if entry.loop.is_closed():
                del self._by_loop[key]

    @property
    def tracked_loops(self) -> int:
        """Number of per-loop pool entries still retained (observability, S8).

        Closed loops are dropped lazily, so this reads as "entries since the last
        reclamation"; :meth:`snapshot` reclaims first, making it exact.
        """
        return len(self._by_loop)

    def classify(self, function_name: str, arguments: dict[str, Any] | None = None) -> str:
        """Return the pool kind (``"nested"`` or ``"leaf"``) for a call."""
        try:
            is_nested = bool(self._nested_predicate(function_name, arguments or {}))
        except Exception:  # noqa: BLE001 — a broken predicate must not break dispatch
            logger.warning("nested_predicate_failed", function_name=function_name)
            is_nested = True  # cautious: the nested pool is the smaller, safer budget
        return self.NESTED if is_nested else self.LEAF

    async def acquire(self, function_name: str, arguments: dict[str, Any] | None = None) -> Permit:
        """Acquire one permit for ``function_name`` and return it."""
        return await self.acquire_kind(self.classify(function_name, arguments))

    async def acquire_kind(self, kind: str) -> Permit:
        """Acquire one permit from the named pool (``"leaf"`` or ``"nested"``)."""
        pool = self._pools_for_running_loop()[kind]
        if kind != self.NESTED:
            await pool.acquire()
            return Permit(kind=kind, pool=pool)

        parent = self._nested_frame.get()
        if parent is not None and parent.pool is not pool:
            parent = None
        while True:
            changed = pool.changed
            if parent is not None and not parent.released and parent.child is None:
                frame = _NestedFrame(pool=pool, parent=parent)
                parent.child = frame
                inherited = True
                break
            if pool.in_flight < pool.limit:
                await pool.acquire()
                frame = _NestedFrame(pool=pool)
                inherited = False
                break
            pool.waiters += 1
            pool.peak_waiters = max(pool.peak_waiters, pool.waiters)
            try:
                await changed.wait()
            finally:
                pool.waiters -= 1

        permit = Permit(kind=kind, pool=pool, inherited=inherited, depth_var=self._nested_frame, _frame=frame)
        permit.token = self._nested_frame.set(frame)
        logger.debug("admission_acquired", kind=kind, in_flight=pool.in_flight, limit=pool.limit, inherited=inherited)
        return permit

    @asynccontextmanager
    async def slot(self, function_name: str, arguments: dict[str, Any] | None = None) -> AsyncIterator[Permit]:
        """Hold one admission permit for the duration of the block."""
        permit = await self.acquire(function_name, arguments)
        try:
            yield permit
        finally:
            permit.release()

    @asynccontextmanager
    async def nested_slot(self) -> AsyncIterator[Permit]:
        """Hold one nested (sub-agent) permit for the duration of the block."""
        permit = await self.acquire_kind(self.NESTED)
        try:
            yield permit
        finally:
            permit.release()

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Current and peak occupancy per kind, across known loops.

        ``peak_in_flight`` is the observable proof that admission really bounded
        concurrency (test/telemetry friendly); ``peak_waiters`` shows how much
        work had to wait for a permit.
        """
        self._drop_closed_loops()
        out: dict[str, Any] = {}
        for kind, limit in ((self.LEAF, self.leaf_limit), (self.NESTED, self.nested_limit)):
            in_flight = 0
            peak_in_flight = 0
            peak_waiters = 0
            for entry in self._by_loop.values():
                pool = entry.pools.get(kind)
                if pool is None:
                    continue
                in_flight += pool.in_flight
                peak_in_flight = max(peak_in_flight, pool.peak_in_flight)
                peak_waiters = max(peak_waiters, pool.peak_waiters)
            out[kind] = {
                "limit": limit,
                "in_flight": in_flight,
                "peak_in_flight": peak_in_flight,
                "peak_waiters": peak_waiters,
            }
        return out

    def reset_peaks(self) -> None:
        """Reset peak counters (observability only; never affects admission)."""
        for entry in self._by_loop.values():
            for pool in entry.pools.values():
                pool.peak_in_flight = pool.in_flight
                pool.peak_waiters = 0

    def close(self) -> None:
        """Drop per-loop pool state (lifecycle cleanup; idempotent).

        Admission is in-process bookkeeping only. Closing never cancels or
        abandons in-flight work, and never reports an unacknowledged cancellation
        as terminal.
        """
        self._by_loop.clear()


_default_controller: AdmissionController | None = None


def get_default_admission_controller() -> AdmissionController:
    """The process-wide default admission controller.

    Shared by the tool executor and nested sub-agent spawns so that a nested
    subtree cannot multiply resource usage beyond one harness-wide budget. Callers
    that need a separate budget (or tighter limits) inject their own controller.
    """
    global _default_controller
    if _default_controller is None:
        _default_controller = AdmissionController()
    return _default_controller
