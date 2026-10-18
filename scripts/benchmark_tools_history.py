#!/usr/bin/env python3
"""Minimal repeatable benchmark: real ToolExecutor tool paths, nested subagent
admission, and journal→prepare history scaling.

Three deliberately narrow lanes against real runtime objects (no runtime edits, no
benchmark framework, no mocks of the objects under test):

``tools`` lane
    Real ``ToolExecutor`` (dana/core/tool/tool_executor.py:42) with a deterministic
    synthetic I/O resource registered through the ``@named_tool``-style registry
    (``{tool_name: (obj, method_name)}`` - dana/core/runtime/base.py:430-457).
    Cases: independent synchronous and asynchronous I/O at N=1,3,8,32 through both
    executor entry points (``execute_tools`` and ``execute_tools_async``), an
    event-loop heartbeat, exact ``tool_call_id``/result checks, ordered conflicting
    mutations, EventBus permission interception (TOOL_CALL block/modify +
    TOOL_RESULT modify), and cancellation observations.

``subagents`` lane
    Real nested-agent fan-out at N=1,4,16: the real ``TaskResource.task`` tool
    (dana/core/resource/task_resource.py:195) dispatching real child ``STARAgent``
    instances (production default runtime/codec, real local timeline repositories)
    through the real ``ToolExecutor`` async batch. Only the provider WIRE is
    substituted (a deterministic in-process ``LLMProvider``), so no network or
    credentials are involved. Records per-call id/result/session checks, spawned
    child count, and the observed admission width (peak simultaneous live
    children), plus a deterministic rendezvous gate that proves all N children can
    be inside their provider call at once.

``history`` lane
    Real SQLite journal (dana/core/session/journal/sqlite.py:98), real
    ``ConversationProjector`` (dana/core/session/projections/conversation.py:95) and
    the real ``AgentSession`` prepare path (``load`` → ``_prepare_agent``,
    dana/core/session/agent_session.py:409,771) at 10/100/1000/5000 prior turns.
    Phases are measured separately: read+decode (``read_facts``), delta read
    (``after_sequence``), project, hydrate (``_populate_timeline``), prepare
    (``_prepare_agent``), and model working set
    (``timeline.to_llm_messages()``). Journal bytes are reported separately from
    the retained model working set.

Output: one JSON document (``--output``) containing raw monotonic duration samples
(``time.perf_counter``, milliseconds), the declared configuration, warmup counts,
per-case sample counts, sample-reduction disclosures, correctness observations and
known failures.

Safety: no network, no provider calls, no secrets. Only synthetic fixtures and
temporary directories are used; every temporary resource (SQLite connections,
asyncio tasks, threads) is closed/removed explicitly and the cleanup is reported.

Usage:
    uv run --no-sync python scripts/benchmark_tools_history.py --output /tmp/bench.json --repetitions 100
    uv run --no-sync python scripts/benchmark_tools_history.py --output /tmp/bench_sub.json --lanes subagents --repetitions 4
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Sequence
import contextlib
from datetime import UTC, datetime
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Declared environment overrides - applied BEFORE importing dana so import-time
# capability detection (dana.common.observable reads LANGFUSE_ENABLED at import)
# is deterministic, matching tests/conftest.py.
#
# DANA_CODE_MCP_ENABLED=0 is a DECLARED DEVIATION from the production default:
# this benchmark configures no MCP server, so leaving MCP wiring on would only
# add an unpredictable network-lease attempt to the per-turn prepare path. The
# effective values are echoed in the output under config.env_flags.
# ---------------------------------------------------------------------------
os.environ.setdefault("LANGFUSE_ENABLED", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("DANA_CODE_MCP_ENABLED", "0")

from dana.common.llm.types import LLMProvider, LLMResponse  # noqa: E402
from dana.config.storage_config import FileStorageConfig  # noqa: E402
from dana.core.agent.star_agent import STARAgent  # noqa: E402
from dana.core.ext.event_bus import EventBus  # noqa: E402
from dana.core.ext.events import TOOL_CALL, TOOL_RESULT  # noqa: E402
from dana.core.resource.task_resource import TaskResource  # noqa: E402
from dana.core.session.agent_session import AgentSession  # noqa: E402
from dana.core.session.journal.models import SessionRecord  # noqa: E402
from dana.core.session.journal.sqlite import SQLiteJournalRepository  # noqa: E402
from dana.core.session.models import FactType, JournalFact, NewJournalFact, OwnerScope  # noqa: E402
from dana.core.session.projections.conversation import ConversationProjector  # noqa: E402
from dana.core.timeline.compressed_timeline import CompressedTimeline  # noqa: E402
from dana.core.timeline.timeline import Timeline  # noqa: E402
from dana.core.tool.tool_executor import ToolExecutor  # noqa: E402
from dana.repositories.local_file_repository import LocalTimelineRepository  # noqa: E402
from dana.repositories.repository_factory import RepositoryFactory, RepositoryType  # noqa: E402


SCHEMA = "dana.benchmark.tools_history/1"
CLOCK = "time.perf_counter (monotonic)"
SEED_SCHEMA_VERSION = 1
# Journal config used by the history lane for both timelines (declared, echoed
# in the output): a large compression trigger + context budget so that
# ``to_llm_messages()`` reflects the retained working set rather than a
# compaction decision. Compaction itself is excluded (no LLM calls here).
CONTEXT_TOKENS = 32000
COMPRESS_TRIGGER_TOKENS = 150000
SEED_BATCH_TURNS = 250
# Nested-subagent lane: child agents use the production default runtime/codec
# (NativeToolsCodec → CodecRuntimeWithNativeToolUse), driven by a deterministic
# in-process provider. Only the WIRE is substituted; the agent, the STAR loop,
# the Task tool, the registry/dispatch and the timeline repositories are real.
CHILD_AGENT_CTOR = "STARAgent(agent_type='bench-child', llm_provider_instance=<controlled>, repository_factory=<local temp>, enable_skills=False, enable_web_search=False, enable_code_execution=False, enable_assistant=False, compress_timeline=False, auto_register=False)"

# Env vars whose literal value is benign to report (never report other values).
_REPORTABLE_FLAG_VALUES = (
    "LANGFUSE_ENABLED",
    "LANGSMITH_TRACING",
    "DANA_CODE_MCP_ENABLED",
    "DANA_CODE_TOOL_CATALOG_ENABLED",
    "DANA_CODE_PERMISSION_PREFLIGHT_ENABLED",
    "DANA_CODE_MULTIMODAL_ENABLED",
    "DANA_MOCK_LLM",
    "DANA_USE_REAL_LLM",
    "DANA_COMPACT_TRIGGER_TOKENS",
    "DANA_DISABLE_TOOL_RESULT_DUMP_RESOURCE",
)
_ENV_PRESENCE_ONLY = (
    "DANA_LLM_PROVIDER",
    "DANA_MODEL",
    "DANA_MCP_SERVERS",
    "DANA_CODE_JOURNAL",
    "DANA_ACP_JOURNAL",
)


# ---------------------------------------------------------------------------
# Generic measurement helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _perf_ms() -> float:
    return time.perf_counter() * 1000.0


def _summarize(samples: Sequence[float]) -> dict[str, Any]:
    """Summary for raw monotonic duration samples (ms).

    p95 is only reported with >= 20 samples; below that the uncertainty is
    disclosed instead of presenting a stable-looking p95 from a handful of runs.
    """
    if not samples:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None, "p95": None, "p95_unavailable_reason": "no samples"}
    ordered = sorted(float(s) for s in samples)
    out: dict[str, Any] = {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
    }
    if len(ordered) >= 20:
        out["p95"] = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    else:
        out["p95"] = None
        out["p95_unavailable_reason"] = f"only {len(ordered)} samples; p95 suppressed (>= 20 required)"
    return out


def _check(checks: list[dict[str, Any]], check_id: str, status: str, expected: Any, observed: Any, note: str = "") -> None:
    checks.append({"id": check_id, "status": status, "expected": expected, "observed": observed, "note": note})


def _failures_from_checks(checks: Sequence[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    out = []
    for c in checks:
        if c["status"] == "pass":
            continue
        out.append(
            {
                "id": c["id"],
                "severity": "defect" if c["status"] == "fail" else "observation",
                "source": source,
                "detail": c["note"] or f"expected={c['expected']!r} observed={c['observed']!r}",
            }
        )
    return out


class Heartbeat:
    """Records event-loop scheduling overshoot against a fixed interval.

    Overshoot = actual tick time - scheduled tick time. A loop that stays free
    keeps overshoot near 0; a loop blocked by in-loop synchronous work shows
    overshoot on the order of the blocking duration.
    """

    def __init__(self, interval_ms: float) -> None:
        self.interval_s = max(interval_ms, 1.0) / 1000.0
        self.overshoots_ms: list[float] = []
        self.ticks = 0
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._next_tick = 0.0

    async def __aenter__(self) -> Heartbeat:
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        # A blocked loop can finish the operation before the overdue timer's
        # coroutine resumes. Record that pending deadline before stopping it.
        overdue_ms = (time.perf_counter() - self._next_tick) * 1000.0
        if overdue_ms > 0:
            self.overshoots_ms.append(overdue_ms)
        self._stop.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        return False

    async def _run(self) -> None:
        next_t = time.perf_counter() + self.interval_s
        self._next_tick = next_t
        while not self._stop.is_set():
            delay = next_t - time.perf_counter()
            if delay > 0:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                if self._stop.is_set():
                    break
            now = time.perf_counter()
            self.overshoots_ms.append(max(0.0, (now - next_t) * 1000.0))
            self.ticks += 1
            next_t += self.interval_s
            if now - next_t > self.interval_s:  # resync after a long stall
                next_t = now + self.interval_s
            self._next_tick = next_t


def _pending_tasks() -> int:
    """Count asyncio tasks other than the current one (orphan-task check)."""
    current = asyncio.current_task()
    return sum(1 for t in asyncio.all_tasks() if t is not current and not t.done())


def _thread_names() -> list[str]:
    return sorted([t.name for t in threading.enumerate()])


# ---------------------------------------------------------------------------
# Tools lane - deterministic synthetic tool surface + real ToolExecutor
# ---------------------------------------------------------------------------


class ToolIoFixture:
    """Synthetic I/O + mutation tool surface registered on the real ToolExecutor.

    ``sync_*`` methods block the calling thread (``time.sleep``); ``async_*``
    methods await (``asyncio.sleep``). ``*_append`` methods mutate shared state so
    ordering and permission-before-effect can be observed exactly.

    Methods carry Google-style docstrings because
    ``Misc.parse_method_signature`` (dana/common/utils/misc.py:202) raises on
    methods without an ``Args:`` section, which makes
    ``validate_and_cast_method_arguments`` return the arguments unchanged. Both
    shapes are exercised: the documented methods for the normal path, and
    ``sync_io_bare`` for the undocumented path.
    """

    resource_id = "bench-io"

    def __init__(self) -> None:
        self.effects: list[dict[str, Any]] = []

    def sync_io(self, token: str, delay_ms: int = 0) -> dict[str, Any]:
        """Synthetic blocking I/O.

        Args:
            token (str): Call identifier echoed back in the result.
            delay_ms (int): Blocking duration in milliseconds.

        Returns:
            dict: token, delay_ms and impl marker.
        """
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
        return {"token": token, "delay_ms": delay_ms, "impl": "sync"}

    async def async_io(self, token: str, delay_ms: int = 0) -> dict[str, Any]:
        """Synthetic awaiting I/O.

        Args:
            token (str): Call identifier echoed back in the result.
            delay_ms (int): Await duration in milliseconds.

        Returns:
            dict: token, delay_ms and impl marker.
        """
        if delay_ms:
            await asyncio.sleep(delay_ms / 1000.0)
        return {"token": token, "delay_ms": delay_ms, "impl": "async"}

    def sync_append(self, index: int, delay_ms: int = 0) -> dict[str, Any]:
        """Append one ordered mutation to shared state, then return its position.

        Args:
            index (int): Submission index of this mutation.
            delay_ms (int): Blocking duration in milliseconds.

        Returns:
            dict: the submitted index and its observed position.
        """
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
        self.effects.append({"index": index, "impl": "sync"})
        return {"index": index, "position": len(self.effects) - 1}

    async def async_append(self, index: int, delay_ms: int = 0) -> dict[str, Any]:
        """Await, then append one ordered mutation to shared state.

        Args:
            index (int): Submission index of this mutation.
            delay_ms (int): Await duration in milliseconds.

        Returns:
            dict: the submitted index and its observed position.
        """
        if delay_ms:
            await asyncio.sleep(delay_ms / 1000.0)
        await asyncio.sleep(0)  # explicit yield so gather can interleave
        self.effects.append({"index": index, "impl": "async"})
        return {"index": index, "position": len(self.effects) - 1}

    def sync_io_bare(self, token: str, delay_ms: Any = 0) -> dict[str, Any]:
        # Intentionally undocumented: no Args section for Misc.parse_method_signature.
        # Tolerates an uncoerced value so the observation is the received TYPE,
        # not an exception from the body.
        if isinstance(delay_ms, (int, float)) and delay_ms:
            time.sleep(delay_ms / 1000.0)
        return {"token": token, "delay_ms": delay_ms, "value_type": type(delay_ms).__name__}


def _make_executor(fixture: ToolIoFixture) -> ToolExecutor:
    """Real ToolExecutor wired to the fixture (registry shape per runtime/base.py:138-141)."""
    registry: dict[str, tuple[Any, str]] = {
        "bench_io_sync": (fixture, "sync_io"),
        "bench_io_async": (fixture, "async_io"),
        "bench_append_sync": (fixture, "sync_append"),
        "bench_append_async": (fixture, "async_append"),
        "bench_io_bare": (fixture, "sync_io_bare"),
    }
    return ToolExecutor(tool_name_registry_getter=lambda: registry)


def _call_prefix(case_id: str) -> str:
    """Deterministic short prefix unique to a case (``n32`` and ``n3`` must not collide)."""
    return hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:6]


def _io_calls(name: str, n: int, delay_ms: float, prefix: str = "c") -> list[dict[str, Any]]:
    return [
        {"tool_call_id": f"{prefix}-{i:03d}", "function": name, "arguments": {"token": f"t{i}", "delay_ms": int(delay_ms)}}
        for i in range(n)
    ]


def _append_calls(name: str, n: int, delay_ms: float, prefix: str = "m") -> list[dict[str, Any]]:
    return [{"tool_call_id": f"{prefix}-{i:03d}", "function": name, "arguments": {"index": i, "delay_ms": int(delay_ms)}} for i in range(n)]


def _verify_io_results(calls: Sequence[dict[str, Any]], results: Sequence[dict[str, Any]]) -> list[str]:
    """Exact per-call ID + result verification (content and identity, not just count)."""
    problems: list[str] = []
    if len(results) != len(calls):
        problems.append(f"result count {len(results)} != call count {len(calls)}")
    for call, res in zip(calls, results, strict=False):
        cid = call["tool_call_id"]
        if res.get("tool_call_id") != cid:
            problems.append(f"{cid}: tool_call_id={res.get('tool_call_id')!r}")
        if res.get("success") is not True:
            problems.append(f"{cid}: success={res.get('success')!r}")
        if res.get("type") != "resource":
            problems.append(f"{cid}: type={res.get('type')!r}")
        if res.get("target") != call["function"]:
            problems.append(f"{cid}: target={res.get('target')!r}")
        inner = res.get("result")
        if not isinstance(inner, dict):
            problems.append(f"{cid}: result is not a dict ({type(inner).__name__})")
            continue
        if inner.get("token") != call["arguments"]["token"]:
            problems.append(f"{cid}: token={inner.get('token')!r}")
        if inner.get("delay_ms") != call["arguments"]["delay_ms"]:
            problems.append(f"{cid}: delay_ms={inner.get('delay_ms')!r}")
        expected_impl = "sync" if call["function"].endswith("_sync") else "async"
        if inner.get("impl") != expected_impl:
            problems.append(f"{cid}: impl={inner.get('impl')!r} expected {expected_impl!r}")
    return problems


def _case(
    case_id: str,
    phase: str,
    config: dict[str, Any],
    samples: Sequence[float],
    *,
    warmups: int,
    full_repetitions: int,
    reduced_reason: str | None,
    runtime_api: str,
    observations: dict[str, Any] | None = None,
    aux_samples: dict[str, Sequence[float]] | None = None,
    checks: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    checks = list(checks or [])
    return {
        "id": case_id,
        "lane": "tools",
        "phase": phase,
        "clock": CLOCK,
        "runtime_api": runtime_api,
        "config": config,
        "warmups": warmups,
        "sample_count": len(samples),
        "sample_policy": {
            "full_repetitions_requested": full_repetitions,
            "reduced": reduced_reason is not None,
            "reason": reduced_reason or "full repetitions",
        },
        "samples_ms": [float(s) for s in samples],
        "summary": _summarize(samples),
        "aux_samples": {k: [float(x) for x in v] for k, v in (aux_samples or {}).items()},
        "aux_summary": {k: _summarize(v) for k, v in (aux_samples or {}).items()},
        "observations": observations or {},
        "checks": checks,
        "known_failures": _failures_from_checks(checks, source="script"),
    }


async def _run_io_case(
    executor: ToolExecutor,
    agent: Any,
    *,
    case_id: str,
    impl: str,
    mode: str,
    n: int,
    delay_ms: float,
    reps: int,
    warmups: int,
    full_repetitions: int,
    reduced_reason: str | None,
    heartbeat_ms: float,
) -> dict[str, Any]:
    tool_name = "bench_io_sync" if impl == "sync" else "bench_io_async"
    samples: list[float] = []
    hb_max: list[float] = []
    hb_ticks: list[float] = []
    first_overshoots: list[float] = []
    problems: list[str] = []
    problems_set = False
    results_count_ok = True

    for rep in range(warmups + reps):
        calls = _io_calls(tool_name, n, delay_ms, prefix=f"{_call_prefix(case_id)}{rep:02d}")
        if mode == "async":
            hb = Heartbeat(heartbeat_ms)
            async with hb:
                t0 = time.perf_counter()
                results = await executor.execute_tools_async(agent, calls)
                elapsed = (time.perf_counter() - t0) * 1000.0
        else:
            hb = None
            t0 = time.perf_counter()
            results = executor.execute_tools(agent, calls, parallel=False)
            elapsed = (time.perf_counter() - t0) * 1000.0
        rep_problems = _verify_io_results(calls, results)
        if len(results) != len(calls):
            results_count_ok = False
        if rep_problems and not problems_set:
            problems = rep_problems
            problems_set = True
        if rep < warmups:
            continue
        samples.append(elapsed)
        if hb is not None:
            hb_max.append(max(hb.overshoots_ms, default=0.0))
            hb_ticks.append(float(hb.ticks))
            if not first_overshoots:
                first_overshoots = list(hb.overshoots_ms)

    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "tool_call_ids_and_results_exact",
        "pass" if not problems else "fail",
        "all ids/results exact",
        problems[:3] if problems else "exact",
        "per-call tool_call_id, success, type, target, token, delay_ms and impl verified on every repetition",
    )
    _check(
        checks,
        "result_count_preserved",
        "pass" if results_count_ok else "fail",
        n,
        "one result per call on every repetition" if results_count_ok else "count mismatch observed",
    )
    if mode == "async":
        max_lag = max(hb_max) if hb_max else 0.0
        _check(
            checks,
            "heartbeat_max_overshoot_ms",
            "observation",
            "< 20 ms target (researcher budget)",
            round(max_lag, 3),
            "B0 observation, not an assertion: in-loop synchronous tool bodies block the loop for their full duration",
        )
    checks_observations = {
        "expected_parallel_behavior": "overlapped"
        if impl == "async"
        else "serialized (in-loop blocking for async mode; no loop involved for sequential mode)",
        "batch_wall_ms_median": _summarize(samples)["median"],
        "io_per_call_ms": delay_ms,
        "ideal_overlapped_ms": delay_ms,
        "ideal_serialized_ms": delay_ms * n,
    }
    return _case(
        case_id,
        phase="tool_execution",
        config={
            "mode": "async (ToolExecutor.execute_tools_async)"
            if mode == "async"
            else "sequential (ToolExecutor.execute_tools, parallel=False)",
            "tool_impl": impl,
            "tool_name": tool_name,
            "n": n,
            "per_call_delay_ms": delay_ms,
            "heartbeat_interval_ms": heartbeat_ms if mode == "async" else None,
        },
        samples=samples,
        warmups=warmups,
        full_repetitions=full_repetitions,
        reduced_reason=reduced_reason,
        runtime_api="ToolExecutor.execute_tools_async(agent, tool_calls)"
        if mode == "async"
        else "ToolExecutor.execute_tools(agent, tool_calls, parallel=False)",
        observations=checks_observations,
        aux_samples={"heartbeat_max_overshoot_ms": hb_max, "heartbeat_ticks": hb_ticks},
        checks=checks,
    ) | {"observations_details": {"heartbeat_overshoots_first_repetition_ms": first_overshoots}}


async def _run_conflict_case(
    executor: ToolExecutor,
    fixture: ToolIoFixture,
    agent: Any,
    *,
    case_id: str,
    impl: str,
    mode: str,
    n: int,
    delay_ms: float,
    reps: int,
    warmups: int,
    full_repetitions: int,
    reduced_reason: str | None,
) -> dict[str, Any]:
    """Ordered conflicting mutations: every call appends to shared state."""
    tool_name = "bench_append_sync" if impl == "sync" else "bench_append_async"
    samples: list[float] = []
    orders: list[list[int]] = []
    count_ok = True
    for rep in range(warmups + reps):
        fixture.effects.clear()
        calls = _append_calls(tool_name, n, delay_ms, prefix=f"{_call_prefix(case_id)}{rep:02d}")
        t0 = time.perf_counter()
        if mode == "async":
            results = await executor.execute_tools_async(agent, calls)
        else:
            results = executor.execute_tools(agent, calls, parallel=False)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if len(results) != n:
            count_ok = False
        if rep < warmups:
            continue
        samples.append(elapsed)
        orders.append([int(e["index"]) for e in fixture.effects])

    submission = list(range(n))
    ordered = bool(orders) and all(o == submission for o in orders)
    inversions = [sum(1 for i, idx in enumerate(o) if idx != i) for o in orders]
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "result_count_preserved",
        "pass" if count_ok else "fail",
        n,
        "one result per call" if count_ok else "count mismatch observed",
    )
    if mode == "async" and impl == "async":
        _check(
            checks,
            "conflicting_mutations_ordered",
            "observation",
            "declared order or documented interleaving",
            f"first_rep_order={orders[0] if orders else None}",
            "concurrent async mutations interleave by design; the ordering contract applies to the sequential path",
        )
    else:
        _check(
            checks,
            "conflicting_mutations_ordered",
            "pass" if ordered else "fail",
            submission,
            "submission order on every repetition" if ordered else orders[:2],
            "effects must appear in submission order for sequential dispatch and for in-loop blocking tools",
        )
    return _case(
        case_id,
        phase="conflicting_mutations",
        config={
            "mode": "async (execute_tools_async)" if mode == "async" else "sequential (parallel=False)",
            "tool_impl": impl,
            "n": n,
            "per_call_delay_ms": delay_ms,
        },
        samples=samples,
        warmups=warmups,
        full_repetitions=full_repetitions,
        reduced_reason=reduced_reason,
        runtime_api="ToolExecutor.execute_tools_async / execute_tools",
        observations={
            "submission_order": submission,
            "observed_orders_first_3": orders[:3],
            "order_inversions_per_repetition": inversions,
            "all_repetitions_ordered": ordered,
        },
        checks=checks,
    )


async def _run_permission_case(
    executor: ToolExecutor,
    fixture: ToolIoFixture,
    *,
    case_id: str,
    mode: str,
    reps: int,
    warmups: int,
    full_repetitions: int,
) -> dict[str, Any]:
    """EventBus interception: TOOL_CALL block/modify then TOOL_RESULT modify."""
    samples: list[float] = []
    blocked_ok = True
    modified_ok = True
    effects_before_after: list[tuple[int, int]] = []
    result_modified_ok = True

    for rep in range(warmups + reps):
        bus = EventBus()

        def _deny_blocked(event: Any) -> dict[str, Any] | None:
            operation = event.payload.get("operation")
            arguments = dict(operation.arguments) if operation is not None else {}
            if arguments.get("index") == 99:
                return {"block": True, "reason": "bench-deny"}
            return None

        def _rewrite_token(event: Any) -> dict[str, Any] | None:
            operation = event.payload.get("operation")
            arguments = dict(operation.arguments) if operation is not None else {}
            if arguments.get("token") == "orig":
                return {"modify": {"arguments": {**arguments, "token": "rewritten"}}}
            return None

        def _tag_result(event: Any) -> dict[str, Any] | None:
            payload = event.payload.get("result")
            inner = payload.get("result") if isinstance(payload, dict) else None
            if isinstance(inner, dict) and inner.get("token") == "tagme":
                return {"modify": {**payload, "result": {**inner, "tagged": True}}}
            return None

        bus.subscribe(TOOL_CALL, _deny_blocked)
        bus.subscribe(TOOL_CALL, _rewrite_token)
        bus.subscribe(TOOL_RESULT, _tag_result)
        agent = SimpleNamespace(event_bus=bus)

        calls = [
            {
                "tool_call_id": "p-block",
                "function": "bench_append_sync",
                "arguments": {"index": 99, "delay_ms": 0},
            },
            {"tool_call_id": "p-mod", "function": "bench_io_sync", "arguments": {"token": "orig", "delay_ms": 0}},
            {"tool_call_id": "p-tag", "function": "bench_io_sync", "arguments": {"token": "tagme", "delay_ms": 0}},
            {"tool_call_id": "p-plain", "function": "bench_io_sync", "arguments": {"token": "plain", "delay_ms": 0}},
        ]
        fixture.effects.clear()
        t0 = time.perf_counter()
        if mode == "async":
            results = await executor.execute_tools_async(agent, calls)
        else:
            results = executor.execute_tools(agent, calls, parallel=False)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if rep < warmups:
            continue
        samples.append(elapsed)
        effects_before_after.append((0, len(fixture.effects)))
        blocked = next((r for r in results if r.get("tool_call_id") == "p-block"), {})
        # The blocked call is denied by the TOOL_CALL handler regardless of which
        # dispatch path would have raised on the missing registry entry.
        blocked_ok = blocked_ok and blocked.get("success") is False
        modified_inner = next((r.get("result") for r in results if r.get("tool_call_id") == "p-mod"), None)
        modified_ok = modified_ok and isinstance(modified_inner, dict) and modified_inner.get("token") == "rewritten"
        tagged_inner = next((r.get("result") for r in results if r.get("tool_call_id") == "p-tag"), None)
        result_modified_ok = result_modified_ok and isinstance(tagged_inner, dict) and tagged_inner.get("tagged") is True

    no_effect = all(after == before for before, after in effects_before_after)
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "permission_deny_blocks_dispatch",
        "pass" if blocked_ok else "fail",
        "policy_block error result",
        "policy_block observed" if blocked_ok else "not blocked",
        "TOOL_CALL handler returning {'block': True}",
    )
    _check(
        checks,
        "permission_deny_precedes_effect",
        "pass" if no_effect else "fail",
        "0 effects",
        [after for _, after in effects_before_after][:3],
        "the blocked call must not reach the tool body",
    )
    _check(
        checks,
        "permission_argument_modify_reaches_tool",
        "pass" if modified_ok else "fail",
        "token='rewritten'",
        "rewritten" if modified_ok else "unchanged",
        "TOOL_CALL {'modify': {'arguments': ...}} path",
    )
    _check(
        checks,
        "permission_result_modify_applied",
        "pass" if result_modified_ok else "fail",
        "tagged=True",
        "tagged" if result_modified_ok else "untagged",
        "TOOL_RESULT {'modify': ...} path",
    )
    return _case(
        case_id,
        phase="permissions",
        config={"mode": mode, "handlers": ["TOOL_CALL block", "TOOL_CALL modify arguments", "TOOL_RESULT modify"]},
        samples=samples,
        warmups=warmups,
        full_repetitions=full_repetitions,
        reduced_reason=None,
        runtime_api="EventBus.subscribe(TOOL_CALL|TOOL_RESULT) + ToolExecutor emit helpers",
        observations={"effects_before_after_per_repetition": effects_before_after},
        checks=checks,
    )


async def _run_cancellation_case(
    executor: ToolExecutor,
    fixture: ToolIoFixture,
    *,
    case_id: str,
    impl: str,
    n: int,
    delay_ms: float,
    cancel_after_ms: float,
    reps: int,
    warmups: int,
    full_repetitions: int,
    reduced_reason: str | None,
) -> dict[str, Any]:
    """Cancellation observation on the async batch entry point.

    Requesting cancellation of the awaiting task cancels ``asyncio.gather``.
    Whether the in-flight tool actually stops is a property of the tool body:
    an in-loop synchronous body cannot be interrupted; an awaiting body can.
    """
    tool_name = "bench_append_sync" if impl == "sync" else "bench_append_async"
    samples: list[float] = []
    outcomes: list[str] = []
    effects: list[int] = []
    for rep in range(warmups + reps):
        fixture.effects.clear()
        calls = _append_calls(tool_name, n, delay_ms, prefix=f"{_call_prefix(case_id)}{rep:02d}")
        task = asyncio.create_task(executor.execute_tools_async(SimpleNamespace(), calls))
        await asyncio.sleep(cancel_after_ms / 1000.0)
        requested = time.perf_counter()
        task.cancel()
        outcome = "completed"
        try:
            await task
        except asyncio.CancelledError:
            outcome = "cancelled"
        except Exception as exc:  # noqa: BLE001 - recorded as an observation
            outcome = f"error:{type(exc).__name__}"
        latency = (time.perf_counter() - requested) * 1000.0
        if rep < warmups:
            continue
        samples.append(latency)
        outcomes.append(outcome)
        effects.append(len(fixture.effects))

    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "cancel_request_to_observation_latency",
        "observation",
        f"< {cancel_after_ms + 50:.0f} ms for cooperative await; longer when the loop is blocked",
        _summarize(samples),
        "B0 observation: the async batch has no cancellation owner of its own; the awaiting caller's cancel reaches gather only when the loop regains control",
    )
    _check(
        checks,
        "in_flight_effects_after_cancel",
        "observation",
        "0 for an awaiting tool; unfinished for an in-loop blocking tool",
        {"effects_per_repetition": effects[:5], "n_tools": n},
        "truthful disposition only: a cancelled await does not undo a synchronous body that already ran",
    )
    return _case(
        case_id,
        phase="cancellation",
        config={"mode": "async batch", "tool_impl": impl, "n": n, "per_call_delay_ms": delay_ms, "cancel_after_ms": cancel_after_ms},
        samples=samples,
        warmups=warmups,
        full_repetitions=full_repetitions,
        reduced_reason=reduced_reason,
        runtime_api="asyncio.Task.cancel() around ToolExecutor.execute_tools_async",
        observations={"outcomes": outcomes, "effects_recorded_per_repetition": effects},
        checks=checks,
    )


async def _run_arg_cast_case(executor: ToolExecutor) -> dict[str, Any]:
    """Argument coercion across the string/int boundary (``validate_and_cast_method_arguments``).

    Two shapes are exercised: a documented tool method whose signature is
    parseable (coercion fires) and an undocumented method whose signature parse
    raises (coercion is skipped, arguments pass through unchanged).
    """
    agent = SimpleNamespace()
    documented: dict[str, list[Any]] = {"samples": [], "observed": []}
    bare: dict[str, list[Any]] = {"samples": [], "observed": []}
    for _rep in range(1 + 5):
        for name, bucket in (("bench_io_sync", documented), ("bench_io_bare", bare)):
            calls = [{"tool_call_id": "cast-0", "function": name, "arguments": {"token": "cast", "delay_ms": "1"}}]
            t0 = time.perf_counter()
            results = executor.execute_tools(agent, calls, parallel=False)
            elapsed = (time.perf_counter() - t0) * 1000.0
            bucket["samples"].append(elapsed)
            inner = results[0].get("result") if results else None
            if isinstance(inner, dict):
                bucket["observed"].append(type(inner.get("delay_ms")).__name__ if name == "bench_io_sync" else inner.get("value_type"))
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "documented_method_string_int_coerced",
        "pass" if all(o == "int" for o in documented["observed"]) else "fail",
        "delay_ms reaches the body as int",
        documented["observed"][:3],
        "a method with a parseable Google-style Args section has its arguments cast to the declared types",
    )
    _check(
        checks,
        "undocumented_method_coercion_skipped",
        "observation",
        "declared behavior for a method whose signature cannot be parsed",
        bare["observed"][:3],
        "Misc.parse_method_signature raises KeyError('args') when the method has no Args section; "
        "validate_and_cast_method_arguments then returns the raw string. Recorded, not asserted: it is an argument-typing observation.",
    )
    return _case(
        "tools.arg_cast.string_int",
        phase="tool_dispatch",
        config={
            "documented_call": "bench_io_sync(token='cast', delay_ms='1')",
            "undocumented_call": "bench_io_bare(token='cast', delay_ms='1')",
        },
        samples=documented["samples"][1:],
        warmups=1,
        full_repetitions=5,
        reduced_reason=None,
        runtime_api="ToolExecutor.execute_tools → tool_executor_helpers.validate_and_cast_method_arguments → Misc.parse_method_signature",
        observations={"undocumented_samples_ms": bare["samples"][1:], "undocumented_observed_types": bare["observed"]},
        checks=checks,
    )


async def _run_tools_lane(args: argparse.Namespace, tool_ns: Sequence[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixture = ToolIoFixture()
    executor = _make_executor(fixture)
    agent = SimpleNamespace()  # no event_bus → no interception on the plain cases
    cases: list[dict[str, Any]] = []
    delay = float(args.tool_sleep_ms)

    def _policy(n: int) -> tuple[int, str | None]:
        """Declared sample policy for the independent-I/O cases.

        Batches reaching ``--expensive-tool-n`` receive ``--expensive-samples``
        instead of ``--repetitions``. Every reduction is disclosed per case,
        with p95 suppressed below 20 samples.
        """
        if n >= args.expensive_tool_n:
            reps = min(args.repetitions, args.expensive_samples)
            return (
                reps,
                f"n={n} >= --expensive-tool-n={args.expensive_tool_n}: {reps} measured samples instead of {args.repetitions} (p95 suppressed below 20 samples)",
            )
        return args.repetitions, None

    for n in tool_ns:
        for mode in ("async", "sequential"):
            for impl in ("sync", "async"):
                if mode == "sequential" and impl == "async":
                    continue  # declared coverage gap: sync executor + async tool uses the safe_asyncio_run thread bridge
                reps, reduced = _policy(n)
                case_id = f"tools.{mode}_{impl}_io.n{n}"
                cases.append(
                    await _run_io_case(
                        executor,
                        agent,
                        case_id=case_id,
                        impl=impl,
                        mode=mode,
                        n=n,
                        delay_ms=delay,
                        reps=reps,
                        warmups=1,
                        full_repetitions=args.repetitions,
                        reduced_reason=reduced,
                        heartbeat_ms=float(args.heartbeat_ms),
                    )
                )

    conflict_n = int(args.conflict_n)
    for mode, impl in (("sequential", "sync"), ("async", "sync"), ("async", "async")):
        cases.append(
            await _run_conflict_case(
                executor,
                fixture,
                agent,
                case_id=f"tools.conflict_{impl}_{mode}.n{conflict_n}",
                impl=impl,
                mode=mode,
                n=conflict_n,
                delay_ms=min(delay, 10.0),
                reps=args.expensive_samples,
                warmups=1,
                full_repetitions=args.repetitions,
                reduced_reason=f"conflict probe uses {args.expensive_samples} measured samples (repeated identical schedules add no information)",
            )
        )

    for mode in ("async", "sequential"):
        cases.append(
            await _run_permission_case(
                executor,
                fixture,
                case_id=f"tools.permissions.{mode}",
                mode=mode,
                reps=min(args.repetitions, 20),
                warmups=1,
                full_repetitions=args.repetitions,
            )
        )

    for impl in ("async", "sync"):
        cases.append(
            await _run_cancellation_case(
                executor,
                fixture,
                case_id=f"tools.cancellation_gather.{impl}_tools",
                impl=impl,
                n=4,
                delay_ms=delay,
                cancel_after_ms=float(args.cancel_after_ms),
                reps=args.expensive_samples,
                warmups=1,
                full_repetitions=args.repetitions,
                reduced_reason=f"cancellation probe uses {args.expensive_samples} measured samples",
            )
        )

    cases.append(await _run_arg_cast_case(executor))

    lane_meta = {
        "runtime_api": "dana.core.tool.tool_executor.ToolExecutor (real, registry dispatch path)",
        "fixture": "synthetic in-module resource; time.sleep for sync_*, asyncio.sleep for async_*",
        "coverage_gaps": [
            "ToolExecutionEngine (dana/core/tool/execution_engine.py) catalog/isolated-worker path is not exercised",
            "MCP remote dispatch and remote cancellation are not exercised (DANA_CODE_MCP_ENABLED=0)",
            "ToolExecutor.execute_tools(parallel=True) ThreadPoolExecutor path is not exercised",
            "sync executor + async tool body (Misc.safe_asyncio_run thread bridge) is not exercised",
            "permission enforcement through the real PolicyEvaluator is not exercised (event-bus interception contract only)",
        ],
    }
    return cases, lane_meta


# ---------------------------------------------------------------------------
# Nested-subagent admission lane - real TaskResource + real STARAgent children
# ---------------------------------------------------------------------------


class ControlledChildProvider(LLMProvider):
    """Deterministic in-process provider for the real child STARAgent.

    Only the WIRE is substituted (no network, no credentials, no tokens billed).
    The child agent, its runtime, the STAR loop, the ``Task`` tool, the
    registry/dispatch path and the timeline repositories are the real runtime
    objects, so the admission/queueing behavior measured here is the runtime's.
    """

    def __init__(
        self,
        child_key: str,
        delay_ms: float = 0.0,
        on_enter: Callable[[], None] | None = None,
        on_exit: Callable[[], None] | None = None,
    ) -> None:
        self.model = "controlled-child"
        self.child_key = child_key
        self.delay_ms = delay_ms
        self.chat_calls = 0
        self.peak_active = 0
        self._active = 0
        self._on_enter = on_enter
        self._on_exit = on_exit
        self.entered: asyncio.Event | None = None
        self.release: asyncio.Event | None = None
        self._entered_count = 0
        self.entered_at: list[float] = []
        self.exited_at: list[float] = []

    @property
    def supports_native_tools(self) -> bool:
        return False  # keep the child on the JSON protocol (no tool schemas needed)

    @property
    def active(self) -> int:
        return self._active

    @property
    def entered_count(self) -> int:
        return self._entered_count

    async def chat(self, messages: list[Any], tools: list[Any] | None = None, **kwargs: Any) -> LLMResponse:
        return await self._respond()

    async def stream(self, messages: list[Any], tools: list[Any] | None = None, **kwargs: Any):
        from dana.common.llm.types import LLMStreamChunk

        response = await self._respond()
        yield LLMStreamChunk(type="text_delta", content=response.content)

    async def _respond(self) -> LLMResponse:
        self._active += 1
        self._entered_count += 1
        self.peak_active = max(self.peak_active, self._active)
        self.entered_at.append(time.perf_counter())
        if self._on_enter is not None:
            self._on_enter()
        if self.entered is not None and self.entered_count == 1:
            self.entered.set()
        try:
            if self.release is not None:
                await self.release.wait()
            elif self.delay_ms:
                await asyncio.sleep(self.delay_ms / 1000.0)
            else:
                # Explicit cooperative yield: a real provider awaits I/O here, which
                # is what lets sibling children be admitted concurrently. Without it
                # the zero-delay fixture would serialize children and report an
                # admission width of 1 for reasons that are an artifact of the fixture.
                await asyncio.sleep(0)
        finally:
            self._active -= 1
            if self._on_exit is not None:
                self._on_exit()
            self.exited_at.append(time.perf_counter())
        self.chat_calls += 1
        return LLMResponse(content='{"done": true, "response": "child-ok"}', model=self.model, finish_reason="stop")


class ChildAgentFactory:
    """Real STARAgent children over a controlled provider + local temp repositories.

    Registers as a ``TaskResource`` factory via ``functools.partial`` so the
    dynamic ``Task`` docstring can read ``.func`` (TaskResource.register_agent
    contract) while every spawn still gets a disjoint agent instance.
    """

    def __init__(self, workspace: Path, delay_ms: float, barrier: bool = False) -> None:
        self.workspace = workspace
        self.delay_ms = delay_ms
        self.barrier = barrier
        self.providers: list[ControlledChildProvider] = []
        self.sessions: list[str] = []
        self.barrier_entered: asyncio.Event | None = asyncio.Event() if barrier else None
        self.barrier_release: asyncio.Event | None = asyncio.Event() if barrier else None
        self._spawn_count = 0
        self._factory = functools.partial(self._build)
        # Shared cross-child concurrency accounting (per-provider counters cannot
        # show admission width: each child owns its own provider instance).
        self._live_children = 0
        self.peak_live_children = 0
        self.live_samples: list[int] = []

    @property
    def factory(self) -> Any:
        return self._factory

    def child_entered(self) -> None:
        self._live_children += 1
        self.peak_live_children = max(self.peak_live_children, self._live_children)
        self.live_samples.append(self._live_children)

    def child_exited(self) -> None:
        self._live_children -= 1

    @property
    def children_entered_provider(self) -> int:
        return sum(1 for p in self.providers if p.entered_count > 0)

    def reset_concurrency(self) -> None:
        self._live_children = 0
        self.peak_live_children = 0
        self.live_samples = []

    def _repository_factory(self) -> RepositoryFactory:
        factory = RepositoryFactory()
        factory.register(RepositoryType.TIMELINE, LocalTimelineRepository, FileStorageConfig(workspace_folder=str(self.workspace)))
        return factory

    def _build(self) -> STARAgent:
        self._spawn_count += 1
        provider = ControlledChildProvider(
            f"child-{self._spawn_count}", delay_ms=self.delay_ms, on_enter=self.child_entered, on_exit=self.child_exited
        )
        if self.barrier:
            provider.entered = self.barrier_entered
            provider.release = self.barrier_release
        self.providers.append(provider)
        agent = STARAgent(
            agent_type="bench-child",
            agent_id=f"bench-child-{self._spawn_count}",
            llm_provider_instance=provider,
            repository_factory=self._repository_factory(),
            auto_register=False,
            enable_skills=False,
            enable_web_search=False,
            enable_code_execution=False,
            enable_assistant=False,
            compress_timeline=False,
        )
        self.sessions.append(agent._session_id)
        return agent

    def spawn_count(self) -> int:
        return self._spawn_count


async def _await_barrier(factory: ChildAgentFactory, expected: int, timeout_s: float) -> bool:
    """Best-effort rendezvous: wait until ``expected`` children reach the provider."""
    if factory.barrier_entered is None:
        return False
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if factory.children_entered_provider >= expected:
            return True
        await asyncio.sleep(0.005)
    return factory.children_entered_provider >= expected


def _task_calls(n: int, prefix: str, prompt: str = "child task") -> list[dict[str, Any]]:
    return [
        {
            "tool_call_id": f"{prefix}-{i:03d}",
            "function": "Task",
            "arguments": {"description": "bench task", "prompt": f"{prompt} {i}", "subagent_type": "bench-child"},
        }
        for i in range(n)
    ]


def _verify_task_results(calls: Sequence[dict[str, Any]], results: Sequence[dict[str, Any]], expected_sessions: int) -> list[str]:
    """Exact per-call verification: each call gets its own result, its own child
    and its own session id (a shared child would recycle a session id)."""
    problems: list[str] = []
    if len(results) != len(calls):
        problems.append(f"result count {len(results)} != call count {len(calls)}")
    session_ids: list[str] = []
    for call, res in zip(calls, results, strict=False):
        cid = call["tool_call_id"]
        if res.get("tool_call_id") != cid:
            problems.append(f"{cid}: tool_call_id={res.get('tool_call_id')!r}")
        if res.get("success") is not True:
            problems.append(f"{cid}: success={res.get('success')!r} result={str(res.get('result'))[:120]!r}")
            continue
        if res.get("type") != "resource":
            problems.append(f"{cid}: type={res.get('type')!r}")
        if res.get("target") != "Task":
            problems.append(f"{cid}: target={res.get('target')!r}")
        text = res.get("result")
        if not isinstance(text, str) or "child-ok" not in text:
            problems.append(f"{cid}: child response missing from result ({str(text)[:80]!r})")
            continue
        marker = "[session_id: "
        if marker not in text:
            problems.append(f"{cid}: session marker missing (resume handle unusable)")
            continue
        session_ids.append(text.rsplit(marker, 1)[1].rstrip("]").strip())
    if len(session_ids) != len(set(session_ids)):
        problems.append(f"session ids not distinct: {session_ids}")
    if expected_sessions and len(set(session_ids)) != expected_sessions:
        problems.append(f"expected {expected_sessions} distinct child sessions, observed {len(set(session_ids))}")
    return problems


async def _run_subagent_lane(args: argparse.Namespace, sizes: Sequence[int], tmp_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for n in sizes:
        reps = args.repetitions if n < args.expensive_subagent_n else min(args.repetitions, args.expensive_samples)
        reduced = None
        if reps != args.repetitions:
            reduced = (
                f"n={n} >= --expensive-subagent-n={args.expensive_subagent_n}: {reps} measured samples instead of "
                f"{args.repetitions} (each sample spawns + runs {n} real child agents) - uncertainty disclosed, p95 suppressed below 20 samples"
            )
        case_dir = tmp_root / f"subagents_n{n}"
        case_dir.mkdir(parents=True, exist_ok=True)
        workspace = case_dir / "child-repos"
        workspace.mkdir(parents=True, exist_ok=True)
        factory = ChildAgentFactory(workspace, delay_ms=0.0)
        resource = TaskResource(resource_id="task", agents={"bench-child": factory.factory})
        executor = ToolExecutor(tool_name_registry_getter=lambda resource=resource: {"Task": (resource, "task")})

        samples: list[float] = []
        peak_entries: list[float] = []
        spawned: list[float] = []
        entered: list[float] = []
        problems: list[str] = []
        sessions_ok = True
        for rep in range(1 + reps):
            provider_start = len(factory.providers)
            factory.reset_concurrency()
            calls = _task_calls(n, f"{_call_prefix(f'subagent.n{n}')}{rep:02d}")
            t0 = time.perf_counter()
            results = await executor.execute_tools_async(SimpleNamespace(), calls)
            elapsed = (time.perf_counter() - t0) * 1000.0
            rep_providers = factory.providers[provider_start:]
            rep_problems = _verify_task_results(calls, results, n)
            if rep_problems and not problems:
                problems = rep_problems
            if len(rep_providers) != n:
                sessions_ok = False
            if rep == 0:
                continue
            samples.append(elapsed)
            peak_entries.append(float(factory.peak_live_children))
            spawned.append(float(len(rep_providers)))
            entered.append(float(sum(1 for p in rep_providers if p.entered_count > 0)))

        checks: list[dict[str, Any]] = []
        _check(
            checks,
            "one_child_agent_per_call",
            "pass" if sessions_ok and not problems else "fail",
            n,
            {"children_per_repetition": spawned[:3], "problems": problems[:3]},
            "each Task call must spawn its own child (a shared instance would corrupt concurrent state and recycle session ids)",
        )
        _check(
            checks,
            "distinct_child_sessions",
            "pass" if not problems else "fail",
            f"{n} distinct [session_id: ...] handles",
            "distinct" if not problems else problems[:3],
            "resume handles must be per-child, not shared",
        )
        _check(
            checks,
            "peak_simultaneous_live_children",
            "observation",
            f"<= {n} (unbounded today: no shared subagent admission cap)",
            _summarize(peak_entries),
            "admission evidence, not an assertion: N children are launched concurrently, so peak live-child concurrency tracks N",
        )
        observations = {
            "n": n,
            "children_per_repetition": _summarize(spawned),
            "children_entered_provider": _summarize(entered),
            "peak_simultaneous_live_children": _summarize(peak_entries),
            "wall_ms_median": _summarize(samples)["median"],
            "wall_ms_ideal_serial_ms": None,
            "admission_evidence": (
                "no shared admission primitive exists on the Task path: every Task call constructs a child and awaits its full "
                "STAR loop, so N calls in one ToolExecutor batch run concurrently. Peak simultaneous live children is the "
                "observed admission width."
            ),
        }
        cases.append(
            {
                "id": f"subagents.concurrent_tasks.n{n}",
                "lane": "subagents",
                "phase": "subagent_admission",
                "clock": CLOCK,
                "runtime_api": "ToolExecutor.execute_tools_async → ToolResource dispatch → TaskResource.task → real child STARAgent.aquery → provider.chat",
                "config": {
                    "n": n,
                    "child_agent_ctor": CHILD_AGENT_CTOR,
                    "child_provider": "ControlledChildProvider (deterministic, in-process, no network)",
                    "tool_call": "Task(description='bench task', prompt=..., subagent_type='bench-child')",
                    "executor_mode": "async batch (asyncio.gather)",
                    "child_repositories": "LocalTimelineRepository under a temporary workspace",
                },
                "warmups": 1,
                "sample_count": len(samples),
                "sample_policy": {
                    "full_repetitions_requested": args.repetitions,
                    "reduced": reduced is not None,
                    "reason": reduced or "full repetitions",
                },
                "samples_ms": samples,
                "summary": _summarize(samples),
                "aux_samples": {
                    "peak_simultaneous_live_children": peak_entries,
                    "children_spawned": spawned,
                    "children_entered_provider": entered,
                },
                "aux_summary": {
                    "peak_simultaneous_live_children": _summarize(peak_entries),
                    "children_spawned": _summarize(spawned),
                    "children_entered_provider": _summarize(entered),
                },
                "observations": observations,
                "checks": checks,
                "known_failures": _failures_from_checks(checks, source="script"),
            }
        )

    # --- deterministic admission gate: how many children can ACTUALLY run at once ---
    gate_n = max(sizes)
    gate_dir = tmp_root / "subagents_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    gate_workspace = gate_dir / "child-repos"
    gate_workspace.mkdir(parents=True, exist_ok=True)
    gate_factory = ChildAgentFactory(gate_workspace, delay_ms=0.0, barrier=True)
    gate_resource = TaskResource(resource_id="task", agents={"bench-child": gate_factory.factory})
    gate_executor = ToolExecutor(tool_name_registry_getter=lambda: {"Task": (gate_resource, "task")})
    gate_checks: list[dict[str, Any]] = []
    try:
        calls = _task_calls(gate_n, "gate")
        gate_task = asyncio.create_task(gate_executor.execute_tools_async(SimpleNamespace(), calls))
        reached = await _await_barrier(gate_factory, gate_n, timeout_s=max(5.0, args.subagent_gate_timeout_s))
        released = True
        if not reached:
            released = False
            # Do NOT leave the batch parked: release the gate so the task can finish.
            if gate_factory.barrier_release is not None:
                gate_factory.barrier_release.set()
        else:
            # All children are inside the provider at the same time: the evidence
            # that N tasks execute concurrently rather than being queued.
            gate_factory.barrier_release.set()
        results = await gate_task
        problems = _verify_task_results(calls, results, gate_n)
        _check(
            gate_checks,
            "all_n_children_run_simultaneously",
            "pass" if reached and not problems else "fail",
            f"{gate_n} children inside the provider at once",
            {
                "children_entered_before_release": gate_factory.children_entered_provider,
                "peak_simultaneous_live_children": gate_factory.peak_live_children,
                "results_verified": not problems,
                "problems": problems[:3],
            },
            "rendezvous evidence: every child is blocked inside its provider call before any is released, so the N tasks are "
            "genuinely in flight together - no admission queue bounds them",
        )
        cases.append(
            {
                "id": f"subagents.simultaneous_gate.n{gate_n}",
                "lane": "subagents",
                "phase": "subagent_admission_gate",
                "clock": CLOCK,
                "runtime_api": "ToolExecutor.execute_tools_async (async batch) with a barrier-released controlled child provider",
                "config": {"n": gate_n, "rendezvous_timeout_s": args.subagent_gate_timeout_s, "child_agent_ctor": CHILD_AGENT_CTOR},
                "warmups": 0,
                "sample_count": 1,
                "sample_policy": {
                    "full_repetitions_requested": args.repetitions,
                    "reduced": True,
                    "reason": "single deterministic rendezvous probe (repeat count carries no information)",
                },
                "samples_ms": [],
                "summary": _summarize([]),
                "aux_samples": {},
                "aux_summary": {},
                "observations": {
                    "children_entered_before_release": gate_factory.children_entered_provider,
                    "peak_simultaneous_live_children": gate_factory.peak_live_children,
                    "barrier_reached": reached,
                    "released_by_probe": released,
                },
                "checks": gate_checks,
                "known_failures": _failures_from_checks(gate_checks, source="script"),
            }
        )
    finally:
        if gate_factory.barrier_release is not None:
            gate_factory.barrier_release.set()

    # --- cross-size fan-out scaling (per-child cost + observed overlap) ---
    wall_by_n = {int(c["config"]["n"]): c["summary"]["median"] for c in cases if c["phase"] == "subagent_admission"}
    peak_by_n = {
        int(c["config"]["n"]): c["observations"]["peak_simultaneous_live_children"]["median"]
        for c in cases
        if c["phase"] == "subagent_admission"
    }
    scaling: dict[str, Any] = {}
    if wall_by_n:
        smallest = min(wall_by_n)
        unit = wall_by_n[smallest] / smallest if smallest else None
        for n in sorted(wall_by_n):
            scaling[str(n)] = {
                "wall_ms_median": wall_by_n[n],
                "peak_simultaneous_live_children": peak_by_n.get(n),
                "wall_ms_per_child": (wall_by_n[n] / n) if n else None,
                "wall_ms_if_fully_serial": (unit * n) if unit is not None else None,
                "observed_overlap_vs_serial": ((unit * n) / wall_by_n[n]) if wall_by_n[n] else None,
            }

    lane_meta = {
        "runtime_api": "TaskResource.task (real) + real STARAgent children (controlled provider) driven through the real ToolExecutor",
        "child_runtime": "production default runtime/codec for STARAgent (NativeToolsCodec → CodecRuntimeWithNativeToolUse)",
        "admission_scaling": scaling,
        "admission_reading": (
            "peak simultaneous live children equals N, so the Task path has NO shared admission cap. Wall time still grows "
            "roughly linearly with N because each child performs substantial synchronous setup (prompt construction, native-tool "
            "registry build, timeline save) around a single provider await; concurrency only overlaps the provider wait, not the "
            "child-side in-loop work."
        ),
        "coverage_gaps": [
            "real provider/network latency is not exercised (deterministic in-process provider; the wire, not the agent, is substituted)",
            "background execution (run_in_background) and max_turns are declared no-ops in TaskResource.task and are not exercised",
            "resume (multi-turn child continuity) is not exercised; only distinct-session spawning is verified",
            "isolated-worker tool execution inside children is not exercised (children use the JSON protocol with no tools)",
            "CPU/memory/fd growth across N children is not measured (no psutil in this environment; /proc is absent on macOS)",
        ],
    }
    return cases, lane_meta


# ---------------------------------------------------------------------------
# History lane - real SQLite journal + projector + AgentSession prepare
# ---------------------------------------------------------------------------


def _payload_text(prefix: str, index: int, size: int) -> str:
    """Deterministic word-shaped payload of exactly ``size`` characters.

    Word-shaped (space-separated) on purpose: ``Timeline._estimate_tokens`` counts
    ``len(content.split())``, and a single unbroken run of characters would make
    the token estimate collapse to ~1 token per message. Word shape makes both the
    token estimate and the character count meaningful, and exercises the timeline's
    token-limited sliding window at large history sizes.
    """
    head = f"{prefix}{index:05d}"
    words = [head]
    word_index = 0
    while len(" ".join(words)) < size:
        words.append(f"w{index % 97:02d}{word_index % 997:03d}")
        word_index += 1
    text = " ".join(words)
    return text[:size]


def _turn_facts(turn: int, message_bytes: int) -> list[NewJournalFact]:
    """The four facts AgentSession journals for one completed text turn."""
    correlation_id = f"turn-{turn:06d}"
    user_text = _payload_text("u", turn, message_bytes)
    assistant_text = _payload_text("a", turn, message_bytes)
    return [
        NewJournalFact(
            fact_type=FactType.TURN_STARTED,
            correlation_id=correlation_id,
            causation_id=None,
            payload={"prompt_summary": user_text[:200]},
        ),
        NewJournalFact(
            fact_type=FactType.USER_CONTENT_FINAL,
            correlation_id=correlation_id,
            causation_id=correlation_id,
            payload={"text": user_text},
        ),
        NewJournalFact(
            fact_type=FactType.ASSISTANT_CONTENT_FINAL,
            correlation_id=correlation_id,
            causation_id=correlation_id,
            payload={"text": assistant_text},
        ),
        NewJournalFact(
            fact_type=FactType.TURN_COMPLETED,
            correlation_id=correlation_id,
            causation_id=correlation_id,
            payload={},
        ),
    ]


async def _seed_session(
    repo: SQLiteJournalRepository,
    scope: OwnerScope,
    session_id: str,
    turns: int,
    message_bytes: int,
) -> dict[str, Any]:
    """Create the session + seed N completed turns (mirrors tests/unit/core/session/test_agent_session.py:100-140)."""
    record = SessionRecord.new(session_id, scope)
    init_facts = [
        JournalFact(
            fact_id=str(uuid4()),
            owner_scope=scope,
            session_id=session_id,
            sequence=1,
            fact_type=FactType.SESSION_CREATED,
            timestamp=datetime.now(UTC),
            correlation_id=str(uuid4()),
            causation_id=None,
            schema_version=SEED_SCHEMA_VERSION,
            payload={},
        )
    ]
    t0 = time.perf_counter()
    await repo.create_session(record, init_facts)
    version = len(init_facts)
    ops = 1
    for start in range(0, turns, SEED_BATCH_TURNS):
        batch: list[NewJournalFact] = []
        for turn in range(start, min(start + SEED_BATCH_TURNS, turns)):
            batch.extend(_turn_facts(turn, message_bytes))
        if not batch:
            continue
        result = await repo.append(scope, session_id, version, batch)
        version = result.new_version
        ops += 1
    seed_ms = (time.perf_counter() - t0) * 1000.0
    return {"turns": turns, "facts_seeded": version, "seed_ms": seed_ms, "append_batches": ops, "final_version": version}


class StubAgent:
    """Minimal agent shape the prepare path needs (no runtime → no tool catalog).

    The history lane isolates the history path: `_build_tool_catalog` returns early
    on a missing `_runtime`, exactly as the researcher report's history probe did.
    """

    def __init__(self, timeline: Any) -> None:
        self._timeline = timeline
        self._runtime = None
        self._resources: list[Any] = []
        self.event_bus = None
        self.agent_id = "bench-history-stub"
        self.agent_type = "bench"
        self._session_id = None

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id


def _plain_timeline() -> Timeline:
    return Timeline(max_context_tokens=CONTEXT_TOKENS)


def _compressed_timeline() -> CompressedTimeline:
    return CompressedTimeline(
        max_context_tokens=CONTEXT_TOKENS,
        max_tokens_until_compression=COMPRESS_TRIGGER_TOKENS,
        compression_enabled=False,  # no LLM/compaction in this lane (declared deviation)
    )


def _message_stats(messages: Sequence[Any], estimator: Callable[[list[Any]], int]) -> dict[str, Any]:
    """Journal-independent model working-set stats for one provider message list.

    ``provider_estimated_tokens`` uses the timeline's OWN estimator
    (``Timeline._estimate_tokens``: word-count heuristic), so the number is
    comparable to the runtime's context budgeting rather than to a BPE tokenizer.
    """
    content_chars = 0
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            content_chars += len(content)
        elif isinstance(content, list):
            content_chars += len(json.dumps(content, default=str))
        else:
            content_chars += len(str(content))
    return {
        "provider_message_count": len(messages),
        "provider_content_chars": content_chars,
        "provider_estimated_tokens": estimator(list(messages)),
    }


def _journal_stats(db_path: Path) -> dict[str, Any]:
    """Read-only inspection of the journal file (separate stdlib connection).

    Best-effort: a locked or partially-checkpointed database must not fail the
    benchmark run, so the read is guarded and the failure is reported inline.
    """
    wal_path = db_path.parent / (db_path.name + "-wal")
    stats: dict[str, Any] = {
        "db_bytes": db_path.stat().st_size if db_path.exists() else None,
        "wal_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
        "inspection_error": None,
    }
    con = None
    try:
        con = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        row = con.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM session_facts").fetchone()
        stats["facts_rows"] = int(row[0])
        stats["payload_bytes"] = int(row[1])
        page_count = con.execute("PRAGMA page_count").fetchone()[0]
        page_size = con.execute("PRAGMA page_size").fetchone()[0]
        stats["page_bytes"] = int(page_count) * int(page_size)
    except Exception as exc:  # noqa: BLE001 - inspection must not break the run
        stats["inspection_error"] = f"{type(exc).__name__}: {exc}"
        stats.setdefault("facts_rows", None)
        stats.setdefault("payload_bytes", None)
    finally:
        if con is not None:
            con.close()
    stats["journal_total_bytes"] = (stats["db_bytes"] or 0) + (stats["wal_bytes"] or 0)
    return stats


async def _run_history_size(
    turns: int,
    args: argparse.Namespace,
    tmp_root: Path,
) -> dict[str, Any]:
    size_dir = tmp_root / f"turns_{turns}"
    size_dir.mkdir(parents=True, exist_ok=True)
    db_path = size_dir / "journal.db"
    session_id = f"bench-{turns}"
    scope = OwnerScope(owner_id="bench", workspace=str(size_dir))

    reps = args.repetitions if turns < args.expensive_turns else min(args.repetitions, args.expensive_samples)
    reduced = None
    if reps != args.repetitions:
        reduced = (
            f"{turns} turns is an expensive case: {reps} measured samples instead of {args.repetitions} "
            "(uncertainty disclosed; p95 suppressed below 20 samples)"
        )

    repo = await SQLiteJournalRepository.open(str(db_path))
    phases: dict[str, dict[str, Any]] = {}
    facts_meta: dict[str, Any] = {}
    observations: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    try:
        facts_meta = await _seed_session(repo, scope, session_id, turns, args.message_bytes)
        projector = ConversationProjector()
        probe_agent_plain = StubAgent(_plain_timeline())
        probe_session = AgentSession(owner_scope=scope, session_id=session_id, repository=repo, agent_factory=lambda: probe_agent_plain)

        samples: dict[str, list[float]] = {
            "read_all": [],
            "read_delta": [],
            "project": [],
            "hydrate_plain": [],
            "hydrate_compressed": [],
            "working_set_plain": [],
            "working_set_compressed": [],
            "prepare": [],
        }
        counts: dict[str, Any] = {}
        prepare_working_set: dict[str, Any] = {}
        delta_rows: list[int] = []
        all_facts: list[JournalFact] = []

        for rep in range(1 + reps):  # first pass is the declared warmup
            t0 = time.perf_counter()
            all_facts = await repo.read_facts(scope, session_id)
            read_all_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            tail = await repo.read_facts(scope, session_id, after_sequence=max(0, 1 + turns * 4 - 4))
            read_delta_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            view = projector.project(all_facts)
            project_ms = (time.perf_counter() - t0) * 1000.0

            plain = StubAgent(_plain_timeline())
            probe_session._agent = plain  # wire the stub directly to isolate the hydrate phase
            t0 = time.perf_counter()
            probe_session._populate_timeline(view)
            hydrate_plain_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            plain_msgs = plain._timeline.to_llm_messages()
            working_set_plain_ms = (time.perf_counter() - t0) * 1000.0

            compressed = StubAgent(_compressed_timeline())
            probe_session._agent = compressed
            t0 = time.perf_counter()
            probe_session._populate_timeline(view)
            hydrate_compressed_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            compressed_msgs = compressed._timeline.to_llm_messages()
            working_set_compressed_ms = (time.perf_counter() - t0) * 1000.0

            prepare_agent = StubAgent(_compressed_timeline())
            prepare_session = AgentSession(
                owner_scope=scope,
                session_id=session_id,
                repository=repo,
                agent_factory=lambda prepare_agent=prepare_agent: prepare_agent,
            )
            t0 = time.perf_counter()
            await prepare_session.load()  # load() -> _prepare_agent() (agent_session.py:409,771)
            prepare_ms = (time.perf_counter() - t0) * 1000.0

            if rep == 0:
                continue
            samples["read_all"].append(read_all_ms)
            samples["read_delta"].append(read_delta_ms)
            samples["project"].append(project_ms)
            samples["hydrate_plain"].append(hydrate_plain_ms)
            samples["hydrate_compressed"].append(hydrate_compressed_ms)
            samples["working_set_plain"].append(working_set_plain_ms)
            samples["working_set_compressed"].append(working_set_compressed_ms)
            samples["prepare"].append(prepare_ms)
            delta_rows.append(len(tail))
            plain_stats = _message_stats(plain_msgs, plain._timeline._estimate_tokens)
            compressed_stats = _message_stats(compressed_msgs, compressed._timeline._estimate_tokens)
            counts = {
                "seeded_facts_read": len(all_facts),
                "projected_messages": len(view.messages),
                "plain_timeline_entries": len(plain._timeline.timeline),
                "compressed_timeline_entries": len(compressed._timeline.timeline),
                "compressed_native_messages": len(compressed._timeline._native_messages),
                "plain_provider_messages": plain_stats["provider_message_count"],
                "plain_provider_content_chars": plain_stats["provider_content_chars"],
                "plain_provider_estimated_tokens": plain_stats["provider_estimated_tokens"],
                "compressed_provider_messages": compressed_stats["provider_message_count"],
                "compressed_provider_content_chars": compressed_stats["provider_content_chars"],
                "compressed_provider_estimated_tokens": compressed_stats["provider_estimated_tokens"],
            }
            prepare_working_set = {
                "prepare_agent_timeline_entries": len(prepare_agent._timeline.timeline),
                "prepare_agent_native_messages": len(prepare_agent._timeline._native_messages),
                "prepare_agent_provider_messages": len(prepare_agent._timeline.to_llm_messages()),
            }
    finally:
        await repo.close()

    stats = _journal_stats(db_path)

    phases["read_all"] = {
        "runtime_api": "SQLiteJournalRepository.read_facts(scope, session_id)",
        "config": {"after_sequence": 0, "facts_expected": facts_meta.get("final_version")},
        "warmups": 1,
        "sample_count": len(samples["read_all"]),
        "samples_ms": samples["read_all"],
        "summary": _summarize(samples["read_all"]),
        "sample_policy": {
            "full_repetitions_requested": args.repetitions,
            "reduced": reduced is not None,
            "reason": reduced or "full repetitions",
        },
    }
    phases["read_delta"] = {
        "runtime_api": "SQLiteJournalRepository.read_facts(scope, session_id, after_sequence=last_turn_start)",
        "config": {"after_sequence": max(0, 1 + turns * 4 - 4), "rows_observed": delta_rows[:3]},
        "warmups": 1,
        "sample_count": len(samples["read_delta"]),
        "samples_ms": samples["read_delta"],
        "summary": _summarize(samples["read_delta"]),
        "sample_policy": {
            "full_reductions_note": "delta read exists in the repository API; this measures the mechanism, not an integrated incremental projector"
        },
    }
    phases["project"] = {
        "runtime_api": "ConversationProjector.project(facts, provider_key=None)",
        "config": {"facts_in": facts_meta.get("final_version")},
        "warmups": 1,
        "sample_count": len(samples["project"]),
        "samples_ms": samples["project"],
        "summary": _summarize(samples["project"]),
        "sample_policy": {"reduced": reduced is not None, "reason": reduced or "full repetitions"},
    }
    phases["hydrate_plain"] = {
        "runtime_api": "AgentSession._populate_timeline(view) on a plain Timeline",
        "config": {"timeline": "dana.core.timeline.timeline.Timeline", "max_context_tokens": CONTEXT_TOKENS},
        "warmups": 1,
        "sample_count": len(samples["hydrate_plain"]),
        "samples_ms": samples["hydrate_plain"],
        "summary": _summarize(samples["hydrate_plain"]),
        "sample_policy": {"reduced": reduced is not None, "reason": reduced or "full repetitions"},
    }
    phases["hydrate_compressed"] = {
        "runtime_api": "AgentSession._populate_timeline(view) on the production CompressedTimeline",
        "config": {"timeline": "dana.core.timeline.compressed_timeline.CompressedTimeline", "compression_enabled": False},
        "warmups": 1,
        "sample_count": len(samples["hydrate_compressed"]),
        "samples_ms": samples["hydrate_compressed"],
        "summary": _summarize(samples["hydrate_compressed"]),
        "sample_policy": {"reduced": reduced is not None, "reason": reduced or "full repetitions"},
    }
    phases["working_set_plain"] = {
        "runtime_api": "Timeline.to_llm_messages() over the hydrated entries",
        "config": {"max_context_tokens": CONTEXT_TOKENS},
        "warmups": 1,
        "sample_count": len(samples["working_set_plain"]),
        "samples_ms": samples["working_set_plain"],
        "summary": _summarize(samples["working_set_plain"]),
        "sample_policy": {"reduced": reduced is not None, "reason": reduced or "full repetitions"},
    }
    phases["working_set_compressed"] = {
        "runtime_api": "CompressedTimeline.to_llm_messages() (reads _native_messages)",
        "config": {"compression_enabled": False},
        "warmups": 1,
        "sample_count": len(samples["working_set_compressed"]),
        "samples_ms": samples["working_set_compressed"],
        "summary": _summarize(samples["working_set_compressed"]),
        "sample_policy": {"reduced": reduced is not None, "reason": reduced or "full repetitions"},
    }
    phases["prepare"] = {
        "runtime_api": "AgentSession.load() → AgentSession._prepare_agent() (fresh session per sample)",
        "config": {
            "timeline": "CompressedTimeline (production default _build_timeline config shape)",
            "tool_catalog": "skipped (stub agent has no _runtime) - history path isolated",
            "mcp": "disabled via DANA_CODE_MCP_ENABLED=0",
        },
        "warmups": 1,
        "sample_count": len(samples["prepare"]),
        "samples_ms": samples["prepare"],
        "summary": _summarize(samples["prepare"]),
        "sample_policy": {
            "full_repetitions_requested": args.repetitions,
            "reduced": reduced is not None,
            "reason": reduced or "full repetitions",
        },
        "note": "includes read_all + project + hydrate + timeline wiring; the sub-phase rows above are separate invocations and must not be summed to this value",
    }

    # --- correctness observations ---
    expected_facts = 1 + turns * 4
    _check(
        checks,
        "prepare_reads_all_seeded_facts",
        "pass" if counts.get("seeded_facts_read") == expected_facts else "fail",
        expected_facts,
        counts.get("seeded_facts_read"),
        "full journal read on the ordinary prepare path (B0 behavior: every turn rereads everything)",
    )
    _check(
        checks,
        "projection_message_count",
        "pass" if counts.get("projected_messages") == turns * 2 else "fail",
        turns * 2,
        counts.get("projected_messages"),
        "one user + one assistant message per completed turn",
    )
    plain_messages = counts.get("plain_provider_messages", 0)
    plain_bound = counts.get("plain_provider_estimated_tokens", 0) > CONTEXT_TOKENS
    _check(
        checks,
        "plain_working_set_within_context_budget",
        "pass" if plain_messages <= turns * 2 else "fail",
        f"<= {turns * 2} messages from the hydrated entries",
        plain_messages,
        "plain Timeline.to_llm_messages() over the hydrated entries (sliding window may trim)",
    )
    _check(
        checks,
        "plain_window_is_token_bound_not_entry_bound",
        "observation",
        f"unbounded would be {turns * 2} messages",
        {
            "provider_messages": plain_messages,
            "provider_estimated_tokens": counts.get("plain_provider_estimated_tokens"),
            "context_budget_tokens": CONTEXT_TOKENS,
            "token_window_bound": plain_bound,
            "unbounded_estimated_tokens": turns * 2 * max(1, int(args.message_bytes / 6)) * 1.3,
        },
        "reported, not asserted: the sliding window is a token-budget effect, not a journal-size effect",
    )
    compressed_messages = counts.get("compressed_provider_messages", 0)
    _check(
        checks,
        "compressed_timeline_provider_messages_present",
        "fail" if compressed_messages == 0 and turns > 0 else "pass",
        f"> 0 (expected {turns * 2} after hydrate)",
        compressed_messages,
        "B0 hydration gap: _populate_timeline mutates CompressedTimeline.timeline but not CompressedTimeline._native_messages, "
        "which is what to_llm_messages() reads (researcher report §3.7 / P0). Recorded as a known failure, not a script bug.",
    )
    _check(
        checks,
        "delta_read_returns_tail_only",
        "pass" if all(r <= 4 for r in delta_rows) and delta_rows else "fail",
        "<= 4 rows",
        delta_rows[:3],
        "read_facts(after_sequence=...) exists and returns only the tail - foundation for incremental projection",
    )
    _check(
        checks,
        "prepare_agent_working_set_compressed",
        "observation",
        f"{turns * 2} provider messages after prepare",
        prepare_working_set,
        "same gap observed through the real prepare path (prepare agent is a production-shaped CompressedTimeline)",
    )

    observations.update(
        {
            "message_bytes_declared": args.message_bytes,
            "turns": turns,
            "facts_per_turn": 4,
            "phase_sum_median_ms": round(
                sum(_summarize(samples[k])["median"] or 0.0 for k in ("read_all", "project", "hydrate_compressed")), 3
            ),
            "prepare_median_ms": _summarize(samples["prepare"])["median"],
            "journal_bytes_is_not_model_working_set": {
                "journal_total_bytes": stats["journal_total_bytes"],
                "journal_payload_bytes": stats["payload_bytes"],
                "model_working_set_plain_chars": counts.get("plain_provider_content_chars"),
                "model_working_set_plain_messages": counts.get("plain_provider_messages"),
                "model_working_set_plain_estimated_tokens": counts.get("plain_provider_estimated_tokens"),
                "model_working_set_compressed_messages_after_prepare": prepare_working_set.get("prepare_agent_provider_messages"),
            },
        }
    )

    return {
        "id": f"history.turns_{turns}",
        "lane": "history",
        "clock": CLOCK,
        "config": {
            "turns": turns,
            "message_bytes": args.message_bytes,
            "facts_per_turn": 4,
            "timeline_context_tokens": CONTEXT_TOKENS,
            "compression_enabled": False,
            "seed_strategy": f"{SEED_BATCH_TURNS} turns per append transaction",
        },
        "warmups": 1,
        "sample_count": len(samples["prepare"]),
        "sample_policy": {
            "full_repetitions_requested": args.repetitions,
            "reduced": reduced is not None,
            "reason": reduced or "full repetitions",
        },
        "samples_ms": samples["prepare"],
        "summary": _summarize(samples["prepare"]),
        "phases": phases,
        "counts": counts,
        "seed": facts_meta,
        "journal": stats,
        "prepare_working_set": prepare_working_set,
        "observations": observations,
        "checks": checks,
        "known_failures": _failures_from_checks(checks, source="script"),
    }


# ---------------------------------------------------------------------------
# Environment / provenance / main
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    head = ROOT / ".git" / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value.startswith("ref:"):
        ref = ROOT / ".git" / value.split(" ", 1)[1].strip()
        try:
            return ref.read_text(encoding="utf-8").strip()
        except OSError:
            return value
    return value


def _env_flags() -> dict[str, Any]:
    flags: dict[str, Any] = {}
    for name in _REPORTABLE_FLAG_VALUES:
        flags[name] = os.environ.get(name)
    for name in _ENV_PRESENCE_ONLY:
        flags[name] = "<set>" if os.environ.get(name) is not None else None
    return flags


def _environment() -> dict[str, Any]:
    import importlib.metadata as importlib_metadata

    try:
        dana_version = importlib_metadata.version("dana-agent")
    except Exception:  # noqa: BLE001 - provenance best effort
        dana_version = None
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "dana_agent_version": dana_version,
        "dana_import_path": str(Path(__file__).resolve().parent.parent / "dana"),
        "repo_head": _git_head(),
        "uv_lock_sha256": _file_sha256(ROOT / "uv.lock"),
        "env_flags": _env_flags(),
    }


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    lanes = {lane.strip() for lane in args.lanes.split(",") if lane.strip()}
    unknown = lanes - {"tools", "subagents", "history"}
    if unknown or not lanes:
        raise SystemExit(f"--lanes must be a non-empty subset of tools,subagents,history (got: {args.lanes!r}; unknown: {sorted(unknown)})")
    tmp_root = (
        Path(tempfile.mkdtemp(prefix="dana-bench-tools-history-"))
        if args.temp_root is None
        else Path(args.temp_root).expanduser().resolve() / f"dana-bench-{os.getpid()}"
    )
    tmp_root.mkdir(parents=True, exist_ok=True)

    threads_before = _thread_names()
    tool_cases: list[dict[str, Any]] = []
    history_cases: list[dict[str, Any]] = []
    subagent_cases: list[dict[str, Any]] = []
    lane_meta: dict[str, Any] = {}
    cleanup: dict[str, Any] = {"temp_root": str(tmp_root), "keep_temp": bool(args.keep_temp)}

    try:
        if "tools" in lanes:
            tool_cases, lane_meta["tools"] = await _run_tools_lane(args, _parse_int_list(args.tool_n))
            cleanup["orphan_tasks_after_tools"] = _pending_tasks()
        if "subagents" in lanes:
            subagent_cases, lane_meta["subagents"] = await _run_subagent_lane(args, _parse_int_list(args.subagent_n), tmp_root)
            cleanup["orphan_tasks_after_subagents"] = _pending_tasks()
        if "history" in lanes:
            for turns in _parse_int_list(args.sizes):
                history_cases.append(await _run_history_size(turns, args, tmp_root))
            lane_meta["history"] = {
                "runtime_api": "SQLiteJournalRepository + ConversationProjector + AgentSession.load/_prepare_agent + Timeline.to_llm_messages",
                "coverage_gaps": [
                    "PostgreSQL journal (dana/core/session/journal/postgres.py) not exercised",
                    "Compaction/cheap-shrink path not exercised (compression_enabled=False, no LLM)",
                    "Real STARAgent prompt construction (tool schemas, system prompt, message conversion) is out of scope for this lane",
                    "Tool-catalog cost inside _prepare_agent is excluded (stub agent has no _runtime)",
                    "Cross-session concurrency and RSS/fd growth are not measured",
                ],
            }
    finally:
        cleanup["orphan_tasks_after_all"] = _pending_tasks()
        if args.keep_temp:
            cleanup["removed"] = False
            cleanup["exists_after"] = tmp_root.exists()
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)
            cleanup["removed"] = True
            cleanup["exists_after"] = tmp_root.exists()
    cleanup["threads_before_count"] = len(threads_before)
    cleanup["threads_after"] = _thread_names()
    cleanup["thread_count_delta"] = len(cleanup["threads_after"]) - len(threads_before)

    all_checks = [c for case in tool_cases + history_cases + subagent_cases for c in case.get("checks", [])]
    status_counts = {"pass": 0, "fail": 0, "observation": 0}
    for c in all_checks:
        status_counts[c["status"]] = status_counts.get(c["status"], 0) + 1

    return {
        "schema": SCHEMA,
        "generated_at": _now_iso(),
        "clock": CLOCK,
        "reproduction": {
            "argv": sys.argv,
            "command": f"uv run --no-sync python scripts/benchmark_tools_history.py --output {args.output} --repetitions {args.repetitions}",
            "paid_requests": False,
            "network": "none (no provider, no MCP, no geolocation; synthetic fixtures only)",
            "writes": ["--output path", "--temp-root/OS temp dir (removed unless --keep-temp)"],
        },
        "config": {
            "repetitions": args.repetitions,
            "expensive_samples": args.expensive_samples,
            "expensive_turns": args.expensive_turns,
            "expensive_tool_n": args.expensive_tool_n,
            "tool_n": _parse_int_list(args.tool_n),
            "tool_sleep_ms": args.tool_sleep_ms,
            "heartbeat_ms": args.heartbeat_ms,
            "sizes": _parse_int_list(args.sizes),
            "message_bytes": args.message_bytes,
            "conflict_n": args.conflict_n,
            "cancel_after_ms": args.cancel_after_ms,
            "subagent_n": _parse_int_list(args.subagent_n),
            "expensive_subagent_n": args.expensive_subagent_n,
            "subagent_gate_timeout_s": args.subagent_gate_timeout_s,
            "lanes": sorted(lanes),
            "declared_deviations": [
                "DANA_CODE_MCP_ENABLED=0 (no MCP server configured for the benchmark)",
                "LANGFUSE_ENABLED=false, LANGSMITH_TRACING=false (tracing off)",
                "history lane: CompressedTimeline(compression_enabled=False) - no compaction/LLM call",
                "history lane: tool-catalog build excluded via a stub agent without _runtime",
                "tools lane: no production agent/LLM is involved; the executor is driven directly",
                "subagents lane: child STARAgents use a deterministic in-process provider (no network/credentials); the agent, STAR loop, Task tool and repositories are real",
            ],
        },
        "environment": _environment(),
        "lanes": {
            "tools": {"cases": tool_cases, **lane_meta.get("tools", {})},
            "subagents": {"cases": subagent_cases, **lane_meta.get("subagents", {})},
            "history": {"cases": history_cases, **lane_meta.get("history", {})},
        },
        "checks_summary": status_counts,
        "known_failures": [f for case in tool_cases + history_cases + subagent_cases for f in case.get("known_failures", [])],
        "cleanup": cleanup,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tool execution + journal/prepare history benchmark (JSON output, raw monotonic samples).")
    parser.add_argument("--output", required=True, help="Path to write the JSON result document.")
    parser.add_argument("--repetitions", type=int, default=100, help="Measured repetitions for cheap local cases (default: 100).")
    parser.add_argument("--expensive-samples", type=int, default=5, help="Measured samples for expensive cases (default: 5).")
    parser.add_argument("--expensive-turns", type=int, default=1000, help="History sizes >= this use --expensive-samples (default: 1000).")
    parser.add_argument("--expensive-tool-n", type=int, default=8, help="Tool batch sizes >= this use --expensive-samples (default: 8).")
    parser.add_argument("--tool-n", default="1,3,8,32", help="Comma-separated independent-tool batch sizes.")
    parser.add_argument("--tool-sleep-ms", type=float, default=150.0, help="Per-call synthetic I/O delay in ms (default: 150).")
    parser.add_argument("--heartbeat-ms", type=float, default=10.0, help="Event-loop heartbeat interval in ms (default: 10).")
    parser.add_argument("--sizes", default="10,100,1000,5000", help="Comma-separated prior-turn counts for the history lane.")
    parser.add_argument("--message-bytes", type=int, default=512, help="Synthetic user/assistant message size in bytes.")
    parser.add_argument("--conflict-n", type=int, default=8, help="Conflicting-mutation batch size.")
    parser.add_argument("--cancel-after-ms", type=float, default=30.0, help="Delay before cancelling the async tool batch.")
    parser.add_argument("--subagent-n", default="1,4,16", help="Comma-separated nested-subagent counts for the subagents lane.")
    parser.add_argument(
        "--expensive-subagent-n",
        type=int,
        default=8,
        help="Subagent counts >= this use --expensive-samples (default: 8; each sample spawns n real child agents).",
    )
    parser.add_argument(
        "--subagent-gate-timeout-s",
        type=float,
        default=20.0,
        help="Rendezvous timeout for the simultaneous-child admission gate (default: 20s).",
    )
    parser.add_argument("--lanes", default="tools,history", help="Comma-separated lanes to run (tools,subagents,history).")
    parser.add_argument("--temp-root", default=None, help="Parent directory for temporary state (default: OS temp dir).")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary directories for inspection.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    payload = asyncio.run(_run(args))
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    summary = {
        "output": str(output),
        "tool_cases": len(payload["lanes"]["tools"]["cases"]),
        "subagent_cases": len(payload["lanes"]["subagents"]["cases"]),
        "history_cases": len(payload["lanes"]["history"]["cases"]),
        "checks_summary": payload["checks_summary"],
        "known_failures": len(payload["known_failures"]),
        "cleanup": payload["cleanup"],
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
