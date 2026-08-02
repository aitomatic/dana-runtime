"""S3 Tool Execution Engine — scenarios from sprint/plans/S3-tool-execution-engine.md.

Each test maps 1:1 to a row in the plan's given/when/then table (S3.1..S3.15).
Slice 3.1 covers Operation/ToolIdentity construction + derivation (S3.1).
Later slices append tests for wiring (3.2), policy (3.3), demo (3.4).

Async cases use asyncio.run to avoid a pytest-asyncio plugin dependency.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

from dana.core.ext.event_bus import Event, EventBus
from dana.core.ext.events import TOOL_CALL, TOOL_RESULT
from dana.core.ext.guard import install_guard
from dana.core.ext.operation import Operation, ToolIdentity, build_operation
from dana.core.ext.permission import PermissionPolicy
from dana.core.tool.tool_executor import ToolExecutor


# ---------------------------------------------------------------------------
# Shared fixtures for wiring/policy/demo slices
# ---------------------------------------------------------------------------


class _RecordingTool:
    """A real @named_tool-style object: ``run(q=...)`` records calls + returns."""

    resource_id = "rec-tool"

    def __init__(self, retval: Any = "ok") -> None:
        self.retval = retval
        self.calls: list[dict] = []

    def run(self, q: str = "default") -> str:
        self.calls.append({"q": q})
        return self.retval


class _AgentWithBus:
    """Minimal agent exposing a real ``event_bus`` (no other attrs needed for
    the registry fast path, which never touches ``agent`` beyond the bus)."""

    def __init__(self) -> None:
        self.event_bus = EventBus()


def _build(retval: Any = "ok") -> tuple[_AgentWithBus, ToolExecutor, _RecordingTool]:
    tool = _RecordingTool(retval=retval)
    registry = {"tool": (tool, "run")}
    executor = ToolExecutor(tool_name_registry_getter=lambda: registry)
    agent = _AgentWithBus()
    return agent, executor, tool


def _call(tool_call_id: str = "tc1", **arguments) -> dict:
    return {"function": "tool", "arguments": dict(arguments), "tool_call_id": tool_call_id}


# ---------------------------------------------------------------------------
# Slice 3.1 — Operation / ToolIdentity construction + derivation (S3.1)
# ---------------------------------------------------------------------------


def test_s31_build_operation_from_registry_hit():
    """S3.1: registry hit → source = resource_id, arguments copied."""
    resource = MagicMock()
    resource.resource_id = "web-search"
    registry = {"web_search": (resource, "search")}

    tool_call = {"function": "web_search", "arguments": {"q": "dana"}, "tool_call_id": "tc1"}
    op = build_operation(tool_call, registry)

    assert op.tool_identity.name == "web_search"
    assert op.tool_identity.source == "web-search"
    assert op.arguments == {"q": "dana"}
    # arguments is a copy — mutating the Operation must not affect the call dict
    assert op.arguments is not tool_call["arguments"]


def test_s31_build_operation_no_registry_hit():
    """S3.1: registry miss → source = None (generic ClassName:method fallback)."""
    tool_call = {"function": "MyClass:my_method", "arguments": {"x": 1}}
    op = build_operation(tool_call, registry={})

    assert op.tool_identity.name == "MyClass:my_method"
    assert op.tool_identity.source is None
    assert op.arguments == {"x": 1}


def test_s31_build_operation_source_falls_back_to_class_name():
    """No resource_id/object_id on the object → source = class name."""
    obj = MagicMock(spec=[])  # no resource_id / object_id attrs
    registry = {"tool": (obj, "run")}
    op = build_operation({"function": "tool", "arguments": {}}, registry)
    assert op.tool_identity.source == type(obj).__name__


def test_s31_build_operation_missing_fields():
    """Missing function/arguments → name='', arguments={}, no crash."""
    op = build_operation({}, registry={})
    assert op.tool_identity.name == ""
    assert op.tool_identity.source is None
    assert op.arguments == {}


def test_s31_operation_is_frozen():
    """Operation/ToolIdentity are immutable (reassignment + item mutation)."""
    op = Operation(ToolIdentity(name="t", source=None), {"a": 1})
    with pytest.raises(FrozenInstanceError):
        op.arguments = {}  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        op.tool_identity.name = "x"  # type: ignore[misc]
    # [Fix 6] arguments is a read-only mapping — item mutation raises TypeError
    with pytest.raises(TypeError):
        op.arguments["a"] = 2  # type: ignore[index]
    # reads still work
    assert op.arguments["a"] == 1


# ---------------------------------------------------------------------------
# Slice 3.2 — wiring: emit/block/modify/parity/never-raise/parallel (S3.2..S3.9)
# ---------------------------------------------------------------------------


def test_s32_no_bus_is_noop():
    """S3.2: agent without a real event_bus → dispatch normal, 0 emit, 0 crash."""
    tool = _RecordingTool(retval="done")
    registry = {"tool": (tool, "run")}
    executor = ToolExecutor(tool_name_registry_getter=lambda: registry)
    agent = MagicMock(spec=[])  # getattr(agent, "event_bus", None) -> None

    result = executor._execute_single_call(agent, _call(q="keep"))

    assert result["success"] is True
    assert result["result"] == "done"
    assert tool.calls == [{"q": "keep"}]


def test_s33_tool_call_handler_logs_async():
    """S3.3: a tool_call handler observes operation + tool_call_id (async path)."""
    agent, executor, tool = _build()
    seen: list[dict] = []

    def _log(event: Event):
        seen.append({"id": event.payload["tool_call_id"], "name": event.payload["operation"].tool_identity.name})
        return None

    agent.event_bus.subscribe(TOOL_CALL, _log)
    result = asyncio.run(executor._execute_single_call_async(agent, _call(q="x")))

    assert result["success"] is True
    assert seen == [{"id": "tc1", "name": "tool"}]


def test_s34_block_handler_prevents_dispatch():
    """S3.4: tool_call handler {block:True,reason} → policy_block error, tool NOT run."""
    agent, executor, tool = _build()
    agent.event_bus.subscribe(TOOL_CALL, lambda _e: {"block": True, "reason": "denied"})

    result = executor._execute_single_call(agent, _call(q="x"))

    assert result["success"] is False
    assert result["type"] == "policy_block"
    assert "denied" in result["result"]
    assert tool.calls == []  # dispatch never happened


def test_s35_modify_arguments_handler():
    """S3.5: tool_call handler {modify:{arguments:...}} → tool runs with new args."""
    agent, executor, tool = _build()
    agent.event_bus.subscribe(TOOL_CALL, lambda _e: {"modify": {"arguments": {"q": "patched"}}})

    result = executor._execute_single_call(agent, _call(q="original"))

    assert result["success"] is True
    assert tool.calls == [{"q": "patched"}]


def test_s36_modify_result_handler():
    """S3.6: tool_result handler {modify:new_result} → result replaced."""
    agent, executor, tool = _build(retval="raw")
    agent.event_bus.subscribe(
        TOOL_RESULT, lambda _e: {"modify": {"type": "resource", "target": "tool", "result": "patched", "success": True}}
    )

    result = executor._execute_single_call(agent, _call())

    assert result["result"] == "patched"
    assert result["success"] is True


def test_s37_handler_raise_is_degraded_not_crash():
    """S3.7: tool_call handler raises → bus catches, tool runs with original args, returns result."""
    agent, executor, tool = _build(retval="ok")

    def _boom(_event: Event):
        raise RuntimeError("boom")

    agent.event_bus.subscribe(TOOL_CALL, _boom)

    result = executor._execute_single_call(agent, _call(q="keep"))

    assert result["success"] is True
    assert result["result"] == "ok"
    assert tool.calls == [{"q": "keep"}]  # ran with original args


def test_s38_parallel_block_isolates_sibling():
    """S3.8: blocking one tool in a concurrent batch does NOT block siblings."""

    class _Two:
        def __init__(self) -> None:
            self.a = _RecordingTool(retval="A")
            self.b = _RecordingTool(retval="B")

    duo = _Two()
    registry = {"tool_a": (duo.a, "run"), "tool_b": (duo.b, "run")}
    executor = ToolExecutor(tool_name_registry_getter=lambda: registry)
    agent = _AgentWithBus()
    # Block only tool_a
    agent.event_bus.subscribe(
        TOOL_CALL,
        lambda e: {"block": True, "reason": "no-a"} if e.payload["operation"].tool_identity.name == "tool_a" else None,
    )

    calls = [
        {"function": "tool_a", "arguments": {}, "tool_call_id": "a"},
        {"function": "tool_b", "arguments": {}, "tool_call_id": "b"},
    ]
    results = asyncio.run(executor.execute_tools_async(agent, calls))

    by_id = {r["tool_call_id"]: r for r in results}
    assert by_id["a"]["success"] is False and by_id["a"]["type"] == "policy_block"
    assert by_id["b"]["success"] is True and by_id["b"]["result"] == "B"
    assert duo.a.calls == [] and duo.b.calls == [{"q": "default"}]


def test_s39_sync_async_block_parity():
    """S3.9: same block setup yields the same policy_block via sync and async."""

    def fresh():
        tool = _RecordingTool()
        registry = {"tool": (tool, "run")}
        executor = ToolExecutor(tool_name_registry_getter=lambda: registry)
        agent = _AgentWithBus()
        agent.event_bus.subscribe(TOOL_CALL, lambda _e: {"block": True, "reason": "parity"})
        return executor, agent

    sync_exec, sync_agent = fresh()
    async_exec, async_agent = fresh()

    sync_res = sync_exec._execute_single_call(sync_agent, _call())
    async_res = asyncio.run(async_exec._execute_single_call_async(async_agent, _call()))

    assert sync_res["type"] == "policy_block" and async_res["type"] == "policy_block"
    assert sync_res["success"] is False and async_res["success"] is False
    assert sync_res["result"] == async_res["result"]


# ---------------------------------------------------------------------------
# Slice 3.3 — PermissionPolicy deny-only (S3.10..S3.12)
# ---------------------------------------------------------------------------


class _KWTool:
    """Tool capturing **kwargs (used by policy tests: bash/write/ls shapes)."""

    def __init__(self, resource_id: str = "kw", retval: str = "ran") -> None:
        self.resource_id = resource_id
        self.retval = retval
        self.calls: list[dict] = []

    def run(self, **kwargs) -> str:
        self.calls.append(dict(kwargs))
        return self.retval


def _policy_with_common_rules() -> PermissionPolicy:
    """The two demo deny rules from the plan: rm -rf + protected path.

    Mirrors the shipped ``guard.py``; ``str(...)`` coercion defeats list-typed
    args bypassing the substring check (see test_fix2_guard_blocks_list_command).
    """
    policy = PermissionPolicy()
    policy.deny(
        lambda op: "rm -rf blocked" if op.tool_identity.name == "bash_tool" and "rm -rf" in str(op.arguments.get("command", "")) else None
    )
    policy.deny(
        lambda op: "protected path"
        if op.tool_identity.name in ("write", "edit") and str(op.arguments.get("path", "")) in {".env", "node_modules"}
        else None
    )
    return policy


def test_s310_policy_blocks_rm_rf():
    """S3.10: PermissionPolicy deny rm -rf → bash_tool 'rm -rf /tmp' blocked."""
    tool = _KWTool("bash")
    executor = ToolExecutor(tool_name_registry_getter=lambda: {"bash_tool": (tool, "run")})
    agent = _AgentWithBus()
    agent.event_bus.subscribe(TOOL_CALL, _policy_with_common_rules().on_tool_call)

    result = executor._execute_single_call(
        agent, {"function": "bash_tool", "arguments": {"command": "rm -rf /tmp/x"}, "tool_call_id": "b1"}
    )

    assert result["success"] is False
    assert result["type"] == "policy_block"
    assert result["result"].endswith("rm -rf blocked")
    assert tool.calls == []


def test_s311_policy_blocks_protected_path():
    """S3.11: PermissionPolicy deny .env → write path '.env' blocked."""
    tool = _KWTool("write")
    executor = ToolExecutor(tool_name_registry_getter=lambda: {"write": (tool, "run")})
    agent = _AgentWithBus()
    agent.event_bus.subscribe(TOOL_CALL, _policy_with_common_rules().on_tool_call)

    result = executor._execute_single_call(agent, {"function": "write", "arguments": {"path": ".env"}, "tool_call_id": "w1"})

    assert result["success"] is False
    assert result["type"] == "policy_block"
    assert result["result"].endswith("protected path")
    assert tool.calls == []


def test_s312_policy_allows_when_no_rule_matches():
    """S3.12: PermissionPolicy with rules but no match → ls passes through, tool runs."""
    tool = _KWTool("ls")
    executor = ToolExecutor(tool_name_registry_getter=lambda: {"ls": (tool, "run")})
    agent = _AgentWithBus()
    agent.event_bus.subscribe(TOOL_CALL, _policy_with_common_rules().on_tool_call)

    result = executor._execute_single_call(agent, {"function": "ls", "arguments": {"path": "."}, "tool_call_id": "l1"})

    assert result["success"] is True
    assert result["result"] == "ran"
    assert tool.calls == [{"path": "."}]


def test_s313_policy_first_matching_rule_wins():
    """Unit check: deny rules evaluate in order; first non-None reason wins."""
    policy = PermissionPolicy()
    policy.deny(lambda op: None)  # allow
    policy.deny(lambda op: "second")  # matches
    policy.deny(lambda op: "third")  # would also match but unreachable

    op = Operation(ToolIdentity(name="t"), {})
    assert policy.check(op) == "second"
    assert policy.on_tool_call(Event(TOOL_CALL, {"operation": op})) == {"block": True, "reason": "second"}


# ---------------------------------------------------------------------------
# Slice 3.4 — demo guard + scaffold deprecation (S3.13..S3.15)
# ---------------------------------------------------------------------------


def test_s313_guard_e2e_two_blocks_one_pass():
    """S3.13: install_guard → batch [rm -rf, write .env, ls] → 2 blocked, ls passes."""
    bash = _KWTool("bash")
    write = _KWTool("write")
    ls = _KWTool("ls")
    registry = {
        "bash_tool": (bash, "run"),
        "write": (write, "run"),
        "ls": (ls, "run"),
    }
    executor = ToolExecutor(tool_name_registry_getter=lambda: registry)
    agent = _AgentWithBus()
    install_guard(agent.event_bus)

    calls = [
        {"function": "bash_tool", "arguments": {"command": "rm -rf /tmp/x"}, "tool_call_id": "rm"},
        {"function": "write", "arguments": {"path": ".env"}, "tool_call_id": "env"},
        {"function": "ls", "arguments": {"path": "."}, "tool_call_id": "ls"},
    ]
    results = executor.execute_tools(agent, calls)
    by_id = {r["tool_call_id"]: r for r in results}

    assert by_id["rm"]["type"] == "policy_block" and by_id["rm"]["success"] is False
    assert by_id["env"]["type"] == "policy_block" and by_id["env"]["success"] is False
    assert by_id["ls"]["success"] is True and by_id["ls"]["result"] == "ran"
    assert bash.calls == [] and write.calls == []
    assert ls.calls == [{"path": "."}]


def test_s314_scaffold_removed():
    """S3.14: constructor has no hooks/approval; attrs gone; Protocols unimportable."""
    params = inspect.signature(ToolExecutor.__init__).parameters
    assert "hooks" not in params
    assert "approval" not in params

    executor = ToolExecutor()
    assert not hasattr(executor, "_hooks")
    assert not hasattr(executor, "_approval")

    import dana.core.runtime as runtime

    for name in ("ToolHookProtocol", "ApprovalProtocol"):
        assert not hasattr(runtime, name), f"{name} should be removed from runtime"
        with pytest.raises(ImportError):
            from dana.core.runtime.protocols import (  # noqa: F401
                ToolHookProtocol,
            )


def test_s315_full_core_suite_is_the_regression_gate():
    """S3.15: regression is the full tests/unit/core/ run (no inline re-run).

    This test exists to map the plan's S3.15 row to a test id; the actual gate
    is `uv run pytest tests/unit/core/` (see CI / Makefile). We sanity-check the
    parallel executor still wires correctly through the new emit path.
    """
    tool = _RecordingTool(retval="ok")
    executor = ToolExecutor(tool_name_registry_getter=lambda: {"tool": (tool, "run")})
    agent = MagicMock(spec=[])  # no bus → no-op emit path (regression-equivalent)

    results = executor.execute_tools(agent, [_call(q="x")], parallel=True)
    assert results[0]["success"] is True
    assert tool.calls == [{"q": "x"}]


# ---------------------------------------------------------------------------
# Adversarial fix-regression tests (Stage 3 accepted findings)
# ---------------------------------------------------------------------------


def test_fix1_non_dict_modify_does_not_crash_batch():
    """[Fix 1] A tool_result handler returning a NON-DICT modify must NOT crash
    the batch wrapper (``result['tool_call_id'] = ...``) nor corrupt the result.
    Never-raise is preserved; the malformed modify is ignored."""
    tool = _RecordingTool(retval="real")
    executor = ToolExecutor(tool_name_registry_getter=lambda: {"tool": (tool, "run")})
    agent = _AgentWithBus()
    agent.event_bus.subscribe(TOOL_RESULT, lambda _e: {"modify": "GARBAGE-STRING"})

    # Sync batch
    calls = [{"function": "tool", "arguments": {}, "tool_call_id": "t1"}]
    results = executor.execute_tools(agent, calls)
    assert results[0]["success"] is True
    assert results[0]["result"] == "real"  # original result kept
    assert results[0]["tool_call_id"] == "t1"  # propagation still works

    # Async batch
    agent2 = _AgentWithBus()
    agent2.event_bus.subscribe(TOOL_RESULT, lambda _e: {"modify": 12345})
    async_results = asyncio.run(executor.execute_tools_async(agent2, calls))
    assert async_results[0]["success"] is True
    assert async_results[0]["result"] == "real"


def test_fix2_guard_blocks_list_typed_command():
    """[Fix 2] The shipped guard must block rm -rf even when ``command`` arrives
    as a list (membership-vs-substring bypass). Uses the real guard.py policy."""
    from dana.core.ext.guard import create_guard_policy

    tool = _KWTool("bash")
    executor = ToolExecutor(tool_name_registry_getter=lambda: {"bash_tool": (tool, "run")})
    agent = _AgentWithBus()
    agent.event_bus.subscribe(TOOL_CALL, create_guard_policy().on_tool_call)

    result = executor._execute_single_call(agent, {"function": "bash_tool", "arguments": {"command": ["rm -rf /tmp"]}, "tool_call_id": "b"})

    assert result["success"] is False
    assert result["type"] == "policy_block"
    assert tool.calls == []


# ---------------------------------------------------------------------------
# Adversarial fix-regression: deferred Lows [3] sync-parallel emit, [5] strict bool
# ---------------------------------------------------------------------------


def test_fix3_sync_parallel_emits_per_call():
    """[Fix 3] sync parallel=True (ThreadPoolExecutor) routes through the bus:
    the tool_call handler fires once per tool, results are correct, no crash.
    Covers the concurrency path untested by S3.8 (async gather only)."""
    tools = {n: (_KWTool(n, retval=n), "run") for n in ("a", "b", "c")}
    executor = ToolExecutor(tool_name_registry_getter=lambda: tools)
    agent = _AgentWithBus()
    seen: list[str] = []
    agent.event_bus.subscribe(TOOL_CALL, lambda e: seen.append(e.payload["operation"].tool_identity.name) or None)

    calls = [{"function": n, "arguments": {}, "tool_call_id": n} for n in ("a", "b", "c")]
    results = executor.execute_tools(agent, calls, parallel=True)

    assert {r["tool_call_id"]: r["result"] for r in results} == {"a": "a", "b": "b", "c": "c"}
    assert sorted(seen) == ["a", "b", "c"]  # handler invoked once per tool (concurrent reads, no mutation)


@pytest.mark.parametrize("blocky", ["yes", 1, "false", ["True"]])
def test_fix5_non_bool_block_does_not_block(blocky):
    """[Fix 5] Only literal ``True`` blocks. Truthy non-bool (e.g. ``"yes"``,
    ``"false"``) must pass through — truthiness would wrongly block on ``"false"``."""
    tool = _RecordingTool(retval="ran")
    executor = ToolExecutor(tool_name_registry_getter=lambda: {"tool": (tool, "run")})
    agent = _AgentWithBus()
    agent.event_bus.subscribe(TOOL_CALL, lambda _e: {"block": blocky, "reason": "x"})

    result = executor._execute_single_call(agent, _call())

    assert result["success"] is True
    assert result["result"] == "ran"
    assert tool.calls == [{"q": "default"}]
