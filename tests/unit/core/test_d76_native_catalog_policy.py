"""D7.6 — native-tool catalog for policy classification + TOOL_CALL hard-deny hook.

Tests the three D7.6 pieces:
  1. build_native_tool_catalog classifies the agent's native tools (normal ->
     non-sensitive; unknown -> sensitive, fail-cautious).
  2. build_policy_operation(catalog=) + create_default_hard_policy classify
     correctly (normal proceeds; destructive-on-protected-path blocked;
     unknown blocked).
  3. AgentSession._on_tool_call (the TOOL_CALL EventBus hook) returns
     {"block": True} ONLY on PolicyDecision.DENY; ALLOW/NEEDS_PROMPT proceed;
     None-catalog pass-through (no P0 regression).
  4. Per-turn pinned catalog version (AC #1).
"""

from __future__ import annotations

import pytest

from dana.core.policy.effects import EffectKind
from dana.core.policy.evaluator import PolicyEvaluator
from dana.core.policy.hard_policy import create_default_hard_policy
from dana.core.policy.modes import PermissionMode
from dana.core.policy.operations import build_policy_operation
from dana.core.policy.scope import OwnerScope
from dana.core.session.agent_session import AgentSession
from dana.core.session.models import OwnerScope as SessionOwnerScope
from dana.core.tool.native_catalog import (
    build_native_tool_catalog,
)


# ---------------------------------------------------------------------------
# Synthetic native-tool schemas (the shape runtime._native_tools produces)
# ---------------------------------------------------------------------------
def _native_schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": f"tool {name}", "parameters": {"type": "object", "properties": {}}},
    }


KNOWN_TOOLS = [
    "Read",
    "Grep",
    "Glob",
    "read_tool_result",
    "bash__get_task_output",
    "bash__list_tasks",
    "TaskOutput",
    "todo__todo_write",
    "Edit",
    "Write",
    "Task",
    "bash__execute",
    "bash__kill_task",
    "Skill",
    "call_mcp_tool",
]


class _FakeRepo:
    """Minimal async fake journal repository (no aiosqlite needed)."""

    async def create_session(self, *a, **k):
        return None

    async def read_facts(self, *a, **k):
        return []

    async def append(self, *a, **k):
        from types import SimpleNamespace

        return SimpleNamespace(new_version=0, appended_facts=[])


class _GrantStoreMock:
    """Empty grant store (no grants) -> falls through to mode/prompt."""

    async def find_matching_grants(self, scope, operation):
        from types import SimpleNamespace

        return SimpleNamespace(matched=None)


# ---------------------------------------------------------------------------
# Piece 1 — catalog classification
# ---------------------------------------------------------------------------
class TestNativeCatalogClassification:
    def test_all_known_native_tools_classified(self):
        catalog = build_native_tool_catalog([_native_schema(n) for n in KNOWN_TOOLS])
        for name in KNOWN_TOOLS:
            assert catalog.get(name) is not None, f"missing entry for {name}"

    def test_normal_tools_are_non_sensitive(self):
        catalog = build_native_tool_catalog([_native_schema(n) for n in KNOWN_TOOLS])
        for name in ["Read", "Grep", "Glob", "read_tool_result", "todo__todo_write"]:
            entry = catalog.get(name)
            assert entry.effects.is_sensitive is False, f"{name} should be non-sensitive"

    def test_file_mutating_and_shell_tools_non_sensitive_but_effectful(self):
        catalog = build_native_tool_catalog([_native_schema(n) for n in KNOWN_TOOLS])
        # Non-sensitive (flow to mode/grant/prompt), but carry real effect kinds
        # so the hard policy can still catch destructive-on-protected-path.
        assert catalog.get("Edit").effects.is_sensitive is False
        assert any(e.kind is EffectKind.MODIFY for e in catalog.get("Edit").effects.effects)
        assert catalog.get("Write").effects.is_sensitive is False
        assert any(e.kind is EffectKind.CREATE for e in catalog.get("Write").effects.effects)
        assert any(e.kind is EffectKind.EXECUTE for e in catalog.get("bash__execute").effects.effects)

    def test_unknown_tool_is_sensitive_fail_cautious(self):
        catalog = build_native_tool_catalog([_native_schema("mystery_tool")])
        entry = catalog.get("mystery_tool")
        assert entry.effects.is_sensitive is True, "unknown tool must be fail-cautious sensitive"

    def test_version_pinned(self):
        catalog = build_native_tool_catalog([_native_schema("Read")], version=7)
        assert catalog.version == 7


# ---------------------------------------------------------------------------
# Piece 1/2 — policy classification via build_policy_operation + hard policy
# ---------------------------------------------------------------------------
class TestPolicyClassification:
    def setup_method(self):
        self.catalog = build_native_tool_catalog([_native_schema(n) for n in KNOWN_TOOLS])
        self.evaluator = PolicyEvaluator(
            create_default_hard_policy(),
            grant_store=_GrantStoreMock(),
            mode=PermissionMode.DEFAULT,
        )
        self.scope = OwnerScope(owner_id="user", workspace="/ws")

    def _op(self, fn: str, args: dict | None = None):
        return build_policy_operation(
            {"function": fn, "arguments": args or {}},
            catalog=self.catalog,
            owner="user",
            workspace="/ws",
        )

    @pytest.mark.asyncio
    async def test_normal_read_tool_not_hard_denied(self):
        from dana.core.policy.evaluator import PolicyDecision

        result = await self.evaluator.evaluate(self._op("Read"), self.scope)
        assert result.decision is not PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_unknown_tool_hard_denied(self):
        from dana.core.policy.evaluator import PolicyDecision

        result = await self.evaluator.evaluate(self._op("mystery_tool"), self.scope)
        assert result.decision is PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_edit_on_protected_path_hard_denied(self):
        from dana.core.policy.evaluator import PolicyDecision

        result = await self.evaluator.evaluate(self._op("Edit", {"path": ".env"}), self.scope)
        assert result.decision is PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_edit_on_normal_path_not_hard_denied(self):
        from dana.core.policy.evaluator import PolicyDecision

        result = await self.evaluator.evaluate(self._op("Edit", {"path": "src/main.py"}), self.scope)
        assert result.decision is not PolicyDecision.DENY


# ---------------------------------------------------------------------------
# Piece 2 — AgentSession._on_tool_call hook (the live hard-deny gate)
# ---------------------------------------------------------------------------
class _PolicyEvalMock:
    """Mock PolicyEvaluator: returns a configured decision."""

    def __init__(self, decision):
        self._decision = decision
        self.set_mode = lambda mode: None

    async def evaluate(self, op, scope):
        from types import SimpleNamespace

        return SimpleNamespace(decision=self._decision, reason="mock")


def _make_session_with_catalog(catalog, decision=None, evaluator=None):
    """Build a real AgentSession and inject catalog + (optional) evaluator."""

    repo = _FakeRepo()
    scope = SessionOwnerScope(owner_id="user", workspace="/ws")
    session = AgentSession(owner_scope=scope, session_id="s1", repository=repo)
    session._tool_catalog = catalog
    if evaluator is not None:
        session._policy_evaluator = evaluator
    return session


def _tool_call_event(fn: str, args: dict | None = None):
    from types import MappingProxyType

    from dana.core.ext.event_bus import Event
    from dana.core.ext.operation import Operation, ToolIdentity

    op = Operation(tool_identity=ToolIdentity(name=fn, source="native"), arguments=MappingProxyType(dict(args or {})))
    return Event("tool_call", {"tool_call_id": "tc1", "operation": op})


class TestToolCallPolicyHook:
    @pytest.mark.asyncio
    async def test_deny_blocks(self):
        from dana.core.policy.evaluator import PolicyDecision

        catalog = build_native_tool_catalog([_native_schema(n) for n in KNOWN_TOOLS])
        session = _make_session_with_catalog(catalog, evaluator=_PolicyEvalMock(PolicyDecision.DENY))
        out = await session._on_tool_call(_tool_call_event("mystery_tool"))
        assert out == {"block": True, "reason": "denied: mock"}

    @pytest.mark.asyncio
    async def test_allow_proceeds(self):
        from dana.core.policy.evaluator import PolicyDecision

        catalog = build_native_tool_catalog([_native_schema(n) for n in KNOWN_TOOLS])
        session = _make_session_with_catalog(catalog, evaluator=_PolicyEvalMock(PolicyDecision.ALLOW))
        out = await session._on_tool_call(_tool_call_event("Read"))
        assert out is None

    @pytest.mark.asyncio
    async def test_needs_prompt_proceeds(self):
        from dana.core.policy.evaluator import PolicyDecision

        catalog = build_native_tool_catalog([_native_schema(n) for n in KNOWN_TOOLS])
        session = _make_session_with_catalog(catalog, evaluator=_PolicyEvalMock(PolicyDecision.NEEDS_PROMPT))
        out = await session._on_tool_call(_tool_call_event("Edit"))
        assert out is None  # NEEDS_PROMPT -> proceed (interactive prompt is a host follow-up)

    @pytest.mark.asyncio
    async def test_needs_prompt_callback_allow_proceeds(self):
        # D7.6 follow-up: a host prompt callback that allows -> proceed.
        from types import SimpleNamespace

        from dana.core.policy.evaluator import PolicyDecision

        catalog = build_native_tool_catalog([_native_schema(n) for n in KNOWN_TOOLS])
        session = _make_session_with_catalog(catalog, evaluator=_PolicyEvalMock(PolicyDecision.NEEDS_PROMPT))
        called = []

        async def _allow(op):
            called.append(op.tool_identity.name)
            return SimpleNamespace(allowed=True, reason="allowed once (user)")

        session.set_permission_prompt_callback(_allow)
        out = await session._on_tool_call(_tool_call_event("Edit"))
        assert out is None  # allowed -> proceed
        assert called == ["Edit"]  # callback invoked with the operation

    @pytest.mark.asyncio
    async def test_needs_prompt_callback_deny_blocks(self):
        # D7.6 follow-up: a host prompt callback that denies -> block.
        from types import SimpleNamespace

        from dana.core.policy.evaluator import PolicyDecision

        catalog = build_native_tool_catalog([_native_schema(n) for n in KNOWN_TOOLS])
        session = _make_session_with_catalog(catalog, evaluator=_PolicyEvalMock(PolicyDecision.NEEDS_PROMPT))

        async def _deny(op):
            return SimpleNamespace(allowed=False, reason="denied once (user)")

        session.set_permission_prompt_callback(_deny)
        out = await session._on_tool_call(_tool_call_event("Edit"))
        assert out == {"block": True, "reason": "denied: denied once (user)"}

    @pytest.mark.asyncio
    async def test_no_catalog_pass_through(self):
        # preflight-on + catalog-off (None) must NOT deny everything (P0 guard).
        from dana.core.policy.evaluator import PolicyDecision

        session = _make_session_with_catalog(None, evaluator=_PolicyEvalMock(PolicyDecision.DENY))
        out = await session._on_tool_call(_tool_call_event("Read"))
        assert out is None  # no catalog -> cannot classify -> pass-through

    @pytest.mark.asyncio
    async def test_no_evaluator_pass_through(self):
        catalog = build_native_tool_catalog([_native_schema(n) for n in KNOWN_TOOLS])
        session = _make_session_with_catalog(catalog, evaluator=None)
        out = await session._on_tool_call(_tool_call_event("Read"))
        assert out is None


# ---------------------------------------------------------------------------
# Piece 3 — per-turn version pin
# ---------------------------------------------------------------------------
class TestPerTurnVersion:
    @pytest.mark.asyncio
    async def test_version_increments_each_build(self):
        # _build_tool_catalog is gated on tool_catalog_enabled() (default on).
        session = _make_session_with_catalog(None)

        # Provide a fake agent with a runtime + native_tools (read-only schemas).
        from types import SimpleNamespace

        session._agent = SimpleNamespace(_runtime=SimpleNamespace(_native_tools=[_native_schema("Read"), _native_schema("Edit")]))
        await session._build_tool_catalog()
        v1 = session._catalog_version
        assert v1 >= 1 and session.tool_catalog.version == v1
        await session._build_tool_catalog()
        assert session._catalog_version == v1 + 1
        assert session.tool_catalog.version == v1 + 1

    @pytest.mark.asyncio
    async def test_no_native_tools_no_catalog(self):
        session = _make_session_with_catalog(None)
        from types import SimpleNamespace

        session._agent = SimpleNamespace(_runtime=SimpleNamespace(_native_tools=None))
        await session._build_tool_catalog()
        assert session.tool_catalog is None
