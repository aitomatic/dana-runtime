"""D7.3 — Capability Inheritance & Slash Commands tests (AC #1–#6).

No live LLM. Covers:
- CLIPermissionAdapter decision surface (AC #2): allow / always (persist) /
  deny (hard) / deny-once (user) / unrecognized (fail-closed) / durable grant.
- Slash-command handlers (AC #3, #6): /status, /model (no-arg + busy-reject +
  switch), /permissions (empty + with grants), /reset, /compact, /help.
- Capability rollback flags (AC #1, #4, #5, #6): flag-off paths.
- AgentSession accessors (shared infra): set_policy_evaluator + public reads.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dana.apps.code import commands as cmds
from dana.apps.code.permissions import CLIPermissionAdapter
from dana.config.code_capabilities import ALL_FLAGS
from dana.core.policy.evaluator import PolicyDecision, PolicyResult
from dana.core.policy.grants import GrantDecision, PolicyGrant
from dana.core.policy.scope import OwnerScope
from dana.core.session.agent_session import AgentSession


SCOPE = OwnerScope(owner_id="tester", workspace="/ws")


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _FakeEvaluator:
    """Fake PolicyEvaluator returning a canned decision."""

    def __init__(self, decision: PolicyDecision, reason: str = "") -> None:
        self._decision = decision
        self._reason = reason
        from unittest.mock import Mock

        self.set_mode = Mock()  # type: ignore[assignment]

    async def evaluate(self, op, scope):
        return PolicyResult(decision=self._decision, reason=self._reason)


class _FakeGrantStore:
    """Records created grants; returns a canned list."""

    def __init__(self, grants=None) -> None:
        self.created: list[PolicyGrant] = []
        self._grants = grants or []

    async def create_grant(self, grant):
        self.created.append(grant)
        return grant

    async def list_grants(self, scope):
        return list(self._grants)


def _adapter(
    decision: PolicyDecision, *, reason: str = "", prompt_reply: str | None = None, grants=None
) -> tuple[CLIPermissionAdapter, _FakeGrantStore]:
    store = _FakeGrantStore(grants)
    evaluator = _FakeEvaluator(decision, reason=reason)
    prompt = (lambda _msg: prompt_reply) if prompt_reply is not None else None
    return CLIPermissionAdapter(evaluator, store, SCOPE, prompt=prompt), store


TOOL_CALL = {"function": "write_file", "arguments": {"path": "/tmp/a.txt"}}


# ---------------------------------------------------------------------------
# CLIPermissionAdapter (AC #2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_permission_allow_no_prompt():
    adapter, store = _adapter(PolicyDecision.ALLOW, reason="durable grant")
    verdict = await adapter.request(TOOL_CALL)
    assert verdict.allowed is True
    assert verdict.persisted is False
    assert store.created == []  # no new grant on an existing allow


@pytest.mark.asyncio
async def test_permission_hard_deny():
    adapter, store = _adapter(PolicyDecision.DENY, reason="hard policy")
    verdict = await adapter.request(TOOL_CALL)
    assert verdict.allowed is False
    assert "hard policy" in verdict.reason
    assert store.created == []


@pytest.mark.asyncio
async def test_permission_allow_once_user_choice():
    adapter, store = _adapter(PolicyDecision.NEEDS_PROMPT, prompt_reply="1")
    verdict = await adapter.request(TOOL_CALL)
    assert verdict.allowed is True
    assert verdict.persisted is False
    assert store.created == []  # "once" does not persist


@pytest.mark.asyncio
async def test_permission_deny_once_user_choice():
    adapter, store = _adapter(PolicyDecision.NEEDS_PROMPT, prompt_reply="3")
    verdict = await adapter.request(TOOL_CALL)
    assert verdict.allowed is False
    assert verdict.persisted is False
    assert store.created == []


@pytest.mark.asyncio
async def test_permission_allow_always_persists_durable_grant():
    adapter, store = _adapter(PolicyDecision.NEEDS_PROMPT, prompt_reply="2")
    verdict = await adapter.request(TOOL_CALL)
    assert verdict.allowed is True
    assert verdict.persisted is True
    assert len(store.created) >= 1
    assert all(g.decision is GrantDecision.ALLOW for g in store.created)


@pytest.mark.asyncio
async def test_permission_deny_always_persists_durable_grant():
    adapter, store = _adapter(PolicyDecision.NEEDS_PROMPT, prompt_reply="4")
    verdict = await adapter.request(TOOL_CALL)
    assert verdict.allowed is False
    assert verdict.persisted is True
    assert all(g.decision is GrantDecision.REJECT for g in store.created)


@pytest.mark.asyncio
async def test_permission_unrecognized_choice_fail_closed():
    adapter, store = _adapter(PolicyDecision.NEEDS_PROMPT, prompt_reply="nope")
    verdict = await adapter.request(TOOL_CALL)
    assert verdict.allowed is False
    assert "fail-closed" in verdict.reason
    assert store.created == []


@pytest.mark.asyncio
async def test_permission_default_prompt_eof_denies(monkeypatch):
    # The default prompt returns "3" (deny once) on EOF → fail-closed.
    import builtins

    def boom(_msg):
        raise EOFError

    monkeypatch.setattr(builtins, "input", boom)
    adapter, store = _adapter(PolicyDecision.NEEDS_PROMPT)  # prompt=None → default
    verdict = await adapter.request(TOOL_CALL)
    assert verdict.allowed is False  # "3" → deny once
    assert store.created == []


# ---------------------------------------------------------------------------
# Slash-command handlers (AC #3, #6)
# ---------------------------------------------------------------------------


def _fake_app(*, agent_session=None, grant_store=None):
    return SimpleNamespace(
        agent_session=agent_session,
        agent=None,
        renderer=SimpleNamespace(verbose=True),
        _grant_store=grant_store,
        _close_repo=AsyncMock(),
        _initialize_session=AsyncMock(),
    )


def _fake_session(*, locked=False, provider="openai", model="gpt-5", sid="s-1", version=7):
    ns = SimpleNamespace(
        session_id=sid,
        version=version,
        owner_scope=SCOPE,
        current_provider=provider,
        current_model=model,
        permission_mode=SimpleNamespace(value="default"),
        _lock=SimpleNamespace(locked=lambda: locked),
        append_fact=AsyncMock(),
    )

    def _rebind(t, p, r):
        ns.current_provider = t.provider
        ns.current_model = t.model

    ns.rebind_model = _rebind
    return ns


def test_help_text_lists_all_commands():
    for name in ("/help", "/compact", "/status", "/model", "/permissions", "/reset", "/exit"):
        assert name in cmds.HELP_TEXT


def test_compact_toggle_flips_renderer():
    app = _fake_app()
    out = cmds.compact_toggle(app)
    assert app.renderer.verbose is False
    assert "compact" in out
    out2 = cmds.compact_toggle(app)
    assert app.renderer.verbose is True
    assert "verbose" in out2


def test_status_lines_agentsession_path():
    app = _fake_app(agent_session=_fake_session())
    out = cmds.status_lines(app)
    assert "Session: s-1" in out
    assert "Provider: openai" in out
    assert "Model: gpt-5" in out
    assert "Journal version: 7" in out
    assert "Permission mode: default" in out


@pytest.mark.asyncio
async def test_permissions_empty():
    app = _fake_app(agent_session=_fake_session(), grant_store=_FakeGrantStore())
    out = await cmds.list_permissions_async(app)
    assert "No active durable grants" in out


@pytest.mark.asyncio
async def test_permissions_with_grants():
    grant = PolicyGrant(
        grant_id="g1",
        owner_scope=SCOPE,
        decision=GrantDecision.ALLOW,
        tool_identity="write_file",
        effect_kind=SimpleNamespace(value="write"),
        location="/tmp",
        created_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
    )
    store = _FakeGrantStore(grants=[grant])
    app = _fake_app(agent_session=_fake_session(), grant_store=store)
    out = await cmds.list_permissions_async(app)
    assert "write_file" in out
    assert "allow" in out
    assert "/tmp" in out


@pytest.mark.asyncio
async def test_permissions_no_session():
    app = _fake_app(agent_session=None)
    out = await cmds.list_permissions_async(app)
    assert "AgentSession path only" in out


@pytest.mark.asyncio
async def test_permissions_no_grant_store():
    app = _fake_app(agent_session=_fake_session(), grant_store=None)
    out = await cmds.list_permissions_async(app)
    assert "preflight disabled" in out.lower()


@pytest.mark.asyncio
async def test_reset_agentsession_path():
    app = _fake_app(agent_session=_fake_session())
    # reset rebuilds: closes repo, re-inits, reports new session id
    app._initialize_session = AsyncMock(side_effect=lambda: setattr(app, "agent_session", _fake_session(sid="s-2")))
    out = await cmds.reset_session(app)
    app._close_repo.assert_awaited_once()
    app._initialize_session.assert_awaited_once()
    assert "s-2" in out


@pytest.mark.asyncio
async def test_model_no_arg_lists_targets():
    app = _fake_app(agent_session=_fake_session())
    out = await cmds.switch_model(app, "model")
    assert "Current: openai/gpt-5" in out
    assert "Available:" in out
    # default catalog has at least one target
    assert "/" in out.split("Available:")[1]


@pytest.mark.asyncio
async def test_model_busy_reject():
    app = _fake_app(agent_session=_fake_session(locked=True))
    out = await cmds.switch_model(app, "model anthropic/claude-sonnet-4")
    assert "turn is in progress" in out.lower()


@pytest.mark.asyncio
async def test_model_switch_atomic(monkeypatch):
    monkeypatch.setenv("DANA_MODEL_CATALOG", '[{"provider":"anthropic","model":"claude-sonnet-4"}]')
    s = _fake_session(provider="openai", model="gpt-5")
    app = _fake_app(agent_session=s)
    out = await cmds.switch_model(app, "model anthropic/claude-sonnet-4")
    assert "Switched to anthropic/claude-sonnet-4" in out
    assert s.current_provider == "anthropic"
    assert s.current_model == "claude-sonnet-4"
    # M2: a MODEL_CHANGED fact must be journaled after a successful switch
    # (ADR-002 durability — mirrors ACP session/set_session_model).
    s.append_fact.assert_called_once()
    fact = s.append_fact.call_args.args[0]
    from dana.core.session.models import FactType

    assert fact.fact_type is FactType.MODEL_CHANGED
    assert fact.payload == {"provider": "anthropic", "model": "claude-sonnet-4"}


@pytest.mark.asyncio
async def test_model_switch_failure_no_journal(monkeypatch):
    """A failed switch must NOT journal a MODEL_CHANGED fact."""
    monkeypatch.setenv("DANA_MODEL_CATALOG", '[{"provider":"anthropic","model":"claude-sonnet-4"}]')
    s = _fake_session(provider="openai", model="gpt-5")
    # Sabotage the switch so rebind raises → switcher.switch returns failure.
    s.rebind_model = Mock(side_effect=RuntimeError("boom"))
    app = _fake_app(agent_session=s)
    out = await cmds.switch_model(app, "model anthropic/claude-sonnet-4")
    assert "Model switch failed" in out
    s.append_fact.assert_not_called()


@pytest.mark.asyncio
async def test_model_no_arg_current_env_fallback(monkeypatch):
    """When no provider/model is bound, /model Current falls back to env (parity with /status)."""
    monkeypatch.setenv("DANA_LLM_PROVIDER", "azure")
    monkeypatch.setenv("DANA_MODEL", "gpt-5.4")
    monkeypatch.setenv("DANA_MODEL_CATALOG", '[{"provider":"anthropic","model":"claude-sonnet-4"}]')
    s = _fake_session(provider=None, model=None)
    app = _fake_app(agent_session=s)
    out = await cmds.switch_model(app, "model")
    assert "Current: azure/gpt-5.4" in out


@pytest.mark.asyncio
async def test_model_switch_invalid_no_slash():
    app = _fake_app(agent_session=_fake_session())
    out = await cmds.switch_model(app, "model bogus")
    assert "Invalid model" in out


@pytest.mark.asyncio
async def test_model_switch_unknown_target(monkeypatch):
    monkeypatch.setenv("DANA_MODEL_CATALOG", '[{"provider":"anthropic","model":"claude-sonnet-4"}]')
    app = _fake_app(agent_session=_fake_session())
    out = await cmds.switch_model(app, "model openai/gpt-9")
    assert "Unknown model target" in out


# ---------------------------------------------------------------------------
# Capability rollback flags (AC #1, #4, #5, #6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_switch_disabled_flag(monkeypatch):
    monkeypatch.setenv("DANA_CODE_MODEL_SWITCH_ENABLED", "0")
    app = _fake_app(agent_session=_fake_session())
    out = await cmds.switch_model(app, "model anthropic/claude-sonnet-4")
    assert "disabled" in out.lower()


def test_all_flags_default_on(monkeypatch):
    from dana.config import code_capabilities as caps

    for name in ALL_FLAGS:
        monkeypatch.delenv(name, raising=False)
    assert caps.permission_preflight_enabled() is True
    assert caps.model_switch_enabled() is True
    assert caps.tool_catalog_enabled() is True
    assert caps.mcp_enabled() is True
    assert caps.multimodal_enabled() is True


def test_each_flag_off_disables(monkeypatch):
    from dana.config import code_capabilities as caps

    monkeypatch.setenv("DANA_CODE_TOOL_CATALOG_ENABLED", "0")
    assert caps.tool_catalog_enabled() is False
    monkeypatch.setenv("DANA_CODE_MCP_ENABLED", "0")
    assert caps.mcp_enabled() is False
    monkeypatch.setenv("DANA_CODE_MULTIMODAL_ENABLED", "0")
    assert caps.multimodal_enabled() is False


# ---------------------------------------------------------------------------
# AgentSession accessors (shared infra — fixes ACP too)
# ---------------------------------------------------------------------------


class _NullRepo:
    """Minimal repo stub — no methods called during construction."""

    pass


def test_agentsession_accessors_and_set_policy_evaluator():
    s = AgentSession(owner_scope=SCOPE, session_id="abc", repository=_NullRepo())
    assert s.session_id == "abc"
    assert s.owner_scope is SCOPE
    assert s.version == 0  # no facts yet
    assert s.policy_evaluator is None

    evaluator = SimpleNamespace(set_mode=Mock())
    s.set_policy_evaluator(evaluator)
    assert s.policy_evaluator is evaluator
    evaluator.set_mode.assert_called_once()


# ---------------------------------------------------------------------------
# D6: Multimodal input parsing (AC #5) — _build_prompt_blocks
# ---------------------------------------------------------------------------


def _make_app(*, provider=None):
    """Build a DanaCodeApp with a fake AgentSession for _build_prompt_blocks tests."""
    from dana.apps.code.code_app import DanaCodeApp

    app = DanaCodeApp()
    app.agent_session = _fake_session(provider=provider)
    app.renderer = SimpleNamespace(verbose=True)
    return app


def test_prompt_blocks_text_only_no_at(tmp_path, monkeypatch):
    """Plain text with no @-token → single TextBlock, content_blocks=None."""
    monkeypatch.delenv("DANA_CODE_MULTIMODAL_ENABLED", raising=False)
    app = _make_app()
    blocks, content = app._build_prompt_blocks("hello world")
    assert len(blocks) == 1
    assert blocks[0].text == "hello world"
    assert content is None


def test_prompt_blocks_flag_off_disables_parsing(tmp_path, monkeypatch):
    """DANA_CODE_MULTIMODAL_ENABLED=0 → even a real @path is left as text."""
    img = tmp_path / "pic.png"
    img.write_bytes(b"fake-png")
    monkeypatch.setenv("DANA_CODE_MULTIMODAL_ENABLED", "0")
    app = _make_app()
    blocks, content = app._build_prompt_blocks(f"look @{img}")
    assert content is None
    assert blocks[0].text == f"look @{img}"


def test_prompt_blocks_image_attachment(tmp_path, monkeypatch):
    """A real @path to a png → image content block in the payload."""
    monkeypatch.setenv("DANA_CODE_MULTIMODAL_ENABLED", "1")
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n fake")
    app = _make_app(provider=None)  # fresh session → no capability check
    blocks, content = app._build_prompt_blocks(f"see this @{img}")
    assert content is not None
    assert any(b.get("type") == "image" for b in content)
    img_block = next(b for b in content if b.get("type") == "image")
    assert "png" in img_block["media_type"]
    # bytes are base64-encoded in the payload (journal-serializable)
    assert isinstance(img_block["data"], str)
    assert len(blocks) == 1
    assert "see this" in blocks[0].text


def test_prompt_blocks_file_resource_attachment(tmp_path, monkeypatch):
    """A non-image @path → file_resource block (not image)."""
    monkeypatch.setenv("DANA_CODE_MULTIMODAL_ENABLED", "1")
    doc = tmp_path / "notes.txt"
    doc.write_text("hello")
    app = _make_app(provider=None)
    blocks, content = app._build_prompt_blocks(f"read @{doc}")
    assert content is not None
    assert any(b.get("type") == "file_resource" for b in content)
    fr = next(b for b in content if b.get("type") == "file_resource")
    assert fr["uri"] == str(doc)


def test_prompt_blocks_nonexistent_at_stays_text(tmp_path, monkeypatch):
    """An @token that is not a real file stays literal (no false positive)."""
    monkeypatch.setenv("DANA_CODE_MULTIMODAL_ENABLED", "1")
    app = _make_app(provider=None)
    blocks, content = app._build_prompt_blocks("email me@test.com and @/no/such/file")
    assert content is None
    assert "me@test.com" in blocks[0].text
    assert "@/no/such/file" in blocks[0].text


def test_prompt_blocks_provider_capability_rejects_unsupported(tmp_path, monkeypatch):
    """ADR-009: an unsupported provider + image attachment raises before the turn."""
    monkeypatch.setenv("DANA_CODE_MULTIMODAL_ENABLED", "1")
    img = tmp_path / "pic.png"
    img.write_bytes(b"fake")
    app = _make_app(provider="unsupported-co")
    from dana.core.content.validation import ProviderCapabilityError

    with pytest.raises(ProviderCapabilityError):
        app._build_prompt_blocks(f"@{img}")


def test_prompt_blocks_provider_capability_allows_supported(tmp_path, monkeypatch):
    """A supported provider (e.g. openai) + image attachment passes."""
    monkeypatch.setenv("DANA_CODE_MULTIMODAL_ENABLED", "1")
    img = tmp_path / "pic.png"
    img.write_bytes(b"fake")
    app = _make_app(provider="openai")
    blocks, content = app._build_prompt_blocks(f"@{img}")
    assert content is not None
    assert any(b.get("type") == "image" for b in content)


def test_normalized_blocks_to_text_blocks_shared_helper():
    """The shared content helper (moved from ACP) round-trips text+image."""
    from dana.core.content.blocks import normalized_blocks_to_text_blocks

    blocks = [
        {"type": "text", "text": "hi"},
        {"type": "image", "media_type": "image/png", "data": b"\x89PNG"},
    ]
    text_blocks, payload = normalized_blocks_to_text_blocks(blocks)
    assert len(text_blocks) == 1
    assert "hi" in text_blocks[0].text
    assert "[Image: image/png]" in text_blocks[0].text
    assert payload[0]["type"] == "text"
    assert payload[1]["type"] == "image"
    assert isinstance(payload[1]["data"], str)  # base64-encoded
