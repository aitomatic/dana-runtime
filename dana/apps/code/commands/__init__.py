"""Slash-command handlers for dana-code (D7.3, AC #3/#6).

Extracted from ``DanaCodeApp._handle_command`` so command logic is independently
testable and ``code_app.py`` keeps only a thin dispatch seam. Each handler takes
the app (for state access) and returns a user-facing message string; the caller
decides continue/exit. Async handlers (``reset_session``, ``switch_model``,
``list_permissions_async``) are coroutines; the rest are sync.
"""

from __future__ import annotations

import os
from typing import Any


HELP_TEXT = """Commands:
  /help          Show this help
  /compact       Toggle verbose output
  /status        Show session, model, and mode info
  /model         List configured models, or switch: /model provider/model
  /permissions   List active durable permission grants
  /reset         Start a fresh session (clears in-memory history)
  /exit          Exit
"""


def _model_switching_enabled() -> bool:
    """DANA_CODE_MODEL_SWITCH_ENABLED (default on)."""
    from dana.config.code_capabilities import model_switch_enabled

    return model_switch_enabled()


def _model_catalog() -> Any:
    """Build the model catalog from env (parity with dana-acp)."""
    import json

    from dana.core.model.catalog import ModelCatalog, ModelTarget

    raw = os.environ.get("DANA_MODEL_CATALOG")
    if raw:
        targets = [ModelTarget(**t) for t in json.loads(raw)]
    else:
        targets = [ModelTarget(provider="anthropic", model="claude-sonnet-4")]
    return ModelCatalog(targets)


def compact_toggle(app: Any) -> str:
    assert app.renderer is not None
    app.renderer.verbose = not app.renderer.verbose
    mode = "verbose" if app.renderer.verbose else "compact"
    return f"\nOutput mode: {mode}\n"


def status_lines(app: Any) -> str:
    """Format /status for whichever path is active."""
    out = ["\n"]
    if app.agent_session is not None:
        s = app.agent_session
        out.append(f"Session: {s.session_id}")
        out.append(f"Provider: {s.current_provider or os.environ.get('DANA_LLM_PROVIDER', 'unknown')}")
        out.append(f"Model: {s.current_model or os.environ.get('DANA_MODEL', 'unknown')}")
        out.append(f"Permission mode: {s.permission_mode.value}")
        out.append(f"Journal version: {s.version}")
    elif app.agent is not None:
        state = app.agent.get_state()
        out.append(f"Agent: {state.get('object_id', 'unknown')}")
        out.append(f"Provider: {app.agent._llm_config.get('provider', 'unknown')}")
        out.append(f"Model: {app.agent._llm_config.get('model', 'unknown')}")
        out.append(f"Timeline entries: {state.get('timeline_entries', 0)}")
    out.append("")
    return "\n".join(out)


async def list_permissions_async(app: Any) -> str:
    """/permissions: list active durable grants (async — awaited by the REPL)."""
    if app.agent_session is None:
        return "\n/permissions is available on the AgentSession path only.\n"
    store = getattr(app, "_grant_store", None)
    if store is None:
        return "\nNo permission grant store wired (preflight disabled).\n"
    grants = await store.list_grants(app.agent_session.owner_scope)
    if not grants:
        return "\nNo active durable grants.\n"
    lines = ["\nActive durable grants:"]
    for g in grants:
        lines.append(f"  {g.decision.value:7} {g.tool_identity} ({g.effect_kind.value}) {g.location or '*'}")
    lines.append("")
    return "\n".join(lines)


async def reset_session(app: Any) -> str:
    """/reset on the AgentSession path: start a fresh session (journal semantics).

    The old journal repository is closed and a new session id is minted on the
    same repository path; prior turns remain durable in the journal under their
    old session id (ADR-002 — journal is the sole durable authority).
    """
    if app.agent_session is None:
        # Legacy path: clear in-memory timeline.
        app.agent._timeline.timeline.clear()
        return "\nConversation history reset.\n"
    await app._close_repo()
    await app._initialize_session()
    return f"\nFresh session started: {app.agent_session.session_id}\n"


async def switch_model(app: Any, arg: str) -> str:
    """/model: list targets, or switch to provider/model (ADR-007 atomic + busy-reject)."""
    if app.agent_session is None:
        return "\n/model is available on the AgentSession path only.\n"
    if not _model_switching_enabled():
        return "\nModel switching is disabled (DANA_CODE_MODEL_SWITCH_ENABLED=0).\n"

    catalog = _model_catalog()
    s = app.agent_session

    # No arg → list configured targets + current.
    target_id = arg[len("model ") :].strip() if arg.startswith("model ") else ""
    if not target_id:
        lines = [f"\nCurrent: {s.current_provider or '?'}/{s.current_model or '?'}"]
        lines.append("Available:")
        for t in catalog.targets:
            lines.append(f"  {t.provider}/{t.model}")
        lines.append("Switch with: /model provider/model\n")
        return "\n".join(lines)

    # Busy check: switching during an active turn is rejected (ADR-007).
    if s._lock.locked():
        return "\n⏳ Cannot switch model — a turn is in progress. Wait for it to finish.\n"

    if "/" not in target_id:
        return f"\nInvalid model '{target_id}' (expected 'provider/model').\n"
    provider, model = target_id.split("/", 1)
    target = catalog.get(provider, model)
    if target is None:
        return f"\nUnknown model target: {target_id!r}\n"

    from dana.core.model.switching import ModelSwitcher

    # Mirror ACP's _build_provider_client/_build_model_runtime stubs (D4 parity).
    # Real provider construction is deferred; the stub carries target identity so
    # rebind_model updates session provider/model atomically (ADR-007).
    def _build_provider(t: Any) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(provider=t.provider, model=t.model, config=t.config or {})

    def _build_runtime(t: Any, p: Any) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(provider=t.provider, model=t.model)

    switcher = ModelSwitcher(
        build_provider=_build_provider,
        build_runtime=_build_runtime,
        apply_switch=lambda t, p, r: s.rebind_model(t, p, r),
    )
    result = switcher.switch(target)
    if not result.success:
        return f"\nModel switch failed: {result.error}\n"
    return f"\nSwitched to {s.current_provider}/{s.current_model}\n"
