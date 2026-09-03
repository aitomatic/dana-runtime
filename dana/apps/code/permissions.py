"""CLI permission prompt adapter (D7.3, AC #2).

The in-process analog of ACP ``session/request_permission``: instead of
returning ``PermissionOption``s to a remote host, this adapter prompts the
terminal user directly. It evaluates an operation through the shared
:class:`~dana.core.policy.evaluator.PolicyEvaluator` (ADR-006 precedence:
hard deny → durable grant → permission mode → interactive prompt →
fail-closed) and persists durable grants on "always" decisions.

The adapter owns only the *decision surface*; grant precedence and
``affected_locations`` matching belong to the PolicyEvaluator / GrantStore.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from dana.core.policy.evaluator import PolicyDecision
from dana.core.policy.grants import GrantDecision, PolicyGrant
from dana.core.policy.operations import build_policy_operation
from dana.core.policy.scope import OwnerScope


@dataclass(frozen=True)
class PermissionVerdict:
    """Outcome of a CLI permission request."""

    allowed: bool
    reason: str
    persisted: bool = False  # True if a durable grant was created


class _PromptFn(Protocol):
    """Callable that presents a prompt and returns the user's choice string."""

    def __call__(self, prompt: str) -> str: ...


def _default_prompt(prompt: str) -> str:
    """Interactive terminal prompt reading from stdin."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return "3"  # fail-closed → deny once


class CLIPermissionAdapter:
    """Interactive CLI permission-decision adapter (ADR-006).

    Mirrors the contract of ``DanaACPAgent.request_permission`` but resolves
    the decision locally by prompting the user. Hard-deny and durable-grant
    precedence are enforced by the evaluator before any prompt is shown.
    """

    def __init__(
        self,
        evaluator: Any,
        grant_store: Any,
        scope: OwnerScope,
        prompt: _PromptFn | None = None,
        catalog_getter: Any = None,
    ) -> None:
        self._evaluator = evaluator
        self._grant_store = grant_store
        self._scope = scope
        self._prompt = prompt or _default_prompt
        # D7.6: optional catalog getter (lambda -> ToolCatalog | None) so this
        # host adapter classifies tools the same way as the live TOOL_CALL hook.
        # None (default) -> no catalog -> unknown/sensitive (fail-cautious).
        self._catalog_getter = catalog_getter

    async def request(self, tool_call: dict[str, Any]) -> PermissionVerdict:
        """Evaluate a tool call through the policy and return a verdict.

        ``tool_call`` mirrors the ACP shape: ``{"function": <name>, "arguments": {...}}``.
        """
        op = build_policy_operation(
            tool_call,
            catalog=(self._catalog_getter() if self._catalog_getter is not None else None),
            owner=self._scope.owner_id,
            workspace=self._scope.workspace,
        )
        result = await self._evaluator.evaluate(op, self._scope)

        if result.decision is PolicyDecision.DENY:
            return PermissionVerdict(allowed=False, reason=f"denied: {result.reason}")
        if result.decision is PolicyDecision.ALLOW:
            return PermissionVerdict(allowed=True, reason=result.reason)

        # NEEDS_PROMPT — ask the terminal user.
        return await self._prompt_and_maybe_persist(op)

    async def prompt_and_persist(self, op: Any) -> PermissionVerdict:
        """Prompt the terminal user for a NEEDS_PROMPT operation (host hook).

        Entry point for the live TOOL_CALL permission hook on AgentSession: the
        hook has already evaluated the policy and reached NEEDS_PROMPT; this
        method prompts the user (allow once/always/deny once/always) and
        persists a durable grant on an 'always' choice. The sync input() prompt
        runs off-thread (asyncio.to_thread) so the event loop is not blocked.
        """
        return await self._prompt_and_maybe_persist(op)

    async def _prompt_and_maybe_persist(self, op: Any) -> PermissionVerdict:
        name = op.tool_identity.name
        locs = ", ".join(op.affected_locations) if op.affected_locations else "(any)"
        prompt = f"\n🔐 Tool '{name}' wants to run (affects: {locs}).\n  [1] allow once  [2] allow always  [3] deny once  [4] deny always: "
        choice = await asyncio.to_thread(self._prompt, prompt)

        if choice == "1":
            return PermissionVerdict(allowed=True, reason="allowed once (user)")
        if choice == "3":
            return PermissionVerdict(allowed=False, reason="denied once (user)")

        decision: GrantDecision
        if choice == "2":
            decision = GrantDecision.ALLOW
        elif choice == "4":
            decision = GrantDecision.REJECT
        else:
            # Unrecognized input → fail-closed (ADR-006).
            return PermissionVerdict(allowed=False, reason="unrecognized choice (fail-closed)")

        # Persist a durable grant for each effect kind declared by the operation
        # so future calls of the same tool+effect skip the prompt.
        for eff in op.effects.effects:
            grant = PolicyGrant(
                grant_id=str(uuid4()),
                owner_scope=self._scope,
                decision=decision,
                tool_identity=name,
                effect_kind=eff.kind,
                location="",  # any location for this tool+effect
                created_at=datetime.now(UTC),
                reason="durable grant from dana-code CLI",
            )
            with contextlib.suppress(Exception):
                await self._grant_store.create_grant(grant)

        return PermissionVerdict(
            allowed=decision is GrantDecision.ALLOW,
            reason=f"{'allowed' if decision is GrantDecision.ALLOW else 'denied'} always (persisted grant)",
            persisted=True,
        )
