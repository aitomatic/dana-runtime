"""Policy evaluator — chains HardPolicy, GrantStore, and PermissionMode.

Per ADR-006 decision precedence:
    hard deny → durable reject grant → durable allow grant →
    permission mode → interactive prompt → fail-closed

The evaluator orchestrates the full precedence chain. Each layer can
short-circuit: hard deny blocks immediately; a matching reject grant
blocks; a matching allow grant permits; the permission mode may auto-allow;
otherwise the operation needs an interactive prompt (or is denied if
prompting is not possible — fail-closed).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from dana.core.policy.grants import GrantDecision, GrantStore
from dana.core.policy.hard_policy import HardPolicy
from dana.core.policy.modes import PermissionMode
from dana.core.policy.operations import Operation
from dana.core.policy.scope import OwnerScope


class PolicyDecision(Enum):
    ALLOW = auto()
    DENY = auto()
    NEEDS_PROMPT = auto()


@dataclass(frozen=True)
class PolicyResult:
    decision: PolicyDecision
    reason: str = ""
    matched_grant_id: str | None = None


class PolicyEvaluator:
    """Evaluates an Operation through the full ADR-006 precedence chain.

    Usage:
        evaluator = PolicyEvaluator(hard_policy, grant_store, mode)
        result = await evaluator.evaluate(operation, scope)
    """

    def __init__(
        self,
        hard_policy: HardPolicy,
        grant_store: GrantStore,
        mode: PermissionMode = PermissionMode.DEFAULT,
    ) -> None:
        self._hard_policy = hard_policy
        self._grant_store = grant_store
        self._mode = mode

    @property
    def mode(self) -> PermissionMode:
        return self._mode

    def set_mode(self, mode: PermissionMode) -> None:
        """Change the permission mode (ADR-013: outside an active turn)."""
        self._mode = mode

    async def evaluate(
        self,
        operation: Operation,
        scope: OwnerScope,
    ) -> PolicyResult:
        """Evaluate an operation through the full precedence chain.

        Returns:
            PolicyResult with decision ALLOW, DENY, or NEEDS_PROMPT.
        """
        # 1. Hard deny (always wins)
        hard_reason = self._hard_policy.check(operation)
        if hard_reason is not None:
            return PolicyResult(PolicyDecision.DENY, reason=hard_reason)

        # 2. Durable grants (reject before allow)
        grant_match = await self._grant_store.find_matching_grants(scope, operation)
        if grant_match.matched is not None:
            if grant_match.decision is GrantDecision.REJECT:
                return PolicyResult(
                    PolicyDecision.DENY,
                    reason=f"rejected by grant {grant_match.grant_id}",
                    matched_grant_id=grant_match.grant_id,
                )
            return PolicyResult(
                PolicyDecision.ALLOW,
                reason=f"allowed by grant {grant_match.grant_id}",
                matched_grant_id=grant_match.grant_id,
            )

        # 3. Permission mode
        effect_kinds = frozenset(e.kind for e in operation.effects.effects)
        if self._mode.allows_without_prompt(effect_kinds):
            return PolicyResult(
                PolicyDecision.ALLOW,
                reason=f"auto-allowed by mode {self._mode.value}",
            )

        # 4. Interactive prompt needed (or fail-closed if not possible)
        return PolicyResult(PolicyDecision.NEEDS_PROMPT, reason="no matching grant or mode auto-allow")
