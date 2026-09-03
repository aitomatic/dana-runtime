"""Policy Preflight — reports all predictable missing grants before work begins.

Per ADR-006: Policy Preflight reports all predictable missing grants before work
begins but never grants access and never replaces invocation-time enforcement
for dynamic Operations. Fail-closed fallback is the last resort.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dana.core.policy.evaluator import PolicyDecision, PolicyEvaluator
from dana.core.policy.grants import GrantStore
from dana.core.policy.operations import Operation
from dana.core.policy.scope import OwnerScope


@dataclass(frozen=True)
class PreflightResult:
    """Result of a policy preflight check.

    ``can_proceed``     — True if all predictable operations have matching grants
                          or are auto-allowed by the permission mode.
    ``missing_grants``  — List of tool identities that would need a grant or
                          interactive prompt at invocation time.
    ``hard_denied``     — List of tool identities that are hard-denied and would
                          be blocked regardless of grants.
    """

    can_proceed: bool
    missing_grants: list[str] = field(default_factory=list)
    hard_denied: list[str] = field(default_factory=list)


async def run_preflight(
    evaluator: PolicyEvaluator,
    grant_store: GrantStore,
    scope: OwnerScope,
    operations: list[Operation],
) -> PreflightResult:
    """Run a policy preflight for a list of predictable operations.

    Evaluates each operation through the full precedence chain. Reports
    which tools would need a grant or prompt at invocation time, and which
    are hard-denied.

    Args:
        evaluator: The PolicyEvaluator to use.
        grant_store: The GrantStore to check for existing grants.
        scope: The OwnerScope for the session.
        operations: The list of predictable Operations to check.

    Returns:
        A PreflightResult with the findings.
    """
    missing_grants: list[str] = []
    hard_denied: list[str] = []

    for op in operations:
        result = await evaluator.evaluate(op, scope)

        if result.decision is PolicyDecision.DENY:
            hard_denied.append(op.tool_identity.name)
        elif result.decision is PolicyDecision.NEEDS_PROMPT:
            missing_grants.append(op.tool_identity.name)

    return PreflightResult(
        can_proceed=len(hard_denied) == 0 and len(missing_grants) == 0,
        missing_grants=missing_grants,
        hard_denied=hard_denied,
    )
