"""Hard-deny-wins policy enforcement.

Per ADR-006: hard deny wins under all conditions — a hard deny rule overrides
any grant, permission mode, or other policy decision. This is the final
authority: if any hard deny rule matches, the operation is blocked regardless
of what other policies say.

Hard deny rules are evaluated first, before any other policy. If a hard deny
matches, the operation is blocked immediately with no further evaluation.
"""

from __future__ import annotations

from collections.abc import Callable

from dana.core.policy.effects import EffectKind
from dana.core.policy.operations import Operation


# A hard deny rule: inspect an Operation, return a human-readable reason to
# block, or None to allow. First matching rule (registration order) wins.
HardDenyRule = Callable[[Operation], str | None]


class HardPolicy:
    """Hard-deny-wins policy — evaluated before any other policy.

    Hard deny rules are registered with ``deny()`` and evaluated in order.
    The first matching rule returns a block reason; if no rule matches, the
    operation passes through for further evaluation by other policies.

    Hard deny wins under ALL conditions: no grant, permission mode, or
    override can bypass a hard deny.
    """

    def __init__(self) -> None:
        self._rules: list[HardDenyRule] = []

    def deny(self, rule: HardDenyRule) -> None:
        """Register a hard deny rule. First matching rule wins."""
        self._rules.append(rule)

    def check(self, operation: Operation) -> str | None:
        """Evaluate all hard deny rules against the operation.

        Returns the first matching deny reason, or None if all rules allow.

        Hard deny wins: if any rule matches, the operation is blocked
        regardless of what other policies (grants, modes, etc.) say.
        """
        for rule in self._rules:
            reason = rule(operation)
            if reason:
                return str(reason)
        return None

    def is_blocked(self, operation: Operation) -> bool:
        """Convenience: returns True if any hard deny rule matches."""
        return self.check(operation) is not None


def create_default_hard_policy() -> HardPolicy:
    """Create the default hard-deny policy with built-in safety rules.

    These rules represent non-negotiable safety constraints that cannot
    be overridden by any grant or permission mode.

    Returns:
        A ``HardPolicy`` instance with default deny rules.
    """
    policy = HardPolicy()

    # Block unknown/sensitive tools (fail cautious per ADR-006)
    policy.deny(lambda op: "unknown tool — sensitive effect metadata" if op.effects.is_sensitive else None)

    # Block destructive operations on protected paths
    _PROTECTED_PATHS = frozenset({".env", "node_modules"})
    _DESTRUCTIVE_KINDS = {EffectKind.DELETE, EffectKind.MODIFY, EffectKind.CREATE}

    def _has_destructive_effect(op: Operation) -> bool:
        return any(e.kind in _DESTRUCTIVE_KINDS for e in op.effects.effects)

    policy.deny(
        lambda op: "hard deny: protected path"
        if _has_destructive_effect(op)
        and any(p in str(op.arguments.get("path", "")) or p in str(op.arguments.get("command", "")) for p in _PROTECTED_PATHS)
        else None
    )

    # Block rm -rf in bash commands (defense-in-depth)
    policy.deny(
        lambda op: "hard deny: rm -rf blocked"
        if op.tool_identity.name == "bash__execute" and "rm -rf" in str(op.arguments.get("command", ""))
        else None
    )

    return policy
