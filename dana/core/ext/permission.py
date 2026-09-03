"""PermissionPolicy — deny-only policy v0.1 (M3).

A PermissionPolicy is NOT special to the executor: it is just a ``tool_call``
subscriber that returns ``{"block": True, "reason": ...}`` when a deny rule
matches. Consumers instantiate it, register deny rules, and subscribe it:

    policy = PermissionPolicy()
    policy.deny(lambda op: "rm -rf blocked"
                if op.tool_identity.name == "bash_tool"
                and "rm -rf" in op.arguments.get("command", "")
                else None)
    agent.event_bus.subscribe(TOOL_CALL, policy.on_tool_call)

The executor only ever respects a ``block`` from *any* handler — it does not
know a policy exists (maximum decoupling). v0.1 is **deny-only**: allow is the
default; no confirmation/approval mode. The richer model (Permission Mode +
Policy Grant, the removed approval scaffold revived as an event handler) is
deferred to vNext — see sprint/plans/S3-tool-execution-engine.md §vNext.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dana.core.ext.event_bus import HandlerResult
from dana.core.ext.operation import Operation


# A deny rule: inspect an Operation, return a human-readable reason to block,
# or None to allow. First matching rule (registration order) wins.
DenyRule = Callable[[Operation], str | None]


class PermissionPolicy:
    """Deny-only permission policy, consumed as a ``tool_call`` event handler."""

    def __init__(self) -> None:
        self._rules: list[DenyRule] = []

    def deny(self, rule: DenyRule) -> None:
        """Register a deny rule. First matching rule (in registration order) wins."""
        self._rules.append(rule)

    def check(self, operation: Operation) -> str | None:
        """Return the first matching deny reason, or None if all rules allow."""
        for rule in self._rules:
            reason = rule(operation)
            if reason:
                return str(reason)
        return None

    def on_tool_call(self, event: Any) -> HandlerResult | None:
        """EventBus handler. Subscribe this for the ``TOOL_CALL`` event.

        Returning ``{"block": True, "reason": ...}`` makes the executor skip
        dispatch and surface a ``policy_block`` tool_result; returning ``None``
        is a pass-through (allow).
        """
        operation: Operation = event.payload["operation"]
        reason = self.check(operation)
        return {"block": True, "reason": reason} if reason else None
