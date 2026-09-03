"""Reference deny-only guard policy (M3 demo / S4 discovery target).

A bundled, importable example of a ``PermissionPolicy`` wired to the two demo
deny rules from the sprint plan (`rm -rf` + protected path). S4 (extension
auto-discovery) will later pick extensions up from ``.dana/extensions/``; until
then this is instantiated explicitly:

    from dana.core.ext.guard import install_guard
    install_guard(agent.event_bus)

This module only builds + subscribes a policy — it contains no executor logic.
The executor remains agnostic to policies (it only respects a ``block`` from
any ``tool_call`` handler).
"""

from __future__ import annotations

from dana.core.ext.event_bus import EventBus
from dana.core.ext.events import TOOL_CALL
from dana.core.ext.permission import PermissionPolicy


# Paths the guard refuses to let write/edit touch.
_PROTECTED_PATHS = frozenset({".env", "node_modules"})


def create_guard_policy() -> PermissionPolicy:
    """Build the demo deny-only policy: block `rm -rf` and protected-path writes."""
    policy = PermissionPolicy()
    policy.deny(
        lambda op: "rm -rf blocked" if op.tool_identity.name == "bash_tool" and "rm -rf" in str(op.arguments.get("command", "")) else None
    )
    policy.deny(
        lambda op: "protected path"
        if op.tool_identity.name in ("write", "edit") and op.arguments.get("path", "") in _PROTECTED_PATHS
        else None
    )
    return policy


def install_guard(bus: EventBus) -> PermissionPolicy:
    """Subscribe the guard policy to ``TOOL_CALL`` on ``bus``. Returns the policy."""
    policy = create_guard_policy()
    bus.subscribe(TOOL_CALL, policy.on_tool_call)
    return policy
