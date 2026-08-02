"""dana v2.0 extensibility substrate.

EventBus backbone (milestone S1) + Operation/PermissionPolicy (milestone S3).
See sprint/plans/{S1-eventbus-substrate,S3-tool-execution-engine}.md.
"""

from dana.core.ext import events
from dana.core.ext.event_bus import Event, EventBus, EventHandler, HandlerResult, Subscription
from dana.core.ext.operation import Operation, ToolIdentity, build_operation
from dana.core.ext.permission import PermissionPolicy


__all__ = [
    "Event",
    "EventBus",
    "EventHandler",
    "HandlerResult",
    "Subscription",
    "Operation",
    "ToolIdentity",
    "build_operation",
    "PermissionPolicy",
    "events",
]
