"""dana v2.0 extensibility substrate.

EventBus backbone (milestone S1). See sprint/plans/S1-eventbus-substrate.md.
"""

from dana.core.ext import events
from dana.core.ext.event_bus import Event, EventBus, EventHandler, HandlerResult, Subscription


__all__ = [
    "Event",
    "EventBus",
    "EventHandler",
    "HandlerResult",
    "Subscription",
    "events",
]
