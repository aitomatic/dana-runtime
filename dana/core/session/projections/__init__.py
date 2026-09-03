"""Projection package — pure projectors turning ordered Journal Facts into views.

- :class:`ConversationProjector` produces the model-facing :class:`ConversationView`.
- :class:`HostEventProjector` produces the host-visible :class:`HostEvent` stream.

Both projectors are pure: deterministic given the same ordered facts.
"""

from __future__ import annotations

from dana.core.session.projections.conversation import ConversationProjector, ConversationView
from dana.core.session.projections.host_events import HostEvent, HostEventProjector, HostEventType


__all__ = [
    "ConversationProjector",
    "ConversationView",
    "HostEvent",
    "HostEventProjector",
    "HostEventType",
]
