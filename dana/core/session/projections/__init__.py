"""Projection package — pure projectors turning ordered Journal Facts into views.

- :class:`ConversationProjector` produces the model-facing :class:`ConversationView`.
- :class:`ConversationProjectionState` is the incremental fold state behind it, for
  sessions that fold only the facts appended since their last cursor.
- :class:`HostEventProjector` produces the host-visible :class:`HostEvent` stream.

Both projectors are pure: deterministic given the same ordered facts.
"""

from __future__ import annotations

from dana.core.session.projections.conversation import (
    CONVERSATION_STATE_SCHEMA,
    ConversationProjectionState,
    ConversationProjector,
    ConversationProviderMismatch,
    ConversationPrunedStateError,
    ConversationView,
)
from dana.core.session.projections.host_events import HostEvent, HostEventProjector, HostEventType


__all__ = [
    "CONVERSATION_STATE_SCHEMA",
    "ConversationProjectionState",
    "ConversationProjector",
    "ConversationProviderMismatch",
    "ConversationPrunedStateError",
    "ConversationView",
    "HostEvent",
    "HostEventProjector",
    "HostEventType",
]
