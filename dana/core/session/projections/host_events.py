"""
Host-event projection — the host-visible event stream of a Session Journal.

Projects ordered :class:`JournalFact` values into ordered :class:`HostEvent`
values for host display. Unlike the Conversation projection, partial
interrupted assistant text REMAINS visible to the host: every
``ASSISTANT_CONTENT_CHUNK`` and ``ASSISTANT_CONTENT_FINAL`` fact emits an event
regardless of the turn's terminal status.

D1 is text-only, so host events cover the lifecycle and text streaming only;
thought, tool, permission, model, mode, and MCP events belong to later phases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dana.core.session.models import FactType, JournalFact, JSONValue


class HostEventType(Enum):
    """Host-visible event kinds (D1 text-only lifecycle + streaming set)."""

    SESSION_CREATED = "session_created"
    SESSION_LOADED = "session_loaded"
    SESSION_RESUMED = "session_resumed"
    TURN_STARTED = "turn_started"
    USER_MESSAGE = "user_message"
    ASSISTANT_CONTENT_CHUNK = "assistant_content_chunk"
    ASSISTANT_CONTENT_FINAL = "assistant_content_final"
    TURN_COMPLETED = "turn_completed"
    TURN_INTERRUPTED = "turn_interrupted"
    TURN_ERROR = "turn_error"
    TURN_CANCELLED = "turn_cancelled"


@dataclass(frozen=True, slots=True)
class HostEvent:
    """A single host-visible event projected from a Journal Fact.

    Attributes:
        event_type: The host-visible kind.
        sequence: The fact sequence this event was projected from.
        correlation_id: Carried verbatim from the source fact.
        timestamp: Carried verbatim from the source fact.
        text: Display text for text-bearing events (user/assistant chunk/final);
            ``None`` otherwise.
        metadata: Remaining structured payload data (everything except ``text``
            for text-bearing events; the whole payload otherwise).
    """

    event_type: HostEventType
    sequence: int
    correlation_id: str
    timestamp: datetime
    text: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


# D1 facts that map to a host event. LEGACY_TIMELINE_MIGRATED is intentionally
# absent: it is an internal migration marker, not a host-visible event.
_FACT_TO_EVENT: Mapping[FactType, HostEventType] = {
    FactType.SESSION_CREATED: HostEventType.SESSION_CREATED,
    FactType.SESSION_LOADED: HostEventType.SESSION_LOADED,
    FactType.SESSION_RESUMED: HostEventType.SESSION_RESUMED,
    FactType.TURN_STARTED: HostEventType.TURN_STARTED,
    FactType.USER_CONTENT_FINAL: HostEventType.USER_MESSAGE,
    FactType.ASSISTANT_CONTENT_CHUNK: HostEventType.ASSISTANT_CONTENT_CHUNK,
    FactType.ASSISTANT_CONTENT_FINAL: HostEventType.ASSISTANT_CONTENT_FINAL,
    FactType.TURN_COMPLETED: HostEventType.TURN_COMPLETED,
    FactType.TURN_INTERRUPTED: HostEventType.TURN_INTERRUPTED,
    FactType.TURN_ERROR: HostEventType.TURN_ERROR,
    FactType.TURN_CANCELLED: HostEventType.TURN_CANCELLED,
}

# Fact types whose payload carries a displayable "text" field.
_TEXT_FACTS = frozenset(
    {
        FactType.USER_CONTENT_FINAL,
        FactType.ASSISTANT_CONTENT_CHUNK,
        FactType.ASSISTANT_CONTENT_FINAL,
    }
)


class HostEventProjector:
    """Pure projector turning ordered Journal Facts into host-visible events.

    Faithful to input order (does not reorder). Partial interrupted assistant
    text stays visible: terminal status never suppresses a chunk/final event.
    """

    def project(self, facts: Sequence[JournalFact]) -> list[HostEvent]:
        """Project ordered facts into host-visible events."""
        events: list[HostEvent] = []
        for fact in facts:
            event_type = _FACT_TO_EVENT.get(fact.fact_type)
            if event_type is None:
                continue
            text: str | None = None
            metadata: dict[str, JSONValue] = {}
            if fact.fact_type in _TEXT_FACTS:
                text = str(fact.payload["text"])
                for key, value in fact.payload.items():
                    if key != "text":
                        metadata[key] = value
            else:
                for key, value in fact.payload.items():
                    metadata[key] = value
            events.append(
                HostEvent(
                    event_type=event_type,
                    sequence=fact.sequence,
                    correlation_id=fact.correlation_id,
                    timestamp=fact.timestamp,
                    text=text,
                    metadata=metadata,
                )
            )
        return events
