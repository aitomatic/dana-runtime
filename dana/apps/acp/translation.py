"""Translation helpers between Dana HostEvents and ACP SessionUpdates.

Dana's core session layer speaks :class:`HostEvent`; ACP speaks JSON-RPC
``session_update`` notifications with typed update chunks. This module is the
ONLY place where the two meet, keeping ACP types out of STAR core.

Chunk semantics: every ``ASSISTANT_CONTENT_CHUNK`` and ``USER_MESSAGE`` event
maps to an ACP delta (the client accumulates). ``ASSISTANT_CONTENT_FINAL`` is
NOT re-sent as a delta because the individual chunks already carried the text;
sending the full text again would duplicate it on the client side. Lifecycle
events (``TURN_*``, ``SESSION_*``) have no ACP update equivalent in D1 — the
``PromptResponse`` / ``LoadSessionResponse`` itself signals completion.
"""

from __future__ import annotations

from typing import Any

from acp.helpers import update_agent_message_text, update_user_message_text

from dana.core.session.projections.host_events import HostEvent, HostEventType


def host_event_to_acp_update(event: HostEvent) -> Any:
    """Translate a :class:`HostEvent` to an ACP SessionUpdate chunk, or ``None``.

    Returns ``None`` for events with no D1 ACP equivalent (lifecycle events,
    content-final). Text-bearing events become delta chunks.
    """
    if event.event_type is HostEventType.USER_MESSAGE:
        return update_user_message_text(event.text or "")
    if event.event_type is HostEventType.ASSISTANT_CONTENT_CHUNK:
        return update_agent_message_text(event.text or "")
    # ASSISTANT_CONTENT_FINAL: already streamed via chunks — skip to avoid duplication.
    # TURN_*, SESSION_*: no ACP update in D1; the response signals completion.
    return None
