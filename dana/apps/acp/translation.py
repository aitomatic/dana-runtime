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

D2 adds tool lifecycle events: thought, tool-call, tool-update, result, and
cancellation states. These are translated to ACP ``agent_thought_chunk``,
``tool_call``, and ``tool_call_update`` notifications per ADR-013.
"""

from __future__ import annotations

from typing import Any

from acp.helpers import (
    start_tool_call,
    update_agent_message_text,
    update_agent_thought_text,
    update_tool_call,
    update_user_message_text,
)

from dana.core.session.projections.host_events import HostEvent, HostEventType


def host_event_to_acp_update(event: HostEvent) -> Any:
    """Translate a :class:`HostEvent` to an ACP SessionUpdate chunk, or ``None``.

    Returns ``None`` for events with no ACP equivalent (lifecycle events,
    content-final). Text-bearing events become delta chunks. Tool lifecycle
    events become ``tool_call`` or ``tool_call_update`` notifications.
    """
    # --- D1: Text-bearing events ---
    if event.event_type is HostEventType.USER_MESSAGE:
        return update_user_message_text(event.text or "")
    if event.event_type is HostEventType.ASSISTANT_CONTENT_CHUNK:
        return update_agent_message_text(event.text or "")
    # ASSISTANT_CONTENT_FINAL: already streamed via chunks — skip to avoid duplication.
    # TURN_*, SESSION_*: no ACP update in D1; the response signals completion.

    # --- D2: Agent thought ---
    if event.event_type is HostEventType.THOUGHT:
        return update_agent_thought_text(event.text or "")

    # --- D2: Tool lifecycle ---
    if event.event_type is HostEventType.TOOL_REQUESTED:
        return _tool_requested_to_acp(event)
    if event.event_type is HostEventType.TOOL_AUTHORIZED_OR_DENIED:
        return _tool_authorized_or_denied_to_acp(event)
    if event.event_type is HostEventType.TOOL_STARTED:
        return _tool_started_to_acp(event)
    if event.event_type is HostEventType.TOOL_PROGRESS:
        return _tool_progress_to_acp(event)
    if event.event_type is HostEventType.TOOL_CANCELLATION_REQUESTED:
        return _tool_cancellation_requested_to_acp(event)
    if event.event_type in (
        HostEventType.TOOL_RESULT,
        HostEventType.TOOL_FAILURE,
        HostEventType.TOOL_ACKNOWLEDGED,
        HostEventType.TOOL_TIMED_OUT,
        HostEventType.TOOL_EFFECT_UNKNOWN,
    ):
        return _tool_terminal_to_acp(event)

    return None


# ---------------------------------------------------------------------------
# Tool lifecycle translation helpers
# ---------------------------------------------------------------------------


def _tool_requested_to_acp(event: HostEvent) -> Any:
    """Translate a TOOL_REQUESTED event to an ACP ``tool_call`` start notification.

    The ``tool_call`` notification carries the tool's identity, kind, and
    pending status. The client uses this to display a new tool card.
    """
    meta = event.metadata
    tool_name = meta.get("tool_name", "")
    tool_call_id = meta.get("tool_call_id", "")
    kind = meta.get("kind")
    return start_tool_call(
        tool_call_id=tool_call_id,
        title=tool_name,
        kind=kind,
        status="pending",
        raw_input=meta.get("raw_input"),
    )


def _tool_authorized_or_denied_to_acp(event: HostEvent) -> Any:
    """Translate a TOOL_AUTHORIZED_OR_DENIED event to an ACP tool_call_update.

    If the tool was denied, the status is ``failed`` with an error message.
    If authorized, the status remains ``pending`` (the TOOL_STARTED event
    will advance it to ``in_progress``).
    """
    meta = event.metadata
    tool_call_id = meta.get("tool_call_id", "")
    authorized = meta.get("authorized", True)
    if not authorized:
        return update_tool_call(
            tool_call_id=tool_call_id,
            status="failed",
            raw_output={"error": meta.get("reason", "Permission denied")},
        )
    return update_tool_call(
        tool_call_id=tool_call_id,
        status="pending",
    )


def _tool_started_to_acp(event: HostEvent) -> Any:
    """Translate a TOOL_STARTED event to an ACP tool_call_update with in_progress status."""
    meta = event.metadata
    return update_tool_call(
        tool_call_id=meta.get("tool_call_id", ""),
        status="in_progress",
    )


def _tool_progress_to_acp(event: HostEvent) -> Any:
    """Translate a TOOL_PROGRESS event to an ACP tool_call_update with progress content."""
    meta = event.metadata
    return update_tool_call(
        tool_call_id=meta.get("tool_call_id", ""),
        status="in_progress",
        raw_output=meta.get("progress"),
    )


def _tool_cancellation_requested_to_acp(event: HostEvent) -> Any:
    """Translate a TOOL_CANCELLATION_REQUESTED event to an ACP tool_call_update.

    The tool call is being cancelled. The terminal outcome (acknowledged,
    timed-out, effect-unknown) will follow as a separate terminal event.
    """
    meta = event.metadata
    return update_tool_call(
        tool_call_id=meta.get("tool_call_id", ""),
        status="in_progress",
        raw_output={"cancellation": "requested"},
    )


def _tool_terminal_to_acp(event: HostEvent) -> Any:
    """Translate a terminal tool event to an ACP tool_call_update.

    Maps the five terminal outcomes to ACP status:
    - TOOL_RESULT → completed
    - TOOL_FAILURE → failed
    - TOOL_ACKNOWLEDGED → completed (cancellation acknowledged)
    - TOOL_TIMED_OUT → failed (cancellation timed out)
    - TOOL_EFFECT_UNKNOWN → failed (effect unknown)
    """
    meta = event.metadata
    tool_call_id = meta.get("tool_call_id", "")

    if event.event_type is HostEventType.TOOL_RESULT:
        return update_tool_call(
            tool_call_id=tool_call_id,
            status="completed",
            raw_output=meta.get("result"),
        )

    if event.event_type is HostEventType.TOOL_FAILURE:
        return update_tool_call(
            tool_call_id=tool_call_id,
            status="failed",
            raw_output={"error": meta.get("error", "Tool execution failed")},
        )

    if event.event_type is HostEventType.TOOL_ACKNOWLEDGED:
        return update_tool_call(
            tool_call_id=tool_call_id,
            status="completed",
            raw_output={"cancellation": "acknowledged"},
        )

    if event.event_type is HostEventType.TOOL_TIMED_OUT:
        return update_tool_call(
            tool_call_id=tool_call_id,
            status="failed",
            raw_output={"cancellation": "timed_out"},
        )

    # TOOL_EFFECT_UNKNOWN
    return update_tool_call(
        tool_call_id=tool_call_id,
        status="failed",
        raw_output={"cancellation": "effect_unknown"},
    )
