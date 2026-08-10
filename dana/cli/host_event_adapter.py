"""HostEvent → Rich renderer bridge for the dana-code CLI (D7.2).

In-process analog of :func:`dana.apps.acp.translation.host_event_to_acp_update`:
instead of translating :class:`HostEvent` values to ACP JSON-RPC chunks, this
module dispatches them to :class:`RichCLIRenderer` component handlers for
terminal display.

Both the ACP adapter and this CLI bridge consume the SAME ``HostEvent``
projection from :class:`~dana.core.session.agent_session.AgentSession`
(ADR-001 — one host-neutral session, one event model). There is no second
event model for the CLI; only the *rendering* edge differs.

The renderer owns all terminal-side state (spinner, stream buffer, tool cards,
live display); this module is a pure dispatcher plus a few label tables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dana.core.session.projections.host_events import HostEvent, HostEventType


if TYPE_CHECKING:
    from dana.cli.rich_cli_renderer import RichCLIRenderer


# Terminal tool-outcome → (display status, payload key) mirroring ACP
# session/cancel semantics (ADR-005: acknowledged / timed-out / effect-unknown).
TOOL_TERMINAL_STATUS: dict[HostEventType, str] = {
    HostEventType.TOOL_RESULT: "completed",
    HostEventType.TOOL_FAILURE: "failed",
    HostEventType.TOOL_ACKNOWLEDGED: "completed",
    HostEventType.TOOL_TIMED_OUT: "failed",
    HostEventType.TOOL_EFFECT_UNKNOWN: "failed",
}

# Turn-terminal kinds surfaced as banners.
TURN_TERMINAL_KIND: dict[HostEventType, str] = {
    HostEventType.TURN_CANCELLED: "cancelled",
    HostEventType.TURN_INTERRUPTED: "interrupted",
    HostEventType.TURN_ERROR: "error",
}


def render_host_event(renderer: RichCLIRenderer, event: HostEvent) -> None:
    """Dispatch one ``HostEvent`` to the renderer's component handlers.

    Called by :meth:`RichCLIRenderer.handle_host_event`. Unknown / unmapped
    event types are ignored silently (forward-compatible with future phases).
    """
    et = event.event_type

    if et is HostEventType.TURN_STARTED:
        renderer.begin_turn(event)
    elif et is HostEventType.USER_MESSAGE:
        renderer.show_user_message(event)
    elif et is HostEventType.ASSISTANT_CONTENT_CHUNK:
        renderer.stream_assistant_chunk(event)
    elif et is HostEventType.ASSISTANT_CONTENT_FINAL:
        renderer.finish_assistant_response(event)
    elif et is HostEventType.THOUGHT:
        renderer.show_thought(event)
    elif et is HostEventType.TOOL_REQUESTED:
        renderer.show_tool_requested(event)
    elif et is HostEventType.TOOL_AUTHORIZED_OR_DENIED:
        renderer.show_tool_authorization(event)
    elif et is HostEventType.TOOL_STARTED:
        renderer.show_tool_started(event)
    elif et is HostEventType.TOOL_PROGRESS:
        renderer.show_tool_progress(event)
    elif et in TOOL_TERMINAL_STATUS:
        renderer.show_tool_terminal(event, status=TOOL_TERMINAL_STATUS[et])
    elif et is HostEventType.TURN_COMPLETED:
        renderer.complete_turn(event)
    elif et in TURN_TERMINAL_KIND:
        renderer.terminate_turn(event, kind=TURN_TERMINAL_KIND[et])
    # SESSION_* and other lifecycle events: no CLI rendering in D7.


def cancellation_outcome(event: HostEvent) -> str:
    """Human-readable cancellation outcome for a TURN_CANCELLED event.

    Mirrors ADR-005's truthful-terminal contract: the banner must state what
    actually happened to owned work, not just "stopped".
    """
    partial = (event.text or "").strip()
    if partial:
        preview = partial if len(partial) <= 60 else partial[:57] + "…"
        return f"Cancellation acknowledged — partial output preserved ({preview})"
    return "Cancellation acknowledged — no partial output."
