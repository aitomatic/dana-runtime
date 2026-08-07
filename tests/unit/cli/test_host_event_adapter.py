"""D7.2 — HostEvent → RichCLIRenderer bridge unit tests.

Feeds synthetic HostEvent sequences to RichCLIRenderer.handle_host_event against
a captured (StringIO) console. No live LLM. Asserts that each HostEventType
dispatches without crashing and that text-bearing events produce the expected
output, including the truthful cancellation banner (ADR-005) and no-color
graceful degradation.
"""

from __future__ import annotations

from datetime import UTC, datetime
import io

from rich.console import Console

from dana.cli.host_event_adapter import (
    TOOL_TERMINAL_STATUS,
    cancellation_outcome,
)
from dana.cli.rich_cli_renderer import RichCLIRenderer
from dana.core.session.projections.host_events import HostEvent, HostEventType


def _evt(
    et: HostEventType,
    *,
    text: str | None = None,
    metadata: dict | None = None,
    sequence: int = 0,
) -> HostEvent:
    return HostEvent(
        event_type=et,
        sequence=sequence,
        correlation_id="c1",
        timestamp=datetime.now(UTC),
        text=text,
        metadata=metadata or {},
    )


def _renderer(color: bool = False, width: int = 100, verbose: bool = True) -> tuple[RichCLIRenderer, io.StringIO]:
    """Build a renderer writing to a captured buffer.

    ``color=False`` exercises the no-color graceful-degradation path.
    """
    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=color,
        color_system="auto" if color else None,
        width=width,
        highlight=False,
        soft_wrap=True,
    )
    renderer = RichCLIRenderer(console=console, verbose=verbose, show_tool_calls=True)
    return renderer, buf


def _drive(renderer: RichCLIRenderer, events: list[HostEvent]) -> None:
    for event in events:
        renderer.handle_host_event(event)


def test_text_turn_streams_then_finalizes():
    """ASSISTANT_CONTENT_CHUNK accumulates; FINAL prints the full response."""
    renderer, buf = _renderer()
    _drive(
        renderer,
        [
            _evt(HostEventType.TURN_STARTED),
            _evt(HostEventType.ASSISTANT_CONTENT_CHUNK, text="Hello "),
            _evt(HostEventType.ASSISTANT_CONTENT_CHUNK, text="world"),
            _evt(HostEventType.ASSISTANT_CONTENT_FINAL, text="Hello world"),
            _evt(HostEventType.TURN_COMPLETED),
        ],
    )
    out = buf.getvalue()
    # Chunks stream in no-color mode (printed with end="")
    assert "Hello" in out
    # Final response is rendered
    assert "Hello world" in out


def test_tool_lifecycle_renders_result():
    """TOOL_REQUESTED → TOOL_STARTED → TOOL_RESULT produces a tool card + result."""
    renderer, buf = _renderer()
    _drive(
        renderer,
        [
            _evt(HostEventType.TURN_STARTED),
            _evt(
                HostEventType.TOOL_REQUESTED,
                metadata={"tool_call_id": "tc1", "tool_name": "bash", "raw_input": {"command": "ls"}},
            ),
            _evt(HostEventType.TOOL_STARTED, metadata={"tool_call_id": "tc1"}),
            _evt(HostEventType.TOOL_RESULT, metadata={"tool_call_id": "tc1", "result": {"stdout": "a\nb"}}),
            _evt(HostEventType.ASSISTANT_CONTENT_FINAL, text="done"),
            _evt(HostEventType.TURN_COMPLETED),
        ],
    )
    # Tool name was tracked from REQUESTED and reused at terminal (no crash).
    assert renderer._tool_names.get("tc1") == "bash"
    # No exception is the primary assertion; result panel path exercised.


def test_cancellation_banner_is_truthful():
    """TURN_CANCELLED surfaces a truthful outcome banner (ADR-005), incl. partial text."""
    renderer, buf = _renderer()
    _drive(
        renderer,
        [
            _evt(HostEventType.TURN_STARTED),
            _evt(HostEventType.ASSISTANT_CONTENT_CHUNK, text="partial ans"),
            _evt(HostEventType.TURN_CANCELLED, text="partial ans"),
        ],
    )
    out = buf.getvalue()
    assert "Cancellation acknowledged" in out
    # cancellation_outcome helper includes the partial preview
    assert "partial ans" in cancellation_outcome(_evt(HostEventType.TURN_CANCELLED, text="partial ans"))


def test_error_banner_shows_message():
    """TURN_ERROR surfaces the error text."""
    renderer, buf = _renderer()
    _drive(
        renderer,
        [
            _evt(HostEventType.TURN_STARTED),
            _evt(HostEventType.TURN_ERROR, metadata={"error": "model overloaded"}),
        ],
    )
    assert "model overloaded" in buf.getvalue()


def test_tool_failure_terminal():
    """TOOL_FAILURE routes through the terminal handler with failed status."""
    renderer, buf = _renderer()
    _drive(
        renderer,
        [
            _evt(HostEventType.TURN_STARTED),
            _evt(HostEventType.TOOL_REQUESTED, metadata={"tool_call_id": "tc2", "tool_name": "edit"}),
            _evt(HostEventType.TOOL_FAILURE, metadata={"tool_call_id": "tc2", "error": "disk full"}),
            _evt(HostEventType.TURN_COMPLETED),
        ],
    )
    # Did not crash; failure terminal path exercised.


def test_every_hostevent_type_dispatches_without_crash():
    """Every HostEventType in the D7 mapping renders without raising."""
    renderer, buf = _renderer()
    sample_events = [
        _evt(HostEventType.TURN_STARTED),
        _evt(HostEventType.USER_MESSAGE, text="hi"),
        _evt(HostEventType.ASSISTANT_CONTENT_CHUNK, text="x"),
        _evt(HostEventType.ASSISTANT_CONTENT_FINAL, text="x"),
        _evt(HostEventType.THOUGHT, text="hmm"),
        _evt(HostEventType.TOOL_REQUESTED, metadata={"tool_call_id": "t", "tool_name": "n"}),
        _evt(HostEventType.TOOL_AUTHORIZED_OR_DENIED, metadata={"tool_call_id": "t", "authorized": True}),
        _evt(
            HostEventType.TOOL_AUTHORIZED_OR_DENIED, metadata={"tool_call_id": "t2", "tool_name": "n2", "authorized": False, "reason": "no"}
        ),
        _evt(HostEventType.TOOL_STARTED, metadata={"tool_call_id": "t"}),
        _evt(HostEventType.TOOL_PROGRESS, metadata={"tool_call_id": "t", "progress": {"p": 1}}),
        _evt(HostEventType.TOOL_RESULT, metadata={"tool_call_id": "t", "result": "ok"}),
        _evt(HostEventType.TOOL_ACKNOWLEDGED, metadata={"tool_call_id": "t"}),
        _evt(HostEventType.TOOL_TIMED_OUT, metadata={"tool_call_id": "t"}),
        _evt(HostEventType.TOOL_EFFECT_UNKNOWN, metadata={"tool_call_id": "t"}),
        _evt(HostEventType.TURN_COMPLETED),
        _evt(HostEventType.TURN_CANCELLED, text="p"),
        _evt(HostEventType.TURN_ERROR, metadata={"error": "e"}),
    ]
    _drive(renderer, sample_events)  # must not raise


def test_narrow_terminal_does_not_crash():
    """A terminal narrower than 80 cols exercises the degradation path."""
    renderer, buf = _renderer(color=False, width=40)
    assert renderer.is_narrow
    _drive(
        renderer,
        [
            _evt(HostEventType.TURN_STARTED),
            _evt(HostEventType.ASSISTANT_CONTENT_FINAL, text="x" * 200),
            _evt(HostEventType.TURN_COMPLETED),
        ],
    )  # must not raise


def test_color_mode_smoke():
    """Color-capable console path must not crash on a full turn."""
    renderer, buf = _renderer(color=True, width=100)
    _drive(
        renderer,
        [
            _evt(HostEventType.TURN_STARTED),
            _evt(HostEventType.ASSISTANT_CONTENT_CHUNK, text="streaming"),
            _evt(HostEventType.ASSISTANT_CONTENT_FINAL, text="streaming response"),
            _evt(HostEventType.TURN_COMPLETED),
        ],
    )  # must not raise; Live exercised then stopped


def test_tool_terminal_status_table_covers_all_outcomes():
    """The terminal-status dispatch table covers every tool terminal type."""
    expected = {
        HostEventType.TOOL_RESULT,
        HostEventType.TOOL_FAILURE,
        HostEventType.TOOL_ACKNOWLEDGED,
        HostEventType.TOOL_TIMED_OUT,
        HostEventType.TOOL_EFFECT_UNKNOWN,
    }
    assert set(TOOL_TERMINAL_STATUS) == expected
