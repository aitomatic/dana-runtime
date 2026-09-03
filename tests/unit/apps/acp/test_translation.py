"""
Unit tests for ACP translation — HostEvent → ACP SessionUpdate mapping.

Covers D2 tool lifecycle states: thought, tool-call, tool-update, result,
and cancellation states.
"""

from __future__ import annotations

from datetime import datetime

from dana.apps.acp.translation import host_event_to_acp_update
from dana.core.session.projections.host_events import HostEvent, HostEventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_type: HostEventType,
    *,
    text: str | None = None,
    metadata: dict | None = None,
    sequence: int = 1,
    correlation_id: str = "corr-1",
) -> HostEvent:
    return HostEvent(
        event_type=event_type,
        sequence=sequence,
        correlation_id=correlation_id,
        timestamp=datetime(2026, 8, 3, 12, 0, 0),
        text=text,
        metadata=metadata or {},
    )


# ===========================================================================
# 1. D1 text-bearing events (unchanged)
# ===========================================================================


class TestD1TextEvents:
    def test_user_message_returns_user_message_chunk(self) -> None:
        event = _make_event(HostEventType.USER_MESSAGE, text="hello")
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "user_message_chunk"

    def test_assistant_chunk_returns_agent_message_chunk(self) -> None:
        event = _make_event(HostEventType.ASSISTANT_CONTENT_CHUNK, text="world")
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "agent_message_chunk"

    def test_assistant_final_returns_none(self) -> None:
        event = _make_event(HostEventType.ASSISTANT_CONTENT_FINAL, text="full")
        assert host_event_to_acp_update(event) is None

    def test_turn_lifecycle_returns_none(self) -> None:
        for typ in (
            HostEventType.TURN_STARTED,
            HostEventType.TURN_COMPLETED,
            HostEventType.TURN_CANCELLED,
            HostEventType.TURN_ERROR,
            HostEventType.SESSION_CREATED,
            HostEventType.SESSION_LOADED,
        ):
            assert host_event_to_acp_update(_make_event(typ)) is None


# ===========================================================================
# 2. D2: Agent thought
# ===========================================================================


class TestThought:
    def test_thought_returns_agent_thought_chunk(self) -> None:
        event = _make_event(HostEventType.THOUGHT, text="I am thinking...")
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "agent_thought_chunk"
        assert update.content.text == "I am thinking..."

    def test_thought_empty_text(self) -> None:
        event = _make_event(HostEventType.THOUGHT, text="")
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "agent_thought_chunk"


# ===========================================================================
# 3. D2: Tool lifecycle — TOOL_REQUESTED → tool_call start
# ===========================================================================


class TestToolRequested:
    def test_tool_requested_returns_tool_call_start(self) -> None:
        event = _make_event(
            HostEventType.TOOL_REQUESTED,
            metadata={
                "tool_call_id": "tc-1",
                "tool_name": "read_file",
                "kind": "read",
                "raw_input": {"path": "/tmp/test.txt"},
            },
        )
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "tool_call"
        assert update.tool_call_id == "tc-1"
        assert update.title == "read_file"
        assert update.kind == "read"
        assert update.status == "pending"

    def test_tool_requested_no_kind(self) -> None:
        event = _make_event(
            HostEventType.TOOL_REQUESTED,
            metadata={
                "tool_call_id": "tc-2",
                "tool_name": "bash",
            },
        )
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "tool_call"
        assert update.tool_call_id == "tc-2"
        assert update.title == "bash"
        assert update.kind is None


# ===========================================================================
# 4. D2: Tool lifecycle — TOOL_AUTHORIZED_OR_DENIED
# ===========================================================================


class TestToolAuthorizedOrDenied:
    def test_authorized_returns_pending_update(self) -> None:
        event = _make_event(
            HostEventType.TOOL_AUTHORIZED_OR_DENIED,
            metadata={"tool_call_id": "tc-1", "authorized": True},
        )
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "tool_call_update"
        assert update.tool_call_id == "tc-1"
        assert update.status == "pending"

    def test_denied_returns_failed_update(self) -> None:
        event = _make_event(
            HostEventType.TOOL_AUTHORIZED_OR_DENIED,
            metadata={"tool_call_id": "tc-1", "authorized": False, "reason": "Not allowed"},
        )
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "tool_call_update"
        assert update.tool_call_id == "tc-1"
        assert update.status == "failed"


# ===========================================================================
# 5. D2: Tool lifecycle — TOOL_STARTED
# ===========================================================================


class TestToolStarted:
    def test_tool_started_returns_in_progress_update(self) -> None:
        event = _make_event(
            HostEventType.TOOL_STARTED,
            metadata={"tool_call_id": "tc-1"},
        )
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "tool_call_update"
        assert update.tool_call_id == "tc-1"
        assert update.status == "in_progress"


# ===========================================================================
# 6. D2: Tool lifecycle — TOOL_PROGRESS
# ===========================================================================


class TestToolProgress:
    def test_tool_progress_returns_in_progress_with_output(self) -> None:
        event = _make_event(
            HostEventType.TOOL_PROGRESS,
            metadata={"tool_call_id": "tc-1", "progress": {"bytes_read": 1024}},
        )
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "tool_call_update"
        assert update.tool_call_id == "tc-1"
        assert update.status == "in_progress"


# ===========================================================================
# 7. D2: Tool lifecycle — TOOL_CANCELLATION_REQUESTED
# ===========================================================================


class TestToolCancellationRequested:
    def test_cancellation_requested_returns_update(self) -> None:
        event = _make_event(
            HostEventType.TOOL_CANCELLATION_REQUESTED,
            metadata={"tool_call_id": "tc-1"},
        )
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "tool_call_update"
        assert update.tool_call_id == "tc-1"
        assert update.status == "in_progress"


# ===========================================================================
# 8. D2: Tool lifecycle — Terminal states (result, failure, cancellation)
# ===========================================================================


class TestToolTerminal:
    def test_tool_result_returns_completed(self) -> None:
        event = _make_event(
            HostEventType.TOOL_RESULT,
            metadata={"tool_call_id": "tc-1", "result": {"output": "file content"}},
        )
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "tool_call_update"
        assert update.tool_call_id == "tc-1"
        assert update.status == "completed"

    def test_tool_failure_returns_failed(self) -> None:
        event = _make_event(
            HostEventType.TOOL_FAILURE,
            metadata={"tool_call_id": "tc-1", "error": "File not found"},
        )
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "tool_call_update"
        assert update.tool_call_id == "tc-1"
        assert update.status == "failed"

    def test_tool_acknowledged_returns_completed(self) -> None:
        event = _make_event(
            HostEventType.TOOL_ACKNOWLEDGED,
            metadata={"tool_call_id": "tc-1"},
        )
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "tool_call_update"
        assert update.tool_call_id == "tc-1"
        assert update.status == "completed"

    def test_tool_timed_out_returns_failed(self) -> None:
        event = _make_event(
            HostEventType.TOOL_TIMED_OUT,
            metadata={"tool_call_id": "tc-1"},
        )
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "tool_call_update"
        assert update.tool_call_id == "tc-1"
        assert update.status == "failed"

    def test_tool_effect_unknown_returns_failed(self) -> None:
        event = _make_event(
            HostEventType.TOOL_EFFECT_UNKNOWN,
            metadata={"tool_call_id": "tc-1"},
        )
        update = host_event_to_acp_update(event)
        assert update is not None
        assert update.session_update == "tool_call_update"
        assert update.tool_call_id == "tc-1"
        assert update.status == "failed"


# ===========================================================================
# 9. D2: Full tool lifecycle sequence
# ===========================================================================


class TestFullToolLifecycle:
    def test_full_lifecycle_sequence(self) -> None:
        """Simulate a complete tool call lifecycle: request → start → result."""
        events = [
            _make_event(
                HostEventType.TOOL_REQUESTED,
                metadata={"tool_call_id": "tc-1", "tool_name": "read", "kind": "read"},
            ),
            _make_event(
                HostEventType.TOOL_AUTHORIZED_OR_DENIED,
                metadata={"tool_call_id": "tc-1", "authorized": True},
            ),
            _make_event(
                HostEventType.TOOL_STARTED,
                metadata={"tool_call_id": "tc-1"},
            ),
            _make_event(
                HostEventType.TOOL_RESULT,
                metadata={"tool_call_id": "tc-1", "result": {"data": "ok"}},
            ),
        ]
        updates = [host_event_to_acp_update(e) for e in events]
        assert updates[0].session_update == "tool_call"
        assert updates[0].status == "pending"
        assert updates[1].session_update == "tool_call_update"
        assert updates[1].status == "pending"
        assert updates[2].session_update == "tool_call_update"
        assert updates[2].status == "in_progress"
        assert updates[3].session_update == "tool_call_update"
        assert updates[3].status == "completed"

    def test_cancellation_lifecycle_sequence(self) -> None:
        """Simulate a tool call that gets cancelled."""
        events = [
            _make_event(
                HostEventType.TOOL_REQUESTED,
                metadata={"tool_call_id": "tc-2", "tool_name": "bash"},
            ),
            _make_event(
                HostEventType.TOOL_STARTED,
                metadata={"tool_call_id": "tc-2"},
            ),
            _make_event(
                HostEventType.TOOL_CANCELLATION_REQUESTED,
                metadata={"tool_call_id": "tc-2"},
            ),
            _make_event(
                HostEventType.TOOL_ACKNOWLEDGED,
                metadata={"tool_call_id": "tc-2"},
            ),
        ]
        updates = [host_event_to_acp_update(e) for e in events]
        assert updates[0].status == "pending"
        assert updates[1].status == "in_progress"
        assert updates[2].status == "in_progress"
        assert updates[3].status == "completed"
