"""
Unit tests for HostEventProjector — D2 tool lifecycle fact projection.

Verifies that tool journal facts (TOOL_REQUESTED, TOOL_STARTED, TOOL_RESULT,
TOOL_FAILURE, TOOL_ACKNOWLEDGED, TOOL_TIMED_OUT, TOOL_EFFECT_UNKNOWN, etc.)
are correctly projected to HostEvent values.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from dana.core.session.models import FactType, JournalFact, OwnerScope
from dana.core.session.projections.host_events import HostEventProjector, HostEventType


# ---------------------------------------------------------------------------
# Fact factory
# ---------------------------------------------------------------------------


def _owner() -> OwnerScope:
    return OwnerScope(owner_id="owner-1", workspace="ws-1")


@pytest.fixture
def make_fact():
    """Build JournalFacts with auto-incrementing sequence, isolated per test."""
    counter = 0

    def _make(
        fact_type: FactType,
        *,
        correlation_id: str = "turn-1",
        payload: dict | None = None,
    ) -> JournalFact:
        nonlocal counter
        counter += 1
        return JournalFact(
            fact_id=f"fact-{counter}",
            owner_scope=_owner(),
            session_id="sess-1",
            sequence=counter,
            fact_type=fact_type,
            timestamp=datetime(2026, 8, 3, 12, 0, 0),
            correlation_id=correlation_id,
            causation_id=None,
            schema_version=1,
            payload=payload if payload is not None else {},
        )

    return _make


# ===========================================================================
# 1. Each tool fact type maps to the correct host event type
# ===========================================================================


class TestToolFactToEventMapping:
    def test_tool_requested_maps_correctly(self, make_fact) -> None:
        events = HostEventProjector().project([make_fact(FactType.TOOL_REQUESTED, payload={"tool_call_id": "tc-1", "tool_name": "read"})])
        assert len(events) == 1
        assert events[0].event_type is HostEventType.TOOL_REQUESTED
        assert events[0].metadata["tool_call_id"] == "tc-1"
        assert events[0].metadata["tool_name"] == "read"

    def test_tool_authorized_or_denied_maps_correctly(self, make_fact) -> None:
        events = HostEventProjector().project(
            [make_fact(FactType.TOOL_AUTHORIZED_OR_DENIED, payload={"tool_call_id": "tc-1", "authorized": True})]
        )
        assert len(events) == 1
        assert events[0].event_type is HostEventType.TOOL_AUTHORIZED_OR_DENIED
        assert events[0].metadata["authorized"] is True

    def test_tool_started_maps_correctly(self, make_fact) -> None:
        events = HostEventProjector().project([make_fact(FactType.TOOL_STARTED, payload={"tool_call_id": "tc-1"})])
        assert len(events) == 1
        assert events[0].event_type is HostEventType.TOOL_STARTED

    def test_tool_progress_maps_correctly(self, make_fact) -> None:
        events = HostEventProjector().project(
            [make_fact(FactType.TOOL_PROGRESS, payload={"tool_call_id": "tc-1", "progress": {"pct": 50}})]
        )
        assert len(events) == 1
        assert events[0].event_type is HostEventType.TOOL_PROGRESS
        assert events[0].metadata["progress"] == {"pct": 50}

    def test_tool_cancellation_requested_maps_correctly(self, make_fact) -> None:
        events = HostEventProjector().project([make_fact(FactType.TOOL_CANCELLATION_REQUESTED, payload={"tool_call_id": "tc-1"})])
        assert len(events) == 1
        assert events[0].event_type is HostEventType.TOOL_CANCELLATION_REQUESTED

    def test_tool_result_maps_correctly(self, make_fact) -> None:
        events = HostEventProjector().project([make_fact(FactType.TOOL_RESULT, payload={"tool_call_id": "tc-1", "result": {"data": "ok"}})])
        assert len(events) == 1
        assert events[0].event_type is HostEventType.TOOL_RESULT
        assert events[0].metadata["result"] == {"data": "ok"}

    def test_tool_failure_maps_correctly(self, make_fact) -> None:
        events = HostEventProjector().project([make_fact(FactType.TOOL_FAILURE, payload={"tool_call_id": "tc-1", "error": "boom"})])
        assert len(events) == 1
        assert events[0].event_type is HostEventType.TOOL_FAILURE
        assert events[0].metadata["error"] == "boom"

    def test_tool_acknowledged_maps_correctly(self, make_fact) -> None:
        events = HostEventProjector().project([make_fact(FactType.TOOL_ACKNOWLEDGED, payload={"tool_call_id": "tc-1"})])
        assert len(events) == 1
        assert events[0].event_type is HostEventType.TOOL_ACKNOWLEDGED

    def test_tool_timed_out_maps_correctly(self, make_fact) -> None:
        events = HostEventProjector().project([make_fact(FactType.TOOL_TIMED_OUT, payload={"tool_call_id": "tc-1"})])
        assert len(events) == 1
        assert events[0].event_type is HostEventType.TOOL_TIMED_OUT

    def test_tool_effect_unknown_maps_correctly(self, make_fact) -> None:
        events = HostEventProjector().project([make_fact(FactType.TOOL_EFFECT_UNKNOWN, payload={"tool_call_id": "tc-1"})])
        assert len(events) == 1
        assert events[0].event_type is HostEventType.TOOL_EFFECT_UNKNOWN


# ===========================================================================
# 2. Tool lifecycle ordering
# ===========================================================================


class TestToolLifecycleOrdering:
    def test_tool_lifecycle_in_sequence_order(self, make_fact) -> None:
        facts = [
            make_fact(FactType.TOOL_REQUESTED, payload={"tool_call_id": "tc-1", "tool_name": "read"}),
            make_fact(FactType.TOOL_AUTHORIZED_OR_DENIED, payload={"tool_call_id": "tc-1", "authorized": True}),
            make_fact(FactType.TOOL_STARTED, payload={"tool_call_id": "tc-1"}),
            make_fact(FactType.TOOL_RESULT, payload={"tool_call_id": "tc-1", "result": {"data": "ok"}}),
        ]
        events = HostEventProjector().project(facts)
        assert len(events) == 4
        expected_types = [
            HostEventType.TOOL_REQUESTED,
            HostEventType.TOOL_AUTHORIZED_OR_DENIED,
            HostEventType.TOOL_STARTED,
            HostEventType.TOOL_RESULT,
        ]
        assert [e.event_type for e in events] == expected_types
        assert [e.sequence for e in events] == [1, 2, 3, 4]

    def test_cancellation_lifecycle_in_order(self, make_fact) -> None:
        facts = [
            make_fact(FactType.TOOL_REQUESTED, payload={"tool_call_id": "tc-1", "tool_name": "bash"}),
            make_fact(FactType.TOOL_STARTED, payload={"tool_call_id": "tc-1"}),
            make_fact(FactType.TOOL_CANCELLATION_REQUESTED, payload={"tool_call_id": "tc-1"}),
            make_fact(FactType.TOOL_ACKNOWLEDGED, payload={"tool_call_id": "tc-1"}),
        ]
        events = HostEventProjector().project(facts)
        assert len(events) == 4
        assert events[2].event_type is HostEventType.TOOL_CANCELLATION_REQUESTED
        assert events[3].event_type is HostEventType.TOOL_ACKNOWLEDGED


# ===========================================================================
# 3. Mixed D1 + D2 facts
# ===========================================================================


class TestMixedD1AndD2:
    def test_mixed_facts_preserve_order(self, make_fact) -> None:
        facts = [
            make_fact(FactType.TURN_STARTED),
            make_fact(FactType.USER_CONTENT_FINAL, payload={"text": "hello"}),
            make_fact(FactType.TOOL_REQUESTED, payload={"tool_call_id": "tc-1", "tool_name": "read"}),
            make_fact(FactType.TOOL_STARTED, payload={"tool_call_id": "tc-1"}),
            make_fact(FactType.ASSISTANT_CONTENT_CHUNK, payload={"text": "result:", "index": 0}),
            make_fact(FactType.TOOL_RESULT, payload={"tool_call_id": "tc-1", "result": {"data": "ok"}}),
            make_fact(FactType.ASSISTANT_CONTENT_CHUNK, payload={"text": " done", "index": 1}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, payload={"text": "result: done"}),
            make_fact(FactType.TURN_COMPLETED),
        ]
        events = HostEventProjector().project(facts)
        assert len(events) == 9
        # Verify interleaving: tool events appear between text events
        types = [e.event_type for e in events]
        assert types[0] is HostEventType.TURN_STARTED
        assert types[1] is HostEventType.USER_MESSAGE
        assert types[2] is HostEventType.TOOL_REQUESTED
        assert types[3] is HostEventType.TOOL_STARTED
        assert types[4] is HostEventType.ASSISTANT_CONTENT_CHUNK
        assert types[5] is HostEventType.TOOL_RESULT
        assert types[6] is HostEventType.ASSISTANT_CONTENT_CHUNK
        assert types[7] is HostEventType.ASSISTANT_CONTENT_FINAL
        assert types[8] is HostEventType.TURN_COMPLETED

    def test_tool_facts_carry_correlation_id(self, make_fact) -> None:
        facts = [
            make_fact(FactType.TOOL_REQUESTED, correlation_id="turn-1", payload={"tool_call_id": "tc-1", "tool_name": "read"}),
            make_fact(FactType.TOOL_RESULT, correlation_id="turn-1", payload={"tool_call_id": "tc-1", "result": {"data": "ok"}}),
        ]
        events = HostEventProjector().project(facts)
        assert all(e.correlation_id == "turn-1" for e in events)
