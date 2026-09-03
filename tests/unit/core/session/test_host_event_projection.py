"""
Unit tests for HostEventProjector — host-visible event projection of journal facts.

Key distinction from ConversationProjector: partial interrupted assistant text
REMAINS visible to hosts. Every text-bearing fact emits an event regardless of
terminal status.

Also includes the canonical-parity test verifying both projectors agree on fact
ordering, correlation_id tracking, and sequence numbers.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from dana.core.session.models import FactType, JournalFact, OwnerScope
from dana.core.session.projections.conversation import ConversationProjector
from dana.core.session.projections.host_events import HostEventProjector, HostEventType


# ---------------------------------------------------------------------------
# Fact factory
# ---------------------------------------------------------------------------


def _owner() -> OwnerScope:
    return OwnerScope(owner_id="owner-1", workspace="ws-1")


# Fact types whose payload carries a displayable "text" field.
_TEXT_FACT_TYPES = frozenset({FactType.USER_CONTENT_FINAL, FactType.ASSISTANT_CONTENT_CHUNK, FactType.ASSISTANT_CONTENT_FINAL})


@pytest.fixture
def make_fact():
    """Build JournalFacts with auto-incrementing sequence, isolated per test."""

    counter = 0

    def _make(
        fact_type: FactType,
        *,
        correlation_id: str = "turn-1",
        payload: dict | None = None,
        protected_payload: bytes | None = None,
    ) -> JournalFact:
        nonlocal counter
        counter += 1
        return JournalFact(
            fact_id=f"fact-{counter}",
            owner_scope=_owner(),
            session_id="sess-1",
            sequence=counter,
            fact_type=fact_type,
            timestamp=datetime(2026, 7, 16, 12, 0, 0),
            correlation_id=correlation_id,
            causation_id=None,
            schema_version=1,
            payload=payload if payload is not None else {},
            protected_payload=protected_payload,
        )

    return _make


# ===========================================================================
# 1. all event types
# ===========================================================================


class TestAllEventTypes:
    def test_each_fact_type_maps_to_correct_event(self, make_fact) -> None:
        pairs = [
            (FactType.SESSION_CREATED, HostEventType.SESSION_CREATED),
            (FactType.SESSION_LOADED, HostEventType.SESSION_LOADED),
            (FactType.SESSION_RESUMED, HostEventType.SESSION_RESUMED),
            (FactType.TURN_STARTED, HostEventType.TURN_STARTED),
            (FactType.USER_CONTENT_FINAL, HostEventType.USER_MESSAGE),
            (FactType.ASSISTANT_CONTENT_CHUNK, HostEventType.ASSISTANT_CONTENT_CHUNK),
            (FactType.ASSISTANT_CONTENT_FINAL, HostEventType.ASSISTANT_CONTENT_FINAL),
            (FactType.TURN_COMPLETED, HostEventType.TURN_COMPLETED),
            (FactType.TURN_INTERRUPTED, HostEventType.TURN_INTERRUPTED),
            (FactType.TURN_ERROR, HostEventType.TURN_ERROR),
            (FactType.TURN_CANCELLED, HostEventType.TURN_CANCELLED),
        ]
        for fact_type, expected_event in pairs:
            payload = {"text": "x"} if fact_type in _TEXT_FACT_TYPES else None
            events = HostEventProjector().project([make_fact(fact_type, payload=payload)])
            assert events[0].event_type is expected_event

    def test_text_fact_missing_text_raises_keyerror(self, make_fact) -> None:
        # Text-bearing facts are trusted journal content; a missing "text" is a
        # malformed fact and must fail fast rather than silently degrade to "".
        import pytest

        for fact_type in _TEXT_FACT_TYPES:
            with pytest.raises(KeyError):
                HostEventProjector().project([make_fact(fact_type, payload={})])

    def test_legacy_migrated_produces_no_event(self, make_fact) -> None:
        events = HostEventProjector().project([make_fact(FactType.LEGACY_TIMELINE_MIGRATED)])
        assert events == []

    def test_user_message_carries_text(self, make_fact) -> None:
        events = HostEventProjector().project([make_fact(FactType.USER_CONTENT_FINAL, payload={"text": "hello"})])
        assert events[0].text == "hello"
        assert events[0].event_type is HostEventType.USER_MESSAGE

    def test_chunk_carries_text_and_index_metadata(self, make_fact) -> None:
        events = HostEventProjector().project([make_fact(FactType.ASSISTANT_CONTENT_CHUNK, payload={"text": "part", "index": 2})])
        assert events[0].text == "part"
        assert events[0].metadata == {"index": 2}

    def test_assistant_final_carries_text(self, make_fact) -> None:
        events = HostEventProjector().project([make_fact(FactType.ASSISTANT_CONTENT_FINAL, payload={"text": "full"})])
        assert events[0].text == "full"

    def test_turn_error_carries_error_in_metadata(self, make_fact) -> None:
        events = HostEventProjector().project([make_fact(FactType.TURN_ERROR, payload={"error": "boom"})])
        assert events[0].event_type is HostEventType.TURN_ERROR
        assert events[0].metadata == {"error": "boom"}

    def test_event_carries_correlation_and_timestamp(self, make_fact) -> None:
        fact = make_fact(FactType.TURN_STARTED, correlation_id="corr-x")
        events = HostEventProjector().project([fact])
        assert events[0].correlation_id == "corr-x"
        assert events[0].timestamp == fact.timestamp


# ===========================================================================
# 2. ordering
# ===========================================================================


class TestOrdering:
    def test_events_in_sequence_order(self, make_fact) -> None:
        facts = [
            make_fact(FactType.SESSION_CREATED),
            make_fact(FactType.USER_CONTENT_FINAL, payload={"text": "q"}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, payload={"text": "a"}),
            make_fact(FactType.TURN_COMPLETED),
        ]
        events = HostEventProjector().project(facts)
        assert [e.sequence for e in events] == [1, 2, 3, 4]

    def test_out_of_order_input_preserved_as_given(self, make_fact) -> None:
        # Projector is faithful to the order it receives (caller is responsible
        # for handing it ordered facts); it must not silently reorder.
        a = make_fact(FactType.TURN_COMPLETED, correlation_id="c1")
        b = make_fact(FactType.USER_CONTENT_FINAL, correlation_id="c2", payload={"text": "q"})
        events = HostEventProjector().project([a, b])
        assert [e.sequence for e in events] == [a.sequence, b.sequence]


# ===========================================================================
# 3. interrupted text remains visible
# ===========================================================================


class TestInterruptedTextRemainsVisible:
    def test_interrupted_final_still_emits_event(self, make_fact) -> None:
        facts = [
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, payload={"text": "partial answer"}),
            make_fact(FactType.TURN_INTERRUPTED),
        ]
        events = HostEventProjector().project(facts)
        final_events = [e for e in events if e.event_type is HostEventType.ASSISTANT_CONTENT_FINAL]
        assert len(final_events) == 1
        assert final_events[0].text == "partial answer"
        interrupted = [e for e in events if e.event_type is HostEventType.TURN_INTERRUPTED]
        assert len(interrupted) == 1

    def test_conversation_excludes_but_host_includes(self, make_fact) -> None:
        facts = [
            make_fact(FactType.USER_CONTENT_FINAL, payload={"text": "q"}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, payload={"text": "partial"}),
            make_fact(FactType.TURN_INTERRUPTED),
        ]
        view = ConversationProjector().project(facts)
        events = HostEventProjector().project(facts)
        # Conversation: no assistant message.
        assert [m for m in view.messages if m.role == "assistant"] == []
        # Host: assistant-final event present.
        assert any(e.event_type is HostEventType.ASSISTANT_CONTENT_FINAL for e in events)


# ===========================================================================
# 4. chunk events
# ===========================================================================


class TestChunkEvents:
    def test_each_chunk_emits_individual_event(self, make_fact) -> None:
        facts = [
            make_fact(FactType.ASSISTANT_CONTENT_CHUNK, payload={"text": "one", "index": 0}),
            make_fact(FactType.ASSISTANT_CONTENT_CHUNK, payload={"text": "two", "index": 1}),
            make_fact(FactType.ASSISTANT_CONTENT_CHUNK, payload={"text": "three", "index": 2}),
        ]
        events = HostEventProjector().project(facts)
        assert len(events) == 3
        assert [e.text for e in events] == ["one", "two", "three"]
        assert [e.metadata["index"] for e in events] == [0, 1, 2]
        assert all(e.event_type is HostEventType.ASSISTANT_CONTENT_CHUNK for e in events)


# ===========================================================================
# Canonical parity across projectors
# ===========================================================================


class TestCanonicalParity:
    def test_sequence_and_correlation_agree(self, make_fact) -> None:
        facts = [
            make_fact(FactType.SESSION_CREATED, correlation_id="sess"),
            make_fact(FactType.TURN_STARTED, correlation_id="turn-1"),
            make_fact(FactType.USER_CONTENT_FINAL, correlation_id="turn-1", payload={"text": "q1"}),
            make_fact(FactType.ASSISTANT_CONTENT_CHUNK, correlation_id="turn-1", payload={"text": "c", "index": 0}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="turn-1", payload={"text": "a1"}),
            make_fact(FactType.TURN_COMPLETED, correlation_id="turn-1"),
            make_fact(FactType.TURN_INTERRUPTED, correlation_id="turn-2"),
        ]
        view = ConversationProjector().project(facts)
        events = HostEventProjector().project(facts)

        # last_sequence equals the highest sequence among facts/events.
        assert view.last_sequence == max(f.sequence for f in facts)
        assert view.last_sequence == max(e.sequence for e in events)

        # Every event correlation_id is drawn from the fact set.
        fact_corrs = {f.correlation_id for f in facts}
        assert {e.correlation_id for e in events}.issubset(fact_corrs)

        # Events are ordered by their fact sequence (the shared canonical order).
        assert [e.sequence for e in events] == sorted(e.sequence for e in events)

    def test_no_facts_both_empty_consistent(self) -> None:
        view = ConversationProjector().project([])
        events = HostEventProjector().project([])
        assert view.last_sequence == 0
        assert events == []
        assert view.messages == ()

    def test_host_event_is_frozen(self, make_fact) -> None:
        from dataclasses import FrozenInstanceError

        events = HostEventProjector().project([make_fact(FactType.TURN_STARTED)])
        with pytest.raises(FrozenInstanceError):
            events[0].text = "x"  # type: ignore[misc]
