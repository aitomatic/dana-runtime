"""
Unit tests for ConversationProjector — model-facing projection of journal facts.

Covers the Task 3 contract:
  1. chunk/final assembly (assistant message uses FINAL text, not chunks)
  2. exactly-one-terminal validation (one TURN_COMPLETED -> one assistant message)
  3. interrupted partial visibility (no completed message; observation set)
  4. completed-message exclusion (interrupted text excluded from messages)
  5. multiple turns (ordered user+assistant)
  6. replay state decryption (most recent protected_payload; None cases)
  7. replay fingerprint mismatch (wrong key -> raises, no silent swallow)
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pytest

from dana.common.llm.types import LLMMessage
from dana.core.session.models import FactType, JournalFact, OwnerScope
from dana.core.session.projections.conversation import ConversationProjector
from dana.core.session.protected_state import ProtectedStateCodec


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


def _codec_with_key(key: bytes) -> ProtectedStateCodec:
    class _FixedProvider:
        def key(self) -> bytes:
            return key

    return ProtectedStateCodec(_FixedProvider())


def _completed_turn(
    make_fact,
    *,
    correlation_id: str,
    user_text: str,
    assistant_text: str,
    chunks: Sequence[str] = (),
) -> list[JournalFact]:
    """Build a complete turn: user + chunks + final + completed (shared corr)."""
    facts: list[JournalFact] = [
        make_fact(FactType.USER_CONTENT_FINAL, correlation_id=correlation_id, payload={"text": user_text}),
    ]
    for i, chunk in enumerate(chunks):
        facts.append(
            make_fact(
                FactType.ASSISTANT_CONTENT_CHUNK,
                correlation_id=correlation_id,
                payload={"text": chunk, "index": i},
            )
        )
    facts.append(make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id=correlation_id, payload={"text": assistant_text}))
    facts.append(make_fact(FactType.TURN_COMPLETED, correlation_id=correlation_id))
    return facts


# ===========================================================================
# 1. chunk/final assembly
# ===========================================================================


class TestChunkFinalAssembly:
    def test_assistant_message_uses_final_text_not_chunks(self, make_fact) -> None:
        facts = _completed_turn(
            make_fact,
            correlation_id="turn-1",
            user_text="hello",
            assistant_text="the full response",
            chunks=["the ", "full ", "response"],
        )
        view = ConversationProjector().project(facts)
        assistant_msgs = [m for m in view.messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "the full response"

    def test_chunks_alone_produce_no_assistant_message(self, make_fact) -> None:
        facts = [
            make_fact(FactType.ASSISTANT_CONTENT_CHUNK, payload={"text": "chunk1", "index": 0}),
            make_fact(FactType.ASSISTANT_CONTENT_CHUNK, payload={"text": "chunk2", "index": 1}),
        ]
        view = ConversationProjector().project(facts)
        assert view.messages == ()


# ===========================================================================
# 2. exactly-one-terminal validation
# ===========================================================================


class TestExactlyOneTerminal:
    def test_one_completed_one_assistant_message(self, make_fact) -> None:
        facts = _completed_turn(make_fact, correlation_id="turn-1", user_text="q", assistant_text="a")
        view = ConversationProjector().project(facts)
        assistant_msgs = [m for m in view.messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1

    def test_completed_without_final_emits_no_assistant_message(self, make_fact) -> None:
        facts = [
            make_fact(FactType.USER_CONTENT_FINAL, payload={"text": "q"}),
            make_fact(FactType.TURN_COMPLETED),
        ]
        view = ConversationProjector().project(facts)
        assert [m for m in view.messages if m.role == "assistant"] == []
        assert len(view.messages) == 1
        assert view.messages[0].role == "user"


# ===========================================================================
# 3. interrupted partial visibility
# ===========================================================================


class TestInterruptedPartialVisibility:
    def test_interrupted_final_not_in_messages(self, make_fact) -> None:
        facts = [
            make_fact(FactType.USER_CONTENT_FINAL, payload={"text": "q"}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, payload={"text": "partial answer"}),
            make_fact(FactType.TURN_INTERRUPTED),
        ]
        view = ConversationProjector().project(facts)
        assert [m for m in view.messages if m.role == "assistant"] == []

    def test_interrupted_sets_observation(self, make_fact) -> None:
        facts = [
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, payload={"text": "partial"}),
            make_fact(FactType.TURN_INTERRUPTED),
        ]
        view = ConversationProjector().project(facts)
        assert view.interruption_observation == ("The previous turn was interrupted; do not assume unfinished effects completed.")

    def test_completed_does_not_set_observation(self, make_fact) -> None:
        facts = _completed_turn(make_fact, correlation_id="turn-1", user_text="q", assistant_text="a")
        view = ConversationProjector().project(facts)
        assert view.interruption_observation is None

    def test_dangling_final_without_terminal_excluded(self, make_fact) -> None:
        # Final emitted but no terminal fact at all: uncommitted -> excluded.
        facts = [
            make_fact(FactType.USER_CONTENT_FINAL, payload={"text": "q"}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, payload={"text": "no terminal"}),
        ]
        view = ConversationProjector().project(facts)
        assert [m for m in view.messages if m.role == "assistant"] == []
        assert view.interruption_observation is None


# ===========================================================================
# 4. completed-message exclusion (explicit content check)
# ===========================================================================


class TestCompletedMessageExclusion:
    def test_interrupted_text_absent_from_messages(self, make_fact) -> None:
        partial = "PARTIAL-SENTINEL"
        facts = [
            make_fact(FactType.USER_CONTENT_FINAL, payload={"text": "q"}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, payload={"text": partial}),
            make_fact(FactType.TURN_INTERRUPTED),
        ]
        view = ConversationProjector().project(facts)
        for m in view.messages:
            assert partial not in (m.content if isinstance(m.content, str) else "")

    def test_committed_text_present_from_messages(self, make_fact) -> None:
        committed = "COMMITTED-SENTINEL"
        facts = _completed_turn(make_fact, correlation_id="turn-1", user_text="q", assistant_text=committed)
        view = ConversationProjector().project(facts)
        assistant_contents = [m.content for m in view.messages if m.role == "assistant"]
        assert committed in assistant_contents


# ===========================================================================
# 5. multiple turns
# ===========================================================================


class TestMultipleTurns:
    def test_two_complete_turns_ordered(self, make_fact) -> None:
        facts = [
            *_completed_turn(make_fact, correlation_id="turn-1", user_text="q1", assistant_text="a1"),
            *_completed_turn(make_fact, correlation_id="turn-2", user_text="q2", assistant_text="a2"),
        ]
        view = ConversationProjector().project(facts)
        roles = [m.role for m in view.messages]
        contents = [m.content for m in view.messages]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert contents == ["q1", "a1", "q2", "a2"]

    def test_interleaved_interrupted_then_completed(self, make_fact) -> None:
        # First turn interrupted (no assistant message), second turn completes.
        facts = [
            make_fact(FactType.USER_CONTENT_FINAL, correlation_id="turn-1", payload={"text": "q1"}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="turn-1", payload={"text": "partial"}),
            make_fact(FactType.TURN_INTERRUPTED, correlation_id="turn-1"),
            *_completed_turn(make_fact, correlation_id="turn-2", user_text="q2", assistant_text="a2"),
        ]
        view = ConversationProjector().project(facts)
        roles = [m.role for m in view.messages]
        contents = [m.content for m in view.messages]
        # The interrupted assistant text must NOT appear; only q1, q2, a2.
        assert roles == ["user", "user", "assistant"]
        assert contents == ["q1", "q2", "a2"]
        # The later completed turn clears the earlier interruption observation.
        assert view.interruption_observation is None


# ===========================================================================
# 5b. interruption_observation reflects the most recent terminated turn
# ===========================================================================


class TestInterruptionObservationIsLastTerminatedTurn:
    def test_interruption_cleared_by_later_completed_turn(self, make_fact) -> None:
        # turn-1 interrupted, turn-2 completed: observation must be None because
        # the LAST terminated turn completed normally.
        facts = [
            make_fact(FactType.USER_CONTENT_FINAL, correlation_id="turn-1", payload={"text": "u1"}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="turn-1", payload={"text": "final1"}),
            make_fact(FactType.TURN_INTERRUPTED, correlation_id="turn-1"),
            make_fact(FactType.USER_CONTENT_FINAL, correlation_id="turn-2", payload={"text": "u2"}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="turn-2", payload={"text": "final2"}),
            make_fact(FactType.TURN_COMPLETED, correlation_id="turn-2"),
        ]
        view = ConversationProjector().project(facts)
        # turn-1 assistant text excluded (interrupted); turn-2 included (completed).
        contents = [m.content for m in view.messages]
        assert contents == ["u1", "u2", "final2"]
        assert "final1" not in contents
        # Last terminated turn completed -> no observation.
        assert view.interruption_observation is None

    def test_interruption_is_last_terminated_turn(self, make_fact) -> None:
        # turn-1 completed, turn-2 interrupted: observation IS set because the
        # LAST terminated turn was interrupted.
        facts = [
            make_fact(FactType.USER_CONTENT_FINAL, correlation_id="turn-1", payload={"text": "u1"}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="turn-1", payload={"text": "final1"}),
            make_fact(FactType.TURN_COMPLETED, correlation_id="turn-1"),
            make_fact(FactType.USER_CONTENT_FINAL, correlation_id="turn-2", payload={"text": "u2"}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="turn-2", payload={"text": "final2"}),
            make_fact(FactType.TURN_INTERRUPTED, correlation_id="turn-2"),
        ]
        view = ConversationProjector().project(facts)
        # turn-1 assistant text committed; turn-2 excluded (interrupted).
        contents = [m.content for m in view.messages]
        assert contents == ["u1", "final1", "u2"]
        assert "final2" not in contents
        # Last terminated turn interrupted -> observation set.
        assert view.interruption_observation == ("The previous turn was interrupted; do not assume unfinished effects completed.")

    def test_interruption_cleared_by_later_error_turn(self, make_fact) -> None:
        # An earlier interruption must be cleared by a later errored turn (a
        # terminal state): the last terminated turn did not leave effects pending.
        facts = [
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="turn-1", payload={"text": "partial"}),
            make_fact(FactType.TURN_INTERRUPTED, correlation_id="turn-1"),
            make_fact(FactType.TURN_ERROR, correlation_id="turn-2", payload={"error": "boom"}),
        ]
        view = ConversationProjector().project(facts)
        assert view.interruption_observation is None

    def test_interruption_cleared_by_later_cancelled_turn(self, make_fact) -> None:
        # An earlier interruption must be cleared by a later cancelled turn.
        facts = [
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="turn-1", payload={"text": "partial"}),
            make_fact(FactType.TURN_INTERRUPTED, correlation_id="turn-1"),
            make_fact(FactType.TURN_CANCELLED, correlation_id="turn-2"),
        ]
        view = ConversationProjector().project(facts)
        assert view.interruption_observation is None


# ===========================================================================
# 6. replay state
# ===========================================================================


class TestReplayState:
    def test_decrypts_most_recent_protected_payload(self, make_fact) -> None:
        codec = _codec_with_key(b"replay-key-material")
        first = codec.encrypt(b"replay-v1")
        second = codec.encrypt(b"replay-v2")
        facts = [
            make_fact(FactType.TURN_COMPLETED, correlation_id="turn-1", protected_payload=first),
            make_fact(FactType.TURN_COMPLETED, correlation_id="turn-2", protected_payload=second),
        ]
        view = ConversationProjector(protected_state_codec=codec).project(facts)
        assert view.replay_state == b"replay-v2"

    def test_no_codec_yields_none(self, make_fact) -> None:
        codec = _codec_with_key(b"replay-key-material")
        ciphertext = codec.encrypt(b"replay")
        facts = [
            make_fact(
                FactType.TURN_COMPLETED,
                protected_payload=ciphertext,
            )
        ]
        view = ConversationProjector().project(facts)
        assert view.replay_state is None

    def test_no_protected_payload_yields_none(self, make_fact) -> None:
        facts = _completed_turn(make_fact, correlation_id="turn-1", user_text="q", assistant_text="a")
        view = ConversationProjector(protected_state_codec=_codec_with_key(b"k")).project(facts)
        assert view.replay_state is None


# ===========================================================================
# 7. replay fingerprint mismatch
# ===========================================================================


class TestReplayFingerprintMismatch:
    def test_wrong_key_raises_invalid_tag(self, make_fact) -> None:
        from cryptography.exceptions import InvalidTag

        codec_a = _codec_with_key(b"key-a-material")
        codec_b = _codec_with_key(b"key-b-material")
        ciphertext = codec_a.encrypt(b"provider-replay-state")
        facts = [
            make_fact(
                FactType.TURN_COMPLETED,
                protected_payload=ciphertext,
            )
        ]
        projector = ConversationProjector(protected_state_codec=codec_b)
        with pytest.raises(InvalidTag):
            projector.project(facts)


# ===========================================================================
# View shape / last_sequence
# ===========================================================================


class TestViewShape:
    def test_empty_facts(self) -> None:
        view = ConversationProjector().project([])
        assert view.messages == ()
        assert view.replay_state is None
        assert view.interruption_observation is None
        assert view.last_sequence == 0

    def test_last_sequence_tracks_high(self, make_fact) -> None:
        facts = _completed_turn(make_fact, correlation_id="turn-1", user_text="q", assistant_text="a")
        view = ConversationProjector().project(facts)
        assert view.last_sequence == len(facts)

    def test_view_is_frozen(self, make_fact) -> None:
        from dataclasses import FrozenInstanceError

        facts = _completed_turn(make_fact, correlation_id="turn-1", user_text="q", assistant_text="a")
        view = ConversationProjector().project(facts)
        with pytest.raises(FrozenInstanceError):
            view.interruption_observation = "x"  # type: ignore[misc]

    def test_messages_are_llm_message_instances(self, make_fact) -> None:
        facts = _completed_turn(make_fact, correlation_id="turn-1", user_text="q", assistant_text="a")
        view = ConversationProjector().project(facts)
        assert all(isinstance(m, LLMMessage) for m in view.messages)


# ===========================================================================
# D4: Model change tracking
# ===========================================================================


class TestModelChangeTracking:
    """MODEL_CHANGED facts are projected into model_changes, current_provider, current_model."""

    def test_no_model_changes(self, make_fact) -> None:
        facts = _completed_turn(make_fact, correlation_id="turn-1", user_text="q", assistant_text="a")
        view = ConversationProjector().project(facts)
        assert view.model_changes == ()
        assert view.current_provider is None
        assert view.current_model is None

    def test_single_model_change(self, make_fact) -> None:
        facts = [
            make_fact(
                FactType.MODEL_CHANGED,
                payload={"provider": "anthropic", "model": "claude-sonnet-4"},
            ),
        ]
        view = ConversationProjector().project(facts)
        assert len(view.model_changes) == 1
        assert view.model_changes[0]["provider"] == "anthropic"
        assert view.model_changes[0]["model"] == "claude-sonnet-4"
        assert view.current_provider == "anthropic"
        assert view.current_model == "claude-sonnet-4"

    def test_multiple_model_changes(self, make_fact) -> None:
        facts = [
            make_fact(
                FactType.MODEL_CHANGED,
                correlation_id="switch-1",
                payload={"provider": "anthropic", "model": "claude-sonnet-4"},
            ),
            make_fact(
                FactType.MODEL_CHANGED,
                correlation_id="switch-2",
                payload={"provider": "openai", "model": "gpt-4o"},
            ),
        ]
        view = ConversationProjector().project(facts)
        assert len(view.model_changes) == 2
        assert view.model_changes[0]["provider"] == "anthropic"
        assert view.model_changes[1]["provider"] == "openai"
        # Current is the most recent
        assert view.current_provider == "openai"
        assert view.current_model == "gpt-4o"

    def test_model_change_between_turns(self, make_fact) -> None:
        """Messages from both pre- and post-switch providers appear in conversation."""
        facts = [
            make_fact(FactType.USER_CONTENT_FINAL, correlation_id="turn-1", payload={"text": "hello from anthropic"}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="turn-1", payload={"text": "hi there"}),
            make_fact(FactType.TURN_COMPLETED, correlation_id="turn-1"),
            make_fact(
                FactType.MODEL_CHANGED,
                correlation_id="switch-1",
                payload={"provider": "openai", "model": "gpt-4o"},
            ),
            make_fact(FactType.USER_CONTENT_FINAL, correlation_id="turn-2", payload={"text": "hello from openai"}),
            make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="turn-2", payload={"text": "hello again"}),
            make_fact(FactType.TURN_COMPLETED, correlation_id="turn-2"),
        ]
        view = ConversationProjector().project(facts)
        # All messages from both providers are in the conversation
        assert len(view.messages) == 4
        assert view.messages[0].content == "hello from anthropic"
        assert view.messages[3].content == "hello again"
        # Model changes tracked
        assert len(view.model_changes) == 1
        assert view.current_provider == "openai"
        assert view.current_model == "gpt-4o"


# ===========================================================================
# D4: Protected state compatibility gating
# ===========================================================================


class TestProtectedStateCompatibility:
    """Protected replay state is included only when provider_key matches."""

    def _codec_with_key(self, key: bytes):
        class _FixedProvider:
            def key(self) -> bytes:
                return key

        from dana.core.session.protected_state import ProtectedStateCodec

        return ProtectedStateCodec(_FixedProvider())

    def test_incompatible_provider_excludes_replay_state(self, make_fact) -> None:
        """Protected state from a different provider is excluded from projection."""
        codec = self._codec_with_key(b"test-key-32-bytes-ok-for-testing!")
        ciphertext = codec.encrypt(b"anthropic-replay-state")
        facts = [
            make_fact(
                FactType.MODEL_CHANGED,
                payload={"provider": "anthropic", "model": "claude-sonnet-4"},
            ),
            make_fact(
                FactType.TURN_COMPLETED,
                correlation_id="turn-1",
                protected_payload=ciphertext,
            ),
            make_fact(
                FactType.MODEL_CHANGED,
                correlation_id="switch-1",
                payload={"provider": "openai", "model": "gpt-4o"},
            ),
        ]
        # Project with provider_key="openai" — anthropic's protected state is excluded
        view = ConversationProjector(protected_state_codec=codec).project(facts, provider_key="openai")
        assert view.replay_state is None

    def test_compatible_provider_includes_replay_state(self, make_fact) -> None:
        """Protected state from the current provider is included."""
        codec = self._codec_with_key(b"test-key-32-bytes-ok-for-testing!")
        ciphertext = codec.encrypt(b"openai-replay-state")
        facts = [
            make_fact(
                FactType.MODEL_CHANGED,
                payload={"provider": "openai", "model": "gpt-4o"},
            ),
            make_fact(
                FactType.TURN_COMPLETED,
                correlation_id="turn-1",
                protected_payload=ciphertext,
            ),
        ]
        view = ConversationProjector(protected_state_codec=codec).project(facts, provider_key="openai")
        assert view.replay_state == b"openai-replay-state"

    def test_no_provider_key_includes_all(self, make_fact) -> None:
        """Without provider_key, all protected state is included (pre-switch compatibility)."""
        codec = self._codec_with_key(b"test-key-32-bytes-ok-for-testing!")
        ciphertext = codec.encrypt(b"any-replay-state")
        facts = [
            make_fact(
                FactType.TURN_COMPLETED,
                correlation_id="turn-1",
                protected_payload=ciphertext,
            ),
        ]
        view = ConversationProjector(protected_state_codec=codec).project(facts, provider_key=None)
        assert view.replay_state == b"any-replay-state"

    def test_switch_to_same_provider_includes_replay_state(self, make_fact) -> None:
        """Switching to the same provider keeps replay state compatible."""
        codec = self._codec_with_key(b"test-key-32-bytes-ok-for-testing!")
        ciphertext = codec.encrypt(b"anthropic-replay-state")
        facts = [
            make_fact(
                FactType.MODEL_CHANGED,
                payload={"provider": "anthropic", "model": "claude-sonnet-4"},
            ),
            make_fact(
                FactType.TURN_COMPLETED,
                correlation_id="turn-1",
                protected_payload=ciphertext,
            ),
            make_fact(
                FactType.MODEL_CHANGED,
                correlation_id="switch-1",
                payload={"provider": "anthropic", "model": "claude-haiku-3"},
            ),
        ]
        view = ConversationProjector(protected_state_codec=codec).project(facts, provider_key="anthropic")
        assert view.replay_state == b"anthropic-replay-state"
