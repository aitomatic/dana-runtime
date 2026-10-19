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
import json

import pytest

from dana.common.llm.types import LLMMessage
from dana.core.session.models import FactType, JournalFact, OwnerScope
from dana.core.session.projections.conversation import (
    ConversationProjectionState,
    ConversationProjector,
    ConversationProviderMismatch,
    ConversationPrunedStateError,
)
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


def test_completed_tool_result_survives_interrupted_turn_without_inventing_pending_effect(make_fact):
    import json

    facts = [
        make_fact(FactType.USER_CONTENT_FINAL, payload={"text": "inspect files"}),
        make_fact(FactType.TOOL_REQUESTED, payload={"tool_call_id": "done", "tool_name": "read_file", "arguments": {"path": "a"}}),
        make_fact(FactType.TOOL_RESULT, payload={"tool_call_id": "done", "result": {"content": "verified contents"}}),
        make_fact(FactType.TOOL_REQUESTED, payload={"tool_call_id": "pending", "tool_name": "write_file", "arguments": {"path": "b"}}),
        make_fact(FactType.TURN_INTERRUPTED),
    ]
    view = ConversationProjector().project(facts)
    assert [message.role for message in view.messages] == ["user", "assistant", "tool"]
    assert view.messages[1].tool_calls == [{"id": "done", "name": "read_file", "arguments": {"path": "a"}}]
    assert view.messages[2].tool_call_id == "done"
    assert json.loads(view.messages[2].content) == {"content": "verified contents"}
    assert view.interruption_observation is not None


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_native_replay_is_committed_and_provider_scoped(make_fact, completed, provider):
    import json

    from dana.core.timeline.native_message import NativeMessage

    codec = _codec_with_key(b"k" * 32)
    reasoning = [{"type": "reasoning", "id": "rs_synthetic", "encrypted_content": "opaque-synthetic"}]
    envelope = {
        "schema": "dana.native-turn/1",
        "messages": [
            NativeMessage(role="user", content="question").to_dict(),
            NativeMessage(
                role="assistant",
                content="answer",
                metadata={"reasoning_items": reasoning, "fingerprint": "openai:gpt-5:test", "response_id": "resp_synthetic"},
            ).to_dict(),
        ],
    }
    facts = [
        make_fact(FactType.USER_CONTENT_FINAL, payload={"text": "question"}),
        make_fact(
            FactType.ASSISTANT_CONTENT_FINAL,
            payload={"text": "answer", "provider": "openai", "model": "gpt-5"},
            protected_payload=codec.encrypt(json.dumps(envelope).encode()),
        ),
    ]
    if completed:
        facts.append(make_fact(FactType.TURN_COMPLETED))
    view = ConversationProjector(codec).project(facts, provider_key=provider)
    assert [message.content for message in view.messages] == (["question", "answer"] if completed else ["question"])
    assert [message.reasoning_items for message in view.messages] == (
        [None, reasoning] if completed and provider == "openai" else [None] * len(view.messages)
    )
    assert (view.replay_state is not None) == (provider == "openai")


def test_protected_turn_completion_does_not_erase_interleaved_user_input(make_fact):
    import json

    codec = _codec_with_key(b"k" * 32)
    envelope = {
        "schema": "dana.native-turn/1",
        "messages": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "first answer"}],
    }
    facts = [
        make_fact(FactType.USER_CONTENT_FINAL, correlation_id="first", payload={"text": "first"}),
        make_fact(
            FactType.ASSISTANT_CONTENT_FINAL,
            correlation_id="first",
            payload={"text": "first answer"},
            protected_payload=codec.encrypt(json.dumps(envelope).encode()),
        ),
        make_fact(FactType.USER_CONTENT_FINAL, correlation_id="second", payload={"text": "second"}),
        make_fact(FactType.TURN_COMPLETED, correlation_id="first"),
    ]
    view = ConversationProjector(codec).project(facts)
    assert [message.content for message in view.messages] == ["first", "first answer", "second"]


# ===========================================================================
# 8. Incremental fold: batch boundaries are invisible
# ===========================================================================


def _rich_fact_stream(make_fact, codec) -> list[JournalFact]:
    """A stream exercising every cross-batch fold carrier at once.

    turn-1: user + a completed tool pair (its result arrives in a later batch
    than its request) + native envelope -> committed as a native turn.
    turn-2: user, interleaved BEFORE turn-1 completes, then turn-1's native
    replacement shrinks the vector ahead of it, then its own plain final.
    turn-3: model switch to another provider (gates the openai replay payload
    out), then a multimodal user block, then an interruption.
    """
    import json

    envelope_1 = {
        "schema": "dana.native-turn/1",
        "messages": [
            {"role": "user", "content": "u1"},
            {
                "role": "assistant",
                "content": "a1",
                "metadata": {"reasoning_items": [{"type": "reasoning", "id": "rs_1"}], "fingerprint": "openai:gpt-5:x"},
            },
        ],
    }
    envelope_2 = {
        "schema": "dana.native-turn/1",
        "messages": [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}],
    }
    envelope_3 = {
        "schema": "dana.native-turn/1",
        "messages": [{"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"}],
    }
    return [
        make_fact(FactType.USER_CONTENT_FINAL, correlation_id="turn-1", payload={"text": "u1"}),
        make_fact(
            FactType.TOOL_REQUESTED,
            correlation_id="turn-1",
            payload={"tool_call_id": "c1", "tool_name": "read_file", "arguments": {"path": "a"}},
        ),
        make_fact(FactType.USER_CONTENT_FINAL, correlation_id="turn-2", payload={"text": "u2"}),
        make_fact(FactType.TOOL_RESULT, correlation_id="turn-1", payload={"tool_call_id": "c1", "result": {"content": "ok"}}),
        make_fact(
            FactType.ASSISTANT_CONTENT_FINAL,
            correlation_id="turn-1",
            payload={"text": "a1", "provider": "openai", "model": "gpt-5"},
            protected_payload=codec.encrypt(json.dumps(envelope_1).encode()),
        ),
        make_fact(FactType.TURN_COMPLETED, correlation_id="turn-1"),
        make_fact(
            FactType.ASSISTANT_CONTENT_FINAL,
            correlation_id="turn-2",
            payload={"text": "a2", "provider": "openai", "model": "gpt-5"},
            protected_payload=codec.encrypt(json.dumps(envelope_2).encode()),
        ),
        make_fact(FactType.TURN_COMPLETED, correlation_id="turn-2"),
        make_fact(FactType.MODEL_CHANGED, correlation_id="switch-1", payload={"provider": "azure", "model": "gpt-5"}),
        make_fact(
            FactType.USER_CONTENT_FINAL,
            correlation_id="turn-3",
            payload={
                "text": "",
                "content_blocks": [
                    {"type": "text", "text": "see image"},
                    {"type": "image", "media_type": "image/png", "path": "/tmp/i.png"},
                ],
            },
        ),
        make_fact(
            FactType.ASSISTANT_CONTENT_FINAL,
            correlation_id="turn-3",
            payload={"text": "a3", "provider": "azure", "model": "gpt-5"},
            protected_payload=codec.encrypt(json.dumps(envelope_3).encode()),
        ),
        make_fact(FactType.TURN_INTERRUPTED, correlation_id="turn-3"),
    ]


class TestIncrementalFoldEquivalence:
    """Any split of the fact stream folds to exactly the one-shot projection."""

    def test_split_at_every_boundary_matches_one_shot(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        facts = _rich_fact_stream(make_fact, codec)
        one_shot = ConversationProjector(codec).project(facts, provider_key="azure")
        for cut in range(len(facts) + 1):
            projector = ConversationProjector(codec)
            state = projector.new_state(provider_key="azure")
            state.apply(facts[:cut])
            state.apply(facts[cut:])
            assert state.view() == one_shot, f"cut={cut}"
            assert state.last_sequence == len(facts)

    def test_single_fact_batches_match_one_shot(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        facts = _rich_fact_stream(make_fact, codec)
        one_shot = ConversationProjector(codec).project(facts, provider_key="azure")
        state = ConversationProjector(codec).new_state(provider_key="azure")
        for fact in facts:
            state.apply([fact])
        assert state.view() == one_shot

    def test_empty_delta_is_noop(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        facts = _rich_fact_stream(make_fact, codec)
        state = ConversationProjector(codec).new_state(provider_key="azure")
        state.apply(facts)
        cursor = state.last_sequence
        assert state.apply([]) == cursor
        assert state.view() == ConversationProjector(codec).project(facts, provider_key="azure")


class TestCrossBatchPendingState:
    """Pending final/tool/native state survives a batch boundary."""

    def test_final_in_one_batch_completes_in_the_next(self, make_fact) -> None:
        first = make_fact(FactType.USER_CONTENT_FINAL, correlation_id="t1", payload={"text": "q"})
        final = make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="t1", payload={"text": "answer"})
        done = make_fact(FactType.TURN_COMPLETED, correlation_id="t1")
        state = ConversationProjector().new_state()
        state.apply([first, final])
        assert [m.role for m in state.view().messages] == ["user"]  # uncommitted
        state.apply([done])
        assert [(m.role, m.content) for m in state.view().messages] == [("user", "q"), ("assistant", "answer")]

    def test_incomplete_tool_pair_completes_in_a_later_batch(self, make_fact) -> None:
        import json

        requested = make_fact(
            FactType.TOOL_REQUESTED,
            correlation_id="t1",
            payload={"tool_call_id": "c1", "tool_name": "read_file", "arguments": {"path": "a"}},
        )
        result = make_fact(FactType.TOOL_FAILURE, correlation_id="t1", payload={"tool_call_id": "c1", "error": "boom"})
        state = ConversationProjector().new_state()
        state.apply([requested])
        state.apply([])
        assert state.view().messages == ()  # no terminal fact yet: nothing projected
        state.apply([result])
        messages = state.view().messages
        assert [m.role for m in messages] == ["assistant", "tool"]
        assert messages[0].tool_calls == [{"id": "c1", "name": "read_file", "arguments": {"path": "a"}}]
        assert json.loads(messages[1].content) == {"status": "tool_failure", "error": "boom"}
        assert messages[1].tool_call_id == "c1"

    def test_native_envelope_arrives_before_its_terminal(self, make_fact) -> None:
        import json

        codec = _codec_with_key(b"k" * 32)
        envelope = {
            "schema": "dana.native-turn/1",
            "messages": [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}],
        }
        facts = [
            make_fact(FactType.USER_CONTENT_FINAL, correlation_id="t1", payload={"text": "u1"}),
            make_fact(FactType.USER_CONTENT_FINAL, correlation_id="t2", payload={"text": "u2"}),
            make_fact(
                FactType.ASSISTANT_CONTENT_FINAL,
                correlation_id="t1",
                payload={"text": "a1"},
                protected_payload=codec.encrypt(json.dumps(envelope).encode()),
            ),
        ]
        state = ConversationProjector(codec).new_state()
        state.apply(facts[:2])
        state.apply(facts[2:])
        state.apply([make_fact(FactType.TURN_COMPLETED, correlation_id="t1")])
        assert [m.content for m in state.view().messages] == ["u1", "a1", "u2"]

    def test_interruption_observation_carries_across_batches(self, make_fact) -> None:
        partial = make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="t1", payload={"text": "partial"})
        interrupted = make_fact(FactType.TURN_INTERRUPTED, correlation_id="t1")
        state = ConversationProjector().new_state()
        state.apply([partial])
        assert state.view().interruption_observation is None
        state.apply([interrupted])
        assert state.view().interruption_observation == "The previous turn was interrupted; do not assume unfinished effects completed."
        state.apply([make_fact(FactType.TURN_CANCELLED, correlation_id="t2")])
        assert state.view().interruption_observation is None

    def test_model_change_gating_carries_across_batches(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        payload = codec.encrypt(b"openai-replay")
        state = ConversationProjector(codec).new_state(provider_key="openai")
        state.apply([make_fact(FactType.MODEL_CHANGED, payload={"provider": "openai", "model": "gpt-5"})])
        state.apply([make_fact(FactType.TURN_COMPLETED, correlation_id="t1", protected_payload=payload)])
        assert state.view().replay_state == b"openai-replay"
        state.apply([make_fact(FactType.MODEL_CHANGED, payload={"provider": "azure", "model": "gpt-5"})])
        assert state.view().replay_state is None

    def test_native_replacement_falls_back_when_an_earlier_turn_shrank(self, make_fact) -> None:
        """A turn whose block shifted below its recorded start still folds correctly.

        turn-1 projects three messages (user + tool pair) but its native envelope
        carries two, so committing turn-1 shifts turn-2's user message one slot
        earlier than turn-2's recorded start. The suffix scan alone would miss it;
        the fold must fall back and match the one-shot projection.
        """
        import json

        codec = _codec_with_key(b"k" * 32)
        envelope_1 = {
            "schema": "dana.native-turn/1",
            "messages": [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}],
        }
        envelope_2 = {
            "schema": "dana.native-turn/1",
            "messages": [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}],
        }
        facts = [
            make_fact(FactType.USER_CONTENT_FINAL, correlation_id="t1", payload={"text": "u1"}),
            make_fact(FactType.TOOL_REQUESTED, correlation_id="t1", payload={"tool_call_id": "c1", "tool_name": "f", "arguments": {}}),
            make_fact(FactType.TOOL_RESULT, correlation_id="t1", payload={"tool_call_id": "c1", "result": {"ok": True}}),
            make_fact(FactType.USER_CONTENT_FINAL, correlation_id="t2", payload={"text": "u2"}),
            make_fact(
                FactType.ASSISTANT_CONTENT_FINAL,
                correlation_id="t1",
                payload={"text": "a1"},
                protected_payload=codec.encrypt(json.dumps(envelope_1).encode()),
            ),
            make_fact(FactType.TURN_COMPLETED, correlation_id="t1"),
            make_fact(
                FactType.ASSISTANT_CONTENT_FINAL,
                correlation_id="t2",
                payload={"text": "a2"},
                protected_payload=codec.encrypt(json.dumps(envelope_2).encode()),
            ),
            make_fact(FactType.TURN_COMPLETED, correlation_id="t2"),
        ]
        one_shot = ConversationProjector(codec).project(facts)
        state = ConversationProjector(codec).new_state()
        state.apply(facts)
        assert state.view() == one_shot
        assert [m.content for m in state.view().messages] == ["u1", "a1", "u2", "a2"]


class TestStateSerialization:
    """to_dict/from_dict round-trip the fold state without leaking plaintext replay."""

    def test_round_trip_reproduces_the_view(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        facts = _rich_fact_stream(make_fact, codec)
        state = ConversationProjector(codec).new_state(provider_key="azure")
        state.apply(facts)
        blob = state.to_dict()
        json.dumps(blob)  # must be JSON-safe
        restored = ConversationProjectionState.from_dict(blob, protected_state_codec=codec)
        assert restored.last_sequence == state.last_sequence
        assert restored.provider_key == "azure"
        assert restored.view() == state.view()

    def test_round_trip_is_json_stable_and_deterministic(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        facts = _rich_fact_stream(make_fact, codec)
        state = ConversationProjector(codec).new_state(provider_key="azure")
        state.apply(facts)
        once = json.dumps(state.to_dict(), sort_keys=True)
        twice = json.dumps(state.to_dict(), sort_keys=True)
        assert once == twice
        assert json.loads(once)["schema"] == "dana.conversation-projection/1"

    def test_serialization_never_carries_plaintext_replay(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        secret = b"RAW-REPLAY-SENTINEL"
        payload = codec.encrypt(secret)
        state = ConversationProjector(codec).new_state(provider_key="openai")
        state.apply(
            [
                make_fact(FactType.MODEL_CHANGED, payload={"provider": "openai", "model": "gpt-5"}),
                make_fact(FactType.TURN_COMPLETED, correlation_id="t1", protected_payload=payload),
            ]
        )
        blob = state.to_dict()
        assert state.view().replay_state == secret
        assert secret.decode() not in json.dumps(blob)
        # The ciphertext itself is preserved so a restored state can re-decrypt.
        assert blob["last_replay_ciphertext"] is not None
        restored = ConversationProjectionState.from_dict(blob, protected_state_codec=codec)
        assert restored.view().replay_state == secret

    def test_resume_folds_the_tail_from_the_restored_cursor(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        facts = _rich_fact_stream(make_fact, codec)
        head, tail = facts[:6], facts[6:]
        state = ConversationProjector(codec).new_state(provider_key="azure")
        state.apply(head)
        restored = ConversationProjectionState.from_dict(state.to_dict(), protected_state_codec=codec)
        assert restored.last_sequence == head[-1].sequence
        restored.apply(tail)
        assert restored.view() == ConversationProjector(codec).project(facts, provider_key="azure")

    def test_unknown_schema_is_rejected(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        blob = state.to_dict()
        blob["schema"] = "dana.conversation-projection/999"
        with pytest.raises(ValueError, match="unsupported conversation projection state schema"):
            ConversationProjectionState.from_dict(blob)

    def test_missing_schema_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported conversation projection state schema"):
            ConversationProjectionState.from_dict({})

    def test_malformed_fields_are_rejected(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        state.apply(_completed_turn(make_fact, correlation_id="t1", user_text="q", assistant_text="a"))
        blob = state.to_dict()
        for field, broken in (
            ("last_sequence", "nope"),
            ("message_turns", ["only-one"]),
            ("messages", 7),
            ("model_changes", [1]),
            ("pending_final", {"t": 3}),
            ("open_turns", [7]),
        ):
            damaged = dict(blob)
            damaged[field] = broken
            with pytest.raises(ValueError):
                ConversationProjectionState.from_dict(damaged)

    def test_restored_state_without_codec_keeps_ciphertext_only(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        payload = codec.encrypt(b"replay")
        state = ConversationProjector(codec).new_state(provider_key="openai")
        state.apply(
            [
                make_fact(FactType.MODEL_CHANGED, payload={"provider": "openai", "model": "gpt-5"}),
                make_fact(FactType.TURN_COMPLETED, correlation_id="t1", protected_payload=payload),
            ]
        )
        assert state.view().replay_state == b"replay"
        restored = ConversationProjectionState.from_dict(state.to_dict())
        assert restored.view().replay_state is None
        assert restored.provider_key == "openai"

    def test_view_rejects_a_conflicting_provider_key(self, make_fact) -> None:
        state = ConversationProjector().new_state(provider_key="openai")
        with pytest.raises(ConversationProviderMismatch):
            state.view(provider_key="azure")
        # The matching key, or no re-gating request, is accepted.
        assert state.view(provider_key="openai").messages == ()
        assert state.view().messages == ()

    def test_projector_project_equals_new_state_fold(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        facts = _rich_fact_stream(make_fact, codec)
        projector = ConversationProjector(codec)
        state = projector.new_state(provider_key="azure")
        state.apply(facts)
        assert projector.project(facts, provider_key="azure") == state.view()


class TestBoundedStateAccess:
    """Checkpoint bookkeeping must not require materializing the full view."""

    def test_message_count_and_messages_after(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        state.apply(_completed_turn(make_fact, correlation_id="t1", user_text="q1", assistant_text="a1"))
        boundary = state.message_count
        assert boundary == 2
        assert state.messages_after(boundary) == ()

        state.apply(_completed_turn(make_fact, correlation_id="t2", user_text="q2", assistant_text="a2"))
        tail = state.messages_after(boundary)
        assert [m.content for m in tail] == ["q2", "a2"]
        # The prefix boundary is stable: later completed turns append after it.
        assert [m.content for m in state.messages_after(0)] == ["q1", "a1", "q2", "a2"]
        assert state.messages_after(state.message_count) == ()
        assert state.message_count == 4

    def test_accessors_match_the_view(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        facts = _rich_fact_stream(make_fact, codec)
        state = ConversationProjector(codec).new_state(provider_key="azure")
        state.apply(facts)
        view = state.view()
        assert state.message_count == len(view.messages)
        assert state.interruption_observation == view.interruption_observation
        assert state.current_provider == view.current_provider
        assert state.current_model == view.current_model
        assert state.last_sequence == view.last_sequence

    def test_open_correlation_ids_track_unterminated_work(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        assert state.open_correlation_ids == ()
        state.apply([make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="t1", payload={"text": "partial"})])
        assert state.open_correlation_ids == ("t1",)
        state.apply(
            [
                make_fact(FactType.TOOL_REQUESTED, correlation_id="t2", payload={"tool_call_id": "c1", "tool_name": "f", "arguments": {}}),
                make_fact(FactType.USER_CONTENT_FINAL, correlation_id="t3", payload={"text": "u3"}),
            ]
        )
        assert state.open_correlation_ids == ("t1", "t2", "t3")
        state.apply([make_fact(FactType.TURN_COMPLETED, correlation_id="t1")])
        state.apply([make_fact(FactType.TOOL_RESULT, correlation_id="t2", payload={"tool_call_id": "c1", "result": {}})])
        # A tool terminal closes the TOOL CALL, not the turn: t2 is still in
        # flight until its own TURN_COMPLETED arrives.
        assert state.open_correlation_ids == ("t2", "t3")
        state.apply([make_fact(FactType.TURN_COMPLETED, correlation_id="t2")])
        assert state.open_correlation_ids == ("t3",)
        state.apply([make_fact(FactType.TURN_CANCELLED, correlation_id="t3")])
        assert state.open_correlation_ids == ()

    def test_user_only_turn_is_open(self, make_fact) -> None:
        """A turn with nothing but its user fact is open: a native envelope can still rewrite it."""
        state = ConversationProjector().new_state()
        state.apply([make_fact(FactType.TURN_STARTED, correlation_id="t1", payload={})])
        assert state.open_correlation_ids == ("t1",)
        state.apply([make_fact(FactType.USER_CONTENT_FINAL, correlation_id="t1", payload={"text": "u1"})])
        assert state.open_correlation_ids == ("t1",)
        assert state.open_correlation_ids == tuple(sorted(state.open_correlation_ids))

    def test_open_turns_round_trip_through_serialization(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        state.apply(
            [
                make_fact(FactType.USER_CONTENT_FINAL, correlation_id="done", payload={"text": "u"}),
                make_fact(FactType.TURN_COMPLETED, correlation_id="done"),
                make_fact(FactType.USER_CONTENT_FINAL, correlation_id="inflight", payload={"text": "u2"}),
            ]
        )
        assert state.open_correlation_ids == ("inflight",)
        restored = ConversationProjectionState.from_dict(state.to_dict()).open_correlation_ids
        assert restored == ("inflight",)

    def test_message_objects_are_shared_not_copied(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        state.apply(_completed_turn(make_fact, correlation_id="t1", user_text="q1", assistant_text="a1"))
        tail = state.messages_after(0)
        assert state.view().messages[0] is tail[0]

    def test_prefix_survives_pending_final_then_terminal(self, make_fact) -> None:
        """A completed turn's messages are never restructured by later facts."""
        codec = _codec_with_key(b"k" * 32)
        state = ConversationProjector(codec).new_state()
        state.apply(_completed_turn(make_fact, correlation_id="t1", user_text="q1", assistant_text="a1"))
        boundary = state.message_count
        prefix = state.messages_after(0)
        # A later turn, in every shape (plain, native, multimodal, interrupted).
        state.apply(
            [
                make_fact(
                    FactType.USER_CONTENT_FINAL,
                    correlation_id="t2",
                    payload={"text": "u2", "content_blocks": [{"type": "text", "text": "u2"}]},
                ),
                make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="t2", payload={"text": "a2"}),
                make_fact(FactType.TURN_COMPLETED, correlation_id="t2"),
                make_fact(FactType.TURN_INTERRUPTED, correlation_id="t3"),
            ]
        )
        assert state.messages_after(0)[:boundary] == prefix
        tail = state.messages_after(boundary)
        assert tail[0].content == [{"type": "text", "text": "u2"}]  # multimodal block list
        assert tail[1].content == "a2"
        assert len(tail) == 2


class TestRewriteRevision:
    """The ephemeral revision is an O(delta) invalidation signal for mirrored contexts."""

    def test_pure_appends_do_not_bump_the_revision(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        state.apply([make_fact(FactType.USER_CONTENT_FINAL, correlation_id="t1", payload={"text": "u1"})])
        baseline = state.rewrite_revision
        # Plain completed turn: append only (final + completed in one batch).
        state.apply(
            [
                make_fact(FactType.ASSISTANT_CONTENT_FINAL, correlation_id="t1", payload={"text": "a1"}),
                make_fact(FactType.TURN_COMPLETED, correlation_id="t1"),
            ]
        )
        assert state.rewrite_revision == baseline
        # Many user-only pending turns, still append only.
        state.apply([make_fact(FactType.USER_CONTENT_FINAL, correlation_id=f"p{i}", payload={"text": "x"}) for i in range(5)])
        assert state.rewrite_revision == baseline
        assert state.message_count == 7

    def test_native_final_without_a_user_fact_falls_back_without_bumping(self, make_fact) -> None:
        """A native envelope with no folded user fact has no turn start to rewrite."""
        codec = _codec_with_key(b"k" * 32)
        envelope = {"schema": "dana.native-turn/1", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}
        state = ConversationProjector(codec).new_state()
        baseline = state.rewrite_revision
        state.apply(
            [
                make_fact(
                    FactType.ASSISTANT_CONTENT_FINAL,
                    correlation_id="t1",
                    payload={"text": "a"},
                    protected_payload=codec.encrypt(json.dumps(envelope).encode()),
                ),
                make_fact(FactType.TURN_COMPLETED, correlation_id="t1"),
            ]
        )
        # No user fact -> no turn start -> the pending final text is used as-is.
        assert state.rewrite_revision == baseline
        assert [m.content for m in state.view().messages] == ["a"]

    def test_replacing_messages_from_a_previous_batch_bumps_the_revision(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        envelope = {"schema": "dana.native-turn/1", "messages": [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]}
        state = ConversationProjector(codec).new_state()
        # The user fact is folded in its own batch, so its message predates the
        # native commit batch.
        state.apply([make_fact(FactType.USER_CONTENT_FINAL, correlation_id="t1", payload={"text": "u1"})])
        baseline = state.rewrite_revision
        state.apply(
            [
                make_fact(
                    FactType.ASSISTANT_CONTENT_FINAL,
                    correlation_id="t1",
                    payload={"text": "a1"},
                    protected_payload=codec.encrypt(json.dumps(envelope).encode()),
                ),
                make_fact(FactType.TURN_COMPLETED, correlation_id="t1"),
            ]
        )
        assert state.rewrite_revision == baseline + 1
        assert [m.content for m in state.view().messages] == ["u1", "a1"]

    def test_same_batch_native_replacement_does_not_bump(self, make_fact) -> None:
        """A whole turn folded in one batch is pure tail: no revision change."""
        codec = _codec_with_key(b"k" * 32)
        envelope = {"schema": "dana.native-turn/1", "messages": [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]}
        state = ConversationProjector(codec).new_state()
        baseline = state.rewrite_revision
        state.apply(
            [
                make_fact(FactType.USER_CONTENT_FINAL, correlation_id="t1", payload={"text": "u1"}),
                make_fact(
                    FactType.ASSISTANT_CONTENT_FINAL,
                    correlation_id="t1",
                    payload={"text": "a1"},
                    protected_payload=codec.encrypt(json.dumps(envelope).encode()),
                ),
                make_fact(FactType.TURN_COMPLETED, correlation_id="t1"),
            ]
        )
        assert state.rewrite_revision == baseline
        assert [m.content for m in state.view().messages] == ["u1", "a1"]

    def test_discard_bumps_the_revision_and_from_dict_resets_it(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        state.apply(_completed_turn(make_fact, correlation_id="t1", user_text="q", assistant_text="a"))
        baseline = state.rewrite_revision
        state.discard_closed_history()
        assert state.rewrite_revision == baseline + 1
        # Ephemeral: not serialized, and a restored state starts fresh.
        restored = ConversationProjectionState.from_dict(state.to_dict())
        assert restored.rewrite_revision == 0
        assert "rewrite_revision" not in state.to_dict()

    def test_empty_discard_does_not_bump(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        baseline = state.rewrite_revision
        assert state.discard_closed_history() == 0
        assert state.rewrite_revision == baseline


class TestDiscardClosedHistory:
    """discard_closed_history bounds the retained state without breaking the fold."""

    def test_refuses_while_a_turn_is_open(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        state.apply([make_fact(FactType.USER_CONTENT_FINAL, correlation_id="t1", payload={"text": "u1"})])
        with pytest.raises(ValueError, match="cannot discard history while turns are open"):
            state.discard_closed_history()
        # Nothing was dropped by the refused call.
        assert state.message_count == 1
        assert state.retained_message_count == 1

    def test_drops_retained_messages_and_keeps_canonical_count(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        state.apply(_completed_turn(make_fact, correlation_id="t1", user_text="q1", assistant_text="a1"))
        state.apply(_completed_turn(make_fact, correlation_id="t2", user_text="q2", assistant_text="a2"))
        assert state.message_count == 4
        assert state.discard_closed_history() == 4
        assert state.retained_message_count == 0
        assert state.message_count == 4  # canonical count survives the prune
        assert state.discarded_prefix == 4
        assert state.messages_after(4) == ()

    def test_folds_only_the_new_tail_after_a_prune(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        state.apply(_completed_turn(make_fact, correlation_id="t1", user_text="q1", assistant_text="a1"))
        boundary = state.message_count
        state.discard_closed_history()
        state.apply(_completed_turn(make_fact, correlation_id="t2", user_text="q2", assistant_text="a2"))
        assert [m.content for m in state.messages_after(boundary)] == ["q2", "a2"]
        assert state.retained_message_count == 2
        assert state.message_count == 4

    def test_pruned_serialization_is_bounded(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        facts = _rich_fact_stream(make_fact, codec)
        full = ConversationProjector(codec).new_state(provider_key="azure")
        full.apply(facts)
        state = ConversationProjector(codec).new_state(provider_key="azure")
        state.apply(facts)
        state.discard_closed_history()
        blob = state.to_dict()
        assert blob["messages"] == []
        assert blob["message_turns"] == []
        assert blob["turn_starts"] == {}
        assert blob["discarded_prefix"] == full.message_count
        # Cursor, provider identity and replay ciphertext all survive the prune.
        assert blob["last_sequence"] == full.last_sequence
        assert blob["provider_key"] == "azure"
        assert blob["last_replay_ciphertext"] == full.to_dict()["last_replay_ciphertext"]
        assert len(json.dumps(blob)) < len(json.dumps(full.to_dict()))

    def test_pruned_state_round_trips_and_keeps_folding_the_tail(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        facts = _rich_fact_stream(make_fact, codec)
        state = ConversationProjector(codec).new_state(provider_key="azure")
        state.apply(facts)
        boundary = state.message_count
        state.discard_closed_history()
        restored = ConversationProjectionState.from_dict(state.to_dict(), protected_state_codec=codec)
        assert restored.message_count == boundary
        assert restored.discarded_prefix == boundary
        assert restored.open_correlation_ids == ()
        tail = _completed_turn(make_fact, correlation_id="t9", user_text="q9", assistant_text="a9")
        restored.apply(tail)
        assert [m.content for m in restored.messages_after(boundary)] == ["q9", "a9"]

    def test_pruned_state_keeps_provider_gating(self, make_fact) -> None:
        codec = _codec_with_key(b"k" * 32)
        payload = codec.encrypt(b"openai-replay")
        state = ConversationProjector(codec).new_state(provider_key="openai")
        state.apply(
            [
                make_fact(FactType.MODEL_CHANGED, payload={"provider": "openai", "model": "gpt-5"}),
                make_fact(FactType.TURN_COMPLETED, correlation_id="t1", protected_payload=payload),
            ]
        )
        state.discard_closed_history()
        assert state.view(allow_partial=True).replay_state == b"openai-replay"
        # A provider switch after the prune still gates the retained ciphertext out.
        state.apply([make_fact(FactType.MODEL_CHANGED, payload={"provider": "azure", "model": "gpt-5"})])
        assert state.view(allow_partial=True).replay_state is None

    def test_view_refuses_the_full_history_after_a_prune(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        state.apply(_completed_turn(make_fact, correlation_id="t1", user_text="q", assistant_text="a"))
        state.discard_closed_history()
        with pytest.raises(ConversationPrunedStateError, match="dropped 2 closed messages"):
            state.view()
        assert [m.content for m in state.view(allow_partial=True).messages] == []

    def test_model_change_log_is_bounded_after_a_prune(self, make_fact) -> None:
        state = ConversationProjector().new_state()
        for index in range(5):
            state.apply(
                [make_fact(FactType.MODEL_CHANGED, correlation_id=f"m{index}", payload={"provider": "openai", "model": f"gpt-{index}"})]
            )
        assert len(state.view().model_changes) == 5
        state.discard_closed_history()
        changes = state.view(allow_partial=True).model_changes
        assert len(changes) == 1 and changes[0]["model"] == "gpt-4"
        assert state.current_provider == "openai" and state.current_model == "gpt-4"
