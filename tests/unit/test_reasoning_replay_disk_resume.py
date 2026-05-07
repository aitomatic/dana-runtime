"""End-to-end test for disk-resume reasoning replay.

The Phase 5 live verify ran a single in-process session. This test closes the
gap by exercising the *resume from disk* path:

    process A: capture → persist → process exits
    process B: load timeline.json → build LLMMessage[] → provider input[]

For this to work, replay state must round-trip through:
  TimelineEntry.metadata  (legacy load path)
  NativeMessage.metadata  (native load path)
  → LLMMessage.reasoning_items / .reasoning_fingerprint
  → openai_messages dict carrier keys
  → splice in _convert_to_responses_input

The test validates both load formats (legacy entries and native messages).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from dana.common.llm.providers.openai_compatible_base import OpenAICompatibleProvider
from dana.core.timeline.compressed_timeline import CompressedTimeline
from dana.core.timeline.native_message import NativeMessage
from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType


_REASONING_ITEM = {
    "type": "reasoning",
    "id": "rs_disk_001",
    "summary": [{"type": "summary_text", "text": "thought from prior process."}],
    "encrypted_content": "enc_disk_blob",
}

FINGERPRINT = "azure:gpt-5:f3b74bde"


def _make_provider(fingerprint=FINGERPRINT):
    """Same shape used in the in-process tests."""
    p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    p.model = "gpt-5"
    p.client = MagicMock()
    p._use_responses_api = True
    p._include_unsupported = False
    type(p).fingerprint = property(lambda self, fp=fingerprint: fp)
    return p


def _build_persisted_entries() -> list[TimelineEntry]:
    """Simulate what process A's timeline looked like before it exited:
    a user message, a reasoning entry with metadata, and a response."""
    user = TimelineEntry(
        entry_type=TimelineEntryType.USER_MESSAGE,
        content="Solve the gold-box puzzle.",
    )
    thoughts = TimelineEntry(
        entry_type=TimelineEntryType.AGENT_THOUGHTS,
        content="The gold is in B because exactly one label is true.",
        metadata={
            "reasoning_items": [_REASONING_ITEM],
            "fingerprint": FINGERPRINT,
            "response_id": "resp_001",
        },
    )
    response = TimelineEntry(
        entry_type=TimelineEntryType.AGENT_RESPONSE,
        content="The gold is in box B.",
    )
    return [user, thoughts, response]


class TestLegacyFormatDiskResume:
    """Process A persists TimelineEntry dicts; process B loads them via
    load_timeline(entries=[...dicts...])."""

    def test_full_chain_replays_after_disk_load(self):
        # Process A: build timeline, persist as dicts (simulates timeline.json)
        entries = _build_persisted_entries()
        persisted = [e.to_dict() for e in entries]

        # Process B: fresh CompressedTimeline, load from persisted dicts
        timeline = CompressedTimeline()
        timeline.load_from_entries(entries=persisted)  # type: ignore[arg-type]

        # Build LLMMessages — must carry reasoning_items
        llm_messages = timeline.to_llm_messages()
        thoughts_msg = next(m for m in llm_messages if m.role == "assistant" and m.reasoning_items)
        assert thoughts_msg.reasoning_items == [_REASONING_ITEM]
        assert thoughts_msg.reasoning_fingerprint == FINGERPRINT
        assert thoughts_msg.response_id == "resp_001"

        # Provider must splice raw items into Responses API input[]
        provider = _make_provider(fingerprint=FINGERPRINT)
        # Simulate process B sending a new turn: append a fresh user message
        from dana.common.llm.types import LLMMessage

        full_messages = list(llm_messages) + [LLMMessage(role="user", content="What if B's label changed?")]

        _, openai_messages = provider.prepare_messages(full_messages)
        result = provider._convert_to_responses_input(openai_messages)

        # Order: user → reasoning_item → assistant → user (new turn)
        types_or_roles = [r.get("type") or r.get("role") for r in result]
        assert "reasoning" in types_or_roles
        # The reasoning item we expect is the one persisted from process A
        reasoning_idx = types_or_roles.index("reasoning")
        assert result[reasoning_idx]["id"] == "rs_disk_001"
        assert result[reasoning_idx]["encrypted_content"] == "enc_disk_blob"


class TestNativeFormatDiskResume:
    """Process A persists NativeMessage dicts; process B loads them via
    load_timeline(entries=[{role: ..., metadata: ...}, ...]) — the native path
    detected by ``role in first_entry and not type``."""

    def test_native_format_round_trip_preserves_replay_state(self):
        # Build NativeMessage equivalent of persisted state
        nm_user = NativeMessage(role="user", content="Solve the puzzle.")
        nm_thoughts = NativeMessage(
            role="assistant",
            content="The gold is in B...",
            metadata={
                "reasoning_items": [_REASONING_ITEM],
                "fingerprint": FINGERPRINT,
                "response_id": "resp_001",
            },
        )
        nm_response = NativeMessage(role="assistant", content="The gold is in box B.")

        persisted_native = [nm_user.to_dict(), nm_thoughts.to_dict(), nm_response.to_dict()]

        # Process B: fresh CompressedTimeline, load from native-format dicts
        timeline = CompressedTimeline()
        timeline.load_from_entries(entries=persisted_native)  # type: ignore[arg-type]

        # NativeMessage round-trip must preserve metadata
        thoughts_native = next(m for m in timeline.native_messages if m.metadata.get("reasoning_items"))
        assert thoughts_native.metadata["reasoning_items"] == [_REASONING_ITEM]
        assert thoughts_native.metadata["fingerprint"] == FINGERPRINT

        # to_llm_messages → LLMMessage with reasoning fields populated
        llm_messages = timeline.to_llm_messages()
        thoughts_llm = next(m for m in llm_messages if m.role == "assistant" and m.reasoning_items)
        assert thoughts_llm.reasoning_items == [_REASONING_ITEM]
        assert thoughts_llm.reasoning_fingerprint == FINGERPRINT


class TestFingerprintMismatchAfterResume:
    """Loaded a timeline captured by a *different* provider (e.g. Azure → OpenAI
    migration). Replay must be skipped — items stay in metadata for diagnostics
    but never reach input[]."""

    def test_cross_provider_resume_does_not_replay(self):
        entries = _build_persisted_entries()  # captured under FINGERPRINT (azure)
        persisted = [e.to_dict() for e in entries]

        timeline = CompressedTimeline()
        timeline.load_from_entries(entries=persisted)  # type: ignore[arg-type]

        # Provider with a DIFFERENT fingerprint (simulates migration / wrong client)
        provider = _make_provider(fingerprint="openai:gpt-5:99999999")

        from dana.common.llm.types import LLMMessage

        full_messages = list(timeline.to_llm_messages()) + [LLMMessage(role="user", content="Continue.")]
        _, openai_messages = provider.prepare_messages(full_messages)
        result = provider._convert_to_responses_input(openai_messages)

        types_or_roles = [r.get("type") or r.get("role") for r in result]
        # No reasoning item spliced; carriers stripped from assistant messages
        assert "reasoning" not in types_or_roles
        for entry in result:
            assert "_reasoning_items" not in entry
            assert "_reasoning_fingerprint" not in entry


class TestKillSwitchAfterDiskResume:
    """Operator can flip LLM_REASONING_REPLAY=0 after the timeline was
    captured — replay must be inert even when fingerprint matches and items
    exist on disk."""

    def test_kill_switch_disables_replay_on_resumed_timeline(self, monkeypatch):
        monkeypatch.setenv("LLM_REASONING_REPLAY", "0")

        entries = _build_persisted_entries()
        persisted = [e.to_dict() for e in entries]
        timeline = CompressedTimeline()
        timeline.load_from_entries(entries=persisted)  # type: ignore[arg-type]

        provider = _make_provider(fingerprint=FINGERPRINT)

        from dana.common.llm.types import LLMMessage

        full_messages = list(timeline.to_llm_messages()) + [LLMMessage(role="user", content="Continue.")]
        _, openai_messages = provider.prepare_messages(full_messages)
        result = provider._convert_to_responses_input(openai_messages)

        types_or_roles = [r.get("type") or r.get("role") for r in result]
        assert "reasoning" not in types_or_roles  # kill switch wins over fingerprint match
