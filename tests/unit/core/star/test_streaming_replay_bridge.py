"""Streaming replay-bridge tests: terminal carrier -> timeline metadata -> legacy snapshot.

Three seams are covered without network or live models:

1. ``_think_stream`` consumes ``LLMStreamChunk(type="response", response=...)`` and
   preserves its ``reasoning_items``/``response_id`` through the real runtime
   parser into ``_record_think_results`` -> ``TimelineEntry.metadata``.
2. Providers that emit no terminal carrier keep their existing delta-derived
   behavior (text still becomes the response; no metadata is invented).
3. Actual plain and compacted legacy snapshots withhold opaque replay carriers
   for journal-owned sessions without mutating live provider context.

The marker is deliberately class-agnostic: tests set it on the agent instance
(the same way ``AgentSession`` does) and never patch a shared class attribute.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dana.common.llm.types import LLMResponse, LLMStreamChunk
from dana.core.timeline.compressed_timeline import CompressedTimeline
from dana.core.timeline.timeline import Timeline, TimelineEntry, TimelineEntryType


_REASONING_ITEM = {
    "type": "reasoning",
    "id": "rs_stream_001",
    "summary": [{"type": "summary_text", "text": "Checked the options."}],
    "encrypted_content": "opaque_stream_blob",
}


def _make_agent():
    """Minimal STARAgent with the real codec runtime, no registry/LLM side effects."""
    from dana.core.agent.star_agent import STARAgent

    return STARAgent(
        agent_type="test-agent",
        auto_register=False,
        enable_skills=False,
        enable_web_search=False,
        enable_code_execution=False,
        enable_assistant=False,
        compress_timeline=False,
    )


def _wire_stream(agent, chunks, parse_side_effects=False):
    """Point the agent's real runtime at a scripted chunk stream.

    Only the stream source and the prompt builder are replaced; ``parse_response``,
    ``validate_done_output``, and ``_record_think_results`` stay real so the test
    exercises the actual bridge rather than a mock of it.
    """

    async def _fake_stream(_messages):
        for chunk in chunks:
            yield chunk

    agent._runtime.build_prompt = MagicMock(return_value=[])
    agent._runtime._llm_caller = MagicMock()
    agent._runtime._llm_caller.call_llm_stream = _fake_stream
    return agent


async def _run_think_stream(agent):
    result_holder: dict = {}
    trace_percepts = {"timeline": agent._timeline}
    events = []
    with patch.object(agent, "_maybe_compress_timeline_async", new=AsyncMock()):
        async for event in agent._think_stream(trace_percepts, result_holder):
            events.append(event)
    return events, result_holder


def _entry_type_value(entry) -> str:
    """Compare by ``.value`` since the same enum can load under two module paths."""
    return entry.entry_type.value if hasattr(entry.entry_type, "value") else str(entry.entry_type)


def _thoughts_entries(agent):
    return [entry for entry in agent._timeline.timeline if _entry_type_value(entry) == "agent_thoughts"]


class TestTerminalCarrierReachesTimelineMetadata:
    """The carrier's non-delta fields survive into the persisted timeline entry."""

    @pytest.mark.asyncio
    async def test_reasoning_items_and_response_id_reach_metadata(self):
        agent = _make_agent()
        _wire_stream(
            agent,
            [
                LLMStreamChunk(type="text_delta", content="42"),
                LLMStreamChunk(
                    type="response",
                    response=LLMResponse(
                        content="42",
                        model="gpt-5-nano",
                        finish_reason="stop",
                        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                        reasoning_items=[dict(_REASONING_ITEM)],
                        response_id="resp_stream_001",
                    ),
                ),
            ],
        )

        events, result_holder = await _run_think_stream(agent)

        thoughts = _thoughts_entries(agent)
        assert len(thoughts) == 1
        metadata = thoughts[0].metadata
        assert metadata["reasoning_items"] == [_REASONING_ITEM]
        assert metadata["response_id"] == "resp_stream_001"
        # Fingerprint is resolved from the provider, absent here -> present but None.
        assert "fingerprint" in metadata
        # Same state must be visible where AgentSession reads it (native messages),
        # since that copy is what the journal payload is built from.
        native_thoughts = next(m for m in agent._timeline.native_messages if m.metadata.get("reasoning_items"))
        assert native_thoughts.metadata["reasoning_items"] == [_REASONING_ITEM]
        assert native_thoughts.metadata["response_id"] == "resp_stream_001"
        # The turn still completed normally for the host.
        assert [event.event_type.value for event in events].count("text_delta") == 1
        assert result_holder["trace_thoughts"]["trace_thoughts"]["response"] == "42"

    @pytest.mark.asyncio
    async def test_tool_call_turn_keeps_reasoning_alongside_tools(self):
        agent = _make_agent()
        _wire_stream(
            agent,
            [
                LLMStreamChunk(type="thinking", content="Let me look."),
                LLMStreamChunk(type="tool_use", tool_call={"id": "call_1", "name": "read_file", "input": {"path": "a"}}),
                LLMStreamChunk(
                    type="response",
                    response=LLMResponse(
                        content="",
                        model="gpt-5-nano",
                        finish_reason="tool_calls",
                        reasoning_items=[dict(_REASONING_ITEM)],
                        response_id="resp_tool_turn",
                    ),
                ),
            ],
        )

        await _run_think_stream(agent)

        thoughts = _thoughts_entries(agent)
        assert thoughts, "reasoning entry must persist on a tool-call turn"
        assert thoughts[0].metadata["reasoning_items"] == [_REASONING_ITEM]
        assert thoughts[0].metadata["response_id"] == "resp_tool_turn"
        assert thoughts[0].content == "Let me look."

    @pytest.mark.asyncio
    async def test_carrier_content_does_not_duplicate_streamed_text(self):
        """Text arrives as deltas; the carrier must not append it a second time."""
        agent = _make_agent()
        _wire_stream(
            agent,
            [
                LLMStreamChunk(type="text_delta", content="Hello "),
                LLMStreamChunk(type="text_delta", content="world"),
                LLMStreamChunk(type="response", response=LLMResponse(content="Hello world", model="m", finish_reason="stop")),
            ],
        )

        _, result_holder = await _run_think_stream(agent)

        assert result_holder["trace_thoughts"]["trace_thoughts"]["response"] == "Hello world"
        responses = [e for e in agent._timeline.timeline if _entry_type_value(e) == "agent_response"]
        assert len(responses) == 1
        assert responses[0].content == "Hello world"

    @pytest.mark.asyncio
    async def test_carrier_without_deltas_still_produces_a_response(self):
        """A provider that only reports the final object must not lose the answer."""
        agent = _make_agent()
        _wire_stream(
            agent,
            [
                LLMStreamChunk(
                    type="response",
                    response=LLMResponse(content="final answer", model="m", finish_reason="stop"),
                ),
            ],
        )

        _, result_holder = await _run_think_stream(agent)

        assert result_holder["trace_thoughts"]["trace_thoughts"]["response"] == "final answer"


class TestLegacyProviderWithoutCarrier:
    """No terminal carrier -> pre-existing delta-derived behavior, unchanged."""

    @pytest.mark.asyncio
    async def test_text_only_stream_has_no_replay_metadata(self):
        agent = _make_agent()
        _wire_stream(agent, [LLMStreamChunk(type="text_delta", content="just text")])

        _, result_holder = await _run_think_stream(agent)

        assert result_holder["trace_thoughts"]["trace_thoughts"]["response"] == "just text"
        assert _thoughts_entries(agent) == []

    @pytest.mark.asyncio
    async def test_none_payload_carrier_is_ignored(self):
        agent = _make_agent()
        _wire_stream(
            agent,
            [
                LLMStreamChunk(type="text_delta", content="plain"),
                LLMStreamChunk(type="response", response=None),
            ],
        )

        _, result_holder = await _run_think_stream(agent)

        assert result_holder["trace_thoughts"]["trace_thoughts"]["response"] == "plain"
        assert _thoughts_entries(agent) == []

    @pytest.mark.asyncio
    async def test_tool_call_stream_keeps_delta_derived_finish_reason(self):
        agent = _make_agent()
        _wire_stream(
            agent,
            [LLMStreamChunk(type="tool_use", tool_call={"id": "call_9", "name": "noop", "input": {}})],
        )

        _, result_holder = await _run_think_stream(agent)

        assert result_holder["trace_thoughts"]["trace_thoughts"]["tool_calls"]


@pytest.mark.parametrize("journal_owned", [False, True])
@pytest.mark.parametrize(("compressed", "compacted"), [(False, False), (True, False), (True, True)])
def test_legacy_snapshot_keeps_replay_only_for_legacy_owners(tmp_path, journal_owned, compressed, compacted):
    from datetime import datetime

    from dana.config.storage_config import FileStorageConfig
    from dana.repositories.local_file_repository import LocalTimelineRepository

    agent = _make_agent()
    agent._journal_owned_protected_replay = journal_owned
    repository = LocalTimelineRepository(FileStorageConfig(workspace_folder=str(tmp_path)), agent)
    timeline = CompressedTimeline(max_context_tokens=4000, agent=None, compression_enabled=False) if compressed else Timeline()
    timeline._agent = agent
    timeline._repository = repository
    entry = TimelineEntry(
        entry_type=TimelineEntryType.AGENT_THOUGHTS,
        content="Checked the options.",
        metadata={
            "reasoning_items": [_REASONING_ITEM],
            "fingerprint": "openai:gpt-5:abcd1234",
            "response_id": "resp_stream_001",
        },
    )
    timeline.add_entry(entry)
    if compressed and compacted:
        timeline._last_compression_at = datetime(2026, 10, 17, 12)
    timeline.save("session-1")

    snapshot_path = (
        tmp_path
        / agent.object_id
        / "sessions"
        / ("session-1__compact__20261017T120000_000000" if compressed and compacted else "session-1")
        / "timeline.json"
    )
    snapshot_text = snapshot_path.read_text()
    snapshot = json.loads(snapshot_text)["entries"][0]
    assert ("opaque_stream_blob" in snapshot_text) is (not journal_owned)
    assert snapshot["content"] == "Checked the options."
    assert snapshot["metadata"].get("reasoning_items") == (None if journal_owned else [_REASONING_ITEM])
    assert snapshot["metadata"]["fingerprint"] == "openai:gpt-5:abcd1234"
    assert snapshot["metadata"]["response_id"] == "resp_stream_001"
    assert entry.metadata["reasoning_items"] == [_REASONING_ITEM]
    assert timeline.to_llm_messages()[0].reasoning_items == [_REASONING_ITEM]
