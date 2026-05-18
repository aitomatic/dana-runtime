"""Tests for AGENT_THOUGHTS metadata persistence in _record_think_results.

Covers Phase 2 of plans/260507-1829-reasoning-state-replay:
- _build_thinking_metadata empty when no reasoning items
- _build_thinking_metadata populates reasoning_items + fingerprint + response_id
- _provider_fingerprint defensive when llm_client / provider absent
- _record_think_results attaches metadata on direct-answer branch
- _record_think_results attaches metadata on tool-call branch (reasoning entry only)
- TimelineEntry with reasoning_items metadata round-trips through to_dict/from_dict
"""

from __future__ import annotations

from unittest.mock import MagicMock

from dana.core.timeline.timeline import Timeline, TimelineEntry, TimelineEntryType


def _make_agent(provider_fingerprint: str | None = "azure:gpt-5:abcd1234"):
    """Minimal STARAgent with a stubbed provider exposing .fingerprint."""
    from dana.core.agent.star_agent import STARAgent

    agent = STARAgent(
        agent_type="test-agent",
        auto_register=False,
        enable_skills=False,
        enable_web_search=False,
        enable_code_execution=False,
        enable_assistant=False,
        compress_timeline=False,
    )
    if provider_fingerprint is not None:
        client = MagicMock()
        provider = MagicMock()
        provider.fingerprint = provider_fingerprint
        client.provider = provider
        agent._llm_client = client
    else:
        agent._llm_client = None
    return agent


_RAW_REASONING_ITEM = {
    "type": "reasoning",
    "id": "rs_test_001",
    "summary": [{"type": "summary_text", "text": "Step one."}],
    "encrypted_content": "enc_blob_xyz",
}


class TestBuildThinkingMetadata:
    def test_empty_when_no_items(self):
        agent = _make_agent()
        assert agent._build_thinking_metadata(None, None) == {}
        assert agent._build_thinking_metadata([], "resp_x") == {}

    def test_populates_all_keys_when_items_present(self):
        agent = _make_agent(provider_fingerprint="azure:gpt-5:cafe1234")
        meta = agent._build_thinking_metadata([_RAW_REASONING_ITEM], "resp_xyz")

        assert meta["reasoning_items"] == [_RAW_REASONING_ITEM]
        assert meta["fingerprint"] == "azure:gpt-5:cafe1234"
        assert meta["response_id"] == "resp_xyz"

    def test_fingerprint_none_when_provider_unavailable(self):
        agent = _make_agent(provider_fingerprint=None)
        meta = agent._build_thinking_metadata([_RAW_REASONING_ITEM], "resp_xyz")

        # Items still persist; fingerprint just missing — replay path will skip
        assert meta["reasoning_items"] == [_RAW_REASONING_ITEM]
        assert meta["fingerprint"] is None


class TestProviderFingerprintDefensive:
    def test_returns_none_when_llm_client_none(self):
        agent = _make_agent(provider_fingerprint=None)
        assert agent._provider_fingerprint() is None

    def test_returns_none_when_provider_lacks_fingerprint(self):
        from dana.core.agent.star_agent import STARAgent

        agent = STARAgent(agent_type="t", auto_register=False, enable_skills=False, compress_timeline=False)
        client = MagicMock()
        # Provider has no fingerprint attribute (e.g. Anthropic provider before this PR)
        provider = MagicMock(spec=[])
        client.provider = provider
        agent._llm_client = client

        assert agent._provider_fingerprint() is None

    def test_reads_fingerprint_when_present(self):
        agent = _make_agent(provider_fingerprint="azure:gpt-5:deadbeef")
        assert agent._provider_fingerprint() == "azure:gpt-5:deadbeef"


class TestRecordThinkResultsMetadata:
    def _make_timeline(self) -> Timeline:
        return Timeline()

    def test_direct_answer_branch_attaches_metadata(self):
        agent = _make_agent()
        tl = self._make_timeline()

        agent._record_think_results(
            timeline=tl,
            trace_percepts={},
            response="Final answer.",
            reasoning="I considered the problem.",
            tool_calls=[],
            done=True,
            todo_list=None,
            output_state="exit",
            reasoning_items=[_RAW_REASONING_ITEM],
            response_id="resp_001",
        )

        thoughts = [e for e in tl.timeline if e.entry_type == TimelineEntryType.AGENT_THOUGHTS]
        assert len(thoughts) == 1
        meta = thoughts[0].metadata
        assert meta["reasoning_items"] == [_RAW_REASONING_ITEM]
        assert meta["fingerprint"] == "azure:gpt-5:abcd1234"
        assert meta["response_id"] == "resp_001"

    def test_tool_call_branch_attaches_metadata_to_reasoning_only(self):
        agent = _make_agent()
        tl = self._make_timeline()

        agent._record_think_results(
            timeline=tl,
            trace_percepts={},
            response="Calling tool.",
            reasoning="Need to look this up.",
            tool_calls=[{"tool_call_id": "call_1", "function": "search", "arguments": "{}"}],
            done=False,
            todo_list=None,
            output_state="continue",
            reasoning_items=[_RAW_REASONING_ITEM],
            response_id="resp_002",
        )

        thoughts = [e for e in tl.timeline if e.entry_type == TimelineEntryType.AGENT_THOUGHTS]
        # Two AGENT_THOUGHTS entries: reasoning + response. Metadata only on reasoning.
        assert len(thoughts) == 2
        # First entry = reasoning (gets metadata)
        assert thoughts[0].metadata.get("reasoning_items") == [_RAW_REASONING_ITEM]
        assert thoughts[0].content == "Need to look this up."
        # Second entry = response text (no metadata)
        assert thoughts[1].content == "Calling tool."
        assert thoughts[1].metadata == {}

    def test_no_reasoning_items_means_empty_metadata(self):
        agent = _make_agent()
        tl = self._make_timeline()

        agent._record_think_results(
            timeline=tl,
            trace_percepts={},
            response="Direct answer.",
            reasoning="Plain thinking.",
            tool_calls=[],
            done=True,
            todo_list=None,
            output_state="exit",
            reasoning_items=None,
            response_id=None,
        )

        thoughts = [e for e in tl.timeline if e.entry_type == TimelineEntryType.AGENT_THOUGHTS]
        assert len(thoughts) == 1
        assert thoughts[0].metadata == {}


class TestEmptySummaryReasoningPersistence:
    """Regression: GPT-5/o3/o4 turns return a reasoning item (rs_… +
    encrypted_content) with an EMPTY summary on low-summary turns. The summary
    text is empty but the item is still required for cross-turn replay. The
    AGENT_THOUGHTS gate must key on reasoning_items, not on summary text —
    otherwise the encrypted item is dropped and the turn replays without state.
    """

    _EMPTY_SUMMARY_ITEM = {
        "type": "reasoning",
        "id": "rs_empty_summary_001",
        "summary": [],
        "encrypted_content": "gAAAA_enc_blob",
    }

    def test_tool_call_branch_persists_item_when_summary_empty(self):
        agent = _make_agent()
        tl = Timeline()

        agent._record_think_results(
            timeline=tl,
            trace_percepts={},
            response="",
            reasoning="",  # empty summary text
            tool_calls=[{"tool_call_id": "call_X", "function": "bash__execute", "arguments": "{}"}],
            done=False,
            todo_list=None,
            output_state="continue",
            reasoning_items=[self._EMPTY_SUMMARY_ITEM],
            response_id="resp_empty_001",
        )

        thoughts = [e for e in tl.timeline if e.entry_type == TimelineEntryType.AGENT_THOUGHTS]
        assert len(thoughts) == 1
        assert thoughts[0].metadata["reasoning_items"] == [self._EMPTY_SUMMARY_ITEM]
        assert thoughts[0].metadata["response_id"] == "resp_empty_001"

        # AGENT_THOUGHTS must sit immediately before TOOL_CALL — replay ordering.
        types = [e.entry_type for e in tl.timeline]
        assert types == [TimelineEntryType.AGENT_THOUGHTS, TimelineEntryType.TOOL_CALL]

    def test_direct_answer_branch_persists_item_when_summary_empty(self):
        agent = _make_agent()
        tl = Timeline()

        agent._record_think_results(
            timeline=tl,
            trace_percepts={},
            response="Final answer.",
            reasoning="",  # empty summary text
            tool_calls=[],
            done=True,
            todo_list=None,
            output_state="exit",
            reasoning_items=[self._EMPTY_SUMMARY_ITEM],
            response_id="resp_empty_002",
        )

        thoughts = [e for e in tl.timeline if e.entry_type == TimelineEntryType.AGENT_THOUGHTS]
        assert len(thoughts) == 1
        assert thoughts[0].metadata["reasoning_items"] == [self._EMPTY_SUMMARY_ITEM]

    def test_no_entry_when_summary_empty_and_no_items(self):
        """Nothing to persist — no phantom AGENT_THOUGHTS entry."""
        agent = _make_agent()
        tl = Timeline()

        agent._record_think_results(
            timeline=tl,
            trace_percepts={},
            response="",
            reasoning="",
            tool_calls=[{"tool_call_id": "call_X", "function": "bash__execute", "arguments": "{}"}],
            done=False,
            todo_list=None,
            output_state="continue",
            reasoning_items=None,
            response_id=None,
        )

        thoughts = [e for e in tl.timeline if e.entry_type == TimelineEntryType.AGENT_THOUGHTS]
        assert thoughts == []


class TestMetadataJsonRoundTrip:
    def test_reasoning_items_survive_to_dict_from_dict(self):
        entry = TimelineEntry(
            entry_type=TimelineEntryType.AGENT_THOUGHTS,
            content="reasoning text",
            metadata={
                "reasoning_items": [_RAW_REASONING_ITEM],
                "fingerprint": "azure:gpt-5:abcd1234",
                "response_id": "resp_001",
            },
        )

        # Round-trip through serializer
        d = entry.to_dict()
        reloaded = TimelineEntry.from_dict(d)

        assert reloaded.metadata["reasoning_items"] == [_RAW_REASONING_ITEM]
        assert reloaded.metadata["fingerprint"] == "azure:gpt-5:abcd1234"
        assert reloaded.metadata["response_id"] == "resp_001"

    def test_metadata_isolated_per_entry(self):
        """Mutating one entry's metadata must not affect another's — ensures
        we copy the metadata dict in _record_think_results, not aliasing."""
        agent = _make_agent()
        tl = Timeline()

        agent._record_think_results(
            timeline=tl,
            trace_percepts={},
            response="A",
            reasoning="r1",
            tool_calls=[],
            done=True,
            todo_list=None,
            output_state="exit",
            reasoning_items=[_RAW_REASONING_ITEM],
            response_id="resp_A",
        )
        agent._record_think_results(
            timeline=tl,
            trace_percepts={},
            response="B",
            reasoning="r2",
            tool_calls=[],
            done=True,
            todo_list=None,
            output_state="exit",
            reasoning_items=[_RAW_REASONING_ITEM],
            response_id="resp_B",
        )

        thoughts = [e for e in tl.timeline if e.entry_type == TimelineEntryType.AGENT_THOUGHTS]
        assert len(thoughts) == 2
        # Mutating one shouldn't bleed into the other
        thoughts[0].metadata["mutated"] = True
        assert "mutated" not in thoughts[1].metadata
