"""Regression tests: STAR._see ingests canonical multimodal content blocks.

Before the fix, ``_see`` only added a timeline entry when ``caller_message`` was
a ``str``. A canonical content-block list (what ``AgentSession.prompt`` passes
for a multimodal turn) was silently dropped: the model context never contained
the user's image/audio/document blocks, and an empty timeline was built from a
turn that did have input.

The entry content must reach the real provider conversion unchanged — no ``str()``
coercion, no block mutation.
"""

from __future__ import annotations

from dana.common.llm.types import is_multimodal_content
from dana.core.timeline.timeline import TimelineEntryType


def _make_agent():
    """Minimal STARAgent without registry, LLM, or persistence side-effects."""
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


def _blocks() -> list[dict]:
    return [
        {"type": "text", "text": "What does this chart show?"},
        {"type": "image", "media_type": "image/png", "path": "/tmp/chart.png"},
    ]


def _user_entries(agent):
    return [e for e in agent._timeline.timeline if e.entry_type == TimelineEntryType.USER_MESSAGE]


class TestSeeIngestsContentBlocks:
    """A multimodal caller_message becomes the latest USER_MESSAGE entry."""

    def test_content_blocks_reach_the_timeline_intact(self):
        agent = _make_agent()
        blocks = _blocks()

        agent._see({"caller_message": blocks, "caller_type": "human", "caller_id": "user"})

        entries = _user_entries(agent)
        assert len(entries) == 1, "multimodal caller_message must be ingested, not dropped"
        # Identity, not equality of a copy: the blocks the provider converts.
        assert entries[0].content == blocks
        assert entries[0].is_latest_user_message is True

    def test_content_blocks_survive_provider_conversion(self):
        agent = _make_agent()
        blocks = _blocks()

        agent._see({"caller_message": blocks, "caller_type": "human", "caller_id": "user"})
        messages = agent._timeline.to_llm_messages()

        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == blocks
        # Still recognizably multimodal: the image block was not flattened away.
        assert is_multimodal_content(messages[0].content)

    def test_content_blocks_are_not_stringified(self):
        agent = _make_agent()

        agent._see({"caller_message": _blocks(), "caller_type": "human", "caller_id": "user"})

        content = _user_entries(agent)[0].content
        assert not isinstance(content, str), "str() coercion would destroy the blocks the provider needs"

    def test_text_only_message_still_ingested_as_text(self):
        agent = _make_agent()

        agent._see({"caller_message": "plain question", "caller_type": "human", "caller_id": "user"})

        assert [e.content for e in _user_entries(agent)] == ["plain question"]

    def test_message_alias_accepts_content_blocks(self):
        agent = _make_agent()

        result = agent._see({"message": _blocks(), "caller_type": "human", "caller_id": "user"})

        assert _user_entries(agent)[0].content == _blocks()
        # The 'message' alias is consumed (only one entry, no duplicate).
        assert len(_user_entries(agent)) == 1
        assert "message" not in result["trace_percepts"]

    def test_empty_content_blocks_end_the_loop_without_an_entry(self):
        agent = _make_agent()

        result = agent._see({"caller_message": [], "caller_type": "human", "caller_id": "user"})

        assert _user_entries(agent) == []
        assert result["trace_percepts"]["EXIT_STAR_LOOP_FLAG"] is True

    def test_non_message_payloads_are_not_ingested(self):
        """Only text and canonical block lists are messages — no silent coercion."""
        agent = _make_agent()

        for payload in (["plain", "strings"], {"type": "text"}, 17):
            agent._see({"caller_message": payload, "caller_type": "human", "caller_id": "user"})

        assert _user_entries(agent) == []
