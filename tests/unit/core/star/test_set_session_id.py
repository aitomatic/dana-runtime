"""Tests for STARAgent.set_session_id as a real per-session context boundary.

Changing the session id must: flush the outgoing session, rebuild the timeline
(resetting entries AND compaction-tracking state), and rehydrate from the new
session's persisted entries. Unknown session -> empty timeline. Same id -> no-op.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from dana.config.storage_config import FileStorageConfig
from dana.core.agent.star_agent import STARAgent
from dana.core.timeline.compressed_timeline import CompressedTimeline
from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType
from dana.repositories.local_file_repository import LocalTimelineRepository
from dana.repositories.repository_factory import RepositoryFactory, RepositoryType


def _make_factory(workspace: str) -> RepositoryFactory:
    factory = RepositoryFactory()
    factory.register(
        RepositoryType.TIMELINE,
        LocalTimelineRepository,
        FileStorageConfig(workspace_folder=workspace),
    )
    return factory


def _make_agent(tmp_path, session_id: str = "A") -> STARAgent:
    agent = STARAgent(
        agent_type="test-agent",
        agent_id="agent-1",
        auto_register=False,
        enable_skills=False,
        enable_web_search=False,
        enable_code_execution=False,
        enable_assistant=False,
        repository_factory=_make_factory(str(tmp_path)),
    )
    agent._session_id = session_id
    return agent


def _entry(content: str, entry_type: TimelineEntryType = TimelineEntryType.USER_MESSAGE) -> TimelineEntry:
    return TimelineEntry(entry_type=entry_type, content=content)


class TestSetSessionIdBoundary:
    """set_session_id resets and reloads timeline state."""

    def test_rehydrate_unknown_session_is_empty(self, tmp_path):
        agent = _make_agent(tmp_path, session_id="A")
        agent._timeline.add_entry(_entry("hello-A"))

        agent.set_session_id("never-saved")

        assert agent._timeline.timeline == []
        assert agent._timeline._native_messages == []
        assert agent._session_id == "never-saved"

    def test_set_session_id_clears_previous(self, tmp_path):
        agent = _make_agent(tmp_path, session_id="A")
        agent._timeline.add_entry(_entry("a1"))
        agent._timeline.add_entry(_entry("a2"))
        # Simulate a compaction having fired on session A.
        agent._timeline._last_compression_at = datetime(2026, 4, 20, 12, 0, 0)
        agent._timeline._active_compact_session_id = "A__compact__stale"

        agent.set_session_id("B")

        assert agent._timeline.timeline == []
        # Fresh instance -> compaction-tracking state fully reset, not leaked.
        assert agent._timeline._last_compression_at is None
        assert agent._timeline._active_compact_session_id is None
        assert agent._timeline._active_compact_compression_at is None

    def test_resume_reloads_persisted_session(self, tmp_path):
        agent = _make_agent(tmp_path, session_id="A")
        agent._timeline.add_entry(_entry("user-q", TimelineEntryType.USER_MESSAGE))
        agent._timeline.add_entry(_entry("agent-a", TimelineEntryType.AGENT_RESPONSE))
        agent._timeline.save("A")

        agent.set_session_id("B")
        assert agent._timeline.timeline == []

        agent.set_session_id("A")
        contents = [e.content for e in agent._timeline.timeline]
        assert contents == ["user-q", "agent-a"]
        # _native_messages recomputed by read_since.
        assert [m.content for m in agent._timeline._native_messages] == ["user-q", "agent-a"]

    def test_resume_from_compacted_snapshot(self, tmp_path):
        agent = _make_agent(tmp_path, session_id="A")
        tl = agent._timeline
        tl.add_entry(_entry("pre-compact"))
        tl.save("A")

        # Mint a compact session: stamp a fresh compression, swap entries, save.
        tl._last_compression_at = datetime(2026, 4, 20, 12, 0, 0)
        tl.timeline = [_entry("post-compact")]
        tl._native_messages = [tl._timeline_entry_to_native_message(e) for e in tl.timeline]
        tl.save("A")

        agent.set_session_id("B")
        agent.set_session_id("A")

        contents = [e.content for e in agent._timeline.timeline]
        assert "post-compact" in contents
        assert "pre-compact" not in contents
        assert agent._timeline._native_messages

    def test_reload_timeline_false_keeps_timeline(self, tmp_path):
        # Opt-out path: a caller-managed timeline (e.g. a seeded persona entry)
        # survives the session switch — set_session_id only relabels.
        agent = _make_agent(tmp_path, session_id="A")
        original = agent._timeline
        agent._timeline.add_entry(_entry("seeded-persona"))

        agent.set_session_id("B", reload_timeline=False)

        assert agent._timeline is original
        assert agent._session_id == "B"
        assert [e.content for e in agent._timeline.timeline] == ["seeded-persona"]

    def test_reload_timeline_false_does_not_flush_old_session(self, tmp_path):
        # Pure relabel must not persist anything to the outgoing session id.
        agent = _make_agent(tmp_path, session_id="A")
        agent._timeline.add_entry(_entry("kept"))

        agent.set_session_id("B", reload_timeline=False)

        repo = agent._timeline._repository
        assert repo is not None
        assert list(repo.read_session_entries("A")) == []

    def test_same_session_id_is_fast_path(self, tmp_path):
        agent = _make_agent(tmp_path, session_id="A")
        original_timeline = agent._timeline
        agent._timeline.add_entry(_entry("untouched"))

        agent.set_session_id("A")

        assert agent._timeline is original_timeline
        assert [e.content for e in agent._timeline.timeline] == ["untouched"]


class TestRehydrate:
    """CompressedTimeline.rehydrate guards."""

    def test_rehydrate_noop_without_repository(self):
        # agent=None -> _repository is None; rehydrate must not raise.
        timeline = CompressedTimeline()
        timeline.rehydrate()
        assert timeline.timeline == []
        assert timeline._native_messages == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
