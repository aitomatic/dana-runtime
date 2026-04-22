"""Parity test: LocalTimelineRepository and InMemoryTimelineRepository must
produce identical session-id layouts and entry contents when driven through
the same ``CompressedTimeline`` sequence.

This is the strongest architectural guarantee — if behavior matches across
two independent backends, the serializer's abstraction is clean. Any
regression that leaks filesystem-specific assumptions into the serializer
will break this test first.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

import pytest

from dana.config.storage_config import FileStorageConfig
from dana.core.agent import BaseAgent
from dana.core.timeline.compressed_timeline import CompressedTimeline
from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType
from dana.repositories.local_file_repository import LocalTimelineRepository
from dana.repositories.repository_factory import RepositoryFactory, RepositoryType
from tests.fixtures.in_memory_timeline_repository import make_in_memory_factory


class _Agent(BaseAgent):
    def __init__(self, workspace: str, session_id: str = "sess-1"):
        super().__init__(agent_type="test_agent", agent_id="agent-1")
        self._codec = Mock()
        self._codec.__qualname__ = "TestCodec"
        self._storage_config = FileStorageConfig(workspace_folder=workspace)
        self._session_id = session_id


def _make_local_factory(workspace: str) -> RepositoryFactory:
    factory = RepositoryFactory()
    factory.register(
        RepositoryType.TIMELINE,
        LocalTimelineRepository,
        FileStorageConfig(workspace_folder=workspace),
    )
    return factory


def _drive_sequence(factory: RepositoryFactory, agent: _Agent) -> tuple[list[str], dict[str, list[str]]]:
    """Run a standard add/save/compact/resume sequence and return the observed
    (session_ids, entries_by_session) snapshot. Same sequence on both backends
    must yield the same result."""
    tl = CompressedTimeline(agent=agent, repository_factory=factory)

    # Pre-compaction writes.
    tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="u1"))
    tl.save(agent._session_id)

    # First compaction.
    tl._last_compression_at = datetime(2026, 4, 20, 10, 0, 0)
    tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="u2-after-c1"))
    tl.save(agent._session_id)

    # Within-generation update.
    tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content="a2-after-c1"))
    tl.save(agent._session_id)

    # Second compaction.
    tl._last_compression_at = datetime(2026, 4, 20, 12, 30, 0)
    tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="u3-after-c2"))
    tl.save(agent._session_id)

    # Fresh timeline — resume + additional save.
    tl2 = CompressedTimeline(agent=agent, repository_factory=factory)
    list(tl2.read_since(0))
    tl2.add_entry(TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content="a3-after-resume"))
    tl2.save(agent._session_id)

    # Gather final state via repo API only.
    repo = tl2._repository
    assert repo is not None
    all_sessions = sorted(repo.list_sessions())
    entries_by_session = {sid: [e.content for e in repo.read_session_entries(sid)] for sid in all_sessions}
    return all_sessions, entries_by_session


def test_local_and_in_memory_parity(tmp_path):
    """Identical sequences on both backends produce identical layouts."""
    agent_local = _Agent(str(tmp_path / "local"))
    agent_mem = _Agent(str(tmp_path / "mem"))

    local_sessions, local_entries = _drive_sequence(_make_local_factory(str(tmp_path / "local")), agent_local)
    mem_sessions, mem_entries = _drive_sequence(make_in_memory_factory(str(tmp_path / "mem")), agent_mem)

    assert local_sessions == mem_sessions, f"session layout diverged: local={local_sessions}, mem={mem_sessions}"
    assert local_entries == mem_entries, "per-session entry contents diverged"


def test_parity_session_ids_match_expected_compact_convention(tmp_path):
    """Sanity: explicit assertion on the session-id shape produced by the
    sequence — keeps the test readable independent of parity-equality."""
    agent = _Agent(str(tmp_path))
    sessions, _ = _drive_sequence(make_in_memory_factory(str(tmp_path)), agent)
    assert sessions == [
        "sess-1",
        "sess-1__compact__20260420T100000_000000",
        "sess-1__compact__20260420T123000_000000",
    ]


@pytest.mark.parametrize(
    "make_factory",
    [
        pytest.param(_make_local_factory, id="local-fs"),
        pytest.param(make_in_memory_factory, id="in-memory"),
    ],
)
def test_audit_retention_pre_compaction_entries_remain_readable(tmp_path, make_factory):
    """After compaction + resume + new save, the pre-compaction base-session
    entries must still be readable via the repository. This is the audit
    retention guarantee."""
    agent = _Agent(str(tmp_path))
    factory = make_factory(str(tmp_path))
    sessions, entries_by_session = _drive_sequence(factory, agent)

    # Base session must retain its pre-compaction entry unchanged.
    assert entries_by_session["sess-1"] == ["u1"]
    # Compact sessions must exist and retain their generation's entries.
    assert any(s.startswith("sess-1__compact__") for s in sessions)
