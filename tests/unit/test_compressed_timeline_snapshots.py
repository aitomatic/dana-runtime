"""Repository-agnostic compact-session tests for CompressedTimeline.

Covers the GH-1 "store uncompressed alongside compressed" feature through
the repository interface only — no direct file I/O in assertions:
  - Until any compaction fires, save() writes to the caller-supplied base
    session id.
  - Each compaction mints a sibling session ``{base}__compact__{ISO-ts}``,
    visible via ``repo.list_sessions(prefix=...)``.
  - Subsequent saves within a generation update the active compact session
    in place.
  - Read prefers the newest compact session, falls back to base when none.
  - Retention is "keep all" — older compact sessions are never deleted.

Tests are parametrized across two backends (local FS + in-memory) to prove
the behavior is repository-agnostic.
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


FACTORIES = [
    pytest.param(_make_local_factory, id="local-fs"),
    pytest.param(make_in_memory_factory, id="in-memory"),
]


def _make_timeline(agent: _Agent, factory: RepositoryFactory) -> CompressedTimeline:
    return CompressedTimeline(agent=agent, repository_factory=factory)


def _add_entry(tl: CompressedTimeline, role: TimelineEntryType, content: str) -> None:
    tl.add_entry(TimelineEntry(entry_type=role, content=content))


def _compact_prefix(agent: _Agent) -> str:
    return f"{agent._session_id}__compact__"


# ----------------------------------------------------------------------
# Tests — parametrized across both backends
# ----------------------------------------------------------------------


@pytest.mark.parametrize("make_factory", FACTORIES)
def test_save_before_compaction_writes_to_base_session(tmp_path, make_factory):
    """Until any compaction fires, save() writes to the base session id.
    No compact-suffixed sessions exist yet."""
    agent = _Agent(str(tmp_path))
    tl = _make_timeline(agent, make_factory(str(tmp_path)))
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "hi")
    _add_entry(tl, TimelineEntryType.AGENT_RESPONSE, "hello")

    tl.save(agent._session_id)

    repo = tl._repository
    assert repo.list_sessions(prefix=_compact_prefix(agent)) == []
    # Base session should have the entries.
    entries = list(repo.read_session_entries(agent._session_id))
    assert {e.content for e in entries} == {"hi", "hello"}


@pytest.mark.parametrize("make_factory", FACTORIES)
def test_compaction_rolls_new_compact_session(tmp_path, make_factory):
    """After a fresh compaction stamp, the next save mints a compact session."""
    agent = _Agent(str(tmp_path))
    tl = _make_timeline(agent, make_factory(str(tmp_path)))
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u1")
    tl.save(agent._session_id)

    tl._last_compression_at = datetime(2026, 4, 20, 12, 0, 0)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u2")
    tl.save(agent._session_id)

    compact = tl._repository.list_sessions(prefix=_compact_prefix(agent))
    assert compact == [f"{agent._session_id}__compact__20260420T120000_000000"]


@pytest.mark.parametrize("make_factory", FACTORIES)
def test_subsequent_save_same_generation_updates_same_compact_session(tmp_path, make_factory):
    """New entries added between two compactions update the active compact
    session rather than creating another."""
    agent = _Agent(str(tmp_path))
    tl = _make_timeline(agent, make_factory(str(tmp_path)))
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u1")
    tl.save(agent._session_id)

    tl._last_compression_at = datetime(2026, 4, 20, 12, 0, 0)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u2")
    tl.save(agent._session_id)

    _add_entry(tl, TimelineEntryType.AGENT_RESPONSE, "a2")
    tl.save(agent._session_id)  # same generation → same compact session

    repo = tl._repository
    compact = repo.list_sessions(prefix=_compact_prefix(agent))
    assert len(compact) == 1

    contents = {e.content for e in repo.read_session_entries(compact[0])}
    assert "u2" in contents and "a2" in contents


@pytest.mark.parametrize("make_factory", FACTORIES)
def test_second_compaction_mints_another_compact_session_keeping_first(tmp_path, make_factory):
    """Full audit retention: older compact sessions are kept indefinitely."""
    agent = _Agent(str(tmp_path))
    tl = _make_timeline(agent, make_factory(str(tmp_path)))
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u1")
    tl.save(agent._session_id)

    tl._last_compression_at = datetime(2026, 4, 20, 12, 0, 0)
    tl.save(agent._session_id)  # compact-1

    tl._last_compression_at = datetime(2026, 4, 20, 13, 30, 0)
    tl.save(agent._session_id)  # compact-2

    compact = tl._repository.list_sessions(prefix=_compact_prefix(agent))
    assert compact == [
        f"{agent._session_id}__compact__20260420T120000_000000",
        f"{agent._session_id}__compact__20260420T133000_000000",
    ]


@pytest.mark.parametrize("make_factory", FACTORIES)
def test_read_since_prefers_newest_compact_session(tmp_path, make_factory):
    """Fresh timeline on resume reads the newest compact session, not base."""
    agent = _Agent(str(tmp_path))
    factory = make_factory(str(tmp_path))

    # Stage: base session + two compact sessions (old, new).
    tl_seed = _make_timeline(agent, factory)
    _add_entry(tl_seed, TimelineEntryType.USER_MESSAGE, "from-base")
    tl_seed.save(agent._session_id)

    tl_seed._last_compression_at = datetime(2026, 4, 20, 12, 0, 0)
    # Replace the timeline with a marker entry so the compact session gets it.
    tl_seed.timeline = [TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="snap-old")]
    tl_seed.save(agent._session_id)

    tl_seed._last_compression_at = datetime(2026, 4, 20, 15, 0, 0)
    tl_seed.timeline = [TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="snap-new")]
    tl_seed.save(agent._session_id)

    # Fresh timeline — resume via read_since.
    tl2 = _make_timeline(agent, factory)
    loaded = list(tl2.read_since(0))
    assert any(e.content == "snap-new" for e in loaded)
    assert all(e.content != "snap-old" for e in loaded)


@pytest.mark.parametrize("make_factory", FACTORIES)
def test_read_since_falls_back_to_base_when_no_compact_session(tmp_path, make_factory):
    """With no compact sessions, resume reads the base session."""
    agent = _Agent(str(tmp_path))
    factory = make_factory(str(tmp_path))

    tl = _make_timeline(agent, factory)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "only-base")
    tl.save(agent._session_id)

    tl2 = _make_timeline(agent, factory)
    loaded = list(tl2.read_since(0))
    assert [e.content for e in loaded] == ["only-base"]


@pytest.mark.parametrize("make_factory", FACTORIES)
def test_reload_rehydrates_active_compact_session(tmp_path, make_factory):
    """After reload, subsequent saves keep updating the same compact session
    until a new compaction fires — no new session rolled on each save."""
    agent = _Agent(str(tmp_path))
    factory = make_factory(str(tmp_path))

    tl = _make_timeline(agent, factory)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u1")
    tl.save(agent._session_id)

    tl._last_compression_at = datetime(2026, 4, 20, 12, 0, 0)
    tl.save(agent._session_id)  # mints compact session

    # Fresh timeline — simulates process restart.
    tl2 = _make_timeline(agent, factory)
    list(tl2.read_since(0))  # triggers rehydration
    assert tl2._active_compact_session_id == f"{agent._session_id}__compact__20260420T120000_000000"
    assert tl2._active_compact_compression_at == datetime(2026, 4, 20, 12, 0, 0)

    _add_entry(tl2, TimelineEntryType.USER_MESSAGE, "u3-after-reload")
    tl2.save(agent._session_id)

    compact = tl2._repository.list_sessions(prefix=_compact_prefix(agent))
    assert len(compact) == 1, "reload must not mint a new compact session without a fresh compaction"


@pytest.mark.parametrize("make_factory", FACTORIES)
def test_resume_via_load_from_entries_adopts_latest_compact_session(tmp_path, make_factory):
    """Regression: resume through ``load_from_entries`` (bypassing read_since)
    must not clobber the base session. Subsequent saves go through rehydration
    on read_since; here we drive rehydration explicitly to confirm the state
    is picked up before the post-resume save."""
    agent = _Agent(str(tmp_path))
    factory = make_factory(str(tmp_path))

    tl = _make_timeline(agent, factory)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u1")
    tl.save(agent._session_id)

    # Simulate a compaction — mint the first compact session.
    tl._last_compression_at = datetime(2026, 4, 20, 13, 54, 13)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u2-post-compact")
    tl.save(agent._session_id)

    compact_id = f"{agent._session_id}__compact__20260420T135413_000000"
    assert compact_id in tl._repository.list_sessions(prefix=_compact_prefix(agent))

    # Fresh timeline instance. Read entries via the repo and load_from_entries.
    tl2 = _make_timeline(agent, factory)
    entries = list(tl2._repository.read_session_entries(compact_id))
    tl2.load_from_entries(entries)
    # Drive rehydration since load_from_entries alone doesn't touch the repo.
    tl2._rehydrate_active_compact_session()
    assert tl2._active_compact_session_id == compact_id

    # Add a new entry after resume and save — lands in the same compact session.
    _add_entry(tl2, TimelineEntryType.AGENT_RESPONSE, "a3-after-resume")
    tl2.save(agent._session_id)

    contents = {e.content for e in tl2._repository.read_session_entries(compact_id)}
    assert "a3-after-resume" in contents, "post-resume save must target the latest compact session"

    # No new compact session rolled without a fresh compaction.
    compact = tl2._repository.list_sessions(prefix=_compact_prefix(agent))
    assert compact == [compact_id]
