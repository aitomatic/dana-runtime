"""Snapshot-based persistence tests for CompressedTimeline.

Covers the "store uncompressed alongside compressed" feature:
- `timeline.json` is the original, frozen at the first compression.
- Each compression rolls a new `timeline-after-compress-{ISO-ts}.json`.
- Subsequent saves within a generation update the active snapshot in place.
- Load prefers the newest snapshot, falls back to `timeline.json`, then legacy.
- Retention is "keep all" — no rotation.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from unittest.mock import Mock

from dana.config.storage_config import FileStorageConfig
from dana.core.agent import BaseAgent
from dana.core.timeline.compressed_timeline import CompressedTimeline
from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType
from dana.repositories.local_file_repository import LocalTimelineRepository
from dana.repositories.repository_factory import RepositoryFactory, RepositoryType


class _Agent(BaseAgent):
    def __init__(self, workspace: str, session_id: str = "sess-1"):
        super().__init__(agent_type="test_agent", agent_id="agent-1")
        self._codec = Mock()
        self._codec.__qualname__ = "TestCodec"
        self._storage_config = FileStorageConfig(workspace_folder=workspace)
        self._session_id = session_id


def _make_factory(workspace: str) -> RepositoryFactory:
    """Build a RepositoryFactory rooted at ``workspace`` so tests don't pollute
    the user's real storage directory."""
    factory = RepositoryFactory()
    factory.register(RepositoryType.TIMELINE, LocalTimelineRepository, FileStorageConfig(workspace_folder=workspace))
    return factory


def _make_timeline(agent: _Agent) -> CompressedTimeline:
    return CompressedTimeline(
        agent=agent,
        repository_factory=_make_factory(agent._storage_config.workspace_folder),
    )


def _add_entry(tl: CompressedTimeline, role: TimelineEntryType, content: str) -> None:
    tl.add_entry(TimelineEntry(entry_type=role, content=content))


def _session_folder(agent: _Agent) -> Path:
    # Path shape from LocalTimelineRepository._get_events_path:
    # {workspace}/{agent.object_id}/sessions/{session_id}
    return Path(agent._storage_config.workspace_folder) / str(agent.object_id) / "sessions" / agent._session_id


def test_first_save_before_compression_writes_timeline_json(tmp_path):
    """Until any compression fires, save() writes to timeline.json (legacy shape)."""
    agent = _Agent(str(tmp_path))
    tl = _make_timeline(agent)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "hi")
    _add_entry(tl, TimelineEntryType.AGENT_RESPONSE, "hello")

    tl.save(agent._session_id)

    folder = _session_folder(agent)
    assert (folder / "timeline.json").exists()
    assert list(folder.glob("timeline-after-compress-*.json")) == []


def test_compression_rolls_new_snapshot_file(tmp_path):
    """After _apply_compression stamps `_last_compression_at`, the next save
    writes a `timeline-after-compress-{ts}.json` instead of timeline.json."""
    agent = _Agent(str(tmp_path))
    tl = _make_timeline(agent)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u1")
    _add_entry(tl, TimelineEntryType.AGENT_RESPONSE, "a1")
    tl.save(agent._session_id)

    # Simulate a compression by stamping directly — avoids full LLM pipeline.
    tl._last_compression_at = datetime(2026, 4, 20, 12, 0, 0)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u2")
    tl.save(agent._session_id)

    folder = _session_folder(agent)
    snapshots = sorted(folder.glob("timeline-after-compress-*.json"))
    assert len(snapshots) == 1
    assert snapshots[0].name == "timeline-after-compress-20260420T120000.json"
    # `timeline.json` still exists from the pre-compression save but is frozen.
    assert (folder / "timeline.json").exists()


def test_subsequent_save_same_generation_updates_snapshot_in_place(tmp_path):
    """New entries added between two compressions must update the latest snapshot,
    not create another file or overwrite timeline.json."""
    agent = _Agent(str(tmp_path))
    tl = _make_timeline(agent)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u1")
    tl.save(agent._session_id)

    tl._last_compression_at = datetime(2026, 4, 20, 12, 0, 0)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u2")
    tl.save(agent._session_id)  # rolls snapshot #1

    _add_entry(tl, TimelineEntryType.AGENT_RESPONSE, "a2")
    tl.save(agent._session_id)  # same generation → updates snapshot #1

    folder = _session_folder(agent)
    snapshots = sorted(folder.glob("timeline-after-compress-*.json"))
    assert len(snapshots) == 1
    with snapshots[0].open() as f:
        data = json.load(f)
    contents = {e["content"] for e in data["entries"]}
    assert "u2" in contents and "a2" in contents


def test_second_compression_rolls_another_snapshot_keeping_first(tmp_path):
    """Full audit retention: older snapshots are kept indefinitely."""
    agent = _Agent(str(tmp_path))
    tl = _make_timeline(agent)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u1")
    tl.save(agent._session_id)

    tl._last_compression_at = datetime(2026, 4, 20, 12, 0, 0)
    tl.save(agent._session_id)  # snapshot-1

    tl._last_compression_at = datetime(2026, 4, 20, 13, 30, 0)
    tl.save(agent._session_id)  # snapshot-2

    folder = _session_folder(agent)
    snapshots = sorted(folder.glob("timeline-after-compress-*.json"))
    assert [s.name for s in snapshots] == [
        "timeline-after-compress-20260420T120000.json",
        "timeline-after-compress-20260420T133000.json",
    ]


def test_load_prefers_newest_snapshot(tmp_path):
    """Repository.read_session_entries + serializer load should both pick the
    newest snapshot over timeline.json."""
    agent = _Agent(str(tmp_path))
    tl = _make_timeline(agent)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "from-base")
    tl.save(agent._session_id)

    # Write two snapshots manually — newest should win.
    folder = _session_folder(agent)
    for ts, marker in [
        (datetime(2026, 4, 20, 12, 0, 0), "snap-old"),
        (datetime(2026, 4, 20, 15, 0, 0), "snap-new"),
    ]:
        path = folder / f"timeline-after-compress-{ts.strftime('%Y%m%dT%H%M%S')}.json"
        payload = {
            "session_id": agent._session_id,
            "entries": [{"entry_type": "USER_MESSAGE", "content": marker, "timestamp": ts.isoformat(), "metadata": {}}],
            "native_messages": [{"role": "user", "content": marker, "metadata": {}, "timestamp": ts.isoformat()}],
        }
        with path.open("w") as f:
            json.dump(payload, f)

    # Fresh timeline loads via read_since pipeline.
    tl2 = _make_timeline(agent)
    loaded = list(tl2.read_since(0))
    assert any(e.content == "snap-new" for e in loaded)
    assert all(e.content != "snap-old" for e in loaded)
    # Native messages must come from the newest snapshot too.
    assert any(m.content == "snap-new" for m in tl2._native_messages)


def test_load_fallback_to_timeline_json_when_no_snapshot(tmp_path):
    agent = _Agent(str(tmp_path))
    tl = _make_timeline(agent)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "only-base")
    tl.save(agent._session_id)

    tl2 = _make_timeline(agent)
    loaded = list(tl2.read_since(0))
    assert [e.content for e in loaded] == ["only-base"]


def test_reload_rehydrates_active_snapshot_so_next_save_does_not_roll(tmp_path):
    """After reload, subsequent saves keep updating the same snapshot until a
    new compression fires — not rolling a new file on every save."""
    agent = _Agent(str(tmp_path))
    tl = _make_timeline(agent)
    _add_entry(tl, TimelineEntryType.USER_MESSAGE, "u1")
    tl.save(agent._session_id)

    tl._last_compression_at = datetime(2026, 4, 20, 12, 0, 0)
    tl.save(agent._session_id)  # creates snapshot

    # Fresh timeline — simulates process restart.
    tl2 = _make_timeline(agent)
    list(tl2.read_since(0))  # triggers native_messages load → rehydrates state
    assert tl2._active_snapshot_path is not None
    assert tl2._active_snapshot_compression_at == datetime(2026, 4, 20, 12, 0, 0)

    _add_entry(tl2, TimelineEntryType.USER_MESSAGE, "u3-after-reload")
    tl2.save(agent._session_id)

    folder = _session_folder(agent)
    snapshots = sorted(folder.glob("timeline-after-compress-*.json"))
    assert len(snapshots) == 1, "reload must not roll a new snapshot without a fresh compression"
