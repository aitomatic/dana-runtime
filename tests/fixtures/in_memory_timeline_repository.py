"""In-memory TimelineRepositoryProtocol implementation for tests.

Conforms to ``TimelineRepositoryProtocol`` with the three methods needed
by ``CompressedTimeline`` (save, read_session_entries, list_sessions).
No file I/O — lets us prove the serializer is repository-agnostic and
lets tests run fast without tmp_path setup.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from dana.config.storage_config import FileStorageConfig, StorageConfig
from dana.core.agent.base_agent import BaseAgent
from dana.repositories.repository_factory import RepositoryFactory, RepositoryType


if TYPE_CHECKING:
    from dana.core.timeline.timeline import TimelineEntry


class InMemoryTimelineRepository:
    """Dict-of-lists timeline repo. Test-only. No persistence, no validation."""

    def __init__(self, storage_config: StorageConfig | None = None, agent: BaseAgent | None = None):
        self.storage_config = storage_config
        self._agent = agent
        self._sessions: dict[str, list[TimelineEntry]] = {}

    @classmethod
    def instantiate(cls, storage_config: StorageConfig, agent: BaseAgent) -> InMemoryTimelineRepository:
        return cls(storage_config, agent)

    def save(self, session_id: str, entries: list[TimelineEntry]) -> None:
        # Shallow copy mirrors LocalTimelineRepository: callers can mutate the
        # source list without affecting persisted state.
        self._sessions[session_id] = list(entries)

    def read_session_entries(self, session_id: str) -> Iterator[TimelineEntry]:
        yield from self._sessions.get(session_id, [])

    def list_sessions(self, prefix: str = "") -> list[str]:
        ids = list(self._sessions.keys())
        if prefix:
            ids = [s for s in ids if s.startswith(prefix)]
        return sorted(ids)


class _SharedInMemoryRepoCreator:
    """Factory shim that returns the SAME InMemoryTimelineRepository instance
    across multiple ``instantiate`` calls.

    Local FS repos persist state on disk, so independent instances see the
    same state. In-memory repos store state in ``self._sessions`` — giving
    every timeline its own instance would mean empty state on resume. Sharing
    one instance per factory mirrors the filesystem-backed behavior.
    """

    def __init__(self):
        self._instance: InMemoryTimelineRepository | None = None

    def instantiate(self, storage_config: StorageConfig, agent: BaseAgent) -> InMemoryTimelineRepository:
        if self._instance is None:
            self._instance = InMemoryTimelineRepository(storage_config, agent)
        return self._instance


def make_in_memory_factory(workspace: str) -> RepositoryFactory:
    """Return a RepositoryFactory wired to a shared InMemoryTimelineRepository.

    ``workspace`` is accepted for signature parity with the local factory but
    never touched (no file I/O).
    """
    factory = RepositoryFactory()
    factory.register(
        RepositoryType.TIMELINE,
        _SharedInMemoryRepoCreator(),  # type: ignore[arg-type]
        FileStorageConfig(workspace_folder=workspace),
    )
    return factory
