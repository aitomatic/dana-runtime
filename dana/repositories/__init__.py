from .defaults import TimelineRepositoryDefaultsMixin
from .local_file_repository import LocalEventRepository, LocalLearningRepository, LocalPromptRepository, LocalTimelineRepository
from .repository_factory import RepositoryFactory, RepositoryType


__all__ = [
    "LocalEventRepository",
    "LocalLearningRepository",
    "LocalPromptRepository",
    "LocalTimelineRepository",
    "RepositoryFactory",
    "RepositoryType",
    "TimelineRepositoryDefaultsMixin",
    "LangfusePromptRepository",
]


def __getattr__(name: str):
    """Resolve `LangfusePromptRepository` on first use (PEP 562).

    Importing its module costs the Langfuse SDK, and this package is on the
    `import dana` path. The class itself imports without the SDK installed; only
    building a client raises `ImportError`, so it is always available here (it
    used to be `None` when the SDK was missing).
    """
    if name == "LangfusePromptRepository":
        from .langfuse_repository import LangfusePromptRepository

        globals()[name] = LangfusePromptRepository
        return LangfusePromptRepository
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
