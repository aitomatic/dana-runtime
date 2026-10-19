"""
Dana Agent - Domain-Aware Neurosymbolic Agents

This package provides the core agent framework for building and managing
specialized AI agents with domain-specific knowledge and capabilities.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


# `from . import __init__` would resolve to the module type's own `__init__`
# method-wrapper rather than the `dana.__init__` subpackage, so bind it by name.
# Importing it here (not on first use) keeps `init_environment()` running on
# `import dana`, which loads .env and configures the logging fallback.
_core: Any = import_module("dana.__init__")

__all__ = ["LLM", "LLMMessage", "LLMResponse", "STARAgent"]


def __getattr__(name: str) -> Any:
    """Forward top-level agent exports to the `dana.__init__` subpackage.

    Deferred so that `import dana` stays cheap for callers that never build an
    agent. The subpackage resolves each name on first use; this only forwards.
    """
    if name in __all__:
        return getattr(_core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
