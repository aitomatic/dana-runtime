"""
Dana Agent - Domain-Aware Neurosymbolic Agents

This package provides the core agent framework for building and managing
specialized AI agents with domain-specific knowledge and capabilities.
"""

########################################################
# Initialize environment
from .init_environment import init_environment


init_environment()

########################################################
# Export main components
#
# Resolved on first use, not at import. Importing them eagerly costs every
# `import dana`, including `dana-agent --help`, which only parses argv, the
# provider SDKs (anthropic, openai) and the whole agent runtime: roughly
# 0.6 s of work a given process may never need. `__all__` keeps
# `from dana.__init__ import *` and `dir()` intact.
__all__ = ["LLM", "LLMMessage", "LLMResponse", "STARAgent"]

_LAZY_EXPORTS = {
    "LLM": "dana.common.llm.llm",
    "LLMMessage": "dana.common.llm.llm",
    "LLMResponse": "dana.common.llm.llm",
    "STARAgent": "dana.core.agent.star_agent",
}


def __getattr__(name: str) -> object:
    """Import an exported component on first access (PEP 562)."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path), name)
    # Cache for later lookups: __getattr__ is only consulted on a miss.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
