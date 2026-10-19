"""
Core agent module for the Adana agentic architecture.

This module provides the base Agent class and related functionality for
building conversational AI agents with resource and workflow management.

Exports resolve on first use (PEP 562). Eagerly importing them here pulls the
whole agent runtime, because STARAgent imports the timeline package, which
imports the repository package, which imports `dana.core.agent.base_agent`. Any
module below this one in that chain would then trigger a circular import while
this package is half-initialized. Deferring keeps
`from dana.core.agent.base_agent import BaseAgent` usable from inside the
repository layer.
"""

from importlib import import_module
from typing import Any


__all__ = ["AssistantAgent", "BaseAgent", "BaseSTARAgent", "STARAgent"]

_LAZY_EXPORTS = {
    "AssistantAgent": "dana.core.agent.assistant_agent",
    "BaseAgent": "dana.core.agent.base_agent",
    "BaseSTARAgent": "dana.core.agent.base_star_agent",
    "STARAgent": "dana.core.agent.star_agent",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path), name)
    # Cache for later lookups: __getattr__ is only consulted on a miss.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
