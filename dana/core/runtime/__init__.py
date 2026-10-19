from __future__ import annotations

from importlib import import_module
from typing import Any

# Protocol/dataclass exports come from the dependency-free `protocols` module,
# which many low-level modules import as `dana.core.runtime.protocols`. Importing
# them here is cheap and keeps that common path from re-entering this package.
from .protocols import (
    ApprovalResult,
    LLMCallerProtocol,
    ParsedResponse,
    PromptBuilderProtocol,
    ResponseParserProtocol,
    StreamEvent,
    StreamEventType,
    TodoItem,
    ToolExecutorProtocol,
)


__all__ = [
    "AgentRuntime",
    "AnthropicRuntime",
    "OpenAIRuntime",
    "DefaultRuntime",
    "ParsedResponse",
    "TodoItem",
    "ApprovalResult",
    "StreamEvent",
    "StreamEventType",
    "PromptBuilderProtocol",
    "LLMCallerProtocol",
    "ResponseParserProtocol",
    "ToolExecutorProtocol",
    "RuntimeRegistry",
]

# Runtime implementations pull in the provider SDKs and the full execution
# pipeline, and `.base` imports `dana.core.llm.response_parser` while
# `response_parser` itself imports `dana.core.runtime.protocols`. Importing them
# eagerly here therefore made any `dana.core.runtime.protocols` import re-enter
# `.base` and fail with a partial `response_parser`. Resolve on first use.
_LAZY_EXPORTS = {
    "AgentRuntime": "dana.core.runtime.base",
    "AnthropicRuntime": "dana.core.runtime.anthropic",
    "DefaultRuntime": "dana.core.runtime.default",
    "OpenAIRuntime": "dana.core.runtime.openai",
    "RuntimeRegistry": "dana.core.runtime.selector",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
