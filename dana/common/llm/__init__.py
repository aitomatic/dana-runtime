"""Adana LLM public API, resolved on first use without loading unused SDKs."""

from importlib import import_module
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .embedder import Embedder as Embedder
    from .llm import LLM as LLM
    from .types import EmbeddingNotSupportedError as EmbeddingNotSupportedError
    from .types import EmbeddingResponse as EmbeddingResponse
    from .types import LLMMessage as LLMMessage
    from .types import LLMResponse as LLMResponse
    from .types import LLMStreamChunk as LLMStreamChunk
    from .types import ProviderError as ProviderError


__all__ = [
    "Embedder",
    "EmbeddingNotSupportedError",
    "EmbeddingResponse",
    "LLM",
    "LLMMessage",
    "LLMResponse",
    "LLMStreamChunk",
    "ProviderError",
    "enable_llm_debug_logging",
    "disable_llm_debug_logging",
    "get_llm_debug_stats",
]

_LAZY_EXPORTS = {
    "Embedder": "dana.common.llm.embedder",
    "LLM": "dana.common.llm.llm",
    "EmbeddingNotSupportedError": "dana.common.llm.types",
    "EmbeddingResponse": "dana.common.llm.types",
    "LLMMessage": "dana.common.llm.types",
    "LLMResponse": "dana.common.llm.types",
    "LLMStreamChunk": "dana.common.llm.types",
    "ProviderError": "dana.common.llm.types",
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


def enable_llm_debug_logging():
    """Enable bounded metadata diagnostics, excluding prompts and responses."""
    from .debug_logger import enable_debug_logging

    enable_debug_logging()


def disable_llm_debug_logging():
    """Disable LLM diagnostic logging."""
    from .debug_logger import disable_debug_logging

    disable_debug_logging()


def get_llm_debug_stats():
    """Get diagnostic logging statistics without constructing a provider."""
    from .debug_logger import get_debug_logger

    return get_debug_logger().get_log_stats()
