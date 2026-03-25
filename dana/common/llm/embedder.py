"""
Embedder - Unified interface for text embeddings across LLM providers.

Mirrors the LLM class pattern: stateless, provider-agnostic, KISS.

Usage:
    embedder = Embedder(provider="openai")
    vector = await embedder.embed("Hello world")
    response = await embedder.embed_batch(["text1", "text2"])
"""

import asyncio
import atexit


try:
    import structlog
except ModuleNotFoundError:
    import logging

    class _StructLogShim:
        @staticmethod
        def get_logger() -> logging.Logger:
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger("dana")

    structlog = _StructLogShim()

from ..config import config_manager
from .providers.factory import create_provider
from .types import EmbeddingNotSupportedError, EmbeddingResponse, LLMProvider, ProviderError


logger = structlog.get_logger()

# Module-level event loop for sync operations (same pattern as llm.py)
_sync_event_loop: asyncio.AbstractEventLoop | None = None


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for sync operations."""
    global _sync_event_loop
    if _sync_event_loop is None or _sync_event_loop.is_closed():
        _sync_event_loop = asyncio.new_event_loop()
    return _sync_event_loop


def _cleanup_event_loop():
    """Clean up the persistent event loop on exit."""
    global _sync_event_loop
    if _sync_event_loop is not None and not _sync_event_loop.is_closed():
        try:
            pending = asyncio.all_tasks(_sync_event_loop)
            for task in pending:
                task.cancel()
            if pending:
                _sync_event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            _sync_event_loop.close()
        except Exception:
            pass


atexit.register(_cleanup_event_loop)

# Providers known to support embeddings
_EMBEDDING_CAPABLE_PROVIDERS = {"openai", "gemini", "azure"}


class Embedder:
    """
    Stateless embedding interface — KISS principle.

    The Embedder does not maintain state. The caller provides text,
    gets back vectors.

    Usage:
        embedder = Embedder(provider="openai")
        vector = await embedder.embed("Hello world")

        # Batch
        response = await embedder.embed_batch(["text1", "text2"])

        # Sync
        vector = embedder.embed_sync("Hello world")
    """

    def __init__(self, provider: str | LLMProvider | None = None, model: str | None = None):
        """
        Initialize Embedder with a provider.

        Args:
            provider: Provider name ('openai', 'gemini', 'azure') or provider instance.
            model: Embedding model name (defaults to provider's default).
        """
        if isinstance(provider, str):
            self.provider = create_provider(provider, model=model)
            self.provider_name = provider
        elif isinstance(provider, LLMProvider):
            self.provider = provider
            self.provider_name = "custom"
        else:
            # Auto-select first available provider with embedding support
            selected = self._auto_select_provider()
            if selected:
                self.provider = create_provider(selected, model=model)
                self.provider_name = selected
            else:
                raise EmbeddingNotSupportedError(
                    "No embedding-capable provider available. Set OPENAI_API_KEY or GEMINI_API_KEY environment variable."
                )

        if not self.provider.supports_embeddings:
            raise EmbeddingNotSupportedError(
                f"Provider '{self.provider_name}' does not support embeddings. Use one of: {', '.join(_EMBEDDING_CAPABLE_PROVIDERS)}"
            )

        # Override embedding model if explicitly provided
        if model and hasattr(self.provider, "embedding_model"):
            self.provider.embedding_model = model

        self.model = getattr(self.provider, "embedding_model", "unknown")

    @staticmethod
    def _auto_select_provider() -> str | None:
        """Select first available embedding-capable provider by priority."""
        for provider_name, _priority in config_manager.get_available_providers_by_priority():
            if provider_name in _EMBEDDING_CAPABLE_PROVIDERS:
                return provider_name
        return None

    async def embed(self, text: str, **kwargs) -> list[float]:
        """Embed a single text and return the vector."""
        response = await self.embed_response(text, **kwargs)
        return response.embeddings[0]

    async def embed_response(self, text: str, **kwargs) -> EmbeddingResponse:
        """Embed a single text and return full response with metadata."""
        try:
            return await self.provider.embed(text, **kwargs)
        except EmbeddingNotSupportedError:
            raise
        except Exception as e:
            raise ProviderError(f"Embedding failed with {self.provider_name}: {e}") from e

    async def embed_batch(self, texts: list[str], **kwargs) -> EmbeddingResponse:
        """Embed multiple texts and return full response."""
        if not texts:
            raise ValueError("Texts list cannot be empty")
        try:
            return await self.provider.embed_batch(texts, **kwargs)
        except EmbeddingNotSupportedError:
            raise
        except Exception as e:
            raise ProviderError(f"Batch embedding failed with {self.provider_name}: {e}") from e

    def embed_sync(self, text: str, **kwargs) -> list[float]:
        """Synchronous version of embed()."""
        loop = _get_or_create_event_loop()
        return loop.run_until_complete(self.embed(text, **kwargs))

    def embed_batch_sync(self, texts: list[str], **kwargs) -> EmbeddingResponse:
        """Synchronous version of embed_batch()."""
        loop = _get_or_create_event_loop()
        return loop.run_until_complete(self.embed_batch(texts, **kwargs))

    def switch_provider(self, provider: str, model: str | None = None):
        """Switch to a different embedding provider."""
        self.provider = create_provider(provider, model=model)
        self.provider_name = provider
        if not self.provider.supports_embeddings:
            raise EmbeddingNotSupportedError(f"Provider '{provider}' does not support embeddings.")
        if model and hasattr(self.provider, "embedding_model"):
            self.provider.embedding_model = model
        self.model = getattr(self.provider, "embedding_model", "unknown")
        logger.info("Switched embedding provider", provider=provider, model=self.model)

    @staticmethod
    def get_available_providers() -> list[str]:
        """Get list of available embedding-capable providers."""
        return [
            name
            for name in config_manager.get_available_providers()
            if name in _EMBEDDING_CAPABLE_PROVIDERS and config_manager.is_provider_available(name)
        ]
