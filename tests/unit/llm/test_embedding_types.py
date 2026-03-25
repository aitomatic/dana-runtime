"""Tests for embedding types and base class behavior."""

import pytest

from dana.common.llm.types import (
    EmbeddingNotSupportedError,
    EmbeddingResponse,
    LLMProvider,
    ProviderError,
)


class TestEmbeddingResponse:
    """Tests for EmbeddingResponse dataclass."""

    def test_basic_construction(self):
        resp = EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            model="text-embedding-3-small",
            usage={"prompt_tokens": 5, "total_tokens": 5},
            dimensions=3,
        )
        assert resp.embeddings == [[0.1, 0.2, 0.3]]
        assert resp.model == "text-embedding-3-small"
        assert resp.dimensions == 3
        assert resp.usage["prompt_tokens"] == 5

    def test_batch_construction(self):
        resp = EmbeddingResponse(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model="test-model",
            dimensions=2,
        )
        assert len(resp.embeddings) == 2
        assert resp.usage is None

    def test_defaults(self):
        resp = EmbeddingResponse(embeddings=[], model="m")
        assert resp.usage is None
        assert resp.dimensions == 0


class TestEmbeddingNotSupportedError:
    """Tests for error hierarchy."""

    def test_inherits_from_provider_error(self):
        assert issubclass(EmbeddingNotSupportedError, ProviderError)

    def test_message(self):
        err = EmbeddingNotSupportedError("Anthropic does not support embeddings")
        assert "Anthropic" in str(err)


class TestLLMProviderEmbeddingDefaults:
    """Tests for base class embedding defaults."""

    def test_supports_embeddings_false_by_default(self):
        provider = LLMProvider()
        assert provider.supports_embeddings is False

    @pytest.mark.asyncio
    async def test_embed_raises_not_supported(self):
        provider = LLMProvider()
        with pytest.raises(EmbeddingNotSupportedError):
            await provider.embed("test")

    @pytest.mark.asyncio
    async def test_embed_batch_raises_not_supported(self):
        provider = LLMProvider()
        with pytest.raises(EmbeddingNotSupportedError):
            await provider.embed_batch(["test"])
