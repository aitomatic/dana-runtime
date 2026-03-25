"""Tests for the Embedder class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dana.common.llm.embedder import Embedder
from dana.common.llm.types import EmbeddingNotSupportedError, EmbeddingResponse, LLMProvider


def _make_mock_provider(supports_embeddings: bool = True) -> MagicMock:
    """Create a mock LLM provider that satisfies isinstance(x, LLMProvider)."""
    provider = MagicMock(spec=LLMProvider)
    type(provider).supports_embeddings = property(lambda self: supports_embeddings)
    provider.embedding_model = "test-embedding-model" if supports_embeddings else None

    embed_response = EmbeddingResponse(
        embeddings=[[0.1, 0.2, 0.3]],
        model="test-embedding-model",
        usage={"prompt_tokens": 5, "total_tokens": 5},
        dimensions=3,
    )
    provider.embed = AsyncMock(return_value=embed_response)
    provider.embed_batch = AsyncMock(return_value=embed_response)
    return provider


class TestEmbedderInit:
    """Tests for Embedder initialization."""

    def test_init_with_provider_instance(self):
        mock_provider = _make_mock_provider()
        embedder = Embedder(provider=mock_provider)
        assert embedder.provider_name == "custom"
        assert embedder.model == "test-embedding-model"

    def test_init_rejects_non_embedding_provider(self):
        mock_provider = _make_mock_provider(supports_embeddings=False)
        with pytest.raises(EmbeddingNotSupportedError):
            Embedder(provider=mock_provider)

    @patch("dana.common.llm.embedder.config_manager")
    def test_auto_select_no_providers_raises(self, mock_config):
        mock_config.get_available_providers_by_priority.return_value = []
        with pytest.raises(EmbeddingNotSupportedError, match="No embedding-capable provider"):
            Embedder()


class TestEmbedderMethods:
    """Tests for Embedder embed methods."""

    @pytest.mark.asyncio
    async def test_embed_returns_vector(self):
        mock_provider = _make_mock_provider()
        embedder = Embedder(provider=mock_provider)
        result = await embedder.embed("hello")
        assert result == [0.1, 0.2, 0.3]
        mock_provider.embed.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_embed_response_returns_full(self):
        mock_provider = _make_mock_provider()
        embedder = Embedder(provider=mock_provider)
        result = await embedder.embed_response("hello")
        assert isinstance(result, EmbeddingResponse)
        assert result.dimensions == 3

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        batch_response = EmbeddingResponse(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model="test-model",
            dimensions=2,
        )
        mock_provider = _make_mock_provider()
        mock_provider.embed_batch = AsyncMock(return_value=batch_response)
        embedder = Embedder(provider=mock_provider)
        result = await embedder.embed_batch(["a", "b"])
        assert len(result.embeddings) == 2

    @pytest.mark.asyncio
    async def test_embed_batch_empty_raises(self):
        mock_provider = _make_mock_provider()
        embedder = Embedder(provider=mock_provider)
        with pytest.raises(ValueError, match="empty"):
            await embedder.embed_batch([])

    def test_embed_sync(self):
        mock_provider = _make_mock_provider()
        embedder = Embedder(provider=mock_provider)
        result = embedder.embed_sync("hello")
        assert result == [0.1, 0.2, 0.3]


class TestEmbedderSwitchProvider:
    """Tests for provider switching."""

    def test_switch_to_non_embedding_provider_raises(self):
        mock_provider = _make_mock_provider()
        embedder = Embedder(provider=mock_provider)
        with patch("dana.common.llm.embedder.create_provider") as mock_create:
            mock_create.return_value = _make_mock_provider(supports_embeddings=False)
            with pytest.raises(EmbeddingNotSupportedError):
                embedder.switch_provider("anthropic")


class TestEmbedderAvailableProviders:
    """Tests for static helper methods."""

    @patch("dana.common.llm.embedder.config_manager")
    def test_get_available_providers(self, mock_config):
        mock_config.get_available_providers.return_value = ["openai", "anthropic", "gemini"]
        mock_config.is_provider_available.side_effect = lambda p: p in ["openai", "gemini"]
        result = Embedder.get_available_providers()
        assert "openai" in result
        assert "gemini" in result
        assert "anthropic" not in result
