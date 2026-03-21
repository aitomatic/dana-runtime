"""Tests for provider prepare_messages() with multimodal content."""

import pytest

from dana.common.llm.providers.anthropic import AnthropicProvider, prepare_anthropic_messages
from dana.common.llm.types import LLMMessage, LLMProvider
from dana.core.timeline.native_message import NativeMessage


TEXT_BLOCK = {"type": "text", "text": "What is this?"}


@pytest.fixture
def image_block(tmp_path):
    """Path-based image block with real temp file."""
    f = tmp_path / "test.png"
    f.write_bytes(b"\x89PNG" + b"\x00" * 20)
    return {"type": "image", "media_type": "image/png", "path": str(f)}


@pytest.fixture
def mixed_content(image_block):
    return [TEXT_BLOCK, image_block]


# ---------------------------------------------------------------------------
# Anthropic prepare_messages
# ---------------------------------------------------------------------------


class TestAnthropicPrepareMultimodal:
    def test_multimodal_user_message(self, mixed_content):
        """Anthropic prepare_messages converts path-based blocks to wire format."""
        msgs = [LLMMessage(role="user", content=mixed_content)]
        # Create provider directly instead of using prepare_anthropic_messages
        # since that function uses default supports (vision=True)
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.supports_vision = True
        p.supports_audio = False
        p.supports_video = False
        system, result = p.prepare_messages(msgs)
        assert system is None
        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0] == TEXT_BLOCK
        # Image block should be converted to Anthropic wire format (base64-encoded)
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["media_type"] == "image/png"
        assert len(content[1]["source"]["data"]) > 0

    def test_plain_text_user_message_unchanged(self):
        msgs = [LLMMessage(role="user", content="Hello")]
        _, result = prepare_anthropic_messages(msgs)
        assert result[0]["content"] == "Hello"


# ---------------------------------------------------------------------------
# OpenAI prepare_messages
# ---------------------------------------------------------------------------


class TestOpenAIPrepareMultimodal:
    def _make_provider(self):
        from dana.common.llm.providers.openai_compatible_base import OpenAICompatibleProvider

        p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
        p.model = "gpt-4o"
        p.supports_vision = True
        p.supports_audio = False
        p.supports_video = False
        return p

    def test_multimodal_user_message_converts_to_openai_format(self, mixed_content):
        provider = self._make_provider()
        msgs = [LLMMessage(role="user", content=mixed_content)]
        _, result = provider.prepare_messages(msgs)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0] == TEXT_BLOCK
        # Image converted to OpenAI format with base64 data URL
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_plain_text_user_message_unchanged(self):
        provider = self._make_provider()
        msgs = [LLMMessage(role="user", content="Hello")]
        _, result = provider.prepare_messages(msgs)
        assert result[0]["content"] == "Hello"


# ---------------------------------------------------------------------------
# Base LLMProvider fallback
# ---------------------------------------------------------------------------


class TestBaseLLMProviderMultimodalFallback:
    def test_list_content_extracted_as_text(self, mixed_content):
        provider = LLMProvider()
        msgs = [LLMMessage(role="user", content=mixed_content)]
        _, result = provider.prepare_messages(msgs)
        content = result[0]["content"]
        assert isinstance(content, str)
        assert "What is this?" in content
        assert "[image content]" in content


# ---------------------------------------------------------------------------
# NativeMessage multimodal
# ---------------------------------------------------------------------------


class TestNativeMessageMultimodal:
    def test_list_content_creation(self, mixed_content):
        nm = NativeMessage(role="user", content=mixed_content)
        assert isinstance(nm.content, list)
        assert len(nm.content) == 2

    def test_list_content_to_dict(self, mixed_content):
        nm = NativeMessage(role="user", content=mixed_content)
        d = nm.to_dict()
        assert d["content"] == mixed_content
        assert d["role"] == "user"

    def test_list_content_from_dict(self, mixed_content):
        data = {
            "role": "user",
            "content": mixed_content,
            "timestamp": "2026-03-20T15:00:00",
        }
        nm = NativeMessage.from_dict(data)
        assert isinstance(nm.content, list)
        assert nm.content[1]["type"] == "image"

    def test_list_content_to_llm_message(self, mixed_content):
        nm = NativeMessage(role="user", content=mixed_content)
        lm = nm.to_llm_message()
        assert isinstance(lm.content, list)
        assert lm.role == "user"

    def test_string_content_still_works(self):
        nm = NativeMessage(role="user", content="hello")
        assert isinstance(nm.content, str)
        d = nm.to_dict()
        assert d["content"] == "hello"
        lm = nm.to_llm_message()
        assert lm.content == "hello"
