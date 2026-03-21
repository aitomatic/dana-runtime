"""Tests for multimodal content conversion via provider methods and shared helpers."""

import pytest

from dana.common.llm.types import is_multimodal_content, read_media_as_base64
from dana.common.llm.providers.anthropic import AnthropicProvider
from dana.common.llm.providers.openai_compatible_base import OpenAICompatibleProvider


# ---------------------------------------------------------------------------
# Fixtures — path-based media blocks using real temp files
# ---------------------------------------------------------------------------

TEXT_BLOCK = {"type": "text", "text": "Describe this image."}


@pytest.fixture
def image_block(tmp_path):
    """Create a path-based image block with a real temp file."""
    f = tmp_path / "test.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    return {"type": "image", "media_type": "image/png", "path": str(f)}


@pytest.fixture
def audio_block(tmp_path):
    f = tmp_path / "test.wav"
    f.write_bytes(b"RIFF" + b"\x00" * 20)
    return {"type": "audio", "media_type": "audio/wav", "path": str(f)}


@pytest.fixture
def video_block(tmp_path):
    f = tmp_path / "test.mp4"
    f.write_bytes(b"\x00\x00\x00\x1cftyp" + b"\x00" * 20)
    return {"type": "video", "media_type": "video/mp4", "path": str(f)}


@pytest.fixture
def document_block(tmp_path):
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4 minimal")
    return {"type": "document", "media_type": "application/pdf", "path": str(f)}


def _make_anthropic_provider(**caps):
    """Create a bare AnthropicProvider for testing convert_multimodal_content."""
    p = AnthropicProvider.__new__(AnthropicProvider)
    p.supports_vision = caps.get("supports_vision", True)
    p.supports_audio = caps.get("supports_audio", False)
    p.supports_video = caps.get("supports_video", False)
    return p


def _make_openai_provider(**caps):
    """Create a bare OpenAICompatibleProvider for testing convert_multimodal_content."""
    p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    p.model = "gpt-4o"
    p.supports_vision = caps.get("supports_vision", True)
    p.supports_audio = caps.get("supports_audio", False)
    p.supports_video = caps.get("supports_video", False)
    return p


# ---------------------------------------------------------------------------
# is_multimodal_content
# ---------------------------------------------------------------------------


class TestIsMultimodalContent:
    def test_string_content(self):
        assert not is_multimodal_content("hello world")

    def test_text_only_blocks(self):
        assert not is_multimodal_content([TEXT_BLOCK])

    def test_image_blocks(self, image_block):
        assert is_multimodal_content([TEXT_BLOCK, image_block])

    def test_audio_blocks(self, audio_block):
        assert is_multimodal_content([audio_block])

    def test_video_blocks(self, video_block):
        assert is_multimodal_content([video_block])

    def test_document_blocks(self, document_block):
        assert is_multimodal_content([document_block])

    def test_empty_list(self):
        assert not is_multimodal_content([])


# ---------------------------------------------------------------------------
# read_media_as_base64 helper
# ---------------------------------------------------------------------------


class TestReadMediaAsBase64:
    def test_reads_file_and_encodes(self, tmp_path):
        f = tmp_path / "data.bin"
        content = b"hello world"
        f.write_bytes(content)
        block = {"type": "image", "media_type": "image/png", "path": str(f)}

        import base64
        result = read_media_as_base64(block)
        assert result == base64.b64encode(content).decode("utf-8")


# ---------------------------------------------------------------------------
# Anthropic converter (via provider method)
# ---------------------------------------------------------------------------


class TestConvertForAnthropic:
    def test_image_encodes_from_path(self, image_block):
        p = _make_anthropic_provider()
        result = p.convert_multimodal_content([TEXT_BLOCK, image_block])
        assert result[0] == TEXT_BLOCK
        assert result[1]["type"] == "image"
        assert result[1]["source"]["type"] == "base64"
        assert result[1]["source"]["media_type"] == "image/png"
        assert len(result[1]["source"]["data"]) > 0

    def test_document_encodes_from_path(self, document_block):
        p = _make_anthropic_provider()
        result = p.convert_multimodal_content([document_block])
        assert result[0]["type"] == "document"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "application/pdf"

    def test_audio_unsupported(self, audio_block):
        p = _make_anthropic_provider(supports_audio=False)
        result = p.convert_multimodal_content([audio_block])
        assert result[0]["type"] == "text"
        assert "not supported" in result[0]["text"]

    def test_video_unsupported(self, video_block):
        p = _make_anthropic_provider(supports_video=False)
        result = p.convert_multimodal_content([video_block])
        assert result[0]["type"] == "text"
        assert "not supported" in result[0]["text"]

    def test_vision_unsupported_fallback(self, image_block):
        p = _make_anthropic_provider(supports_vision=False)
        result = p.convert_multimodal_content([image_block])
        assert result[0]["type"] == "text"
        assert "not supported" in result[0]["text"]

    def test_mixed_content(self, image_block, audio_block, video_block):
        p = _make_anthropic_provider(supports_vision=True, supports_audio=False, supports_video=False)
        blocks = [TEXT_BLOCK, image_block, audio_block, video_block]
        result = p.convert_multimodal_content(blocks)
        assert result[0] == TEXT_BLOCK
        assert result[1]["type"] == "image"  # encoded from path
        assert result[1]["source"]["type"] == "base64"
        assert result[2]["type"] == "text"  # audio placeholder
        assert result[3]["type"] == "text"  # video placeholder


# ---------------------------------------------------------------------------
# OpenAI Chat Completions converter (via provider method)
# ---------------------------------------------------------------------------


class TestConvertForOpenAI:
    def test_image_base64(self, image_block):
        p = _make_openai_provider()
        result = p.convert_multimodal_content([image_block])
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert result[0]["image_url"]["detail"] == "auto"

    def test_audio_supported(self, audio_block):
        p = _make_openai_provider(supports_audio=True)
        result = p.convert_multimodal_content([audio_block])
        assert result[0]["type"] == "input_audio"
        assert result[0]["input_audio"]["format"] == "wav"
        assert len(result[0]["input_audio"]["data"]) > 0

    def test_audio_mp3_format(self, tmp_path):
        f = tmp_path / "test.mp3"
        f.write_bytes(b"ID3" + b"\x00" * 20)
        mp3_block = {"type": "audio", "media_type": "audio/mp3", "path": str(f)}
        p = _make_openai_provider(supports_audio=True)
        result = p.convert_multimodal_content([mp3_block])
        assert result[0]["input_audio"]["format"] == "mp3"

    def test_audio_mpeg_format(self, tmp_path):
        f = tmp_path / "test.mp3"
        f.write_bytes(b"\xff\xfb" + b"\x00" * 20)
        mpeg_block = {"type": "audio", "media_type": "audio/mpeg", "path": str(f)}
        p = _make_openai_provider(supports_audio=True)
        result = p.convert_multimodal_content([mpeg_block])
        assert result[0]["input_audio"]["format"] == "mp3"

    def test_video_always_unsupported(self, video_block):
        p = _make_openai_provider(supports_video=True)
        result = p.convert_multimodal_content([video_block])
        assert result[0]["type"] == "text"
        assert "not supported" in result[0]["text"]

    def test_document_supported(self, document_block):
        p = _make_openai_provider()
        result = p.convert_multimodal_content([document_block])
        assert result[0]["type"] == "file"
        assert "base64" in result[0]["file"]["file_data"]

    def test_vision_unsupported_fallback(self, image_block):
        p = _make_openai_provider(supports_vision=False)
        result = p.convert_multimodal_content([image_block])
        assert result[0]["type"] == "text"
        assert "not supported" in result[0]["text"]

    def test_text_passthrough(self):
        p = _make_openai_provider()
        result = p.convert_multimodal_content([TEXT_BLOCK])
        assert result[0] == TEXT_BLOCK


# ---------------------------------------------------------------------------
# Placeholder content
# ---------------------------------------------------------------------------


class TestPlaceholder:
    def test_includes_type_and_media(self, video_block):
        p = _make_anthropic_provider(supports_video=False)
        result = p.convert_multimodal_content([video_block])
        text = result[0]["text"]
        assert "video" in text
        assert "video/mp4" in text

    def test_audio_placeholder(self, audio_block):
        p = _make_openai_provider(supports_audio=False)
        result = p.convert_multimodal_content([audio_block])
        text = result[0]["text"]
        assert "audio" in text
        assert "audio/wav" in text
