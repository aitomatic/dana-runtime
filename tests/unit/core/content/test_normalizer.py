"""
Unit tests for content normalizer — multimodal block normalization.

Covers:
- Text block normalization
- Image block normalization (inline data and file path)
- Embedded resource normalization
- File resource normalization
- MIME/size enforcement
- Path traversal detection
- Artifact reference generation for oversized content
"""

from __future__ import annotations

import hashlib

import pytest

from dana.core.content.normalizer import (
    ContentNormalizer,
    NormalizationError,
    NormalizedMediaBlock,
    NormalizedTextBlock,
    UnsupportedBlockType,
)
from dana.core.content.validation import MimeTypeError, OversizedError, TraversalError


# ===========================================================================
# Text block normalization
# ===========================================================================


class TestNormalizeText:
    """Text blocks pass through unchanged."""

    def test_text_block(self) -> None:
        normalizer = ContentNormalizer()
        result = normalizer.normalize([{"type": "text", "text": "hello world"}])
        assert len(result) == 1
        block = result[0]
        assert isinstance(block, NormalizedTextBlock)
        assert block.text == "hello world"

    def test_empty_text(self) -> None:
        normalizer = ContentNormalizer()
        result = normalizer.normalize([{"type": "text", "text": ""}])
        assert len(result) == 1
        assert result[0].text == ""

    def test_text_with_no_text_key(self) -> None:
        normalizer = ContentNormalizer()
        result = normalizer.normalize([{"type": "text"}])
        assert result[0].text == ""


# ===========================================================================
# Image block normalization
# ===========================================================================


class TestNormalizeImage:
    """Image blocks are validated and normalized."""

    def test_image_inline_data(self) -> None:
        normalizer = ContentNormalizer()
        data = b"fake-image-data"
        result = normalizer.normalize([{"type": "image", "media_type": "image/png", "data": data}])
        assert len(result) == 1
        block = result[0]
        assert isinstance(block, NormalizedMediaBlock)
        assert block.type == "image"
        assert block.media_type == "image/png"
        assert block.content == data
        assert block.sha256 == hashlib.sha256(data).hexdigest()
        assert block.size == len(data)
        assert block.artifact_uri is None

    def test_image_from_path(self, tmp_path) -> None:
        normalizer = ContentNormalizer()
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake-image-data")
        result = normalizer.normalize(
            [{"type": "image", "media_type": "image/png", "path": str(img_file)}],
            workspace=str(tmp_path),
        )
        assert len(result) == 1
        block = result[0]
        assert isinstance(block, NormalizedMediaBlock)
        assert block.type == "image"
        assert block.content == b"fake-image-data"

    def test_image_rejected_mime(self) -> None:
        normalizer = ContentNormalizer()
        with pytest.raises(MimeTypeError, match="not allowed"):
            normalizer.normalize([{"type": "image", "media_type": "video/mp4", "data": b"x"}])

    def test_image_no_data_or_path(self) -> None:
        normalizer = ContentNormalizer()
        with pytest.raises(NormalizationError, match="no 'data' or 'path'"):
            normalizer.normalize([{"type": "image", "media_type": "image/png"}])


# ===========================================================================
# Embedded resource normalization
# ===========================================================================


class TestNormalizeEmbeddedResource:
    """Embedded resource blocks are validated."""

    def test_embedded_resource_inline(self) -> None:
        normalizer = ContentNormalizer()
        data = b'{"key": "value"}'
        result = normalizer.normalize(
            [{"type": "embedded_resource", "media_type": "application/json", "data": data}]
        )
        assert len(result) == 1
        block = result[0]
        assert isinstance(block, NormalizedMediaBlock)
        assert block.type == "embedded_resource"
        assert block.content == data

    def test_embedded_resource_rejected_mime(self) -> None:
        normalizer = ContentNormalizer()
        with pytest.raises(MimeTypeError):
            normalizer.normalize(
                [{"type": "embedded_resource", "media_type": "video/mp4", "data": b"x"}]
            )


# ===========================================================================
# File resource normalization
# ===========================================================================


class TestNormalizeFileResource:
    """File resource blocks are validated with path safety."""

    def test_file_resource_from_path(self, tmp_path) -> None:
        normalizer = ContentNormalizer()
        doc_file = tmp_path / "doc.txt"
        doc_file.write_text("hello world")
        result = normalizer.normalize(
            [{"type": "file_resource", "media_type": "text/plain", "path": str(doc_file)}],
            workspace=str(tmp_path),
        )
        assert len(result) == 1
        block = result[0]
        assert isinstance(block, NormalizedMediaBlock)
        assert block.type == "file_resource"
        assert block.content == b"hello world"

    def test_file_resource_traversal_raises(self, tmp_path) -> None:
        normalizer = ContentNormalizer()
        with pytest.raises(TraversalError):
            normalizer.normalize(
                [{"type": "file_resource", "media_type": "text/plain", "path": "/etc/passwd"}],
                workspace=str(tmp_path),
            )

    def test_file_resource_not_found(self, tmp_path) -> None:
        normalizer = ContentNormalizer()
        with pytest.raises(NormalizationError, match="file not found"):
            normalizer.normalize(
                [
                    {
                        "type": "file_resource",
                        "media_type": "text/plain",
                        "path": str(tmp_path / "nonexistent.txt"),
                    }
                ],
                workspace=str(tmp_path),
            )


# ===========================================================================
# Size enforcement
# ===========================================================================


class TestSizeEnforcement:
    """Oversized content is rejected or converted to artifact references."""

    def test_oversized_inline_rejected(self) -> None:
        normalizer = ContentNormalizer(max_block_size=10)
        with pytest.raises(OversizedError):
            normalizer.normalize([{"type": "image", "media_type": "image/png", "data": b"x" * 20}])

    def test_large_content_becomes_artifact_reference(self) -> None:
        normalizer = ContentNormalizer(inline_size_limit=5, max_block_size=100)
        data = b"x" * 20
        result = normalizer.normalize([{"type": "image", "media_type": "image/png", "data": data}])
        block = result[0]
        assert isinstance(block, NormalizedMediaBlock)
        assert block.content == b""  # No inline content
        assert block.artifact_uri is not None
        assert block.artifact_uri.startswith("artifact://")
        assert block.sha256 == hashlib.sha256(data).hexdigest()
        assert block.size == 20

    def test_oversized_file_rejected(self, tmp_path) -> None:
        normalizer = ContentNormalizer(max_block_size=10)
        big_file = tmp_path / "big.txt"
        big_file.write_bytes(b"x" * 20)
        with pytest.raises(OversizedError):
            normalizer.normalize(
                [{"type": "file_resource", "media_type": "text/plain", "path": str(big_file)}],
                workspace=str(tmp_path),
            )


# ===========================================================================
# Unsupported block types
# ===========================================================================


class TestUnsupportedBlockType:
    """Unknown block types raise UnsupportedBlockType."""

    def test_unknown_type_raises(self) -> None:
        normalizer = ContentNormalizer()
        with pytest.raises(UnsupportedBlockType, match="unsupported"):
            normalizer.normalize([{"type": "unknown_type", "data": b"x"}])

    def test_empty_type_raises(self) -> None:
        normalizer = ContentNormalizer()
        with pytest.raises(UnsupportedBlockType):
            normalizer.normalize([{"type": "", "data": b"x"}])


# ===========================================================================
# Multiple blocks
# ===========================================================================


class TestMultipleBlocks:
    """Multiple blocks are normalized in order."""

    def test_mixed_blocks(self) -> None:
        normalizer = ContentNormalizer()
        result = normalizer.normalize(
            [
                {"type": "text", "text": "hello"},
                {"type": "image", "media_type": "image/png", "data": b"img"},
                {"type": "text", "text": "world"},
            ]
        )
        assert len(result) == 3
        assert isinstance(result[0], NormalizedTextBlock)
        assert result[0].text == "hello"
        assert isinstance(result[1], NormalizedMediaBlock)
        assert result[1].type == "image"
        assert isinstance(result[2], NormalizedTextBlock)
        assert result[2].text == "world"
