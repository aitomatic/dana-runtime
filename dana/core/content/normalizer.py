"""
Content normalizer — normalizes raw multimodal content blocks into canonical form.

Per ADR-009 (Multimodal Content and Artifact References):
AgentSession accepts normalized text/image/embedded-resource/file-resource blocks.
Large payloads are artifact references (hash/URI/media-type/size/access-metadata).

The normalizer:
1. Validates MIME type against allowed types
2. Validates payload size against configured boundaries
3. Detects path traversal attempts
4. Computes SHA-256 hash for deduplication
5. Produces artifact references for large payloads
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal

from dana.core.content.validation import (
    validate_mime_type,
    validate_path_safety,
    validate_size,
)


# ---------------------------------------------------------------------------
# Normalized block types
# ---------------------------------------------------------------------------

NormalizedBlockType = Literal["text", "image", "embedded_resource", "file_resource"]


@dataclass(frozen=True, slots=True)
class NormalizedTextBlock:
    """A normalized text content block."""

    type: Literal["text"] = "text"
    text: str = ""


@dataclass(frozen=True, slots=True)
class NormalizedMediaBlock:
    """A normalized media content block (image/audio/video/document).

    When the payload exceeds the inline size threshold, ``artifact_uri`` is set
    and ``content`` is empty.
    """

    type: Literal["image", "embedded_resource", "file_resource"] = "image"
    media_type: str = ""
    content: bytes = b""
    sha256: str = ""
    size: int = 0
    artifact_uri: str | None = None


# Union of all normalized block types
NormalizedBlock = NormalizedTextBlock | NormalizedMediaBlock


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NormalizationError(Exception):
    """Base exception for content normalization failures."""


class UnsupportedBlockType(NormalizationError):
    """Raised when a block type is not supported."""

    def __init__(self, block_type: str) -> None:
        self.block_type = block_type
        super().__init__(f"unsupported content block type: {block_type!r}")


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

# Maximum size for inline content (bytes). Content larger than this becomes an
# artifact reference.
DEFAULT_INLINE_SIZE_LIMIT = 1_000_000  # 1 MB

# Maximum size for any single content block (bytes).
DEFAULT_MAX_BLOCK_SIZE = 100_000_000  # 100 MB

# Allowed MIME type prefixes for each block type.
ALLOWED_IMAGE_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/avif",
        "image/tiff",
        "image/bmp",
    }
)

ALLOWED_DOCUMENT_MIME_TYPES = frozenset(
    {
        "application/pdf",
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/json",
        "application/xml",
        "text/html",
    }
)

ALLOWED_RESOURCE_MIME_TYPES = frozenset(
    {
        "application/octet-stream",
        "application/zip",
        "application/gzip",
        "application/x-tar",
        "application/x-7z-compressed",
    }
)


# ---------------------------------------------------------------------------
# Content normalizer
# ---------------------------------------------------------------------------


class ContentNormalizer:
    """Normalizes raw content blocks into canonical form with validation.

    Usage::

        normalizer = ContentNormalizer()
        blocks = normalizer.normalize(raw_blocks, workspace="/home/user/project")
    """

    def __init__(
        self,
        inline_size_limit: int = DEFAULT_INLINE_SIZE_LIMIT,
        max_block_size: int = DEFAULT_MAX_BLOCK_SIZE,
    ) -> None:
        self._inline_size_limit = inline_size_limit
        self._max_block_size = max_block_size

    def normalize(
        self,
        blocks: list[dict],
        workspace: str | None = None,
    ) -> list[NormalizedBlock]:
        """Normalize a list of raw content blocks.

        Each raw block is a dict with at least a ``type`` key. Supported types:

        - ``text``: ``{"type": "text", "text": "..."}``
        - ``image``: ``{"type": "image", "media_type": "...", "data": b"..."}``
          or ``{"type": "image", "media_type": "...", "path": "..."}``
        - ``embedded_resource``: ``{"type": "embedded_resource", "media_type": "...", "data": b"..."}``
        - ``file_resource``: ``{"type": "file_resource", "media_type": "...", "path": "..."}``

        Args:
            blocks: Raw content blocks to normalize.
            workspace: Optional workspace root path for path-safety checks.

        Returns:
            List of normalized blocks.

        Raises:
            NormalizationError: On unsupported block types.
            MimeTypeError: On unsupported MIME types.
            OversizedError: On payloads exceeding size limits.
            TraversalError: On path traversal attempts.
        """
        normalized: list[NormalizedBlock] = []
        for block in blocks:
            block_type = block.get("type", "")
            if block_type == "text":
                normalized.append(self._normalize_text(block))
            elif block_type == "image":
                normalized.append(self._normalize_image(block, workspace))
            elif block_type == "embedded_resource":
                normalized.append(self._normalize_embedded_resource(block))
            elif block_type == "file_resource":
                normalized.append(self._normalize_file_resource(block, workspace))
            else:
                raise UnsupportedBlockType(block_type)
        return normalized

    def _normalize_text(self, block: dict) -> NormalizedTextBlock:
        """Normalize a text block."""
        text = block.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        return NormalizedTextBlock(text=text)

    def _normalize_image(self, block: dict, workspace: str | None) -> NormalizedMediaBlock:
        """Normalize an image block."""
        media_type = block.get("media_type", "")
        validate_mime_type(media_type, ALLOWED_IMAGE_MIME_TYPES)

        return self._normalize_media_block(block, media_type, "image", workspace)

    def _normalize_embedded_resource(self, block: dict, workspace: str | None = None) -> NormalizedMediaBlock:
        """Normalize an embedded resource block (inline data only).

        Embedded resources are for inline data only. If a ``path`` field is
        present, it is rejected — use ``file_resource`` for file-based content.
        """
        media_type = block.get("media_type", "")
        validate_mime_type(media_type, ALLOWED_DOCUMENT_MIME_TYPES | ALLOWED_RESOURCE_MIME_TYPES)

        if block.get("path") is not None:
            raise NormalizationError("embedded_resource blocks cannot use 'path'; use file_resource instead")

        return self._normalize_media_block(block, media_type, "embedded_resource", workspace)

    def _normalize_file_resource(self, block: dict, workspace: str | None) -> NormalizedMediaBlock:
        """Normalize a file resource block (file path reference)."""
        media_type = block.get("media_type", "")
        validate_mime_type(media_type, ALLOWED_DOCUMENT_MIME_TYPES | ALLOWED_RESOURCE_MIME_TYPES)

        return self._normalize_media_block(block, media_type, "file_resource", workspace)

    def _normalize_media_block(
        self,
        block: dict,
        media_type: str,
        block_type: NormalizedBlockType,
        workspace: str | None,
    ) -> NormalizedMediaBlock:
        """Normalize a media block from either inline data or file path."""
        # Check for path-based content
        path = block.get("path")
        if path is not None:
            return self._normalize_from_path(path, media_type, block_type, workspace)

        # Check for inline data
        data = block.get("data")
        if data is not None:
            return self._normalize_from_data(data, media_type, block_type)

        # No content source found
        raise NormalizationError(f"block of type {block_type!r} has no 'data' or 'path' field")

    def _normalize_from_path(
        self,
        path: str,
        media_type: str,
        block_type: NormalizedBlockType,
        workspace: str | None,
    ) -> NormalizedMediaBlock:
        """Normalize a block whose content is at a file path."""
        # Validate path safety (traversal check)
        validate_path_safety(path, workspace)

        # Resolve the path and check size
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise NormalizationError(f"file not found: {path}")

        size = resolved.stat().st_size
        validate_size(size, self._max_block_size)

        # Read content
        content = resolved.read_bytes()

        # Compute hash
        sha256 = hashlib.sha256(content).hexdigest()

        # Check if content should be inlined or referenced
        if size <= self._inline_size_limit:
            return NormalizedMediaBlock(
                type=block_type,
                media_type=media_type,
                content=content,
                sha256=sha256,
                size=size,
            )

        # Large content — produce artifact reference
        artifact_uri = self._build_artifact_uri(sha256, media_type)
        return NormalizedMediaBlock(
            type=block_type,
            media_type=media_type,
            content=b"",
            sha256=sha256,
            size=size,
            artifact_uri=artifact_uri,
        )

    def _normalize_from_data(
        self,
        data: bytes,
        media_type: str,
        block_type: NormalizedBlockType,
    ) -> NormalizedMediaBlock:
        """Normalize a block with inline data."""
        if not isinstance(data, (bytes, bytearray)):
            if isinstance(data, str):
                data = data.encode("utf-8")
            else:
                data = bytes(data)

        size = len(data)
        validate_size(size, self._max_block_size)

        sha256 = hashlib.sha256(data).hexdigest()

        if size <= self._inline_size_limit:
            return NormalizedMediaBlock(
                type=block_type,
                media_type=media_type,
                content=bytes(data),
                sha256=sha256,
                size=size,
            )

        artifact_uri = self._build_artifact_uri(sha256, media_type)
        return NormalizedMediaBlock(
            type=block_type,
            media_type=media_type,
            content=b"",
            sha256=sha256,
            size=size,
            artifact_uri=artifact_uri,
        )

    @staticmethod
    def _build_artifact_uri(sha256: str, media_type: str) -> str:
        """Build an artifact URI from hash and media type."""
        ext = _mime_to_extension(media_type)
        return f"artifact://{sha256}{ext}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_MIME_EXTENSION_MAP: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/avif": ".avif",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/csv": ".csv",
    "application/json": ".json",
    "application/xml": ".xml",
    "text/html": ".html",
    "application/octet-stream": ".bin",
    "application/zip": ".zip",
    "application/gzip": ".gz",
    "application/x-tar": ".tar",
    "application/x-7z-compressed": ".7z",
}


def _mime_to_extension(media_type: str) -> str:
    """Map a MIME type to a file extension."""
    return _MIME_EXTENSION_MAP.get(media_type, ".bin")
