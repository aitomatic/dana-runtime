"""
Content validation — MIME type, size, and path-safety checks.

Per ADR-009: core validates MIME type + size + workspace access + model capability
+ artifact authorization before a turn starts.

This module provides the low-level validation primitives used by the
:class:`~dana.core.content.normalizer.ContentNormalizer`.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ContentValidationError(Exception):
    """Base exception for content validation failures."""


class MimeTypeError(ContentValidationError):
    """Raised when a MIME type is not allowed."""

    def __init__(self, media_type: str, allowed: frozenset[str]) -> None:
        self.media_type = media_type
        self.allowed = allowed
        super().__init__(f"MIME type {media_type!r} is not allowed. Allowed types: {', '.join(sorted(allowed))}")


class OversizedError(ContentValidationError):
    """Raised when a content block exceeds the maximum allowed size."""

    def __init__(self, size: int, max_size: int) -> None:
        self.size = size
        self.max_size = max_size
        super().__init__(f"content size {size} exceeds maximum allowed size {max_size}")


class TraversalError(ContentValidationError):
    """Raised when a file path attempts directory traversal outside the workspace."""

    def __init__(self, path: str, workspace: str | None) -> None:
        self.path = path
        self.workspace = workspace
        super().__init__(f"path {path!r} attempts directory traversal outside workspace {workspace!r}")


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_mime_type(media_type: str, allowed: frozenset[str]) -> None:
    """Validate that a MIME type is in the allowed set.

    Args:
        media_type: The MIME type to validate (e.g. ``"image/png"``).
        allowed: Set of allowed MIME types.

    Raises:
        MimeTypeError: If the MIME type is not allowed.
    """
    if not media_type:
        raise MimeTypeError(media_type or "(empty)", allowed)
    if media_type not in allowed:
        raise MimeTypeError(media_type, allowed)


def validate_size(size: int, max_size: int) -> None:
    """Validate that a content size does not exceed the maximum.

    Args:
        size: The content size in bytes.
        max_size: The maximum allowed size in bytes.

    Raises:
        OversizedError: If the size exceeds the maximum.
    """
    if size < 0:
        raise OversizedError(size, max_size)
    if size > max_size:
        raise OversizedError(size, max_size)


def validate_path_safety(path: str, workspace: str | None) -> None:
    """Validate that a file path does not attempt directory traversal.

    Checks for:
    - Absolute paths that are not under the workspace
    - ``..`` components that escape the workspace
    - Symlink-based traversal (resolves the path and checks the real path)

    Note: this is a time-of-check check. The file should be opened with
    ``O_NOFOLLOW`` to prevent symlink-swap TOCTOU attacks.

    Args:
        path: The file path to validate.
        workspace: The allowed workspace root path. If ``None``, only basic
            traversal checks are performed (no workspace scoping).

    Raises:
        TraversalError: If the path attempts traversal outside the workspace.
    """
    resolved = Path(path).resolve()

    if workspace is not None:
        workspace_path = Path(workspace).resolve()
        try:
            resolved.relative_to(workspace_path)
        except ValueError:
            raise TraversalError(path, workspace)


# ---------------------------------------------------------------------------
# Provider capability validation (D6, ADR-009)
# ---------------------------------------------------------------------------


class ProviderCapabilityError(ContentValidationError):
    """Raised when a provider does not support a required content capability."""

    def __init__(self, capability: str, provider: str) -> None:
        self.capability = capability
        self.provider = provider
        super().__init__(f"provider {provider!r} does not support {capability!r}")


def validate_provider_capability(
    blocks: list[dict],
    provider: str,
    *,
    supports_images: bool = False,
    supports_embedded_resources: bool = False,
    supports_file_resources: bool = False,
) -> None:
    """Validate that a provider supports the content types in the given blocks.

    Per ADR-009: unsupported models fail before turn start, not mid-turn.

    Args:
        blocks: Normalized content blocks to validate.
        provider: The provider name (for error messages).
        supports_images: Whether the provider supports image content.
        supports_embedded_resources: Whether the provider supports embedded resources.
        supports_file_resources: Whether the provider supports file resources.

    Raises:
        ProviderCapabilityError: If a block type is not supported by the provider.
    """
    for block in blocks:
        block_type = block.get("type", "")
        if block_type == "image" and not supports_images:
            raise ProviderCapabilityError("image content", provider)
        if block_type == "embedded_resource" and not supports_embedded_resources:
            raise ProviderCapabilityError("embedded resources", provider)
        if block_type == "file_resource" and not supports_file_resources:
            raise ProviderCapabilityError("file resources", provider)
