"""
Unit tests for content validation — MIME type, size, and path-safety checks.

Covers:
- MIME type validation (allowed/rejected)
- Size validation (within/over limits)
- Path safety (traversal detection, workspace scoping)
"""

from __future__ import annotations

import pytest

from dana.core.content.validation import (
    MimeTypeError,
    OversizedError,
    TraversalError,
    validate_mime_type,
    validate_path_safety,
    validate_size,
)


# ===========================================================================
# MIME type validation
# ===========================================================================


class TestValidateMimeType:
    """MIME type validation rejects unsupported types."""

    ALLOWED = frozenset({"image/png", "image/jpeg", "text/plain"})

    def test_allowed_mime_passes(self) -> None:
        validate_mime_type("image/png", self.ALLOWED)  # no error

    def test_allowed_text_passes(self) -> None:
        validate_mime_type("text/plain", self.ALLOWED)  # no error

    def test_rejected_mime_raises(self) -> None:
        with pytest.raises(MimeTypeError, match="not allowed"):
            validate_mime_type("application/pdf", self.ALLOWED)

    def test_empty_mime_raises(self) -> None:
        with pytest.raises(MimeTypeError, match="empty"):
            validate_mime_type("", self.ALLOWED)

    def test_error_contains_allowed_types(self) -> None:
        with pytest.raises(MimeTypeError) as exc:
            validate_mime_type("video/mp4", self.ALLOWED)
        assert "image/png" in str(exc.value)
        assert "image/jpeg" in str(exc.value)

    def test_error_has_attributes(self) -> None:
        try:
            validate_mime_type("video/mp4", self.ALLOWED)
        except MimeTypeError as e:
            assert e.media_type == "video/mp4"
            assert e.allowed == self.ALLOWED


# ===========================================================================
# Size validation
# ===========================================================================


class TestValidateSize:
    """Size validation rejects oversized content."""

    def test_valid_size_passes(self) -> None:
        validate_size(100, 1000)  # no error

    def test_exact_max_passes(self) -> None:
        validate_size(1000, 1000)  # no error

    def test_zero_size_passes(self) -> None:
        validate_size(0, 1000)  # no error

    def test_oversized_raises(self) -> None:
        with pytest.raises(OversizedError, match="exceeds maximum"):
            validate_size(1001, 1000)

    def test_negative_size_raises(self) -> None:
        with pytest.raises(OversizedError):
            validate_size(-1, 1000)

    def test_error_has_attributes(self) -> None:
        try:
            validate_size(2000, 1000)
        except OversizedError as e:
            assert e.size == 2000
            assert e.max_size == 1000


# ===========================================================================
# Path safety validation
# ===========================================================================


class TestValidatePathSafety:
    """Path safety detects traversal attempts."""

    def test_normal_path_within_workspace_passes(self, tmp_path) -> None:
        file = tmp_path / "subdir" / "file.txt"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("hello")
        validate_path_safety(str(file), str(tmp_path))  # no error

    def test_path_outside_workspace_raises(self, tmp_path) -> None:
        outside = tmp_path / ".." / "outside.txt"
        outside.write_text("hello")
        with pytest.raises(TraversalError, match="traversal"):
            validate_path_safety(str(outside), str(tmp_path))

    def test_absolute_path_outside_workspace_raises(self, tmp_path) -> None:
        with pytest.raises(TraversalError):
            validate_path_safety("/etc/passwd", str(tmp_path))

    def test_no_workspace_skips_scope_check(self, tmp_path) -> None:
        file = tmp_path / "test.txt"
        file.write_text("hello")
        validate_path_safety(str(file), None)  # no error

    def test_error_has_attributes(self, tmp_path) -> None:
        try:
            validate_path_safety("/etc/passwd", str(tmp_path))
        except TraversalError as e:
            assert e.path == "/etc/passwd"
            assert e.workspace == str(tmp_path)
