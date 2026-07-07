"""Default implementations for repository protocol methods.

External repository implementations that don't override `list_sessions`
get the no-op default (returns empty list), which lets CompressedTimeline
fall back to single-session semantics without breaking.
"""

from __future__ import annotations


class TimelineRepositoryDefaultsMixin:
    """Backward-compat shim for TimelineRepositoryProtocol.list_sessions.

    Inherit from this before the protocol on existing custom repos
    to avoid implementing list_sessions when a flat session model is fine.
    """

    def list_sessions(self, prefix: str = "") -> list[str]:
        _ = prefix  # no-op default; external repos override if they support listing
        return []
