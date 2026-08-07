"""Tool Identity — stable, provider-neutral identity and collision detection.

Per ADR-004: every tool has one stable identity. Duplicate identities or
provider aliases fail catalog construction (fail early, not at call time).
"""

from __future__ import annotations

from dana.core.tool.catalog import ToolCatalog, ToolCatalogEntry, ToolIdentity


__all__ = [
    "ToolIdentity",
    "ToolCatalogEntry",
    "ToolCatalog",
    "check_collision",
]


def check_collision(
    entries: list[ToolCatalogEntry],
) -> list[ToolCatalogEntry]:
    """Validate entries for collisions and return them if clean.

    Raises ``ValueError`` on the first duplicate identity or alias.
    This is a convenience wrapper around ``ToolCatalog(entries)`` for callers
    that want to validate without keeping the catalog.

    Returns:
        The same list of entries (pass-through on success).
    """
    ToolCatalog(entries)  # validates
    return entries
