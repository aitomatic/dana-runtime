"""Effect classification taxonomy for permission policy.

Defines the effect kinds, effect metadata structure, and classification
rules. Unknown effect metadata is treated as sensitive (fail cautious).

Per ADR-006: unknown effect metadata is sensitive (fail cautious).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EffectKind(Enum):
    """Taxonomy of effect kinds a tool invocation may produce.

    Each kind represents a category of side effect. The policy uses these
    to decide whether an operation is allowed or denied.

    ``UNKNOWN`` is the fallback for uncategorized tools — it is always
    treated as sensitive (fail cautious).
    """

    READ = "read"
    WRITE = "write"
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    EXECUTE = "execute"
    NETWORK = "network"
    IDENTITY = "identity"
    PERSISTENCE = "persistence"
    # Unknown/sensitive — fail cautious when no metadata is declared
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Effect:
    """A single effect a tool invocation may produce.

    Attributes:
        kind: The effect kind from the taxonomy.
        target: The target of the effect (e.g. file path, URL, resource name).
        metadata: Optional additional context about the effect.
    """

    kind: EffectKind
    target: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EffectMetadata:
    """Normalized effect metadata declared by a catalog entry.

    Attributes:
        effects: The list of effects the tool may produce.
        is_sensitive: If True, the tool is treated as sensitive regardless
            of its declared effects. This is the mechanism for "fail cautious"
            on unknown/uncategorized tools.
    """

    effects: tuple[Effect, ...] = ()
    is_sensitive: bool = False

    @classmethod
    def unknown(cls) -> EffectMetadata:
        """Create metadata for unknown/uncategorized tools — fail cautious.

        Returns metadata with a single UNKNOWN effect and is_sensitive=True,
        ensuring the policy treats any tool without declared metadata as
        high-risk.
        """
        return cls(
            effects=(Effect(kind=EffectKind.UNKNOWN, target="unknown"),),
            is_sensitive=True,
        )

    @classmethod
    def empty(cls) -> EffectMetadata:
        """Create empty metadata — no declared effects, not sensitive.

        Use for tools that have no side effects (e.g. read-only queries).
        """
        return cls()
