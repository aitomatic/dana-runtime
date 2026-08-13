"""Tool Catalog — session-owned, versioned, single source of truth for tool schemas.

Per ADR-004: the session-owned Tool Catalog is the only source for model-visible
schemas and invocation targets. Each turn pins one immutable catalog version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dana.core.policy.effects import EffectMetadata


@dataclass(frozen=True)
class ToolIdentity:
    """Stable, provider-neutral identity for a tool.

    ``name``   — the function name as it appears in the tool_call dict
                 (the ``@named_tool`` alias or the ``ClassName:method`` string).
    ``source`` — provenance hint: the object's ``resource_id``/``object_id`` for
                 registry hits, else the object's class name, else ``None``.
    """

    name: str
    source: str | None = None


@dataclass(frozen=True)
class ToolCatalogEntry:
    """One entry in the Tool Catalog.

    ``identity``       — stable ToolIdentity (provider-neutral).
    ``schema``         — the OpenAI-compatible tool schema dict.
    ``adapter``        — callable that dispatches invocation to the real object.
    ``aliases``        — provider-specific alias names (e.g. MCP tool names).
    ``effects``        — normalized effect metadata declared at registration
                        (per ADR-004: each catalog entry declares normalized
                        effect metadata at registration).

    Cancellation & isolation (D2 — ADR-005):
    ``cancellable``    — whether the tool supports cooperative cancellation.
                         If True, ``max_latency_ms`` MUST be set (no hidden default).
    ``max_latency_ms`` — max time in ms the tool may take to respond to a
                         cancellation request. Required when ``cancellable=True``.
    ``isolated``       — if True, the tool runs in an isolated worker process
                         (for unsafe mutating tools). Implies ``cancellable=False``.
    ``worker_module``  — Python module path for the isolated worker to import
                         (required when ``isolated=True``).
    ``worker_callable`` — callable name within ``worker_module``, e.g.
                         ``"ClassName.method"`` (required when ``isolated=True``).
    """

    identity: ToolIdentity
    schema: dict[str, Any]
    adapter: Any  # Callable[[dict], Any] — v0.1: typed as Any for simplicity
    aliases: frozenset[str] = frozenset()
    effects: EffectMetadata = EffectMetadata.empty()

    # D2: Cancellation & isolation contract (ADR-005)
    # Default to non-cancellable for backward compatibility with existing
    # entries that don't declare cancellation metadata.
    cancellable: bool = False
    max_latency_ms: int | None = None
    isolated: bool = False
    worker_module: str | None = None
    worker_callable: str | None = None

    def __post_init__(self) -> None:
        """Validate cancellation/isolation contract at construction time."""
        if self.cancellable and self.max_latency_ms is None:
            raise ValueError(f"Tool '{self.identity.name}': cancellable=True requires max_latency_ms (no hidden default per ADR-005)")
        if self.isolated and self.cancellable:
            raise ValueError(f"Tool '{self.identity.name}': isolated=True implies cancellable=False")
        if self.isolated and not self.worker_module:
            raise ValueError(f"Tool '{self.identity.name}': isolated=True requires worker_module")
        if self.isolated and not self.worker_callable:
            raise ValueError(f"Tool '{self.identity.name}': isolated=True requires worker_callable")


class ToolCatalog:
    """Versioned, session-owned catalog of all model-visible tools.

    Built once per turn. Duplicate identities or aliases fail construction.
    """

    def __init__(self, entries: list[ToolCatalogEntry], *, version: int = 0) -> None:
        self._entries = list(entries)
        self._by_name: dict[str, ToolCatalogEntry] = {}
        self._by_identity: dict[ToolIdentity, ToolCatalogEntry] = {}
        # ADR-004: a turn pins one immutable catalog version. Default 0; the
        # host adapter stamps a per-turn version when building the catalog.
        self._version = int(version)

        for entry in entries:
            if entry.identity in self._by_identity:
                raise ValueError(f"Duplicate tool identity: {entry.identity}")
            self._by_identity[entry.identity] = entry
            if entry.identity.name in self._by_name:
                raise ValueError(
                    f"Duplicate tool name: {entry.identity.name} (conflicts with {self._by_name[entry.identity.name].identity})"
                )
            self._by_name[entry.identity.name] = entry
            for alias in entry.aliases:
                if alias in self._by_name:
                    raise ValueError(f"Duplicate alias: {alias} (conflicts with {self._by_name[alias].identity.name})")
                self._by_name[alias] = entry

    @property
    def entries(self) -> list[ToolCatalogEntry]:
        return list(self._entries)

    @property
    def version(self) -> int:
        """The pinned catalog version for this turn (ADR-004). Defaults to 0."""
        return self._version

    def get(self, name: str) -> ToolCatalogEntry | None:
        """Look up an entry by its primary name or alias."""
        return self._by_name.get(name)

    def get_by_identity(self, identity: ToolIdentity) -> ToolCatalogEntry | None:
        """Look up an entry by its stable identity."""
        return self._by_identity.get(identity)

    @classmethod
    def build(cls, entries: list[ToolCatalogEntry]) -> ToolCatalog:
        """Factory: construct a catalog from entries.

        Convenience wrapper around ``__init__``. Subclasses may override to
        add validation or enrichment.
        """
        return cls(entries)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """All schemas for model-visible tool definitions."""
        return [entry.schema for entry in self._entries]
