"""Normalized Operation with effect metadata for policy evaluation.

Per ADR-006: the policy evaluates normalized Operations carrying Tool Identity,
effects, validated arguments, affected locations, owner, workspace, and session
context — not hard-coded tool names.

Per ADR-004: the policy reads effect metadata from the Operation, which comes
from the catalog entry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dana.core.policy.effects import EffectMetadata
from dana.core.tool.catalog import ToolCatalog, ToolIdentity


@dataclass(frozen=True)
class Operation:
    """Normalized, read-only view of a tool invocation for policy evaluation.

    Richer than the event-bus ``Operation`` (``dana.core.ext.operation``):
    carries effect metadata from the catalog entry, affected locations, and
    session context so the policy can make informed allow/deny decisions.

    Attributes:
        tool_identity: Stable, provider-neutral identity from the catalog.
        arguments: Read-only mapping of validated arguments.
        effects: Normalized effect metadata from the catalog entry.
        affected_locations: Resource paths or identifiers this operation
            touches (file paths, URLs, database tables, etc.).
        owner: The agent or user who owns this operation.
        workspace: The workspace context this operation runs in.
        session_context: Additional session-level context for policy evaluation.
    """

    tool_identity: ToolIdentity
    arguments: Mapping[str, Any]
    effects: EffectMetadata
    affected_locations: tuple[str, ...] = ()
    owner: str | None = None
    workspace: str | None = None
    session_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.arguments, MappingProxyType):
            object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


def build_policy_operation(
    tool_call: Mapping[str, Any],
    catalog: ToolCatalog | None = None,
    *,
    owner: str | None = None,
    workspace: str | None = None,
    session_context: dict[str, Any] | None = None,
) -> Operation:
    """Build a policy ``Operation`` from a raw tool_call dict and optional catalog.

    When a ``catalog`` is provided, the effect metadata is read from the
    matching catalog entry (per ADR-004). When no catalog or no match is found,
    the operation is treated as unknown/sensitive (fail cautious).

    ``affected_locations`` is populated from common argument names
    (``path``, ``file``, ``url``, ``target``, ``directory``).

    Args:
        tool_call: The raw tool call dict from the model.
        catalog: Optional ToolCatalog to resolve effect metadata.
        owner: Optional owner identifier.
        workspace: Optional workspace identifier.
        session_context: Optional session context dict.

    Returns:
        A frozen ``Operation`` ready for policy evaluation.
    """
    function_name = tool_call.get("function", "")
    arguments: Mapping[str, Any] = dict(tool_call.get("arguments", {}))

    # Resolve effect metadata from catalog entry (ADR-004)
    effects: EffectMetadata
    tool_identity: ToolIdentity
    if catalog is not None:
        entry = catalog.get(function_name)
        if entry is not None:
            tool_identity = entry.identity
            effects = entry.effects
        else:
            # Tool not in catalog — unknown/sensitive
            tool_identity = ToolIdentity(name=function_name)
            effects = EffectMetadata.unknown()
    else:
        # No catalog wired — unknown/sensitive
        tool_identity = ToolIdentity(name=function_name)
        effects = EffectMetadata.unknown()

    # Extract affected locations from common argument names
    _LOCATION_KEYS = frozenset({"path", "file", "url", "target", "directory"})
    affected_locations: list[str] = []
    for key in _LOCATION_KEYS:
        val = arguments.get(key)
        if isinstance(val, str) and val:
            affected_locations.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item:
                    affected_locations.append(item)

    return Operation(
        tool_identity=tool_identity,
        arguments=arguments,
        effects=effects,
        affected_locations=tuple(affected_locations),
        owner=owner,
        workspace=workspace,
        session_context=session_context or {},
    )
