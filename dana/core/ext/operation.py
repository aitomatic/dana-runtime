"""Operation — normalized view of a tool invocation for the event bus (M3).

Thin v0.1 wrapper (decision locked in sprint/plans/S3-tool-execution-engine.md).
Fields: ``tool_identity`` + ``arguments`` only. Effects/locations are vNext.

A tool intercept handler inspects an ``Operation`` to decide allow/deny/modify.
PermissionPolicy (``permission.py``) is one such handler.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ToolIdentity:
    """Who is being invoked.

    ``name``  — the function name as it appears in the tool_call dict
                (the ``@named_tool`` alias or the ``ClassName:method`` string).
    ``source``— provenance hint: the object's ``resource_id``/``object_id`` for
                registry hits, else the object's class name, else ``None`` for the
                generic ``ClassName:method`` fallback path.
    """

    name: str
    source: str | None = None


@dataclass(frozen=True)
class Operation:
    """Normalized, read-only view of one tool invocation.

    ``arguments`` is normalized to a ``MappingProxyType`` in ``__post_init__``
    regardless of construction path, so item mutation raises ``TypeError`` at
    runtime. This enforces the "do not mutate payload" contract (S1) for the
    Operation specifically. To change a call, return a bus result dict
    ``{"modify": {"arguments": ...}}`` instead.
    """

    tool_identity: ToolIdentity
    arguments: Mapping[str, Any]
    # vNext: effects: list[str], locations: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.arguments, MappingProxyType):
            object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


def build_operation(
    tool_call: Mapping[str, Any],
    registry: Mapping[str, tuple[Any, str]],
) -> Operation:
    """Derive an ``Operation`` from a raw tool_call dict + the @named_tool registry.

    Mirrors the exact derivation in the M3 plan:

    - ``function_name`` = ``tool_call["function"]`` ("" if missing).
    - ``arguments``     = a copy of ``tool_call["arguments"]`` ({} if missing);
                          ``Operation.__post_init__`` wraps it read-only.
    - ``source``        = ``resource_id``/``object_id`` of the registered object,
                          else its class name, else ``None`` (generic fallback path).
    """
    function_name = tool_call.get("function", "")
    arguments: Mapping[str, Any] = dict(tool_call.get("arguments", {}))
    source: str | None = None
    if function_name in registry:
        obj, _method_name = registry[function_name]
        source = getattr(obj, "resource_id", None) or getattr(obj, "object_id", None) or type(obj).__name__
    return Operation(tool_identity=ToolIdentity(name=function_name, source=source), arguments=arguments)
