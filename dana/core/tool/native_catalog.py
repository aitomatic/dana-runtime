"""Native-tool ToolCatalog for policy classification (D7.6 — middle path).

Builds a :class:`ToolCatalog` from the agent's native-tool schemas so the
permission policy can classify each tool (normal -> non-sensitive -> not
hard-denied; unknown -> sensitive -> hard-denied, fail-cautious per ADR-006).

This catalog feeds the **policy classifier only**. It does NOT reroute tool
execution — the STAR loop still calls native tools directly (D7.5 Decision 2 /
ADR: the D2 ``ToolExecutionEngine`` reroute is deferred; native tools are the
canonical live mechanism). The catalog is consumed by
:func:`~dana.core.policy.operations.build_policy_operation` and the
``TOOL_CALL`` permission hook on the agent EventBus.

Per ADR-004 each turn pins one immutable catalog version (see
:meth:`AgentSession` turn wiring / the ``version`` field on :class:`ToolCatalog`).

Effect classification is a **static, explicitly-extended table**. A tool name
NOT in the table falls through to :meth:`EffectMetadata.unknown` ->
``is_sensitive=True`` -> hard-denied. A new native tool MUST be classified
explicitly here — that is the intended fail-cautious safety behavior, not a
bug. See the D7.6 ADR note (Decision Log).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dana.core.policy.effects import Effect, EffectKind, EffectMetadata
from dana.core.tool.catalog import ToolCatalog, ToolCatalogEntry, ToolIdentity


# ---------------------------------------------------------------------------
# Static effect-classification table — extend explicitly when adding tools.
#
# name -> (effect_kinds, is_sensitive)
#
# All known native tools are ``is_sensitive=False``: they flow to the
# mode/grant/prompt layer (NEEDS_PROMPT -> proceed per the D7.5 adjusted ruling;
# interactive prompting is a host-layer follow-up). Hard-deny enforcement stays
# LIVE via: (a) unknown tools (is_sensitive, fail-cautious), (b) destructive
# {DELETE, MODIFY} on protected paths {.env, node_modules}, (c) rm -rf.
# (See the D7.6 ADR note + the 2 filed findings for the protected-path / rm -rf
# gaps, which are pre-existing and out of scope here.)
# ---------------------------------------------------------------------------
_NATIVE_TOOL_EFFECTS: dict[str, tuple[tuple[EffectKind, ...], bool]] = {
    # --- read-only (no side effects) ---
    "Read": ((EffectKind.READ,), False),
    "Grep": ((EffectKind.READ,), False),
    "Glob": ((EffectKind.READ,), False),
    "read_tool_result": ((EffectKind.READ,), False),
    "bash__get_task_output": ((EffectKind.READ,), False),
    "bash__list_tasks": ((EffectKind.READ,), False),
    "TaskOutput": ((EffectKind.READ,), False),
    # --- internal in-memory state (todo list) ---
    "todo__todo_write": ((EffectKind.WRITE,), False),
    # --- file-mutating (hard-deny still fires on protected paths via MODIFY) ---
    "Edit": ((EffectKind.MODIFY,), False),
    "Write": ((EffectKind.CREATE,), False),
    # --- execution / shell / subprocess ---
    "Task": ((EffectKind.EXECUTE,), False),
    "bash__execute": ((EffectKind.EXECUTE,), False),
    "bash__kill_task": ((EffectKind.EXECUTE,), False),
    # --- dynamic (runs user code / external MCP server) ---
    "Skill": ((EffectKind.EXECUTE,), False),
    "call_mcp_tool": ((EffectKind.EXECUTE,), False),
}


def _effect_metadata_for(name: str, mcp_names: frozenset[str] | None = None) -> EffectMetadata:
    """Resolve effect metadata for a native tool name (fail-cautious on miss).

    Namespaced MCP tool names (``server:tool``) registered in ``mcp_names`` are
    classified EXECUTE / non-sensitive (D7 follow-up 2: per-MCP policy) so they
    flow to mode/grant/prompt instead of being hard-denied as unknown.
    """
    spec = _NATIVE_TOOL_EFFECTS.get(name)
    if spec is None:
        if mcp_names and name in mcp_names:
            # D7 follow-up 2: a registered MCP tool -> EXECUTE, non-sensitive, so it
            # flows to mode/grant/prompt instead of being hard-denied as unknown.
            #
            # The effect's TARGET is the server (I9): the server is the honest
            # conflict domain. The harness knows nothing about a server's own
            # resources, so two unannotated mutations of one server may collide
            # while two different servers do not. A per-call *argument* target would
            # be an unfounded guess for an unannotated server (a ``path`` key it
            # ignores would look provably disjoint when it is not); a shared
            # placeholder would over-serialize every server against every other.
            # Read-only tools overlap through their own declared metadata instead
            # (:func:`dana.core.mcp.dispatch_wrapper._mcp_effect_metadata`).
            return EffectMetadata(
                effects=(Effect(kind=EffectKind.EXECUTE, target=name.split(":", 1)[0]),),
                is_sensitive=False,
            )
        # Unknown/unlisted tool -> fail-cautious (is_sensitive=True -> hard-deny).
        return EffectMetadata.unknown()
    kinds, is_sensitive = spec
    if not kinds:
        # No declared effect kinds -> empty (non-sensitive) unless explicitly sensitive.
        return EffectMetadata.unknown() if is_sensitive else EffectMetadata.empty()
    return EffectMetadata(
        effects=tuple(Effect(kind=k) for k in kinds),
        is_sensitive=is_sensitive,
    )


def native_effect_metadata(name: str, mcp_names: frozenset[str] | None = None) -> EffectMetadata:
    """Public resolver for a native tool's effect metadata (fail-cautious on miss).

    Used by the tool executor's batch scheduler to order conflicting effects when
    the session-owned catalog is not wired on the executor. Same static table and
    same ADR-006 fail-cautious fallback as the catalog build below, so classification
    cannot drift between the policy path and the scheduling path.
    """
    return _effect_metadata_for(name, mcp_names)


def build_native_tool_catalog(
    native_tools: list[Mapping[str, Any]],
    *,
    version: int = 0,
    mcp_names: frozenset[str] | None = None,
    mcp_effects: Mapping[str, EffectMetadata] | None = None,
) -> ToolCatalog:
    """Build a policy-classification catalog from the agent's native-tool schemas.

    Each native-tool schema is the OpenAI-compatible dict
    ``{"type": "function", "function": {"name": ..., "parameters": ...}}`` (the
    shape produced by :class:`AgentRuntime._build_native_tools_if_supported`).

    The catalog entry's ``adapter`` is **not executable** — it is ``None``, and the
    ToolExecutor refuses to dispatch a catalog entry whose adapter is not callable
    (``catalog_entry_not_executable``). This is deliberate: the catalog exists to
    *classify* native tools for the permission policy, and the STAR loop calls the
    native tools directly (D7.5 Decision 2). A classification-only entry can never
    silently become a fake tool result.

    Args:
        native_tools: The agent's native-tool schema list (``runtime._native_tools``).
        version: A per-turn pinned catalog version (AC #1 — stable identity +
            per-turn versioned; ADR-004).
        mcp_names: Namespaced MCP tool names (``server:tool``) to classify as
            EXECUTE / non-sensitive (D7 follow-up 2: per-MCP policy). Other
            unknown names stay fail-cautious.
        mcp_effects: Server-declared effects for registered MCP names. Missing
            declarations retain conservative per-server EXECUTE classification.

    Returns:
        A :class:`ToolCatalog` with one entry per native tool, classified for
        policy. Unknown/unlisted tool names are marked sensitive (fail-cautious).
    """
    entries: list[ToolCatalogEntry] = []
    seen: set[str] = set()
    for tool in native_tools:
        if not isinstance(tool, Mapping):
            continue
        fn = tool.get("function", tool)
        name = fn.get("name") if isinstance(fn, Mapping) else tool.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        schema = dict(tool)
        declared = mcp_effects.get(name) if mcp_effects and mcp_names and name in mcp_names else None
        entries.append(
            ToolCatalogEntry(
                identity=ToolIdentity(name=name, source="native"),
                schema=schema,
                # Classification-only: no adapter. The ToolExecutor refuses to
                # execute an entry whose adapter is not callable, so a
                # classification entry can never be dispatched as a tool.
                adapter=None,
                effects=declared or _effect_metadata_for(name, mcp_names),
            )
        )
    return ToolCatalog(entries, version=version)
