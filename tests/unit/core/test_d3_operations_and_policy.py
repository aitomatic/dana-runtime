"""D3 Operations & Effect Metadata — tests for effect classification, operations,
and hard-deny-wins policy enforcement.

Each test maps to one acceptance criterion or edge case from the story.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dana.core.policy.effects import Effect, EffectKind, EffectMetadata
from dana.core.policy.hard_policy import HardPolicy, create_default_hard_policy
from dana.core.policy.operations import Operation, build_policy_operation
from dana.core.tool.catalog import ToolCatalog, ToolCatalogEntry, ToolIdentity


# =========================================================================
# AC #1 — Operations + effect metadata defined
# =========================================================================


class TestEffectKind:
    """EffectKind taxonomy — all expected kinds exist."""

    def test_all_effect_kinds_defined(self):
        """The taxonomy includes all expected effect kinds."""
        kinds = {
            EffectKind.READ,
            EffectKind.WRITE,
            EffectKind.CREATE,
            EffectKind.MODIFY,
            EffectKind.DELETE,
            EffectKind.EXECUTE,
            EffectKind.NETWORK,
            EffectKind.IDENTITY,
            EffectKind.PERSISTENCE,
            EffectKind.UNKNOWN,
        }
        assert set(EffectKind) == kinds

    def test_unknown_is_sensitive_fallback(self):
        """UNKNOWN is the fallback for uncategorized tools."""
        assert EffectKind.UNKNOWN.value == "unknown"


class TestEffect:
    """Effect — a single effect a tool invocation may produce."""

    def test_effect_is_frozen(self):
        """Effect fields cannot be reassigned after construction."""
        effect = Effect(kind=EffectKind.READ, target="/tmp/file")
        with pytest.raises(FrozenInstanceError):
            effect.kind = EffectKind.WRITE  # type: ignore[misc]

    def test_effect_default_target_is_empty(self):
        """target defaults to empty string."""
        effect = Effect(kind=EffectKind.READ)
        assert effect.target == ""

    def test_effect_default_metadata_is_empty(self):
        """metadata defaults to empty dict."""
        effect = Effect(kind=EffectKind.READ)
        assert effect.metadata == {}


class TestEffectMetadata:
    """EffectMetadata — normalized effect metadata for catalog entries."""

    def test_empty_metadata_not_sensitive(self):
        """empty() creates metadata with no effects and not sensitive."""
        meta = EffectMetadata.empty()
        assert meta.effects == ()
        assert meta.is_sensitive is False

    def test_unknown_metadata_is_sensitive(self):
        """unknown() creates metadata with UNKNOWN effect and is_sensitive=True."""
        meta = EffectMetadata.unknown()
        assert len(meta.effects) == 1
        assert meta.effects[0].kind is EffectKind.UNKNOWN
        assert meta.is_sensitive is True

    def test_metadata_is_frozen(self):
        """EffectMetadata fields cannot be reassigned."""
        meta = EffectMetadata.empty()
        with pytest.raises(FrozenInstanceError):
            meta.is_sensitive = True  # type: ignore[misc]

    def test_metadata_with_explicit_effects(self):
        """Metadata can be constructed with explicit effects."""
        effects = (
            Effect(kind=EffectKind.READ, target="/data"),
            Effect(kind=EffectKind.WRITE, target="/data/out"),
        )
        meta = EffectMetadata(effects=effects, is_sensitive=False)
        assert meta.effects == effects
        assert meta.is_sensitive is False


class TestPolicyOperation:
    """Operation — normalized view for policy evaluation."""

    def test_operation_is_frozen(self):
        """Operation fields cannot be reassigned."""
        op = Operation(
            tool_identity=ToolIdentity(name="test"),
            arguments={"a": 1},
            effects=EffectMetadata.empty(),
        )
        with pytest.raises(FrozenInstanceError):
            op.arguments = {}  # type: ignore[misc]

    def test_arguments_are_read_only_mapping(self):
        """arguments is a MappingProxyType — item mutation raises TypeError."""
        op = Operation(
            tool_identity=ToolIdentity(name="test"),
            arguments={"a": 1},
            effects=EffectMetadata.empty(),
        )
        with pytest.raises(TypeError):
            op.arguments["a"] = 2  # type: ignore[index]
        assert op.arguments["a"] == 1

    def test_operation_defaults(self):
        """Optional fields have sensible defaults."""
        op = Operation(
            tool_identity=ToolIdentity(name="test"),
            arguments={},
            effects=EffectMetadata.empty(),
        )
        assert op.affected_locations == ()
        assert op.owner is None
        assert op.workspace is None
        assert op.session_context == {}

    def test_operation_with_full_context(self):
        """Operation carries full session context for policy evaluation."""
        op = Operation(
            tool_identity=ToolIdentity(name="bash_tool", source="bash"),
            arguments={"command": "ls"},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.EXECUTE, target="shell"),),
            ),
            affected_locations=("/tmp",),
            owner="user-1",
            workspace="default",
            session_context={"mode": "auto"},
        )
        assert op.tool_identity.name == "bash_tool"
        assert op.tool_identity.source == "bash"
        assert op.arguments == {"command": "ls"}
        assert op.effects.effects[0].kind is EffectKind.EXECUTE
        assert op.affected_locations == ("/tmp",)
        assert op.owner == "user-1"
        assert op.workspace == "default"
        assert op.session_context == {"mode": "auto"}


class TestBuildPolicyOperation:
    """build_policy_operation — derive Operation from tool_call + catalog."""

    def test_build_with_catalog_hit(self):
        """Catalog hit → effect metadata from entry, identity from entry."""
        entry = ToolCatalogEntry(
            identity=ToolIdentity(name="search", source="web"),
            schema={},
            adapter=lambda args: {},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.READ, target="web"),),
            ),
            cancellable=False,
        )
        catalog = ToolCatalog([entry])

        op = build_policy_operation(
            {"function": "search", "arguments": {"q": "hello"}},
            catalog,
        )

        assert op.tool_identity.name == "search"
        assert op.tool_identity.source == "web"
        assert op.arguments == {"q": "hello"}
        assert op.effects.effects[0].kind is EffectKind.READ

    def test_build_with_catalog_miss(self):
        """Catalog miss → unknown/sensitive effect metadata."""
        entry = ToolCatalogEntry(
            identity=ToolIdentity(name="known_tool"),
            schema={},
            adapter=lambda args: {},
            cancellable=False,
        )
        catalog = ToolCatalog([entry])

        op = build_policy_operation(
            {"function": "unknown_tool", "arguments": {}},
            catalog,
        )

        assert op.tool_identity.name == "unknown_tool"
        assert op.effects.is_sensitive is True
        assert op.effects.effects[0].kind is EffectKind.UNKNOWN

    def test_build_without_catalog(self):
        """No catalog → unknown/sensitive effect metadata."""
        op = build_policy_operation(
            {"function": "some_tool", "arguments": {"x": 1}},
        )

        assert op.tool_identity.name == "some_tool"
        assert op.effects.is_sensitive is True
        assert op.effects.effects[0].kind is EffectKind.UNKNOWN

    def test_build_with_session_context(self):
        """Session context is carried through to the Operation."""
        op = build_policy_operation(
            {"function": "tool", "arguments": {}},
            owner="user-1",
            workspace="prod",
            session_context={"mode": "approval"},
        )

        assert op.owner == "user-1"
        assert op.workspace == "prod"
        assert op.session_context == {"mode": "approval"}

    def test_build_missing_fields(self):
        """Missing function/arguments → name='', arguments={}, no crash."""
        op = build_policy_operation({})
        assert op.tool_identity.name == ""
        assert op.arguments == {}
        assert op.effects.is_sensitive is True


# =========================================================================
# AC #2 — Hard deny wins under all conditions
# =========================================================================


class TestHardPolicy:
    """HardPolicy — hard-deny-wins enforcement."""

    def test_empty_policy_allows(self):
        """A policy with no rules allows all operations."""
        policy = HardPolicy()
        op = Operation(
            tool_identity=ToolIdentity(name="any"),
            arguments={},
            effects=EffectMetadata.empty(),
        )
        assert policy.check(op) is None
        assert policy.is_blocked(op) is False

    def test_single_deny_rule_blocks(self):
        """A matching deny rule returns the reason."""
        policy = HardPolicy()
        policy.deny(lambda op: "blocked" if op.tool_identity.name == "bad" else None)

        bad_op = Operation(
            tool_identity=ToolIdentity(name="bad"),
            arguments={},
            effects=EffectMetadata.empty(),
        )
        good_op = Operation(
            tool_identity=ToolIdentity(name="good"),
            arguments={},
            effects=EffectMetadata.empty(),
        )

        assert policy.check(bad_op) == "blocked"
        assert policy.is_blocked(bad_op) is True
        assert policy.check(good_op) is None
        assert policy.is_blocked(good_op) is False

    def test_first_matching_rule_wins(self):
        """Deny rules evaluate in order; first non-None reason wins."""
        policy = HardPolicy()
        policy.deny(lambda op: None)  # allow
        policy.deny(lambda op: "second")  # matches
        policy.deny(lambda op: "third")  # would match but unreachable

        op = Operation(
            tool_identity=ToolIdentity(name="t"),
            arguments={},
            effects=EffectMetadata.empty(),
        )
        assert policy.check(op) == "second"

    def test_hard_deny_wins_over_any_grant(self):
        """Hard deny blocks even when the operation would otherwise be allowed.

        This is the core ADR-006 rule: hard deny wins under ALL conditions.
        The HardPolicy is evaluated first; if it blocks, no other policy
        (grant, mode, override) can unblock it.
        """
        policy = HardPolicy()
        # Deny everything
        policy.deny(lambda op: "hard deny: always")

        op = Operation(
            tool_identity=ToolIdentity(name="safe_tool"),
            arguments={},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.READ),),
                is_sensitive=False,
            ),
        )

        # Even a read-only, non-sensitive tool is blocked by hard deny
        assert policy.check(op) == "hard deny: always"
        assert policy.is_blocked(op) is True

    def test_hard_deny_blocks_unknown_sensitive(self):
        """Hard deny blocks unknown/sensitive tools (fail cautious)."""
        policy = HardPolicy()
        policy.deny(lambda op: "unknown tool — sensitive" if op.effects.is_sensitive else None)

        op = Operation(
            tool_identity=ToolIdentity(name="unknown"),
            arguments={},
            effects=EffectMetadata.unknown(),
        )
        assert policy.check(op) is not None
        assert policy.is_blocked(op) is True

    def test_hard_deny_does_not_block_known_safe(self):
        """Hard deny does not block tools with known, non-sensitive effects."""
        policy = HardPolicy()
        policy.deny(lambda op: "unknown tool — sensitive" if op.effects.is_sensitive else None)

        op = Operation(
            tool_identity=ToolIdentity(name="safe"),
            arguments={},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.READ),),
                is_sensitive=False,
            ),
        )
        assert policy.check(op) is None
        assert policy.is_blocked(op) is False


class TestDefaultHardPolicy:
    """Default hard policy — built-in safety rules."""

    def test_default_policy_blocks_unknown_tools(self):
        """Default policy blocks tools with unknown/sensitive effect metadata."""
        policy = create_default_hard_policy()

        op = Operation(
            tool_identity=ToolIdentity(name="unknown_tool"),
            arguments={},
            effects=EffectMetadata.unknown(),
        )
        assert policy.is_blocked(op) is True

    def test_default_policy_blocks_rm_rf(self):
        """Default policy blocks rm -rf in bash_tool."""
        policy = create_default_hard_policy()

        op = Operation(
            tool_identity=ToolIdentity(name="bash_tool"),
            arguments={"command": "rm -rf /tmp"},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.EXECUTE, target="shell"),),
            ),
        )
        assert policy.is_blocked(op) is True

    def test_default_policy_allows_safe_bash(self):
        """Default policy allows safe bash commands."""
        policy = create_default_hard_policy()

        op = Operation(
            tool_identity=ToolIdentity(name="bash_tool"),
            arguments={"command": "ls -la"},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.EXECUTE, target="shell"),),
            ),
        )
        assert policy.is_blocked(op) is False

    def test_default_policy_allows_known_read_tool(self):
        """Default policy allows tools with known, non-sensitive effects."""
        policy = create_default_hard_policy()

        op = Operation(
            tool_identity=ToolIdentity(name="read_file"),
            arguments={"path": "/tmp/data"},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.READ, target="file"),),
                is_sensitive=False,
            ),
        )
        assert policy.is_blocked(op) is False


# =========================================================================
# AC #3 — Effect classification tests pass for all effect types
# =========================================================================


class TestEffectClassification:
    """Effect classification — all effect types are classifiable."""

    def test_read_effect(self):
        """READ effect is classified correctly."""
        effect = Effect(kind=EffectKind.READ, target="/data/file.txt")
        assert effect.kind is EffectKind.READ
        assert effect.kind.value == "read"

    def test_write_effect(self):
        """WRITE effect is classified correctly."""
        effect = Effect(kind=EffectKind.WRITE, target="/data/out.txt")
        assert effect.kind is EffectKind.WRITE
        assert effect.kind.value == "write"

    def test_create_effect(self):
        """CREATE effect is classified correctly."""
        effect = Effect(kind=EffectKind.CREATE, target="/new/file")
        assert effect.kind is EffectKind.CREATE
        assert effect.kind.value == "create"

    def test_modify_effect(self):
        """MODIFY effect is classified correctly."""
        effect = Effect(kind=EffectKind.MODIFY, target="/existing/file")
        assert effect.kind is EffectKind.MODIFY
        assert effect.kind.value == "modify"

    def test_delete_effect(self):
        """DELETE effect is classified correctly."""
        effect = Effect(kind=EffectKind.DELETE, target="/old/file")
        assert effect.kind is EffectKind.DELETE
        assert effect.kind.value == "delete"

    def test_execute_effect(self):
        """EXECUTE effect is classified correctly."""
        effect = Effect(kind=EffectKind.EXECUTE, target="shell")
        assert effect.kind is EffectKind.EXECUTE
        assert effect.kind.value == "execute"

    def test_network_effect(self):
        """NETWORK effect is classified correctly."""
        effect = Effect(kind=EffectKind.NETWORK, target="https://api.example.com")
        assert effect.kind is EffectKind.NETWORK
        assert effect.kind.value == "network"

    def test_identity_effect(self):
        """IDENTITY effect is classified correctly."""
        effect = Effect(kind=EffectKind.IDENTITY, target="user-profile")
        assert effect.kind is EffectKind.IDENTITY
        assert effect.kind.value == "identity"

    def test_persistence_effect(self):
        """PERSISTENCE effect is classified correctly."""
        effect = Effect(kind=EffectKind.PERSISTENCE, target="database")
        assert effect.kind is EffectKind.PERSISTENCE
        assert effect.kind.value == "persistence"

    def test_unknown_effect(self):
        """UNKNOWN effect is classified correctly (fail cautious)."""
        effect = Effect(kind=EffectKind.UNKNOWN)
        assert effect.kind is EffectKind.UNKNOWN
        assert effect.kind.value == "unknown"


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Edge cases for operations, effects, and hard policy."""

    def test_unknown_effect_metadata_is_sensitive(self):
        """Unknown effect metadata is treated as sensitive (fail cautious)."""
        meta = EffectMetadata.unknown()
        assert meta.is_sensitive is True

        # A hard policy with a sensitive-tool rule should block it
        policy = HardPolicy()
        policy.deny(lambda op: "sensitive" if op.effects.is_sensitive else None)

        op = Operation(
            tool_identity=ToolIdentity(name="uncategorized"),
            arguments={},
            effects=meta,
        )
        assert policy.is_blocked(op) is True

    def test_conflicting_effect_classifications(self):
        """A tool can have multiple effects of different kinds."""
        effects = (
            Effect(kind=EffectKind.READ, target="/data"),
            Effect(kind=EffectKind.WRITE, target="/data/out"),
        )
        meta = EffectMetadata(effects=effects, is_sensitive=False)

        # Both effects are present
        kinds = {e.kind for e in meta.effects}
        assert EffectKind.READ in kinds
        assert EffectKind.WRITE in kinds

    def test_hard_policy_bypass_rejection(self):
        """Hard policy cannot be bypassed by any argument manipulation.

        The policy evaluates the Operation as-is; argument manipulation
        happens before the Operation reaches the policy.
        """
        policy = HardPolicy()
        policy.deny(lambda op: "block rm -rf" if "rm -rf" in str(op.arguments.get("command", "")) else None)

        # Even with creative argument shapes, the string check catches it
        op = Operation(
            tool_identity=ToolIdentity(name="bash_tool"),
            arguments={"command": ["rm -rf /tmp"]},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.EXECUTE, target="shell"),),
            ),
        )
        # str(["rm -rf /tmp"]) contains "rm -rf"
        assert policy.is_blocked(op) is True

    def test_catalog_entry_with_effects(self):
        """Catalog entry carries effect metadata at registration (ADR-004)."""
        effects = EffectMetadata(
            effects=(
                Effect(kind=EffectKind.READ, target="web"),
                Effect(kind=EffectKind.NETWORK, target="api.example.com"),
            ),
        )
        entry = ToolCatalogEntry(
            identity=ToolIdentity(name="web_search", source="web-search-resource"),
            schema={"type": "function", "function": {"name": "web_search"}},
            adapter=lambda args: {"result": "ok"},
            effects=effects,
            cancellable=False,
        )

        assert entry.effects is effects
        assert len(entry.effects.effects) == 2
        assert entry.effects.effects[0].kind is EffectKind.READ
        assert entry.effects.effects[1].kind is EffectKind.NETWORK

    def test_catalog_entry_default_effects_is_empty(self):
        """Catalog entry defaults to empty (non-sensitive) effects."""
        entry = ToolCatalogEntry(
            identity=ToolIdentity(name="simple"),
            schema={},
            adapter=lambda args: {},
            cancellable=False,
        )
        assert entry.effects == EffectMetadata.empty()
        assert entry.effects.is_sensitive is False
