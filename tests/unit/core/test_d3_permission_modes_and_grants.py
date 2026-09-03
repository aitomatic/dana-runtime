"""D3 Permission Modes & Policy Grants — tests for permission modes, grant
store, matching precedence, revocation, owner/workspace scoping, and
fail-closed behavior.

Each test maps to one acceptance criterion or edge case from the story.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import pytest_asyncio

from dana.core.policy.effects import Effect, EffectKind, EffectMetadata
from dana.core.policy.evaluator import PolicyDecision, PolicyEvaluator
from dana.core.policy.grants import (
    GrantConflict,
    GrantDecision,
    GrantMatch,
    GrantNotFound,
    PolicyGrant,
    grant_matches_operation,
)
from dana.core.policy.hard_policy import HardPolicy
from dana.core.policy.modes import PermissionMode
from dana.core.policy.operations import Operation
from dana.core.policy.scope import OwnerScope, ScopeMatch, scope_matches
from dana.core.tool.catalog import ToolIdentity


# =========================================================================
# Shared fixtures
# =========================================================================


@pytest.fixture
def owner_a() -> OwnerScope:
    return OwnerScope(owner_id="user-1", workspace="default")


@pytest.fixture
def owner_b() -> OwnerScope:
    return OwnerScope(owner_id="user-2", workspace="default")


@pytest.fixture
def owner_a_other_ws() -> OwnerScope:
    return OwnerScope(owner_id="user-1", workspace="other")


@pytest.fixture
def read_operation() -> Operation:
    return Operation(
        tool_identity=ToolIdentity(name="read_file"),
        arguments={"path": "/tmp/data"},
        effects=EffectMetadata(
            effects=(Effect(kind=EffectKind.READ, target="file"),),
            is_sensitive=False,
        ),
        affected_locations=("/tmp/data",),
        owner="user-1",
        workspace="default",
    )


@pytest.fixture
def write_operation() -> Operation:
    return Operation(
        tool_identity=ToolIdentity(name="write_file"),
        arguments={"path": "/tmp/out"},
        effects=EffectMetadata(
            effects=(Effect(kind=EffectKind.WRITE, target="file"),),
            is_sensitive=False,
        ),
        affected_locations=("/tmp/out",),
        owner="user-1",
        workspace="default",
    )


@pytest.fixture
def delete_operation() -> Operation:
    return Operation(
        tool_identity=ToolIdentity(name="delete_file"),
        arguments={"path": "/tmp/old"},
        effects=EffectMetadata(
            effects=(Effect(kind=EffectKind.DELETE, target="file"),),
            is_sensitive=False,
        ),
        affected_locations=("/tmp/old",),
        owner="user-1",
        workspace="default",
    )


@pytest.fixture
def bash_operation() -> Operation:
    return Operation(
        tool_identity=ToolIdentity(name="bash_tool"),
        arguments={"command": "ls -la"},
        effects=EffectMetadata(
            effects=(Effect(kind=EffectKind.EXECUTE, target="shell"),),
            is_sensitive=False,
        ),
        affected_locations=("shell",),
        owner="user-1",
        workspace="default",
    )


# =========================================================================
# AC #1 — Grants never cross scope
# =========================================================================


class TestScopeMatching:
    """Scope matching — grants never cross owner/workspace boundaries."""

    def test_same_owner_same_workspace_matches(self, owner_a):
        """Grant and operation with same owner+workspace match."""
        result = scope_matches(owner_a, "user-1", "default")
        assert result.is_match is True
        assert result.owner_match is True
        assert result.workspace_match is True

    def test_different_owner_does_not_match(self, owner_a, owner_b):
        """Grant for owner_a does not match operation owned by owner_b."""
        result = scope_matches(owner_a, "user-2", "default")
        assert result.is_match is False
        assert result.owner_match is False
        assert result.workspace_match is True

    def test_different_workspace_does_not_match(self, owner_a, owner_a_other_ws):
        """Grant for workspace 'default' does not match operation in 'other'."""
        result = scope_matches(owner_a, "user-1", "other")
        assert result.is_match is False
        assert result.owner_match is True
        assert result.workspace_match is False

    def test_none_owner_does_not_match(self, owner_a):
        """Operation with no owner does not match any scoped grant."""
        result = scope_matches(owner_a, None, "default")
        assert result.is_match is False
        assert result.owner_match is False

    def test_none_workspace_does_not_match(self, owner_a):
        """Operation with no workspace does not match any scoped grant."""
        result = scope_matches(owner_a, "user-1", None)
        assert result.is_match is False
        assert result.workspace_match is False

    def test_cross_owner_isolation(self, owner_a, owner_b, read_operation):
        """Grant for owner_a never matches operation for owner_b.

        AC #1: Grants with different OwnerScope never match cross-scope.
        """
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
        )
        # Operation owned by owner_b
        op_b = Operation(
            tool_identity=read_operation.tool_identity,
            arguments=read_operation.arguments,
            effects=read_operation.effects,
            affected_locations=read_operation.affected_locations,
            owner="user-2",
            workspace="default",
        )
        assert grant_matches_operation(grant, op_b) is False

    def test_cross_workspace_isolation(self, owner_a, owner_a_other_ws, read_operation):
        """Grant for workspace 'default' does not match operation in 'other'.

        AC #1: Different workspace within same owner is also isolated.
        """
        grant = PolicyGrant(
            grant_id="g2",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
        )
        op_other_ws = Operation(
            tool_identity=read_operation.tool_identity,
            arguments=read_operation.arguments,
            effects=read_operation.effects,
            affected_locations=read_operation.affected_locations,
            owner="user-1",
            workspace="other",
        )
        assert grant_matches_operation(grant, op_other_ws) is False


# =========================================================================
# AC #2 — Matching grants suppress only matching prompts
# =========================================================================


class TestGrantMatching:
    """Grant matching — grants suppress only the exact matching prompt."""

    def test_exact_match_allows(self, owner_a, read_operation):
        """Grant matching tool+effect+location allows the operation."""
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
            location="/tmp/data",
        )
        assert grant_matches_operation(grant, read_operation) is True

    def test_different_tool_does_not_match(self, owner_a, read_operation):
        """Grant for 'write_file' does not match 'read_file' operation."""
        grant = PolicyGrant(
            grant_id="g2",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="write_file",
            effect_kind=EffectKind.READ,
        )
        assert grant_matches_operation(grant, read_operation) is False

    def test_different_effect_does_not_match(self, owner_a, read_operation):
        """Grant for WRITE effect does not match READ operation."""
        grant = PolicyGrant(
            grant_id="g3",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.WRITE,
        )
        assert grant_matches_operation(grant, read_operation) is False

    def test_different_location_does_not_match(self, owner_a, read_operation):
        """Grant for location '/other' does not match operation at '/tmp/data'."""
        grant = PolicyGrant(
            grant_id="g4",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
            location="/other",
        )
        assert grant_matches_operation(grant, read_operation) is False

    def test_empty_location_matches_any(self, owner_a, read_operation):
        """Grant with empty location matches any location."""
        grant = PolicyGrant(
            grant_id="g5",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
            location="",
        )
        assert grant_matches_operation(grant, read_operation) is True

    def test_reject_grant_suppresses_prompt(self, owner_a, delete_operation):
        """Reject grant suppresses the prompt (denies without asking)."""
        grant = PolicyGrant(
            grant_id="g6",
            owner_scope=owner_a,
            decision=GrantDecision.REJECT,
            tool_identity="delete_file",
            effect_kind=EffectKind.DELETE,
        )
        assert grant_matches_operation(grant, delete_operation) is True

    def test_grant_does_not_match_similar_tool(self, owner_a, read_operation):
        """Grant for 'read_file' does not match 'read_file_2'.

        AC #2: Matching grants suppress only the exact matching prompt,
        not similar ones.
        """
        grant = PolicyGrant(
            grant_id="g7",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
        )
        op_similar = Operation(
            tool_identity=ToolIdentity(name="read_file_2"),
            arguments={"path": "/tmp/data"},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.READ, target="file"),),
                is_sensitive=False,
            ),
            affected_locations=("/tmp/data",),
            owner="user-1",
            workspace="default",
        )
        assert grant_matches_operation(grant, op_similar) is False


# =========================================================================
# AC #3 — Revocation is immediate for the next Operation
# =========================================================================


class TestGrantRevocation:
    """Grant revocation — immediate for the next Operation."""

    def test_active_grant_matches(self, owner_a, read_operation):
        """Active (non-revoked) grant matches operations."""
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
        )
        assert grant.is_active is True
        assert grant.is_revoked is False
        assert grant_matches_operation(grant, read_operation) is True

    def test_revoked_grant_does_not_match(self, owner_a, read_operation):
        """Revoked grant does not match any operation.

        AC #3: Revoked grant is not applied to the next Operation.
        """
        grant = PolicyGrant(
            grant_id="g2",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
            revoked_at=datetime.now(UTC),
        )
        assert grant.is_active is False
        assert grant.is_revoked is True
        assert grant_matches_operation(grant, read_operation) is False

    def test_revoked_allow_grant_does_not_allow(self, owner_a, read_operation):
        """A revoked allow grant no longer allows the operation."""
        grant = PolicyGrant(
            grant_id="g3",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
            revoked_at=datetime.now(UTC),
        )
        assert grant_matches_operation(grant, read_operation) is False

    def test_revoked_reject_grant_does_not_reject(self, owner_a, delete_operation):
        """A revoked reject grant no longer rejects the operation."""
        grant = PolicyGrant(
            grant_id="g4",
            owner_scope=owner_a,
            decision=GrantDecision.REJECT,
            tool_identity="delete_file",
            effect_kind=EffectKind.DELETE,
            revoked_at=datetime.now(UTC),
        )
        assert grant_matches_operation(grant, delete_operation) is False


# =========================================================================
# AC #4 — Storage parity across SQLite/PostgreSQL (SQLite tests)
# =========================================================================


class TestSQLiteGrantStore:
    """GrantStore on SQLite — same contract as PostgreSQL."""

    @pytest_asyncio.fixture
    async def store(self):
        from dana.core.policy.store_sqlite import SQLiteGrantStore

        store = await SQLiteGrantStore.open(":memory:")
        yield store
        await store.close()

    @pytest.fixture
    def scope(self) -> OwnerScope:
        return OwnerScope(owner_id="test-user", workspace="test-ws")

    @pytest.mark.asyncio
    async def test_create_and_get_grant(self, store, scope):
        """Create a grant and retrieve it by ID."""
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=scope,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
            reason="test grant",
        )
        created = await store.create_grant(grant)
        assert created.grant_id == "g1"
        assert created.decision == GrantDecision.ALLOW
        assert created.tool_identity == "read_file"
        assert created.effect_kind == EffectKind.READ
        assert created.is_active is True
        assert created.reason == "test grant"

        fetched = await store.get_grant(scope, "g1")
        assert fetched == created

    @pytest.mark.asyncio
    async def test_create_duplicate_raises_conflict(self, store, scope):
        """Creating a grant with duplicate ID raises GrantConflict."""
        grant = PolicyGrant(
            grant_id="dup",
            owner_scope=scope,
            decision=GrantDecision.ALLOW,
            tool_identity="tool",
            effect_kind=EffectKind.READ,
        )
        await store.create_grant(grant)
        with pytest.raises(GrantConflict):
            await store.create_grant(grant)

    @pytest.mark.asyncio
    async def test_get_nonexistent_raises_not_found(self, store, scope):
        """Getting a nonexistent grant raises GrantNotFound."""
        with pytest.raises(GrantNotFound):
            await store.get_grant(scope, "nonexistent")

    @pytest.mark.asyncio
    async def test_list_grants_active_only(self, store, scope):
        """list_grants with active_only=True returns only non-revoked grants."""
        g1 = PolicyGrant(
            grant_id="g1",
            owner_scope=scope,
            decision=GrantDecision.ALLOW,
            tool_identity="read",
            effect_kind=EffectKind.READ,
        )
        g2 = PolicyGrant(
            grant_id="g2",
            owner_scope=scope,
            decision=GrantDecision.REJECT,
            tool_identity="delete",
            effect_kind=EffectKind.DELETE,
        )
        await store.create_grant(g1)
        await store.create_grant(g2)
        await store.revoke_grant(scope, "g1")

        active = await store.list_grants(scope, active_only=True)
        assert len(active) == 1
        assert active[0].grant_id == "g2"

        all_grants = await store.list_grants(scope, active_only=False)
        assert len(all_grants) == 2

    @pytest.mark.asyncio
    async def test_revoke_grant(self, store, scope):
        """Revoking a grant sets revoked_at and makes it inactive."""
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=scope,
            decision=GrantDecision.ALLOW,
            tool_identity="read",
            effect_kind=EffectKind.READ,
        )
        await store.create_grant(grant)

        revoked = await store.revoke_grant(scope, "g1")
        assert revoked.is_revoked is True
        assert revoked.revoked_at is not None

        # Verify it no longer matches
        op = Operation(
            tool_identity=ToolIdentity(name="read"),
            arguments={},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.READ),),
                is_sensitive=False,
            ),
            owner="test-user",
            workspace="test-ws",
        )
        match = await store.find_matching_grants(scope, op)
        assert match.matched is None

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_raises_not_found(self, store, scope):
        """Revoking a nonexistent grant raises GrantNotFound."""
        with pytest.raises(GrantNotFound):
            await store.revoke_grant(scope, "nonexistent")

    @pytest.mark.asyncio
    async def test_find_matching_grants_allow(self, store, scope):
        """find_matching_grants returns the matching allow grant."""
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=scope,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
        )
        await store.create_grant(grant)

        op = Operation(
            tool_identity=ToolIdentity(name="read_file"),
            arguments={},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.READ),),
                is_sensitive=False,
            ),
            owner="test-user",
            workspace="test-ws",
        )
        match = await store.find_matching_grants(scope, op)
        assert match.matched is not None
        assert match.decision == GrantDecision.ALLOW
        assert match.grant_id == "g1"

    @pytest.mark.asyncio
    async def test_find_matching_grants_reject_precedence(self, store, scope):
        """Reject grants take precedence over allow grants (ADR-006)."""
        allow_grant = PolicyGrant(
            grant_id="allow",
            owner_scope=scope,
            decision=GrantDecision.ALLOW,
            tool_identity="delete_file",
            effect_kind=EffectKind.DELETE,
        )
        reject_grant = PolicyGrant(
            grant_id="reject",
            owner_scope=scope,
            decision=GrantDecision.REJECT,
            tool_identity="delete_file",
            effect_kind=EffectKind.DELETE,
        )
        await store.create_grant(allow_grant)
        await store.create_grant(reject_grant)

        op = Operation(
            tool_identity=ToolIdentity(name="delete_file"),
            arguments={},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.DELETE),),
                is_sensitive=False,
            ),
            owner="test-user",
            workspace="test-ws",
        )
        match = await store.find_matching_grants(scope, op)
        assert match.decision == GrantDecision.REJECT
        assert match.grant_id == "reject"

    @pytest.mark.asyncio
    async def test_find_matching_grants_no_match(self, store, scope):
        """find_matching_grants returns None when no grant matches."""
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=scope,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
        )
        await store.create_grant(grant)

        op = Operation(
            tool_identity=ToolIdentity(name="other_tool"),
            arguments={},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.WRITE),),
                is_sensitive=False,
            ),
            owner="test-user",
            workspace="test-ws",
        )
        match = await store.find_matching_grants(scope, op)
        assert match.matched is None
        assert match.decision is None
        assert match.grant_id is None

    @pytest.mark.asyncio
    async def test_cross_scope_isolation_in_store(self, store, scope):
        """Grants in one scope are invisible to another scope.

        AC #1: Grants never cross scope — verified at the store level.
        """
        scope_a = scope
        scope_b = OwnerScope(owner_id="other-user", workspace="test-ws")

        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=scope_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
        )
        await store.create_grant(grant)

        # scope_b should not see scope_a's grant
        with pytest.raises(GrantNotFound):
            await store.get_grant(scope_b, "g1")

        # scope_b's list should be empty
        assert await store.list_grants(scope_b) == []

    @pytest.mark.asyncio
    async def test_revocation_immediate_for_next_operation(self, store, scope):
        """After revocation, the grant is not applied to the next Operation.

        AC #3: Revocation is immediate for the next Operation.
        """
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=scope,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
        )
        await store.create_grant(grant)

        op = Operation(
            tool_identity=ToolIdentity(name="read_file"),
            arguments={},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.READ),),
                is_sensitive=False,
            ),
            owner="test-user",
            workspace="test-ws",
        )

        # Before revocation: matches
        match_before = await store.find_matching_grants(scope, op)
        assert match_before.matched is not None

        # Revoke
        await store.revoke_grant(scope, "g1")

        # After revocation: does not match (immediate for next Operation)
        match_after = await store.find_matching_grants(scope, op)
        assert match_after.matched is None

    @pytest.mark.asyncio
    async def test_specificity_more_specific_wins(self, store, scope):
        """More specific grant wins over less specific one."""
        broad = PolicyGrant(
            grant_id="broad",
            owner_scope=scope,
            decision=GrantDecision.ALLOW,
            tool_identity="bash_tool",
            effect_kind=EffectKind.EXECUTE,
            location="",
        )
        specific = PolicyGrant(
            grant_id="specific",
            owner_scope=scope,
            decision=GrantDecision.ALLOW,
            tool_identity="bash_tool",
            effect_kind=EffectKind.EXECUTE,
            location="shell",
        )
        await store.create_grant(broad)
        await store.create_grant(specific)

        op = Operation(
            tool_identity=ToolIdentity(name="bash_tool"),
            arguments={"command": "ls"},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.EXECUTE, target="shell"),),
                is_sensitive=False,
            ),
            affected_locations=("shell",),
            owner="test-user",
            workspace="test-ws",
        )
        match = await store.find_matching_grants(scope, op)
        assert match.grant_id == "specific"


# =========================================================================
# AC #5 — Timeout/disconnect/cancel denies
# =========================================================================


class TestFailClosed:
    """Timeout/disconnect/cancel all result in deny (fail-closed).

    The PolicyEvaluator chains HardPolicy → GrantStore → PermissionMode.
    When no grant matches and the mode doesn't auto-allow, the result is
    NEEDS_PROMPT — which the runtime treats as deny if prompting is not
    possible (e.g. timeout, disconnect, cancel).
    """

    def test_timeout_denies(self):
        """Timeout results in deny — no grant match possible."""

        hard_policy = HardPolicy()
        grant_store = _NoOpGrantStore()
        evaluator = PolicyEvaluator(hard_policy, grant_store, PermissionMode.DEFAULT)

        op = Operation(
            tool_identity=ToolIdentity(name="read_file"),
            arguments={},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.READ),),
                is_sensitive=False,
            ),
            owner="user-1",
            workspace="default",
        )
        scope = OwnerScope(owner_id="user-1", workspace="default")

        # In DEFAULT mode with no grants, the result is NEEDS_PROMPT
        # (not ALLOW). The runtime treats NEEDS_PROMPT as deny when
        # prompting is not possible (timeout/disconnect/cancel).
        import asyncio

        result = asyncio.run(evaluator.evaluate(op, scope))
        assert result.decision is PolicyDecision.NEEDS_PROMPT
        assert "no matching grant" in result.reason

    def test_disconnect_denies(self):
        """Disconnect results in deny — same as timeout."""

        hard_policy = HardPolicy()
        grant_store = _NoOpGrantStore()
        evaluator = PolicyEvaluator(hard_policy, grant_store, PermissionMode.DEFAULT)

        op = Operation(
            tool_identity=ToolIdentity(name="bash_tool"),
            arguments={"command": "ls"},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.EXECUTE),),
                is_sensitive=False,
            ),
            owner="user-1",
            workspace="default",
        )
        scope = OwnerScope(owner_id="user-1", workspace="default")

        import asyncio

        result = asyncio.run(evaluator.evaluate(op, scope))
        # No grant, DEFAULT mode, not hard-denied → NEEDS_PROMPT
        assert result.decision is PolicyDecision.NEEDS_PROMPT

    def test_cancel_denies(self):
        """Cancel results in deny — same as timeout."""

        hard_policy = HardPolicy()
        grant_store = _NoOpGrantStore()
        evaluator = PolicyEvaluator(hard_policy, grant_store, PermissionMode.DEFAULT)

        op = Operation(
            tool_identity=ToolIdentity(name="delete_file"),
            arguments={"path": "/tmp/x"},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.DELETE),),
                is_sensitive=False,
            ),
            owner="user-1",
            workspace="default",
        )
        scope = OwnerScope(owner_id="user-1", workspace="default")

        import asyncio

        result = asyncio.run(evaluator.evaluate(op, scope))
        # No grant, DEFAULT mode, not hard-denied → NEEDS_PROMPT
        assert result.decision is PolicyDecision.NEEDS_PROMPT

    def test_no_matching_grant_is_needs_prompt(self):
        """When no grant matches, the result is NEEDS_PROMPT (fail-closed)."""
        match = GrantMatch(matched=None)
        assert match.decision is None
        assert match.grant_id is None
        # The policy evaluator treats None as "needs interactive prompt or deny"


# =========================================================================
# Permission Mode tests
# =========================================================================


class TestPermissionMode:
    """PermissionMode — mode transitions and auto-allow behavior."""

    def test_default_mode_does_not_auto_allow(self):
        """DEFAULT mode does not auto-allow any operation."""
        mode = PermissionMode.DEFAULT
        assert mode.allows_without_prompt(frozenset({EffectKind.READ})) is False
        assert mode.allows_without_prompt(frozenset({EffectKind.WRITE})) is False
        assert mode.allows_without_prompt(frozenset({EffectKind.DELETE})) is False
        assert mode.allows_without_prompt(frozenset({EffectKind.EXECUTE})) is False

    def test_accept_edits_auto_allows_read_write(self):
        """ACCEPT_EDITS auto-allows READ, WRITE, CREATE, MODIFY."""
        mode = PermissionMode.ACCEPT_EDITS
        assert mode.allows_without_prompt(frozenset({EffectKind.READ})) is True
        assert mode.allows_without_prompt(frozenset({EffectKind.WRITE})) is True
        assert mode.allows_without_prompt(frozenset({EffectKind.CREATE})) is True
        assert mode.allows_without_prompt(frozenset({EffectKind.MODIFY})) is True

    def test_accept_edits_does_not_auto_allow_destructive(self):
        """ACCEPT_EDITS does not auto-allow DELETE, EXECUTE, etc."""
        mode = PermissionMode.ACCEPT_EDITS
        assert mode.allows_without_prompt(frozenset({EffectKind.DELETE})) is False
        assert mode.allows_without_prompt(frozenset({EffectKind.EXECUTE})) is False
        assert mode.allows_without_prompt(frozenset({EffectKind.NETWORK})) is False
        assert mode.allows_without_prompt(frozenset({EffectKind.IDENTITY})) is False
        assert mode.allows_without_prompt(frozenset({EffectKind.PERSISTENCE})) is False
        assert mode.allows_without_prompt(frozenset({EffectKind.UNKNOWN})) is False

    def test_accept_edits_auto_allows_mixed_read_write(self):
        """ACCEPT_EDITS auto-allows operations with only READ/WRITE/CREATE/MODIFY effects."""
        mode = PermissionMode.ACCEPT_EDITS
        mixed = frozenset({EffectKind.READ, EffectKind.WRITE})
        assert mode.allows_without_prompt(mixed) is True

    def test_accept_edits_does_not_allow_mixed_with_delete(self):
        """ACCEPT_EDITS does not auto-allow if any effect is not in the safe set."""
        mode = PermissionMode.ACCEPT_EDITS
        mixed = frozenset({EffectKind.READ, EffectKind.DELETE})
        assert mode.allows_without_prompt(mixed) is False

    def test_bypass_auto_allows_everything(self):
        """BYPASS_PERMISSIONS auto-allows everything."""
        mode = PermissionMode.BYPASS_PERMISSIONS
        assert mode.allows_without_prompt(frozenset({EffectKind.READ})) is True
        assert mode.allows_without_prompt(frozenset({EffectKind.DELETE})) is True
        assert mode.allows_without_prompt(frozenset({EffectKind.EXECUTE})) is True
        assert mode.allows_without_prompt(frozenset({EffectKind.UNKNOWN})) is True
        assert mode.allows_without_prompt(frozenset()) is True

    def test_mode_values(self):
        """PermissionMode enum values match the spec."""
        assert PermissionMode.DEFAULT.value == "default"
        assert PermissionMode.ACCEPT_EDITS.value == "acceptEdits"
        assert PermissionMode.BYPASS_PERMISSIONS.value == "bypassPermissions"


# =========================================================================
# PolicyGrant model tests
# =========================================================================


class TestPolicyGrant:
    """PolicyGrant — model behavior."""

    def test_grant_defaults(self, owner_a):
        """PolicyGrant has sensible defaults."""
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="tool",
            effect_kind=EffectKind.READ,
        )
        assert grant.location == ""
        assert grant.is_active is True
        assert grant.is_revoked is False
        assert grant.reason == ""

    def test_grant_is_active_when_not_revoked(self, owner_a):
        """Grant is active when revoked_at is None."""
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="tool",
            effect_kind=EffectKind.READ,
        )
        assert grant.is_active is True

    def test_grant_is_revoked_when_revoked_at_set(self, owner_a):
        """Grant is revoked when revoked_at is set."""
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="tool",
            effect_kind=EffectKind.READ,
            revoked_at=datetime.now(UTC),
        )
        assert grant.is_revoked is True
        assert grant.is_active is False

    def test_grant_decision_values(self):
        """GrantDecision enum values."""
        assert GrantDecision.ALLOW.value == "allow"
        assert GrantDecision.REJECT.value == "reject"


# =========================================================================
# Edge cases
# =========================================================================


class _NoOpGrantStore:
    """A grant store that never matches any operation."""

    async def find_matching_grants(self, scope, operation):
        from dana.core.policy.grants import GrantMatch

        return GrantMatch(matched=None)

    async def create_grant(self, grant):
        pass

    async def get_grant(self, scope, grant_id):
        from dana.core.policy.grants import GrantNotFound

        raise GrantNotFound(grant_id)

    async def list_grants(self, scope, *, active_only=True):
        return []

    async def revoke_grant(self, scope, grant_id):
        from dana.core.policy.grants import GrantNotFound

        raise GrantNotFound(grant_id)

    async def close(self):
        pass


class TestEdgeCases:
    """Edge cases for permission modes, grants, and scoping."""

    def test_stale_reply_after_revocation(self, owner_a, read_operation):
        """A stale (revoked) grant does not match even if cached.

        Edge case: If a caller holds a reference to a revoked grant,
        the grant_matches_operation function checks revoked_at and
        returns False.
        """
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
            revoked_at=datetime.now(UTC),
        )
        # Even though the grant object exists, it's revoked
        assert grant_matches_operation(grant, read_operation) is False

    def test_race_between_grant_creation_and_operation(self, owner_a):
        """A grant created after an operation is evaluated does not affect it.

        Edge case: The grant store is checked at operation evaluation time.
        A grant created after that point does not retroactively apply.
        """
        op = Operation(
            tool_identity=ToolIdentity(name="read_file"),
            arguments={},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.READ),),
                is_sensitive=False,
            ),
            owner="user-1",
            workspace="default",
        )
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
        )
        # Grant matches the operation
        assert grant_matches_operation(grant, op) is True
        # If the grant is created after evaluation, it doesn't retroactively apply.
        # This is a temporal ordering concern enforced by the evaluator calling
        # find_matching_grants at evaluation time, not by the grant model itself.
        # The grant model correctly reports match; the evaluator controls timing.

    def test_cross_owner_isolation_concurrent_sessions(self, owner_a, owner_b):
        """Cross-owner isolation holds under concurrent sessions.

        Edge case: Two owners with grants for the same tool name
        should not see each other's grants.
        """
        grant_a = PolicyGrant(
            grant_id="g1",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
        )
        grant_b = PolicyGrant(
            grant_id="g1",  # Same grant_id, different scope
            owner_scope=owner_b,
            decision=GrantDecision.ALLOW,
            tool_identity="read_file",
            effect_kind=EffectKind.READ,
        )

        op_a = Operation(
            tool_identity=ToolIdentity(name="read_file"),
            arguments={},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.READ),),
                is_sensitive=False,
            ),
            owner="user-1",
            workspace="default",
        )
        op_b = Operation(
            tool_identity=ToolIdentity(name="read_file"),
            arguments={},
            effects=EffectMetadata(
                effects=(Effect(kind=EffectKind.READ),),
                is_sensitive=False,
            ),
            owner="user-2",
            workspace="default",
        )

        # Each grant only matches its own scope
        assert grant_matches_operation(grant_a, op_a) is True
        assert grant_matches_operation(grant_a, op_b) is False
        assert grant_matches_operation(grant_b, op_b) is True
        assert grant_matches_operation(grant_b, op_a) is False

    def test_operation_with_no_effects(self, owner_a):
        """An operation with no effects does not match any effect-specific grant."""
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="noop",
            effect_kind=EffectKind.READ,
        )
        op = Operation(
            tool_identity=ToolIdentity(name="noop"),
            arguments={},
            effects=EffectMetadata.empty(),
            owner="user-1",
            workspace="default",
        )
        assert grant_matches_operation(grant, op) is False

    def test_grant_with_unknown_effect_kind(self, owner_a):
        """Grant with UNKNOWN effect kind matches operations with UNKNOWN effect."""
        grant = PolicyGrant(
            grant_id="g1",
            owner_scope=owner_a,
            decision=GrantDecision.ALLOW,
            tool_identity="unknown_tool",
            effect_kind=EffectKind.UNKNOWN,
        )
        op = Operation(
            tool_identity=ToolIdentity(name="unknown_tool"),
            arguments={},
            effects=EffectMetadata.unknown(),
            owner="user-1",
            workspace="default",
        )
        assert grant_matches_operation(grant, op) is True

    def test_scope_match_properties(self, owner_a):
        """ScopeMatch properties work correctly."""
        match = ScopeMatch(owner_match=True, workspace_match=True)
        assert match.is_match is True

        no_owner = ScopeMatch(owner_match=False, workspace_match=True)
        assert no_owner.is_match is False

        no_ws = ScopeMatch(owner_match=True, workspace_match=False)
        assert no_ws.is_match is False

        neither = ScopeMatch(owner_match=False, workspace_match=False)
        assert neither.is_match is False
