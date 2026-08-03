"""Policy Grants — durable allow/reject grants for permission policy.

Per ADR-006:
- allow-always/reject-always create revocable Policy Grants.
- Grants default to Owner Scope + workspace + Tool Identity + effect + location.
- Broader grants require explicit operator provisioning.
- Revocation is immediate for the next Operation.
- Timeout/disconnect/cancel denies.

Decision precedence (ADR-006):
    hard deny → durable reject grant → durable allow grant →
    permission mode → interactive prompt → fail-closed

A ``PolicyGrant`` is a durable, revocable rule that either allows or rejects
an operation without interactive prompting. Grants are scoped by owner,
workspace, tool identity, effect kind, and optionally location.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dana.core.policy.effects import EffectKind
from dana.core.policy.scope import OwnerScope, scope_matches


if TYPE_CHECKING:
    from dana.core.policy.operations import Operation


class GrantDecision(Enum):
    """The decision a grant makes about an operation."""

    ALLOW = "allow"
    REJECT = "reject"


class GrantNotFound(KeyError):
    """Raised when a grant is not found by its ID."""


class GrantConflict(ValueError):
    """Raised when a grant creation conflicts with an existing grant."""


@dataclass(frozen=True)
class PolicyGrant:
    """A durable, revocable policy grant.

    Attributes:
        grant_id: Unique identifier for this grant.
        owner_scope: The ``OwnerScope`` this grant belongs to.
        decision: ``ALLOW`` or ``REJECT``.
        tool_identity: The tool name this grant applies to (exact match).
        effect_kind: The ``EffectKind`` this grant applies to.
        location: Optional location pattern (e.g. file path, URL prefix).
            Empty string means "any location" for the tool+effect.
        created_at: When the grant was created.
        revoked_at: When the grant was revoked (None if active).
        reason: Optional human-readable reason for the grant.
    """

    grant_id: str
    owner_scope: OwnerScope
    decision: GrantDecision
    tool_identity: str
    effect_kind: EffectKind
    location: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    revoked_at: datetime | None = None
    reason: str = ""

    @property
    def is_active(self) -> bool:
        """A grant is active if it has not been revoked."""
        return self.revoked_at is None

    @property
    def is_revoked(self) -> bool:
        return self.revoked_at is not None


@dataclass(frozen=True)
class GrantMatch:
    """Result of matching grants against an operation.

    Attributes:
        matched: The highest-precedence matching grant, or None.
        decision: The decision of the matched grant, or None if no match.
        grant_id: The ID of the matched grant, or None.
    """

    matched: PolicyGrant | None = None

    @property
    def decision(self) -> GrantDecision | None:
        return self.matched.decision if self.matched is not None else None

    @property
    def grant_id(self) -> str | None:
        return self.matched.grant_id if self.matched is not None else None


def grant_matches_operation(
    grant: PolicyGrant,
    operation: Operation,
) -> bool:
    """Check whether a grant matches an operation.

    A grant matches an operation when ALL of the following are true:
    1. The grant is active (not revoked).
    2. The grant's scope matches the operation's owner and workspace.
    3. The grant's ``tool_identity`` matches the operation's tool name.
    4. The grant's ``effect_kind`` is present in the operation's effects.
    5. The grant's ``location`` is empty (any location) or matches one of
       the operation's ``affected_locations``.

    Args:
        grant: The ``PolicyGrant`` to check.
        operation: The ``Operation`` to check against.

    Returns:
        True if the grant matches the operation.
    """
    if grant.is_revoked:
        return False

    # Scope check
    scope = scope_matches(grant.owner_scope, operation.owner, operation.workspace)
    if not scope.is_match:
        return False

    # Tool identity check
    if grant.tool_identity != operation.tool_identity.name:
        return False

    # Effect kind check — the grant's effect kind must be present
    # in the operation's declared effects.
    op_effect_kinds = {e.kind for e in operation.effects.effects}
    if grant.effect_kind not in op_effect_kinds:
        return False

    # Location check — empty location means "any location"
    if grant.location:
        if not any(loc == grant.location or loc.startswith(grant.location.rstrip("/") + "/") for loc in operation.affected_locations):
            return False

    return True


@runtime_checkable
class GrantStore(Protocol):
    """Durable, owner-scoped store for Policy Grants.

    Implementations MUST be safe to call from a single async task.
    All operations are scoped by ``OwnerScope``; grants are invisible
    across different scopes.

    Per ADR-003: the same contract is implemented on SQLite and PostgreSQL.
    """

    async def create_grant(self, grant: PolicyGrant) -> PolicyGrant:
        """Persist a new grant.

        Raises ``GrantConflict`` if a grant with the same ``grant_id``
        already exists in the given scope.
        """
        ...

    async def get_grant(self, scope: OwnerScope, grant_id: str) -> PolicyGrant:
        """Load a grant by ID within the given scope.

        Raises ``GrantNotFound`` if no grant with that ID exists in the scope.
        """
        ...

    async def list_grants(
        self,
        scope: OwnerScope,
        *,
        active_only: bool = True,
    ) -> list[PolicyGrant]:
        """List all grants within the given scope.

        Args:
            scope: The ``OwnerScope`` to list grants for.
            active_only: If True (default), only return active (non-revoked) grants.
        """
        ...

    async def revoke_grant(self, scope: OwnerScope, grant_id: str) -> PolicyGrant:
        """Revoke a grant by ID within the given scope.

        Revocation sets ``revoked_at`` to the current time.
        Revocation is immediate for the next Operation (ADR-006).

        Raises ``GrantNotFound`` if no grant with that ID exists in the scope.
        Returns the revoked grant.
        """
        ...

    async def find_matching_grants(
        self,
        scope: OwnerScope,
        operation: Operation,
    ) -> GrantMatch:
        """Find the highest-precedence matching grant for an operation.

        Per ADR-006 decision precedence:
        - Reject grants take precedence over allow grants.
        - Among grants with the same decision, the most specific match wins
          (tool + effect + location > tool + effect > tool only).
        - If multiple grants match at the same specificity, the earliest
          created wins.

        Args:
            scope: The ``OwnerScope`` to search within.
            operation: The ``Operation`` to match against.

        Returns:
            A ``GrantMatch`` with the highest-precedence matching grant, or
            ``GrantMatch(matched=None)`` if no grant matches.
        """
        ...

    async def close(self) -> None:
        """Close the underlying database connection."""
        ...


def _grant_specificity(grant: PolicyGrant) -> int:
    """Compute a specificity score for a grant.

    Higher score = more specific match. Used to pick the best grant
    when multiple match.

    Scoring:
    - Base: 1 point for tool identity match.
    - +1 if effect kind is specified (non-UNKNOWN).
    - +1 if location is specified (non-empty).
    """
    score = 1  # tool identity match
    if grant.effect_kind is not EffectKind.UNKNOWN:
        score += 1
    if grant.location:
        score += 1
    return score
