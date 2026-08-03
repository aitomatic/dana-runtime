"""OwnerScope and workspace scoping for permission policy.

Per ADR-006: grants default to Owner Scope + workspace + Tool Identity +
effect + location; broader grants require explicit operator provisioning.

Per ADR-003: ``OwnerScope`` is required at every storage boundary.

Cross-owner isolation: grants with different ``OwnerScope`` values never
match the same operation. Within the same owner, workspace scoping further
restricts visibility.
"""

from __future__ import annotations

from dataclasses import dataclass

from dana.core.session.models import OwnerScope as SessionOwnerScope


# Re-export the canonical OwnerScope from session models.
# The policy layer uses the same type for consistency.
OwnerScope = SessionOwnerScope


@dataclass(frozen=True)
class ScopeMatch:
    """Result of a scope comparison between a grant and an operation.

    Attributes:
        owner_match: True if the owner_id matches.
        workspace_match: True if the workspace matches.
        is_match: True if both owner and workspace match.
    """

    owner_match: bool
    workspace_match: bool

    @property
    def is_match(self) -> bool:
        return self.owner_match and self.workspace_match


def scope_matches(
    grant_scope: OwnerScope,
    operation_owner: str | None,
    operation_workspace: str | None,
) -> ScopeMatch:
    """Check whether a grant's scope matches an operation's owner/workspace.

    A grant matches an operation when:
    - The grant's ``owner_id`` equals the operation's ``owner``, AND
    - The grant's ``workspace`` equals the operation's ``workspace``.

    If the operation has no owner or workspace, the grant does not match
    (fail closed — an unscoped operation cannot match a scoped grant).

    Args:
        grant_scope: The ``OwnerScope`` the grant was created with.
        operation_owner: The operation's ``owner`` field (may be None).
        operation_workspace: The operation's ``workspace`` field (may be None).

    Returns:
        A ``ScopeMatch`` with individual and combined match flags.
    """
    owner_match = operation_owner is not None and grant_scope.owner_id == operation_owner
    workspace_match = operation_workspace is not None and grant_scope.workspace == operation_workspace
    return ScopeMatch(owner_match=owner_match, workspace_match=workspace_match)
