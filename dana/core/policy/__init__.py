"""Permission policy package — modes, grants, and storage.

Per ADR-006: the policy evaluates normalized Operations through a fixed
precedence: hard deny → durable reject grant → durable allow grant →
permission mode → interactive prompt → fail-closed.

Per ADR-003: the grant store implements the same contract on SQLite and
PostgreSQL; ``OwnerScope`` is required at every storage boundary.
"""

from __future__ import annotations

from dana.core.policy.grants import (
    GrantConflict,
    GrantMatch,
    GrantNotFound,
    GrantStore,
    PolicyGrant,
    grant_matches_operation,
)
from dana.core.policy.modes import PermissionMode
from dana.core.policy.scope import OwnerScope, scope_matches


__all__ = [
    "GrantConflict",
    "GrantMatch",
    "GrantNotFound",
    "GrantStore",
    "OwnerScope",
    "PermissionMode",
    "PolicyGrant",
    "grant_matches_operation",
    "scope_matches",
]
