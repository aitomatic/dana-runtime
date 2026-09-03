"""PostgreSQL adapter for the Policy Grant store.

Per ADR-003: the policy store implements the same contract on SQLite and
PostgreSQL; ``OwnerScope`` is required at every storage boundary.

Uses ``asyncpg`` with a single connection. Writers are serialized via
``SELECT ... FOR UPDATE`` inside a transaction.
"""

from __future__ import annotations

from datetime import UTC, datetime

import asyncpg

from dana.core.policy.effects import EffectKind
from dana.core.policy.grants import (
    GrantConflict,
    GrantDecision,
    GrantMatch,
    GrantNotFound,
    PolicyGrant,
    _grant_specificity,
    grant_matches_operation,
)
from dana.core.policy.operations import Operation
from dana.core.policy.scope import OwnerScope
from dana.core.policy.store_schema import POLICY_POSTGRES_DDL


def _now() -> datetime:
    return datetime.now(UTC)


def _row_to_grant(row: asyncpg.Record) -> PolicyGrant:
    return PolicyGrant(
        grant_id=row["grant_id"],
        owner_scope=OwnerScope(owner_id=row["owner_id"], workspace=row["workspace"]),
        decision=GrantDecision(row["decision"]),
        tool_identity=row["tool_identity"],
        effect_kind=EffectKind(row["effect_kind"]),
        location=row["location"] or "",
        created_at=row["created_at"],
        revoked_at=row["revoked_at"],
        reason=row["reason"] or "",
    )


class PostgresGrantStore:
    """GrantStore backed by PostgreSQL (asyncpg)."""

    def __init__(self, db: asyncpg.Connection) -> None:
        self._db = db

    @classmethod
    async def open(cls, dsn: str) -> PostgresGrantStore:
        """Connect to ``dsn`` and initialize the schema (idempotent)."""
        db = await asyncpg.connect(dsn=dsn)
        try:
            for stmt in POLICY_POSTGRES_DDL:
                await db.execute(stmt)
        except BaseException:
            await db.close()
            raise
        return cls(db)

    # ------------------------------------------------------------------
    # GrantStore protocol
    # ------------------------------------------------------------------

    async def create_grant(self, grant: PolicyGrant) -> PolicyGrant:
        scope = grant.owner_scope
        async with self._db.transaction():
            existing = await self._db.fetchval(
                "SELECT grant_id FROM policy_grants WHERE owner_id=$1 AND workspace=$2 AND grant_id=$3",
                scope.owner_id,
                scope.workspace,
                grant.grant_id,
            )
            if existing is not None:
                raise GrantConflict(f"grant {grant.grant_id!r} already exists for {scope.owner_id!r}/{scope.workspace!r}")

            now = _now()
            try:
                await self._db.execute(
                    """
                    INSERT INTO policy_grants
                        (grant_id, owner_id, workspace, decision, tool_identity, effect_kind,
                         location, created_at, revoked_at, reason)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    grant.grant_id,
                    scope.owner_id,
                    scope.workspace,
                    grant.decision.value,
                    grant.tool_identity,
                    grant.effect_kind.value,
                    grant.location,
                    now,
                    None,
                    grant.reason,
                )
            except asyncpg.exceptions.UniqueViolationError:
                raise GrantConflict(f"grant {grant.grant_id!r} already exists for {scope.owner_id!r}/{scope.workspace!r}") from None

        return await self.get_grant(scope, grant.grant_id)

    async def get_grant(self, scope: OwnerScope, grant_id: str) -> PolicyGrant:
        row = await self._db.fetchrow(
            "SELECT * FROM policy_grants WHERE owner_id=$1 AND workspace=$2 AND grant_id=$3",
            scope.owner_id,
            scope.workspace,
            grant_id,
        )
        if row is None:
            raise GrantNotFound(grant_id)
        return _row_to_grant(row)

    async def list_grants(
        self,
        scope: OwnerScope,
        *,
        active_only: bool = True,
    ) -> list[PolicyGrant]:
        if active_only:
            rows = await self._db.fetch(
                "SELECT * FROM policy_grants WHERE owner_id=$1 AND workspace=$2 AND revoked_at IS NULL ORDER BY created_at ASC",
                scope.owner_id,
                scope.workspace,
            )
        else:
            rows = await self._db.fetch(
                "SELECT * FROM policy_grants WHERE owner_id=$1 AND workspace=$2 ORDER BY created_at ASC",
                scope.owner_id,
                scope.workspace,
            )
        return [_row_to_grant(r) for r in rows]

    async def revoke_grant(self, scope: OwnerScope, grant_id: str) -> PolicyGrant:
        async with self._db.transaction():
            existing = await self._db.fetchrow(
                "SELECT * FROM policy_grants WHERE owner_id=$1 AND workspace=$2 AND grant_id=$3 FOR UPDATE",
                scope.owner_id,
                scope.workspace,
                grant_id,
            )
            if existing is None:
                raise GrantNotFound(grant_id)

            now = _now()
            await self._db.execute(
                "UPDATE policy_grants SET revoked_at=$1 WHERE owner_id=$2 AND workspace=$3 AND grant_id=$4",
                now,
                scope.owner_id,
                scope.workspace,
                grant_id,
            )

        return await self.get_grant(scope, grant_id)

    async def find_matching_grants(
        self,
        scope: OwnerScope,
        operation: Operation,
    ) -> GrantMatch:
        rows = await self._db.fetch(
            "SELECT * FROM policy_grants WHERE owner_id=$1 AND workspace=$2 AND revoked_at IS NULL ORDER BY created_at ASC",
            scope.owner_id,
            scope.workspace,
        )

        best_reject: PolicyGrant | None = None
        best_reject_spec = -1
        best_allow: PolicyGrant | None = None
        best_allow_spec = -1

        for row in rows:
            grant = _row_to_grant(row)
            if not grant_matches_operation(grant, operation):
                continue

            spec = _grant_specificity(grant)
            if grant.decision is GrantDecision.REJECT:
                if spec > best_reject_spec:
                    best_reject = grant
                    best_reject_spec = spec
            else:
                if spec > best_allow_spec:
                    best_allow = grant
                    best_allow_spec = spec

        if best_reject is not None:
            return GrantMatch(matched=best_reject)
        if best_allow is not None:
            return GrantMatch(matched=best_allow)
        return GrantMatch(matched=None)

    async def close(self) -> None:
        await self._db.close()
