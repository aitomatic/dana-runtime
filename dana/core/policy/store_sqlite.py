"""SQLite adapter for the Policy Grant store.

Per ADR-003: the policy store implements the same contract on SQLite and
PostgreSQL; ``OwnerScope`` is required at every storage boundary.

Uses ``aiosqlite`` with WAL mode and ``BEGIN IMMEDIATE`` transactions
for safe concurrent access.
"""

from __future__ import annotations

from datetime import UTC, datetime

import aiosqlite

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
from dana.core.policy.store_schema import POLICY_SQLITE_DDL


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _row_to_grant(row: aiosqlite.Row) -> PolicyGrant:
    revoked_raw = row["revoked_at"]
    return PolicyGrant(
        grant_id=row["grant_id"],
        owner_scope=OwnerScope(owner_id=row["owner_id"], workspace=row["workspace"]),
        decision=GrantDecision(row["decision"]),
        tool_identity=row["tool_identity"],
        effect_kind=EffectKind(row["effect_kind"]),
        location=row["location"] or "",
        created_at=_parse_dt(row["created_at"]),
        revoked_at=_parse_dt(revoked_raw) if revoked_raw else None,
        reason=row["reason"] or "",
    )


class SQLiteGrantStore:
    """GrantStore backed by SQLite (aiosqlite)."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    @classmethod
    async def open(cls, path: str) -> SQLiteGrantStore:
        """Open (or create) the SQLite database at ``path`` and initialize schema."""
        db = await aiosqlite.connect(path)
        try:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA foreign_keys=ON")
            for stmt in POLICY_SQLITE_DDL:
                await db.execute(stmt)
            await db.commit()
        except BaseException:
            await db.close()
            raise
        return cls(db)

    @staticmethod
    def _scope_key(scope: OwnerScope) -> tuple[str, str]:
        return (scope.owner_id, scope.workspace)

    async def _fetchone(self, sql: str, params: tuple[object, ...] = ()) -> aiosqlite.Row | None:
        cursor = await self._db.execute(sql, params)
        try:
            return await cursor.fetchone()
        finally:
            await cursor.close()

    async def _fetchall(self, sql: str, params: tuple[object, ...] = ()) -> list[aiosqlite.Row]:
        cursor = await self._db.execute(sql, params)
        try:
            return await cursor.fetchall()
        finally:
            await cursor.close()

    # ------------------------------------------------------------------
    # GrantStore protocol
    # ------------------------------------------------------------------

    async def create_grant(self, grant: PolicyGrant) -> PolicyGrant:
        scope = grant.owner_scope
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            existing = await self._fetchone(
                "SELECT grant_id FROM policy_grants WHERE owner_id=? AND workspace=? AND grant_id=?",
                (*self._scope_key(scope), grant.grant_id),
            )
            if existing is not None:
                raise GrantConflict(f"grant {grant.grant_id!r} already exists for {scope.owner_id!r}/{scope.workspace!r}")

            now = _now()
            await self._db.execute(
                """
                INSERT INTO policy_grants
                    (grant_id, owner_id, workspace, decision, tool_identity, effect_kind,
                     location, created_at, revoked_at, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    grant.grant_id,
                    scope.owner_id,
                    scope.workspace,
                    grant.decision.value,
                    grant.tool_identity,
                    grant.effect_kind.value,
                    grant.location,
                    _iso(now),
                    None,
                    grant.reason,
                ),
            )
            await self._db.commit()
        except BaseException:
            await self._db.execute("ROLLBACK")
            raise

        return await self.get_grant(scope, grant.grant_id)

    async def get_grant(self, scope: OwnerScope, grant_id: str) -> PolicyGrant:
        row = await self._fetchone(
            "SELECT * FROM policy_grants WHERE owner_id=? AND workspace=? AND grant_id=?",
            (*self._scope_key(scope), grant_id),
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
            rows = await self._fetchall(
                "SELECT * FROM policy_grants WHERE owner_id=? AND workspace=? AND revoked_at IS NULL ORDER BY created_at ASC",
                (*self._scope_key(scope),),
            )
        else:
            rows = await self._fetchall(
                "SELECT * FROM policy_grants WHERE owner_id=? AND workspace=? ORDER BY created_at ASC",
                (*self._scope_key(scope),),
            )
        return [_row_to_grant(r) for r in rows]

    async def revoke_grant(self, scope: OwnerScope, grant_id: str) -> PolicyGrant:
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            existing = await self._fetchone(
                "SELECT * FROM policy_grants WHERE owner_id=? AND workspace=? AND grant_id=?",
                (*self._scope_key(scope), grant_id),
            )
            if existing is None:
                raise GrantNotFound(grant_id)

            now = _now()
            await self._db.execute(
                "UPDATE policy_grants SET revoked_at=? WHERE owner_id=? AND workspace=? AND grant_id=?",
                (_iso(now), *self._scope_key(scope), grant_id),
            )
            await self._db.commit()
        except BaseException:
            await self._db.execute("ROLLBACK")
            raise

        return await self.get_grant(scope, grant_id)

    async def find_matching_grants(
        self,
        scope: OwnerScope,
        operation: Operation,
    ) -> GrantMatch:
        """Find the highest-precedence matching grant for an operation.

        Per ADR-006: reject grants take precedence over allow grants.
        Among same-decision grants, the most specific match wins.
        """
        rows = await self._fetchall(
            "SELECT * FROM policy_grants WHERE owner_id=? AND workspace=? AND revoked_at IS NULL ORDER BY created_at ASC",
            (*self._scope_key(scope),),
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

        # Reject grants take precedence over allow grants (ADR-006)
        if best_reject is not None:
            return GrantMatch(matched=best_reject)
        if best_allow is not None:
            return GrantMatch(matched=best_allow)
        return GrantMatch(matched=None)

    async def close(self) -> None:
        await self._db.close()
