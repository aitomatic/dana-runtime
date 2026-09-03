"""Shared schema definition for Policy Grant store tables.

Both adapters use IDENTICAL logical table/column names so that the public
interface stays backend-agnostic. The only differences are backend-native types:

* SQLite  — ``TEXT`` for JSON columns (read/written via ``json.dumps``).
* Postgres — ``TEXT`` columns (no JSONB needed for the simple grant schema).

Per ADR-003: the policy store implements the same contract on SQLite and
PostgreSQL.
"""

from __future__ import annotations


POLICY_SQLITE_CREATE_GRANTS = """
CREATE TABLE IF NOT EXISTS policy_grants (
    grant_id       TEXT NOT NULL,
    owner_id       TEXT NOT NULL,
    workspace      TEXT NOT NULL,
    decision       TEXT NOT NULL CHECK (decision IN ('allow', 'reject')),
    tool_identity  TEXT NOT NULL,
    effect_kind    TEXT NOT NULL,
    location       TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    revoked_at     TEXT,
    reason         TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (owner_id, workspace, grant_id)
)
"""

POLICY_SQLITE_CREATE_GRANTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_policy_grants_active
    ON policy_grants (owner_id, workspace, revoked_at)
"""

POLICY_SQLITE_DDL = [
    POLICY_SQLITE_CREATE_GRANTS,
    POLICY_SQLITE_CREATE_GRANTS_INDEX,
]


POLICY_POSTGRES_CREATE_GRANTS = """
CREATE TABLE IF NOT EXISTS policy_grants (
    grant_id       TEXT NOT NULL,
    owner_id       TEXT NOT NULL,
    workspace      TEXT NOT NULL,
    decision       TEXT NOT NULL CHECK (decision IN ('allow', 'reject')),
    tool_identity  TEXT NOT NULL,
    effect_kind    TEXT NOT NULL,
    location       TEXT NOT NULL DEFAULT '',
    created_at     TIMESTAMPTZ NOT NULL,
    revoked_at     TIMESTAMPTZ,
    reason         TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (owner_id, workspace, grant_id)
)
"""

POLICY_POSTGRES_CREATE_GRANTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_policy_grants_active
    ON policy_grants (owner_id, workspace, revoked_at)
"""

POLICY_POSTGRES_DDL = [
    POLICY_POSTGRES_CREATE_GRANTS,
    POLICY_POSTGRES_CREATE_GRANTS_INDEX,
]
