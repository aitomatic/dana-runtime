"""
Shared schema definition for Session Journal tables.

Both adapters use IDENTICAL logical table/column names so that the public
interface and the contract tests stay backend-agnostic. The only differences
are backend-native types:

* SQLite  — ``TEXT`` for JSON columns (read/written via ``json.dumps``).
* Postgres — ``JSONB`` for JSON columns (binary JSON, indexable).

A single integer ``SCHEMA_VERSION`` is recorded in the ``schema_version`` row
of a small ``journal_meta`` table on first init. Future migrations will
consult this value. Phase 01 only supports fresh-create; no migration path is
implemented yet (YAGNI).
"""

from __future__ import annotations


# Bumped on any backwards-incompatible change to the table shapes. Phase 01
# ships v1; a future schema change MUST increment this and add a migration.
SCHEMA_VERSION = 2


# --- SQLite DDL -----------------------------------------------------------
# JSON columns are TEXT; serialized with json.dumps / deserialized with json.loads.

SQLITE_CREATE_SESSION_JOURNALS = """
CREATE TABLE IF NOT EXISTS session_journals (
    owner_id    TEXT NOT NULL,
    workspace   TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (owner_id, workspace, session_id)
)
"""

SQLITE_CREATE_SESSION_FACTS = """
CREATE TABLE IF NOT EXISTS session_facts (
    fact_id           TEXT NOT NULL PRIMARY KEY,
    owner_id          TEXT NOT NULL,
    workspace         TEXT NOT NULL,
    session_id        TEXT NOT NULL,
    sequence          INTEGER NOT NULL,
    fact_type         TEXT NOT NULL,
    timestamp         TEXT NOT NULL,
    correlation_id    TEXT NOT NULL,
    causation_id      TEXT,
    schema_version    INTEGER NOT NULL,
    payload           TEXT NOT NULL,
    protected_payload BLOB,
    artifact_refs     TEXT,
    UNIQUE (owner_id, workspace, session_id, sequence),
    FOREIGN KEY (owner_id, workspace, session_id)
        REFERENCES session_journals (owner_id, workspace, session_id) ON DELETE CASCADE
)
"""

SQLITE_CREATE_PROJECTION_CHECKPOINTS = """
CREATE TABLE IF NOT EXISTS projection_checkpoints (
    owner_id       TEXT NOT NULL,
    workspace      TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    projection_name TEXT NOT NULL,
    last_sequence  INTEGER NOT NULL,
    updated_at     TEXT NOT NULL,
    data           TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (owner_id, workspace, session_id, projection_name)
)
"""

SQLITE_CREATE_JOURNAL_META = """
CREATE TABLE IF NOT EXISTS journal_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

SQLITE_DDL = [
    SQLITE_CREATE_SESSION_JOURNALS,
    SQLITE_CREATE_SESSION_FACTS,
    SQLITE_CREATE_PROJECTION_CHECKPOINTS,
    SQLITE_CREATE_JOURNAL_META,
]


# --- Postgres DDL ---------------------------------------------------------
# Same shape, but JSON columns become JSONB for binary storage + indexing.

POSTGRES_CREATE_SESSION_JOURNALS = """
CREATE TABLE IF NOT EXISTS session_journals (
    owner_id    TEXT NOT NULL,
    workspace   TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TIMESTAMPTZ NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (owner_id, workspace, session_id)
)
"""

POSTGRES_CREATE_SESSION_FACTS = """
CREATE TABLE IF NOT EXISTS session_facts (
    fact_id           TEXT NOT NULL PRIMARY KEY,
    owner_id          TEXT NOT NULL,
    workspace         TEXT NOT NULL,
    session_id        TEXT NOT NULL,
    sequence          INTEGER NOT NULL,
    fact_type         TEXT NOT NULL,
    timestamp         TIMESTAMPTZ NOT NULL,
    correlation_id    TEXT NOT NULL,
    causation_id      TEXT,
    schema_version    INTEGER NOT NULL,
    payload           JSONB NOT NULL,
    protected_payload BYTEA,
    artifact_refs     JSONB,
    UNIQUE (owner_id, workspace, session_id, sequence),
    FOREIGN KEY (owner_id, workspace, session_id)
        REFERENCES session_journals (owner_id, workspace, session_id) ON DELETE CASCADE
)
"""

POSTGRES_CREATE_PROJECTION_CHECKPOINTS = """
CREATE TABLE IF NOT EXISTS projection_checkpoints (
    owner_id        TEXT NOT NULL,
    workspace       TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    projection_name TEXT NOT NULL,
    last_sequence   INTEGER NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL,
    data            JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (owner_id, workspace, session_id, projection_name)
)
"""

POSTGRES_CREATE_JOURNAL_META = """
CREATE TABLE IF NOT EXISTS journal_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

POSTGRES_DDL = [
    POSTGRES_CREATE_SESSION_JOURNALS,
    POSTGRES_CREATE_SESSION_FACTS,
    POSTGRES_CREATE_PROJECTION_CHECKPOINTS,
    POSTGRES_CREATE_JOURNAL_META,
]
