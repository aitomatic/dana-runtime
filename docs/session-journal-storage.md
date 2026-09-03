# Session Journal Storage, Migration, Rollback, and Key Rotation

**Version:** 1.0 | **Status:** Active | **Applies to:** Phase 01 (Durable Dana Conversation)

Operational runbook for the Dana Session Journal: SQLite and PostgreSQL
deployment, legacy Timeline migration, rollback, protected-state key
rotation, and PostgreSQL CI configuration.

The Session Journal is the **sole durable authority** for Dana agent
sessions. Every turn — input, streamed chunks, terminal — is appended as a
typed `JournalFact` before the model is invoked and before the response is
returned. See [`docs/system-architecture.md`](system-architecture.md) →
*Session Journal Architecture* for the design.

## 1. Storage backends

### 1.1 SQLite (default)

- **Path:** `DANA_ACP_JOURNAL` (default `~/.dana/journal.db`).
- **WAL mode:** enabled (`PRAGMA journal_mode=WAL`); keeps reader/writer
  concurrency without blocking the agent loop.
- **Foreign keys:** enforced (`PRAGMA foreign_keys=ON`); deleting a session
  header cascades to its facts.
- **Schema versioning:** a `journal_meta.schema_version` row is written on
  first open. Phase 01 supports only the current `SCHEMA_VERSION`; a mismatch
  raises `JournalError` at open.
- **File permissions:** the database and its `-wal` / `-shm` siblings should
  be `0600` (owner read/write only). The runtime creates the parent
  directory if missing but does not chmod the file; deploy with:

  ```bash
  install -d -m 0700 /var/lib/dana
  touch /var/lib/dana/journal.db
  chmod 0600 /var/lib/dana/journal.db
  export DANA_ACP_JOURNAL=/var/lib/dana/journal.db
  ```

- **Backups:** SQLite Online Backup API (e.g. `sqlite3 journal.db ".backup
  /backup/journal-$(date +%F).db"`) is safe to run against a live WAL
  database. Do NOT file-copy the `.db` file alone — copy all three siblings.

### 1.2 PostgreSQL

- **DSN:** `DANA_SESSION_JOURNAL_DSN` (e.g.
  `postgresql://dana:pass@host:5432/dana`). When set, the Postgres adapter
  is used instead of SQLite.
- **Schema:** the JSONB-typed `payload`, `metadata`, and checkpoint `data`
  columns are used; the JSONB GIN index on `(owner_id, workspace,
  session_id)` supports fast per-owner listings.
- **Owner isolation:** the `OwnerScope` (`owner_id` + `workspace`) is part
  of every primary and foreign key. Cross-owner access is impossible at the
  data layer; a missing session in another scope raises `SessionNotFound`,
  not a conflict.
- **Row-level security (RLS):** for multi-tenant Postgres deployments, the
  recommended posture is one database role per `owner_id` with RLS policies:

  ```sql
  ALTER TABLE session_journals ENABLE ROW LEVEL SECURITY;
  ALTER TABLE session_facts    ENABLE ROW LEVEL SECURITY;
  ALTER TABLE projection_checkpoints ENABLE ROW LEVEL SECURITY;

  CREATE POLICY owner_isolation ON session_journals
    USING (owner_id = current_user);
  -- Repeat for session_facts and projection_checkpoints.
  ```

  The runtime continues to scope every query by `owner_id`; RLS is
  defense-in-depth.

## 2. Migration: legacy Timeline → Session Journal

Legacy sessions persisted as `TimelineEntry` lists can be imported into a
journal via
`dana.core.session.legacy_timeline_migration.migrate_legacy_timeline`. The
migration is:

- **Idempotent** — a content-addressed SHA-256 of the canonical source
  entries is recorded on a `LEGACY_TIMELINE_MIGRATED` marker fact;
  re-migrating the identical source is a no-op.
- **Text-only in D1** — `USER_MESSAGE` and `AGENT_RESPONSE` are converted
  to journal facts; thoughts, tools, summaries, and ephemeral context are
  skipped (and counted in `MigrationResult.skipped_entries`).
- **Non-destructive** — the legacy source is not modified; re-running with a
  different source appends a second migration marker.

### Procedure

```python
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.legacy_timeline_migration import migrate_legacy_timeline
from dana.core.session.models import OwnerScope

repo = await SQLiteJournalRepository.open("/var/lib/dana/journal.db")
scope = OwnerScope(owner_id="alice", workspace="/repo")
# The target session MUST already exist; create an empty one first if needed.
result = await migrate_legacy_timeline(repo, scope, "session-id", source_entries)
assert not result.already_migrated
print(f"appended={result.appended} skipped={result.skipped_entries}")
```

Re-running the same call returns `already_migrated=True, appended=0` without
appending anything.

### Compatibility projection

For legacy readers that still consume `TimelineEntry` lists (e.g. behind the
rollback flag), `journal_facts_to_timeline_entries(facts)` projects journal
facts back to the legacy shape. Note: the compatibility projection includes
ALL assistant finals regardless of terminal status, which differs from
`ConversationProjector`'s committed-turn gating — partial output remains
visible in the legacy format.

## 3. Rollback

`DANA_SESSION_JOURNAL_AUTHORITY=0` is the documented rollback switch for the
D1 cutover:

```bash
export DANA_SESSION_JOURNAL_AUTHORITY=0
# Restart DanaACPAgent processes; they will pick up the flag at __init__.
```

Behavior in D1:

- `DanaACPAgent.journal_authority_enabled` returns `False`.
- The flag is the **future wiring point** for full legacy fallback (a
  Timeline-based ACP agent that does not touch the journal). That fallback
  is a large, separately-tracked effort and is **deferred** beyond Phase 01.
- Setting the flag to `0` today is a "stop the world" action. The journal is
  not modified by reading the flag; flipping back to `1` (or unsetting)
  restores journal-backed behavior on the next process start.

### Recovery from a bad migration

If a legacy migration produces incorrect content:

1. Do NOT delete the session — instead, archive it (`archive_session`) so it
   is excluded from `list_sessions` and the health report.
2. Re-run the migration against a NEW session id; the content-addressed
   marker prevents double-migrating the SAME source into the new session.
3. If you must purge, `purge_session` permanently deletes the header and
  all facts (cascades via foreign keys); checkpoints are also cleaned.

## 4. Protected-state key rotation

Provider replay state (e.g. OpenAI `encrypted_content`, reasoning items) is
never placed in the regular `payload`; it travels only in the
`protected_payload` bytes, envelope-encrypted via AES-GCM + HKDF + AAD by
`ProtectedStateCodec` (`dana/core/session/protected_state.py`). The key is
sourced from `DANA_SESSION_STATE_KEY`.

### Key rotation procedure

1. **Generate the new key:**
   ```bash
   NEW_KEY="$(openssl rand -base64 48)"
   ```
2. **Run the rotation pass** — re-encrypt every `protected_payload` from the
   old key to the new key. A reference rotation tool is planned; for D1,
   re-encryption is a one-time maintenance operation:
   - Open the journal with the OLD key (set `DANA_SESSION_STATE_KEY` to the
     old value).
   - For each fact with `protected_payload is not None`, decrypt → re-encrypt
       with the new key → append a replacement fact (or write a one-off
       migration script that updates the row in place; this bypasses the
       journal's append-only contract and MUST be performed offline).
3. **Cut over:** set `DANA_SESSION_STATE_KEY` to `NEW_KEY` and restart the
   agent processes. Old `protected_payload` values encrypted with the prior
   key become undecryptable — make sure the rotation pass covered every row
   before cutting over.

### Loss of the key

If `DANA_SESSION_STATE_KEY` is lost:

- The journal remains **fully readable** for conversation replay, host
  events, and trace views — only `protected_payload` becomes opaque bytes.
- Provider-continuity features (resuming an OpenAI response stream mid-turn)
  will not work for affected sessions until the next user turn refreshes the
  state.

Treat the key as a primary secret. Store it in your secrets manager (Vault,
AWS KMS, etc.), not in source control.

## 5. PostgreSQL CI configuration

The contract suite
`tests/integration/test_session_journal_contract.py` is parameterized over
both the SQLite and PostgreSQL adapters. To run the Postgres path in CI:

1. **Service container** (GitHub Actions example):

   ```yaml
   services:
     postgres:
       image: postgres:16
       env:
         POSTGRES_USER: dana_test
         POSTGRES_PASSWORD: dana_test
         POSTGRES_DB: dana_test
       ports:
         - 5432:5432
       options: >-
         --health-cmd "pg_isready -U dana_test"
         --health-interval 5s
         --health-timeout 5s
         --health-retries 10
   ```

2. **Set the test DSN**:

   ```bash
   export DANA_PG_TEST_DSN="postgresql://dana_test:dana_test@localhost:5432/dana_test"
   ```

   The contract suite picks up the DSN from the environment; when unset,
   the Postgres parameterizations are skipped (so CI without Postgres still
   passes).

3. **Run the suite**:

   ```bash
   uv run pytest tests/integration/test_session_journal_contract.py -q
   ```

4. **RLS smoke test** (optional, recommended for multi-tenant deployments):
   create two roles with the RLS policies from §1.2 and assert cross-owner
   access raises `SessionNotFound`.

## 6. Operational checks (cheat sheet)

```bash
# Connectivity + counts (read-only, redacted):
DANA_SESSION_STATE_KEY=... python -c '
import asyncio
from dana.core.session.health import check_journal_health
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.models import OwnerScope

async def main():
    repo = await SQLiteJournalRepository.open("/var/lib/dana/journal.db")
    print(await check_journal_health(repo, OwnerScope("alice", "/repo")))
    await repo.close()

asyncio.run(main())
'

# Manual legacy migration dry-run (text-only, idempotent):
# Use migrate_legacy_timeline in a REPL against a non-production journal.
```

## 7. Related documents

- [`docs/acp-configuration.md`](acp-configuration.md) — host configuration
  and the ACP protocol surface.
- [`docs/system-architecture.md`](system-architecture.md) → *Session Journal
  Architecture* — design overview.
- [`docs/project-changelog.md`](project-changelog.md) → *D1: Durable Dana
  Conversation* — what shipped in the cutover.
