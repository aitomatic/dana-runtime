# D1 Durable Dana Conversation — Briefing

**Date:** 2026-07-17  
**Status:** Phase 01 shipped  
**Branch:** `feat/acp-agent-session-kernel`

---

## What changed

Dana can now hold a multi-turn text conversation through `dana-acp` that **survives process restarts**. Before this work, every ACP process restart created a blank conversation — the user lost all context. Now the conversation is persisted to a Session Journal (SQLite or PostgreSQL) and automatically resumed on reconnect.

This is the first of six planned deliveries (D1–D6) that progressively deepen the STAR runtime around the ACP protocol.

---

## The one-sentence version

Dana streams a text conversation through `dana-acp`, appends every turn to an append-only Session Journal, and when the process restarts the console calls `session/load` to replay the full history before the user types another word.

---

## New capabilities

### 1. Session Journal — sole durable authority

Every turn is recorded as immutable, typed **Journal Facts** (turn started, user input, assistant chunks, assistant final, terminal). Facts are append-only and retained until explicit session deletion. Two database adapters share one behavioral contract:

| Adapter | Locking | Payloads | Use case |
|---|---|---|---|
| SQLite (default) | WAL + `BEGIN IMMEDIATE` | JSON text | Local / single-process |
| PostgreSQL | `SELECT … FOR UPDATE` | JSONB | Multi-process / cloud |

Both enforce **optimistic concurrency**: each append carries an expected version; a version mismatch raises `JournalConflict`. One active writer per session; other sessions remain concurrent.

### 2. Crash recovery

A turn that started but never received a terminal fact (process killed mid-stream) is detected on next load and marked as an **Interrupted Turn**. Partial assistant output stays visible to the host but is **excluded** from the model-facing Conversation View — the model is told "the previous turn was interrupted" instead of seeing a half-finished answer as complete.

### 3. Protected state encryption

Provider replay material (e.g. OpenAI `encrypted_content`) is envelope-encrypted with AES-GCM (HKDF-derived key, optional AAD binding) and stored in `protected_payload`. It never appears in host-visible or trace projections. The encryption key is sourced from `DANA_SESSION_STATE_KEY`.

### 4. Legacy Timeline migration

Existing `timeline.json` sessions can be imported into the journal via `migrate_legacy_timeline()`. The migration is **idempotent** — a content-addressed SHA-256 fingerprint prevents double-import. Text entries (user/assistant) are converted; tool entries are skipped (D2 will handle tools).

### 5. ACP stdio agent (`dana-acp`)

A new entry point exposes Dana over the Agent Client Protocol:

```
dana-acp  # stdio JSON-RPC server
```

Implements: `initialize`, `session/new`, `session/load`, `session/resume` (unstable), `session/prompt`, `session/cancel`. Stdout carries JSON-RPC frames only; all diagnostics go to stderr.

### 6. Console restart continuation

dana-console now persists the session ID in `sessionStorage`. On WebSocket reconnect (dev hot-reload or process restart), the browser sends the prior session ID and the backend calls `session/load` instead of `new_session` — the user sees the previous conversation continue seamlessly.

---

## Architecture

```
ACP adapter (dana-acp)     CLI adapters (dana-code, adana)     Future hosts
      |                              |                             |
      +-------- AgentSession --------+-----------------------------+
                     |
            one active turn / session
                     |
        +------------+------------+
        |            |            |
  Session Journal  Tool Catalog  Execution Policy
  (sole authority)  (D2)          (D3)
        |
        +-- Conversation View → STARAgent / model
        +-- Host Event View   → ACP / CLI
        +-- Trace View        → exporters (future)
```

`AgentSession` is the only broad host-facing module. ACP types never enter STAR core.

---

## What is NOT changed

| Component | Status |
|---|---|
| `dana-code`, `adana`, `dana-repl` CLIs | **Unchanged** — still use legacy Timeline + `timeline.json` |
| Existing `STARAgent.query()` / `aquery()` | **Unchanged** — source-compatible |
| Existing stored sessions | **Unchanged** — migrated on-demand when first accessed via ACP |
| Tools, permissions, model switching, MCP, attachments | **Deferred** — D2 through D6 |

The CLIs will move to the journal incrementally in later phases. Phase 01 scoped the cutover to `dana-acp` only.

---

## Key files

| Module | Purpose |
|---|---|
| `dana/core/session/models.py` | `OwnerScope`, `JournalFact`, `NewJournalFact`, `FactType`, `ArtifactRef` |
| `dana/core/session/protected_state.py` | `ProtectedStateCodec` (AES-GCM + HKDF), `EnvProtectedStateKeyProvider` |
| `dana/core/session/journal/` | `JournalRepository` protocol, SQLite + PostgreSQL adapters |
| `dana/core/session/projections/` | `ConversationProjector`, `HostEventProjector` |
| `dana/core/session/agent_session.py` | `AgentSession` — serialized turns, streaming, cancel, journal lifecycle |
| `dana/core/session/legacy_timeline_migration.py` | `recover_interrupted_turns`, `migrate_legacy_timeline` |
| `dana/core/session/health.py` | `check_journal_health` — redacted operational report |
| `dana/apps/acp/` | `DanaACPAgent`, translation, `dana-acp` entry point |

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DANA_SESSION_STATE_KEY` | (required for protected state) | Encryption key for provider replay state |
| `DANA_SESSION_JOURNAL_AUTHORITY` | `1` | `0` = legacy compatibility mode (rollback switch) |
| `DANA_TEST_POSTGRES_DSN` | (test only) | PostgreSQL DSN for contract tests |

---

## Test coverage

- **210 session + integration tests** (16 PostgreSQL skipped without DSN)
- **2087 full unit suite** — zero failures, no regressions
- Contract tests cover: create/load, batch append, version conflict, owner isolation, archive/purge, concurrent writers, crash recovery, migration idempotency, ACP framing, burst ordering, cancel, stderr/stdout discipline

---

## What comes next

| Phase | Outcome | Depends on |
|---|---|---|
| D2 | Visible, cancellable tool execution | D1 |
| D3 | Autonomous permission policy (modes + grants) | D1, D2 |
| D4 | Configured model switching | D1 |
| D5 | Configured MCP tools | D1–D3 |
| D6 | Images and file resources | D1, D3 |

See [`docs/project-roadmap.md`](project-roadmap.md) for the full plan.

---

## Further reading

- [ACP Configuration Guide](acp-configuration.md) — how to set up `dana-acp`
- [Session Journal Storage](session-journal-storage.md) — SQLite/PostgreSQL setup, migration, rollback, key rotation
- [System Architecture](system-architecture.md) — Session Journal Architecture section
- [Design Spec](superpowers/specs/2026-07-07-acp-star-adapter-design.md) — approved full design
- [Project Changelog](project-changelog.md) — detailed D1 changelog
