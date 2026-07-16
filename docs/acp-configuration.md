# ACP Configuration

**Version:** 1.0 | **Status:** Active | **Applies to:** Phase 01 (Durable Dana Conversation)

This guide describes how to run Dana as an [Agent Client Protocol](https://agentclientprotocol.org/) (ACP) agent and configure it from a host (such as `dana-console`).

## Installation

Dana is installed with its ACP extras:

```bash
uv pip install dana[acp]
```

The ACP entry point is the stdio agent `dana-acp`, exposed as the module `dana.apps.acp`:

```bash
python -m dana.apps.acp --help
```

## Architecture overview

```
┌──────────────────────┐   JSON-RPC (stdio)   ┌──────────────────────────┐
│  Host (dana-console) │ ◀──────────────────▶ │  DanaACPAgent            │
│  - session storage   │                       │  ├─ AgentSession         │
│  - restart replay    │                       │  │  ├─ STARAgent         │
│  - UI / streaming    │                       │  │  └─ JournalRepository │
└──────────────────────┘                       │  ├─ Session Journal      │
                                                │  │  (SQLite / Postgres)  │
                                                │  └─ Conversation / Host  │
                                                │     Event projectors    │
                                                └──────────────────────────┘
```

`DanaACPAgent` is the sole ACP façade. It translates the protocol calls
(`initialize`, `session/new`, `session/load`, `session/resume`,
`session/prompt`, `session/cancel`) into `AgentSession` operations and
streams `HostEvent`s back to the host as ACP `session_update` notifications.

The **Session Journal** is the sole durable authority: every turn — input,
streamed chunks, terminal — is appended as a typed `JournalFact` before the
model is invoked and before the response is returned. A host restart that
calls `session/load` sees the full conversation replayed before the call
returns.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `DANA_ACP_JOURNAL` | `~/.dana/journal.db` | Path to the SQLite journal database. Ignored when `DANA_SESSION_JOURNAL_DSN` is set. |
| `DANA_SESSION_JOURNAL_DSN` | _(unset)_ | PostgreSQL DSN (e.g. `postgresql://user:pass@host/db`). When set, the Postgres adapter is used instead of SQLite. |
| `DANA_SESSION_STATE_KEY` | _(unset)_ | High-entropy key (32+ bytes) used by `EnvProtectedStateKeyProvider` to envelope-encrypt provider replay state (e.g. OpenAI `encrypted_content`) into `protected_payload`. **Required in production.** |
| `DANA_SESSION_JOURNAL_AUTHORITY` | `1` | Feature flag for the D1 cutover. `1` (default) makes the journal the durable authority. `0` selects legacy compatibility mode (rollback switch; full fallback deferred — see [Rollback](#rollback)). |
| `USER` | _(shell)_ | Default `owner_id` for the `OwnerScope`. Override per-request from the host when multi-tenant. |

## Configuring dana-console

To register Dana as a Custom ACP Agent in `dana-console`:

1. **Install Dana** in the environment `dana-console` runs from (or ensure the `dana-acp` console script is on `PATH`).
2. **Set the protected-state key** in the environment:
   ```bash
   export DANA_SESSION_STATE_KEY="$(openssl rand -base64 48)"
   ```
   This key MUST be the same across restarts; losing it makes existing
   `protected_payload` values undecryptable (the journal remains readable,
   only provider replay state is lost).
3. **Point at a durable journal location** (SQLite default):
   ```bash
   export DANA_ACP_JOURNAL=/var/lib/dana/journal.db
   ```
   Or use Postgres (recommended for shared deployments):
   ```bash
   export DANA_SESSION_JOURNAL_DSN=postgresql://dana@db.local/dana
   ```
4. **Register the agent** with the command `dana-acp` (stdio protocol). The
   host spawns one process per session; restart spawns a fresh process that
   reattaches to the same journal.

## Stdio protocol

Dana speaks ACP over stdio. Stdout is reserved for JSON-RPC frames;
diagnostics are routed to stderr via `structlog`. Supported methods:

| Method | Behavior |
| --- | --- |
| `initialize` | Returns `protocolVersion`, `agentCapabilities.load_session=true`, and the Dana version. |
| `session/new` | Creates a session row + `SESSION_CREATED` fact; returns a fresh `session_id`. |
| `session/load` | Recovers interrupted turns, rehydrates `AgentSession`, replays host events as `session_update` notifications BEFORE returning `LoadSessionResponse`. |
| `session/resume` | Unstable alias for `session/load`. |
| `session/prompt` | Appends `TURN_STARTED` + `USER_CONTENT_FINAL`, streams chunks, terminalizes the turn with exactly one of `TURN_COMPLETED` / `TURN_CANCELLED` / `TURN_ERROR`. |
| `session/cancel` | Sets the cancel event; the active turn terminates as `TURN_CANCELLED`. |

## Restart and recovery behavior

When a host (or the agent subprocess) is killed mid-turn:

1. The journal already contains `TURN_STARTED` and the user content final;
   streamed chunks may have been flushed by the byte/time bound.
2. On the next `session/load`, `AgentSession` calls
   `recover_interrupted_turns`, which detects started-but-unterminated turns
   and appends a typed `TURN_INTERRUPTED` fact for each.
3. The `ConversationProjector` excludes partial assistant output from
   committed messages (partial output remains in the Host Event View) and
   surfaces an interruption observation to the next model turn.
4. The full Host Event stream is replayed to the host as `session_update`
   notifications before `LoadSessionResponse` returns, so the UI shows the
   pre-crash conversation immediately.

Recovery is **idempotent**: running `recover_interrupted_turns` again finds
no remaining interrupted turns (the appended `TURN_INTERRUPTED` is itself a
terminal fact).

## Health checks

Operational health is exposed via
`dana.core.session.health.check_journal_health(repository, scope)`:

```python
from dana.core.session.health import check_journal_health
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.models import OwnerScope

repo = await SQLiteJournalRepository.open("/var/lib/dana/journal.db")
report = await check_journal_health(repo, OwnerScope(owner_id="alice", workspace="/repo"))
print(report.database_connectivity, report.total_sessions, report.projection_lag_max)
```

The report is **redacted**: only counts and booleans are returned — no
owner_ids, session_ids, payloads, or protected payloads appear. Fields:

| Field | Meaning |
| --- | --- |
| `database_connectivity` | Round-trip to `list_sessions` succeeded. |
| `total_sessions` | Non-DELETED session count in scope. |
| `active_sessions` | Sessions in the `ACTIVE` lifecycle state. |
| `archived_sessions` | Sessions in the `ARCHIVED` lifecycle state. |
| `interrupted_turns_detected` | Started-but-unterminated turns observed (detection only; recovery is performed on the next `session/load`). |
| `projection_lag_max` | Maximum gap between journal `version` and projection `last_sequence` across tracked projections. |
| `legacy_migration_markers` | Count of `LEGACY_TIMELINE_MIGRATED` markers (confirms legacy imports landed). |
| `errors` | Per-session error class names (no payloads). |

## Rollback

The `DANA_SESSION_JOURNAL_AUTHORITY=0` flag is the documented rollback
switch for the D1 cutover. Setting it disables journal-backed behavior in
`DanaACPAgent`.

Full legacy fallback (a Timeline-based ACP agent that does not touch the
journal) is **deferred** to a later phase and tracked separately. For D1 the
flag's primary purpose is documentation and future wiring — operators should
treat a flag flip as a "stop the world" action and consult
[`docs/session-journal-storage.md`](session-journal-storage.md) for the
runbook.
