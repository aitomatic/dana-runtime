# Project Changelog

## D1: Durable Dana Conversation (2026-07-17)

The Session Journal is now the sole durable authority for Dana agent
sessions. Every turn — input, streamed chunks, terminal — is appended as a
typed, immutable `JournalFact` before the model is invoked and before the
response is returned. A host (or agent subprocess) restart that calls
`session/load` replays the full conversation before returning.

### Added
- Session Journal with append-only facts (`OwnerScope`, `FactType`,
  `JournalFact`) — `dana/core/session/models.py`.
- SQLite and PostgreSQL adapters with optimistic concurrency control
  (`JournalConflict` on version skew) — `dana/core/session/journal/`.
- Conversation and Host Event projectors; named `ProjectionCheckpoint`
  cursors stored out-of-band of journal facts —
  `dana/core/session/projections/`.
- `AgentSession` with serialized text turns, streaming (byte/time-bounded
  chunk flush), and exactly-one-terminal-fact-per-turn invariant —
  `dana/core/session/agent_session.py`.
- Crash recovery via `recover_interrupted_turns` — detects started-but-
  unterminated turns and appends a typed `TURN_INTERRUPTED` fact for each.
  Idempotent.
- Legacy Timeline migration (`migrate_legacy_timeline`) — imports legacy
  `TimelineEntry` sessions into the journal. Idempotent via a content-
  addressed SHA-256 on a `LEGACY_TIMELINE_MIGRATED` marker fact. Text-only
  in D1.
- ACP stdio agent (`DanaACPAgent`) covering `initialize`, `session/new`,
  `session/load`, `session/resume`, `session/prompt`, `session/cancel`;
  stdout reserved for JSON-RPC frames, diagnostics via `structlog` to
  stderr — `dana/apps/acp/`.
- Console restart continuation — `session/load` replays Host Events as ACP
  `session_update` notifications BEFORE returning, so the host UI shows the
  pre-crash conversation immediately.
- Protected-state envelope encryption (AES-GCM + HKDF + AAD) for provider
  replay material in `protected_payload`, keyed by `DANA_SESSION_STATE_KEY` —
  `dana/core/session/protected_state.py`.
- Operational health check `check_journal_health` returning a fully
  redacted aggregate report (counts and booleans only — no owner_ids,
  session_ids, payloads, or protected payloads) covering database
  connectivity, session counts by status, interrupted-turn detection,
  projection lag, and legacy migration marker counts —
  `dana/core/session/health.py`.
- `DANA_SESSION_JOURNAL_AUTHORITY` feature flag for rollback (default `1`;
  `0` selects legacy compatibility mode; full legacy fallback deferred
  beyond D1).
- Docs: [`docs/acp-configuration.md`](acp-configuration.md),
  [`docs/session-journal-storage.md`](session-journal-storage.md), and a
  *Session Journal Architecture* section in
  [`docs/system-architecture.md`](system-architecture.md).

### Files
- New: `dana/core/session/health.py`,
  `docs/acp-configuration.md`, `docs/session-journal-storage.md`.
- Modified: `dana/apps/acp/agent.py` (feature-flag wiring),
  `docs/system-architecture.md`, `docs/project-roadmap.md`.
- Tests: `tests/unit/core/session/test_health.py`,
  `tests/integration/test_acp_agent.py::TestJournalAuthorityFlag`.

## [Unreleased]

### Added
- Public `STARAgent.override_system_prompt_template(template, persist=False)` API for replacing the complete system prompt template used by LLM requests. The default is an ephemeral, agent-instance override with no repository write; `persist=True` is codec-runtime-only and writes to the configured prompt repository. Because defaults are not merged, replacements must retain required tool-usage and output-format instructions.
- LangSmith as an alternative tracing backend. `@observable` (`dana/common/observable.py`) dispatches to `langsmith.traceable` when `LANGSMITH_TRACING=true` or `DANA_LANGSMITH_ENABLED` truthy; exclusive with Langfuse (LangSmith takes precedence). No call-site changes — all 30+ `@observable` sites traced automatically. Add via `pip install dana-agent[observability]`. LangSmith API key: `LANGSMITH_API_KEY`.
- Single-knob env trigger `DANA_COMPACT_TRIGGER_TOKENS` (default 150000, clamp `[8k, 2M]`) for compression threshold (P3).
- Optional `system_tokens_fn` / `tools_tokens_fn` callbacks on `CompressedTimeline` — fold system-prompt and tools-schema size into `needs_compression()` estimate without coupling to any provider.
- Client-side tool-result stubbing (`cheap_shrink_tool_results()`, P6) with predictive savings gate; opt-in via `enable_cheap_shrink_tool_results`.
- `PromptTooLongError` typed exception; per-provider mapping for Anthropic (`invalid_request_error` + "prompt is too long"), OpenAI-compatible (`context_length_exceeded`). Gemini logs post-hoc WARNING on `MAX_TOKENS` finish (no SDK signal available).
- Reactive compaction in `llm_caller._invoke_llm_sync/async`: PTL catch → `timeline.reactive_compact(attempt)` (drop 5→10→20, forward-orphan pruning, full re-summary) with exponential backoff 1s/3s (P2).
- Per-session circuit breaker with time-based cooldown recovery (`DANA_CIRCUIT_COOLDOWN_SECONDS`, default 300s) and half-open probe; ops escape hatch `CompressedTimeline.reset_circuit()`.
- Kill switch for reactive compaction via `DANA_DISABLE_REACTIVE_COMPACT=1` env var or `CompressedTimelineConfig.enable_reactive_compact=False`.
- `dana/core/timeline/telemetry.py` with `CompressionLogFields` TypedDict (authoritative allowlist) + `new_compaction_id()` helper.
- AST-based unit test `test_log_field_allowlist.py` — fails CI when log `extra={...}` keys drift outside the allowlist.

### Changed
- `@observable` (`dana/common/observable.py`): when no tracing backend is enabled, the decorator now returns the target function unchanged (identity) instead of wrapping it in a passthrough flush layer. No introspection-sensitive call sites affected; `inspect.signature()` on decorated functions now returns the real signature. Langfuse-path tracing behavior is unchanged.
- `@observable` LangSmith path now sanitizes Dana runtime objects before SDK serialization. Live agents/timelines are logged as compact summaries, avoiding `RecursionError` crashes on cyclic `CompressedTimeline` graphs.
- `CompressedTimeline.__init__` default `max_tokens_until_compression` now defers to env trigger (150000) when unset. Explicit value continues to win.
- `CompressedTimelineConfig` gains `enable_cheap_shrink_tool_results`, `cheap_shrink_keep_recent`, `enable_reactive_compact` fields.
- `star_agent._maybe_compress_timeline` (sync + async) re-raises `PromptTooLongError` from summary path instead of swallowing — lets caller-layer retry kick in.

### Files
- New: `dana/core/timeline/compact_trigger.py`, `dana/core/timeline/telemetry.py`
- Modified: `dana/core/timeline/compression_engine.py`, `dana/core/timeline/compressed_timeline.py`, `dana/core/agent/star_agent.py`, `dana/core/llm/llm_caller.py`, `dana/common/llm/types.py`, `dana/common/llm/providers/{anthropic,openai_compatible_base,gemini}.py`
- Tests: `tests/unit/test_compact_trigger.py`, `tests/unit/test_compressed_timeline_callbacks.py`, `tests/unit/test_cheap_shrink.py`, `tests/unit/test_reactive_compact.py`, `tests/unit/test_llm_caller_ptl_retry.py`, `tests/unit/test_log_field_allowlist.py`
- Fixtures: `tests/fixtures/provider_ptl/{anthropic_prompt_too_long,openai_context_length_exceeded,gemini_max_tokens_finish}.json`
