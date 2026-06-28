# Project Changelog

## [Unreleased]

### Added
- **I/O Security Guard** — LLM-Guard integration for input sanitization & output scrubbing (`dana/core/guard/`).
  - GuardService protocol with pluggable implementations (mirrors LLMProvider pattern); injectable via `STARAgent(guard_instance=...)`.
  - Input: block-listed scanners (default `prompt_injection`) trip → request refused (STAR loop skipped); other findings (secrets/PII) sanitized + passed.
  - Output: gated two-stage scrub — rule scanners strip/redact; LLM scrub fires only when a scanner flags (clean output skips the LLM call).
  - Audit via structlog events + timeline metadata; persisted AGENT_RESPONSE content overwritten with scrubbed text (no raw data on disk / replay to LLM).
  - Fail-open: missing/broken models → no-op + warning, never breaks execution. Thread-safe lazy scanner init.
  - Pluggable scanners via registry (register_input_scanner/register_output_scanner).
  - Configuration: `DANA_GUARD_ENABLED`, `DANA_GUARD_INPUT_SCANNERS`, `DANA_GUARD_OUTPUT_SCANNERS`, `DANA_GUARD_SANITIZE_LLM_ENABLED`, `DANA_GUARD_FAIL_MODE`, `DANA_GUARD_BLOCK_ON`, `DANA_GUARD_BLOCK_MESSAGE` env vars.
  - Integrated into STARAgent.query/aquery at single choke point. llm-guard added as core dependency.
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
- `CompressedTimeline.__init__` default `max_tokens_until_compression` now defers to env trigger (150000) when unset. Explicit value continues to win.
- `CompressedTimelineConfig` gains `enable_cheap_shrink_tool_results`, `cheap_shrink_keep_recent`, `enable_reactive_compact` fields.
- `star_agent._maybe_compress_timeline` (sync + async) re-raises `PromptTooLongError` from summary path instead of swallowing — lets caller-layer retry kick in.

### Files
- New: `dana/core/timeline/compact_trigger.py`, `dana/core/timeline/telemetry.py`
- Modified: `dana/core/timeline/compression_engine.py`, `dana/core/timeline/compressed_timeline.py`, `dana/core/agent/star_agent.py`, `dana/core/llm/llm_caller.py`, `dana/common/llm/types.py`, `dana/common/llm/providers/{anthropic,openai_compatible_base,gemini}.py`
- Tests: `tests/unit/test_compact_trigger.py`, `tests/unit/test_compressed_timeline_callbacks.py`, `tests/unit/test_cheap_shrink.py`, `tests/unit/test_reactive_compact.py`, `tests/unit/test_llm_caller_ptl_retry.py`, `tests/unit/test_log_field_allowlist.py`
- Fixtures: `tests/fixtures/provider_ptl/{anthropic_prompt_too_long,openai_context_length_exceeded,gemini_max_tokens_finish}.json`
