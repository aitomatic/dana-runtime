# Project Changelog

## [Unreleased]

### Added
- LangSmith as an alternative tracing backend. `@observable` (`dana/common/observable.py`) dispatches to `langsmith.traceable` when `LANGSMITH_TRACING=true` or `DANA_LANGSMITH_ENABLED` truthy; exclusive with Langfuse (LangSmith takes precedence). No call-site changes — all 30+ `@observable` sites traced automatically. Add via `pip install dana[observability]`. LangSmith API key: `LANGSMITH_API_KEY`.
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
