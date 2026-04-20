"""Structured telemetry contract for compression subsystem.

The `CompressionLogFields` TypedDict is the authoritative allowlist of
field names permitted in `logger.*(..., extra={...})` calls emitted from
`dana/core/timeline/` and `dana/common/llm/`. A CI-style AST test walks
these modules and fails when a new field sneaks in; additions require
editing this TypedDict (code review gate).

No field may carry raw prompt content, tool output, or user data —
counts, booleans, enumerated state, IDs only.
"""

from __future__ import annotations

from typing import TypedDict
import uuid


class CompressionLogFields(TypedDict, total=False):
    # Identifiers
    compaction_id: str
    session_id: str
    turn_id: str
    model: str

    # Event / trigger shape
    reason: str
    tokens_est: int
    threshold: int

    # Shrink
    entries_stubbed: int
    tokens_before: int
    tokens_after: int
    below_threshold: bool

    # Compress
    entries_compressed: int
    summary_tokens: int
    entry_ts_range: list

    # Reactive compact
    attempt: int
    attempt_num: int
    entries_dropped: int

    # Circuit breaker
    consecutive_failures: int
    last_attempt: int
    circuit_state: str
    circuit_opened_at: str
    count_last_hour: int


def new_compaction_id() -> str:
    """Return a fresh v4 UUID string for threading through one compaction flow."""
    return str(uuid.uuid4())
