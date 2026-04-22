"""Unit tests for Phase 1 additions — system/tools callbacks and env trigger."""

from __future__ import annotations

import pytest

from dana.core.timeline import compact_trigger as ct
from dana.core.timeline.compressed_timeline import CompressedTimeline
from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType


@pytest.fixture(autouse=True)
def _reset_trigger_cache():
    ct._reset_cache_for_tests()
    yield
    ct._reset_cache_for_tests()


def _fill_messages(timeline: CompressedTimeline, n: int, per_msg_chars: int) -> None:
    for _ in range(n):
        timeline.add_entry(
            TimelineEntry(
                entry_type=TimelineEntryType.USER_MESSAGE,
                content=("x" * per_msg_chars),
            )
        )
    # Bypass recent-entries floor by ensuring we exceed max_recent_entries_to_keep
    assert len(timeline._native_messages) > timeline.max_recent_entries_to_keep


def test_callbacks_none_fallback_to_messages_only():
    timeline = CompressedTimeline(max_tokens_until_compression=1000, max_recent_entries_to_keep=2)
    # Add messages below threshold
    _fill_messages(timeline, n=5, per_msg_chars=100)  # ~125 tokens
    assert timeline.needs_compression() is False


def test_callbacks_increase_estimate_triggers_compression():
    timeline = CompressedTimeline(
        max_tokens_until_compression=1000,
        max_recent_entries_to_keep=2,
        system_tokens_fn=lambda: 800,
        tools_tokens_fn=lambda: 200,
    )
    # Messages alone ~125 tokens; callbacks add 1000 -> crosses threshold
    _fill_messages(timeline, n=5, per_msg_chars=100)
    assert timeline.needs_compression() is True


def test_callback_raises_treated_as_zero():
    def raiser() -> int:
        raise RuntimeError("boom")

    timeline = CompressedTimeline(
        max_tokens_until_compression=1000,
        max_recent_entries_to_keep=2,
        system_tokens_fn=raiser,
        tools_tokens_fn=lambda: 10,
    )
    _fill_messages(timeline, n=3, per_msg_chars=100)
    # Should not raise; raising callback contributes 0
    assert timeline.needs_compression() is False


def test_callback_negative_treated_as_zero():
    timeline = CompressedTimeline(
        max_tokens_until_compression=1000,
        max_recent_entries_to_keep=2,
        system_tokens_fn=lambda: -500,
    )
    _fill_messages(timeline, n=3, per_msg_chars=100)
    assert timeline.needs_compression() is False


def test_explicit_trigger_takes_precedence(monkeypatch):
    monkeypatch.setenv("DANA_COMPACT_TRIGGER_TOKENS", "100000")
    timeline = CompressedTimeline(max_tokens_until_compression=1000, max_recent_entries_to_keep=2)
    _fill_messages(timeline, n=5, per_msg_chars=2000)  # ~2500 tokens messages
    assert timeline.needs_compression() is True


def test_env_trigger_applied_when_no_explicit(monkeypatch):
    monkeypatch.setenv("DANA_COMPACT_TRIGGER_TOKENS", "100000")
    timeline = CompressedTimeline(max_tokens_until_compression=None, max_recent_entries_to_keep=2)
    _fill_messages(timeline, n=5, per_msg_chars=2000)  # ~2500 tokens, below 100k env
    assert timeline.needs_compression() is False


def test_needs_compression_true_when_sum_at_threshold():
    timeline = CompressedTimeline(
        max_tokens_until_compression=1000,
        max_recent_entries_to_keep=2,
        system_tokens_fn=lambda: 1000,
    )
    _fill_messages(timeline, n=3, per_msg_chars=20)  # tiny messages
    # system_tokens_fn contributes 1000 which alone meets threshold (>=)
    assert timeline.needs_compression() is True
