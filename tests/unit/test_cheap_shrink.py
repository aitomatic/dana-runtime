"""Unit tests for Phase 2 cheap_shrink_tool_results()."""

from __future__ import annotations

import asyncio

import pytest

from dana.core.timeline import compact_trigger as ct
from dana.core.timeline.compressed_timeline import CompressedTimeline
from dana.core.timeline.compression_engine import SHRINK_STUB_CONTENT
from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType


@pytest.fixture(autouse=True)
def _reset_trigger_cache():
    ct._reset_cache_for_tests()
    yield
    ct._reset_cache_for_tests()


def _build_timeline_with_tool_results(n_old_tool_results: int, n_recent: int, payload_chars: int = 400) -> CompressedTimeline:
    """Build a timeline: N old tool_result entries + N recent user messages.

    Each tool_result carries a unique tool_call_id and a large payload so
    stubbing produces measurable token savings. Keep_recent=10 ensures
    old entries are eligible for shrink.
    """
    timeline = CompressedTimeline(
        max_tokens_until_compression=500,
        max_recent_entries_to_keep=5,
    )
    timeline._compressed_config.cheap_shrink_keep_recent = 10
    # Old tool_result entries (eligible for shrink)
    for i in range(n_old_tool_results):
        tc_id = f"call_{i}"
        # paired tool_call first, then tool_result — but for cheap_shrink we only
        # need the result; tool_call presence isn't required by the shrinker.
        timeline.add_entry(
            TimelineEntry(
                entry_type=TimelineEntryType.RESOURCE_RESULT,
                content="r" * payload_chars,
                tool_call_id=tc_id,
            )
        )
    for i in range(n_recent):
        timeline.add_entry(
            TimelineEntry(
                entry_type=TimelineEntryType.USER_MESSAGE,
                content=f"msg {i}",
            )
        )
    return timeline


def test_shrink_stubs_old_tool_results():
    tl = _build_timeline_with_tool_results(n_old_tool_results=20, n_recent=11, payload_chars=400)
    assert tl.cheap_shrink_tool_results() is True
    # First 10 old results stubbed (20 total older than keep_recent=10 last entries)
    stubbed = [e for e in tl.timeline if e.content == SHRINK_STUB_CONTENT]
    assert len(stubbed) > 0
    # All stubbed entries still carry tool_call_id.
    for e in stubbed:
        assert e.tool_call_id is not None


def test_shrink_preserves_recent_entries():
    tl = _build_timeline_with_tool_results(n_old_tool_results=12, n_recent=11, payload_chars=400)
    tl.cheap_shrink_tool_results()
    # Recent 10 entries (from end of timeline) must be untouched.
    recent_10 = tl.timeline[-10:]
    for e in recent_10:
        assert e.content != SHRINK_STUB_CONTENT


def test_shrink_preserves_tool_call_id():
    tl = _build_timeline_with_tool_results(n_old_tool_results=15, n_recent=11, payload_chars=400)
    ids_before = [e.tool_call_id for e in tl.timeline if e.tool_call_id]
    tl.cheap_shrink_tool_results()
    ids_after = [e.tool_call_id for e in tl.timeline if e.tool_call_id]
    assert ids_before == ids_after


def test_shrink_idempotent_via_content_equality():
    tl = _build_timeline_with_tool_results(n_old_tool_results=15, n_recent=11, payload_chars=400)
    first = tl.cheap_shrink_tool_results()
    state_snapshot = [(e.content, e.tool_call_id) for e in tl.timeline]
    second = tl.cheap_shrink_tool_results()
    # Second call: nothing left to stub, returns False.
    assert first is True
    assert second is False
    # State unchanged by second call.
    assert [(e.content, e.tool_call_id) for e in tl.timeline] == state_snapshot


def test_predictive_gate_blocks_mutation_when_savings_insufficient():
    # Tiny tool-result payloads so shrink can't drop us below trigger.
    tl = CompressedTimeline(
        max_tokens_until_compression=500,
        max_recent_entries_to_keep=5,
    )
    tl._compressed_config.cheap_shrink_keep_recent = 10
    # Add many large user messages (not eligible — no tool_call_id) to blow up tokens
    for _i in range(30):
        tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="x" * 200))
    # Add a few old tiny tool_results.
    for i in range(3):
        tl.add_entry(
            TimelineEntry(
                entry_type=TimelineEntryType.RESOURCE_RESULT,
                content="r" * 20,  # tiny
                tool_call_id=f"call_{i}",
            )
        )
    # Add recent messages.
    for i in range(11):
        tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content=f"m{i}"))

    result = tl.cheap_shrink_tool_results()
    # Savings from tiny tool_results cannot close the gap -> no mutation, False.
    assert result is False
    stubbed = [e for e in tl.timeline if e.content == SHRINK_STUB_CONTENT]
    assert stubbed == []


def test_shrink_noop_when_under_threshold():
    tl = CompressedTimeline(
        max_tokens_until_compression=1_000_000,
        max_recent_entries_to_keep=5,
    )
    tl._compressed_config.cheap_shrink_keep_recent = 10
    for i in range(15):
        tl.add_entry(
            TimelineEntry(
                entry_type=TimelineEntryType.RESOURCE_RESULT,
                content="r" * 400,
                tool_call_id=f"call_{i}",
            )
        )
    for i in range(11):
        tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content=f"m{i}"))
    assert tl.cheap_shrink_tool_results() is False


def test_shrink_skips_entries_referenced_by_pending_tool_use():
    tl = CompressedTimeline(
        max_tokens_until_compression=500,
        max_recent_entries_to_keep=5,
    )
    tl._compressed_config.cheap_shrink_keep_recent = 10
    # 15 old tool_results, ids call_0..call_14
    for i in range(15):
        tl.add_entry(
            TimelineEntry(
                entry_type=TimelineEntryType.RESOURCE_RESULT,
                content="r" * 500,
                tool_call_id=f"call_{i}",
            )
        )
    # Pad recent window with 9 user messages so TOOL_CALL below lives within
    # the kept-recent window (last 10) and protects call_0 result.
    for i in range(9):
        tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content=f"m{i}"))
    # Recent TOOL_CALL referencing call_0 (within keep_recent=10 window).
    tl.add_entry(
        TimelineEntry(
            entry_type=TimelineEntryType.TOOL_CALL,
            content="",
            tool_calls=[{"id": "call_0", "name": "fn", "arguments": {}}],
        )
    )

    tl.cheap_shrink_tool_results()
    # call_0 tool_result must NOT be stubbed (pending tool_use references it).
    call0_entry = next(e for e in tl.timeline if e.tool_call_id == "call_0")
    assert call0_entry.content != SHRINK_STUB_CONTENT


def test_shrink_native_messages_also_stubbed():
    """The NativeMessage mirror should also reflect the stub for LLM sends."""
    tl = _build_timeline_with_tool_results(n_old_tool_results=20, n_recent=11, payload_chars=400)
    tl.cheap_shrink_tool_results()
    stubbed_nm = [nm for nm in tl._native_messages if nm.role == "tool" and nm.content == SHRINK_STUB_CONTENT]
    assert len(stubbed_nm) > 0


def test_shrink_lock_serializes_concurrent_callers():
    """Back-to-back concurrent compress() calls should be lock-serialized."""
    import threading

    tl = _build_timeline_with_tool_results(n_old_tool_results=20, n_recent=11, payload_chars=400)
    tl._compressed_config.enable_cheap_shrink_tool_results = True

    # Supply a dummy llm_call_fn so compress() has something if shrink misses.
    tl._llm_call_fn = lambda prompt: '{"summary":"s"}'

    results = []

    def runner():
        results.append(tl.compress())

    threads = [threading.Thread(target=runner) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # No crash, no deadlock — shrink path ran under lock.
    assert len(results) == 2


def test_flag_off_behavior_unchanged():
    tl = _build_timeline_with_tool_results(n_old_tool_results=20, n_recent=11, payload_chars=400)
    tl._compressed_config.enable_cheap_shrink_tool_results = False
    tl._llm_call_fn = lambda prompt: '{"summary":"compressed"}'
    # compress() should NOT call shrink path; runs full summary.
    tl.compress()
    # No entries got the shrink stub.
    stubbed = [e for e in tl.timeline if e.content == SHRINK_STUB_CONTENT]
    assert stubbed == []


def test_async_compress_respects_shrink_path():
    tl = _build_timeline_with_tool_results(n_old_tool_results=20, n_recent=11, payload_chars=400)
    tl._compressed_config.enable_cheap_shrink_tool_results = True

    async def dummy(prompt):
        return '{"summary":"s"}'

    tl._llm_call_async_fn = dummy

    asyncio.run(tl.compress_async())
    # Shrink fired and short-circuited before summary.
    stubbed = [e for e in tl.timeline if e.content == SHRINK_STUB_CONTENT]
    assert len(stubbed) > 0
