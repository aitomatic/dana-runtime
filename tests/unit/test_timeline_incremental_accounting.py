"""Deterministic regressions for CompressedTimeline token accounting + compaction commits.

Two behaviors are pinned here:

1. **Incremental accounting is exact.** ``get_native_token_estimate()`` (the
   value ``needs_compression()`` compares against the trigger) must equal a
   from-scratch recount of the live native list after every mutation kind:
   append, tool-call append, context swap, timestamp clear, and cheap shrink.
   The append-only turn must not pay for a recount at all.

2. **A compaction commit is exact and validated.** Only the entries the
   summarizer saw are replaced by the summary; entries appended while an awaited
   summarizer ran are retained, and a commit whose projection was replaced
   wholesale (hydration/load) is discarded instead of resurrecting stale history.

3. **Mutable inputs are observed, at the honest cost of a recount.** Whenever
   caller-mutable state is still shared with the model-facing projection — the
   public ``native_messages`` getter, a live entry carrying a multimodal content
   list or tool arguments, or a message object that escaped — the cached total is
   no longer trusted and each estimate recomputes from live content. A plain
   string-content, tool-less append stays fully incremental, and a projection
   rebuilt from decoded journal entries that share nothing mutable restores
   incrementality.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import json

import pytest

from dana.core.timeline import compact_trigger as ct
from dana.core.timeline.compressed_timeline import CompressedTimeline
from dana.core.timeline.compression_engine import SHRINK_STUB_CONTENT
from dana.core.timeline.native_message import COMPRESSED_ENTRIES_COUNT_KEY, NATIVE_SOURCE_ENTRY_KEY, NativeMessage, NativeToolCall
from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType


@pytest.fixture(autouse=True)
def _reset_trigger_cache():
    ct._reset_cache_for_tests()
    yield
    ct._reset_cache_for_tests()


def _make(native_total: int = 10000, keep_recent: int = 3) -> CompressedTimeline:
    return CompressedTimeline(
        max_tokens_until_compression=native_total,
        max_recent_entries_to_keep=keep_recent,
        cutoff_when_token_reach=50,
    )


def _user(content: str, *, ts: datetime | None = None) -> TimelineEntry:
    entry = TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content=content)
    if ts is not None:
        entry.timestamp = ts
    return entry


def _recount(tl: CompressedTimeline) -> int:
    """Independent oracle: a from-scratch recount of the live native list.

    Uses the class implementation directly so it stays a true oracle even in the
    tests that patch the bound method to count invocation.
    """
    return CompressedTimeline._estimate_native_messages_list_tokens(tl, tl._native_messages)


def _fill(tl: CompressedTimeline, n: int, prefix: str = "message") -> None:
    """Append ``n`` plainly-sized user turns (long enough to cross a small trigger)."""
    for index in range(n):
        tl.add_entry(_user(f"{prefix} {index} " + "." * 60))


# ---------------------------------------------------------------------------
# 1. Incremental accounting equals a from-scratch recount after each mutation
# ---------------------------------------------------------------------------


def test_appends_keep_total_equal_to_recount():
    tl = _make()
    for index in range(25):
        tl.add_entry(_user(f"turn {index} " + "x" * 30))
        assert tl.get_native_token_estimate() == _recount(tl)


def test_tool_call_entries_contribute_their_arguments():
    tl = _make()
    tl.add_entry(
        TimelineEntry(
            entry_type=TimelineEntryType.TOOL_CALL,
            content="",
            tool_calls=[{"id": "call_1", "name": "search_assets", "arguments": {"query": "ahu setpoint"}}],
        )
    )
    assert tl.get_native_token_estimate() == _recount(tl)

    # The tool result matching that call must land in the same accounting.
    tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.RESOURCE_RESULT, content="result body", tool_call_id="call_1"))
    assert tl.get_native_token_estimate() == _recount(tl)


def test_context_swap_keeps_total_equal_to_recount():
    tl = _make()
    _fill(tl, 8)
    baseline = tl.get_native_token_estimate()

    tl.set_context({"date": "2026-10-18", "timezone": "UTC", "user": "tester"})
    assert tl.get_native_token_estimate() == _recount(tl)

    # A second swap replaces the first ephemeral context rather than stacking it.
    tl.set_context({"date": "2026-10-19"})
    assert tl.get_native_token_estimate() == _recount(tl)
    assert sum(1 for m in tl.native_messages if m.metadata.get("ephemeral")) == 1
    # The new context is shorter, so the total must have moved down, not up.
    assert tl.get_native_token_estimate() < baseline + 100


def test_clear_old_entries_keeps_total_equal_to_recount():
    tl = _make()
    old = datetime(2026, 1, 1)
    for index in range(10):
        tl.add_entry(_user(f"old {index} " + "y" * 20, ts=old))
    cut = datetime(2026, 6, 1)
    for index in range(4):
        tl.add_entry(_user(f"recent {index} " + "z" * 20, ts=cut + timedelta(days=index)))

    removed = tl.clear_old_entries(cut)

    assert removed == 10
    assert tl.get_native_token_estimate() == _recount(tl)


def test_cheap_shrink_keeps_total_equal_to_recount_and_stays_within_trigger():
    tl = CompressedTimeline(max_tokens_until_compression=500, max_recent_entries_to_keep=5)
    tl._compressed_config.cheap_shrink_keep_recent = 4
    tl._compressed_config.enable_cheap_shrink_tool_results = True

    for index in range(20):
        tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.RESOURCE_RESULT, content="r" * 800, tool_call_id=f"call_{index}"))
    for index in range(4):
        tl.add_entry(_user(f"recent {index}"))

    assert tl.needs_compression()
    assert tl.cheap_shrink_tool_results() is True

    # The stub deltas were folded into the total, so it matches a recount...
    assert tl.get_native_token_estimate() == _recount(tl)
    # ...and the shrink actually dropped the timeline under the trigger.
    assert not tl.needs_compression()
    # All 20 old tool results were stubbed, and the trigger test passes on the
    # accounting total rather than a recount.
    stubbed = [e for e in tl.timeline if e.content == SHRINK_STUB_CONTENT]
    assert len(stubbed) == 20
    assert all(m.content == SHRINK_STUB_CONTENT for m in tl.native_messages if m.role == "tool")


def test_append_path_never_recounts_retained_history():
    """The append-only turn must not rescan history: only hydration recomputes."""
    tl = _make()
    _fill(tl, 12)
    # Warm read: the first estimate after a cold start legitimately recomputes once.
    tl.get_native_token_estimate()

    calls = {"n": 0}
    original = tl._estimate_native_messages_list_tokens

    def counting(messages):
        calls["n"] += 1
        return original(messages)

    tl._estimate_native_messages_list_tokens = counting  # type: ignore[method-assign]

    for index in range(30):
        tl.add_entry(_user(f"more {index} " + "q" * 40))
        tl.get_native_token_estimate()
        tl.needs_compression()

    assert calls["n"] == 0, "append-only turns must not recount the retained history"


def test_public_native_messages_in_place_content_edit_is_observed():
    """The public ``native_messages`` list is mutable; a same-length content edit
    must move the estimate and the consumer's compaction gate."""
    tl = CompressedTimeline(max_tokens_until_compression=100, max_recent_entries_to_keep=1)
    tl.native_messages.extend([NativeMessage(role="user", content="aaaa"), NativeMessage(role="assistant", content="bbbb")])
    before = tl.get_native_token_estimate()

    tl.native_messages[0].content = "a" * 800

    assert tl.get_native_token_estimate() == _recount(tl) == 201
    assert before == 2
    assert tl.needs_compression() is True


def test_public_native_message_content_edit_through_held_reference_is_observed():
    """Same, but the caller keeps the list reference and never re-reads the property."""
    tl = CompressedTimeline(max_tokens_until_compression=100, max_recent_entries_to_keep=1)
    held = tl.native_messages
    held.extend([NativeMessage(role="user", content="xxxx"), NativeMessage(role="assistant", content="yyyy")])
    tl.get_native_token_estimate()

    held[0].content = "x" * 400  # same length as nothing; no getter read in between

    assert tl.get_native_token_estimate() == _recount(tl)
    assert tl.needs_compression() is True


def test_public_native_tool_call_arguments_edit_is_observed():
    tl = CompressedTimeline(max_tokens_until_compression=100, max_recent_entries_to_keep=1)
    messages = tl.native_messages
    messages.append(
        NativeMessage(role="assistant", content="", tool_calls=[NativeToolCall(id="call_1", name="search", arguments={"q": "a"})])
    )
    messages.append(NativeMessage(role="tool", content="result", tool_call_id="call_1"))
    tl.get_native_token_estimate()

    messages[0].tool_calls[0].arguments = {"q": "a" * 600}

    assert tl.get_native_token_estimate() == _recount(tl)
    assert tl.needs_compression() is True


def test_escaped_reference_surviving_a_projection_swap_is_still_observed():
    """An escaped message object survives into the next projection (compaction and
    hydration reuse the same instances), so the untrusted mark must be sticky."""
    tl = CompressedTimeline(max_tokens_until_compression=100, max_recent_entries_to_keep=1)
    tl.native_messages.extend([NativeMessage(role="user", content="aaaa"), NativeMessage(role="assistant", content="bbbb")])
    held = tl.native_messages[0]

    # A projection swap (clear_old_entries rebuilds the list) between escape and edit.
    tl.clear_old_entries(datetime.min)
    before = tl.get_native_token_estimate()

    held.content = "a" * 800

    assert tl.get_native_token_estimate() == _recount(tl) == 201
    assert before == 2
    assert tl.needs_compression() is True


def test_escaped_reference_after_a_compaction_commit_is_still_observed():
    """Compaction retains the pre-existing NativeMessage objects, so a reference
    taken before the commit still aliases live state after it."""
    tl = CompressedTimeline(max_tokens_until_compression=100, max_recent_entries_to_keep=1)
    _fill(tl, 20)
    kept_message = tl.native_messages[-1]

    tl.set_llm_call_fn(lambda prompt: json.dumps({"summary": "compacted"}))
    assert tl.compress() > 0

    kept_message.content = "z" * 800

    assert tl.get_native_token_estimate() == _recount(tl)
    assert tl.needs_compression() is True


def test_caller_held_tool_argument_dict_mutation_is_observed():
    """``add_entry`` shares caller-mutable tool arguments; mutating them in place
    must move the estimate even with no ``native_messages`` read at all."""
    arguments = {"q": "a"}
    tl = CompressedTimeline(max_tokens_until_compression=100, max_recent_entries_to_keep=1)
    tl.add_entry(
        TimelineEntry(
            entry_type=TimelineEntryType.TOOL_CALL,
            content="",
            tool_calls=[{"id": "call_1", "name": "search", "arguments": arguments}],
        )
    )
    tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.RESOURCE_RESULT, content="result", tool_call_id="call_1"))
    before = tl.get_native_token_estimate()

    arguments["q"] = "a" * 600  # the caller's own dict, still shared with the message

    assert tl.get_native_token_estimate() == _recount(tl)
    assert tl.needs_compression() is True
    assert before < tl.get_native_token_estimate()


def test_caller_held_multimodal_content_list_mutation_is_observed():
    blocks = [{"type": "text", "text": "hello"}]
    tl = CompressedTimeline(max_tokens_until_compression=100, max_recent_entries_to_keep=1)
    tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content=blocks))
    tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content="ok"))
    tl.get_native_token_estimate()

    blocks[0]["text"] = "x" * 800

    assert tl.get_native_token_estimate() == _recount(tl)


def test_plain_string_appends_stay_incremental():
    """The ordinary agent loop (immutable string content) must not fall back."""
    tl = CompressedTimeline(max_tokens_until_compression=100000, max_recent_entries_to_keep=2)
    _fill(tl, 5)
    tl.get_native_token_estimate()

    calls = {"n": 0}
    original = tl._estimate_native_messages_list_tokens

    def counting(messages):
        calls["n"] += 1
        return original(messages)

    tl._estimate_native_messages_list_tokens = counting  # type: ignore[method-assign]

    for index in range(30):
        tl.add_entry(_user(f"plain {index} " + "." * 40))
        tl.get_native_token_estimate()
        tl.needs_compression()

    assert calls["n"] == 0
    assert tl._native_aliasing_risk is False


def test_incrementality_resumes_after_loading_decoded_entries():
    """A projection built from decoded dicts is solely owned, so the sticky mark
    must be cleared and the ordinary turn must be incremental again."""
    tl = CompressedTimeline(max_tokens_until_compression=100000, max_recent_entries_to_keep=2)
    _ = tl.native_messages
    assert tl._native_aliasing_risk is True

    tl.load_from_entries([_user(f"decoded {index} " + "." * 60).to_dict() for index in range(12)])

    assert tl._native_aliasing_risk is False
    tl.get_native_token_estimate()

    calls = {"n": 0}
    original = tl._estimate_native_messages_list_tokens

    def counting(messages):
        calls["n"] += 1
        return original(messages)

    tl._estimate_native_messages_list_tokens = counting  # type: ignore[method-assign]

    for index in range(10):
        tl.add_entry(_user(f"post-load {index} " + "." * 60))
        assert tl.get_native_token_estimate() == _recount(tl)

    assert calls["n"] == 0, "a freshly installed, solely-owned projection must be incremental"


def test_loading_live_entry_objects_does_not_claim_sole_ownership():
    """Entries passed as live objects are still shared with the caller: a load of
    those must not be treated as a provably unaliased projection."""
    tl = CompressedTimeline(max_tokens_until_compression=100000, max_recent_entries_to_keep=2)
    _ = tl.native_messages  # mark untrusted

    live = TimelineEntry(entry_type=TimelineEntryType.TOOL_CALL, content="", tool_calls=[{"id": "c", "name": "f", "arguments": {}}])
    tl.load_from_entries([live])

    assert tl._native_aliasing_risk is True


def test_loading_only_immutable_content_entries_clears_the_mark():
    """A projection whose entries carry nothing caller-mutable is unaliased, so the
    mark clears even for a live (non-dict) entry — the predicate is about shared
    mutable state, not about where the entry came from."""
    tl = CompressedTimeline(max_tokens_until_compression=100000, max_recent_entries_to_keep=2)
    _ = tl.native_messages  # mark untrusted

    tl.load_from_entries([_user("plain live entry")])

    assert tl._native_aliasing_risk is False


def test_loading_native_messages_that_share_caller_content_list_stays_untrusted():
    """``native_messages=`` decodes the caller's nested objects by reference, so a
    multimodal content list handed in there is still the caller's."""
    blocks = [{"type": "text", "text": "hello"}]
    tl = CompressedTimeline(max_tokens_until_compression=100000, max_recent_entries_to_keep=2)
    tl.load_from_entries([], native_messages=[{"role": "user", "content": blocks, "timestamp": "2026-10-18T00:00:00"}])

    assert tl._native_aliasing_risk is True

    tl.get_native_token_estimate()
    blocks[0]["text"] = "x" * 800

    assert tl.get_native_token_estimate() == _recount(tl)


def test_loading_native_messages_with_immutable_content_stays_incremental():
    tl = CompressedTimeline(max_tokens_until_compression=100000, max_recent_entries_to_keep=2)
    tl.load_from_entries([], native_messages=[{"role": "user", "content": "plain", "timestamp": "2026-10-18T00:00:00"}])

    assert tl._native_aliasing_risk is False
    assert tl.get_native_token_estimate() == _recount(tl)


def test_externally_rebound_native_list_recomputes_instead_of_returning_stale():
    tl = _make()
    _fill(tl, 5)
    stale = tl.get_native_token_estimate()

    # Deliberate external rebinding — the marker no longer describes the list.
    tl._native_messages = tl._native_messages + [tl._timeline_entry_to_native_message(_user("injected " + "w" * 2000))]

    fresh = tl.get_native_token_estimate()
    assert fresh == _recount(tl)
    assert fresh > stale


def test_hydration_and_native_load_install_coherent_totals():
    tl = _make()
    entries = [_user(f"hydrated {index} " + "h" * 40) for index in range(9)]

    tl.load_from_entries(entries)
    assert tl.get_native_token_estimate() == _recount(tl)

    native = [e.to_dict() for e in entries]
    reloaded = _make()
    reloaded.load_from_entries(native)
    assert reloaded.get_native_token_estimate() == _recount(reloaded)


# ---------------------------------------------------------------------------
# 2. Compaction commits: exact partition, no loss, no stale overwrite
# ---------------------------------------------------------------------------


def test_compress_does_not_drop_entries_appended_while_summarizing(monkeypatch):
    """A sync summarizer that lets another thread append must not lose that turn."""
    tl = CompressedTimeline(max_tokens_until_compression=120, max_recent_entries_to_keep=3, cutoff_when_token_reach=50)
    _fill(tl, 12)

    appended: list[TimelineEntry] = []

    def summarizer(prompt: str) -> str:
        # Stand in for any concurrent mutation during the call.
        fresh = _user("arrived-during-summarizing " + "a" * 60)
        appended.append(fresh)
        tl.add_entry(fresh)
        return json.dumps({"summary": "older turns summarized"})

    tl.set_llm_call_fn(summarizer)

    assert tl.compress() > 0

    assert appended[0] in tl.timeline, "entries appended while summarizing must be retained"
    assert any(m.content.startswith("arrived-during-summarizing") for m in tl.native_messages)
    assert tl.get_native_token_estimate() == _recount(tl)


def test_async_compress_retains_entries_appended_during_await():
    tl = CompressedTimeline(max_tokens_until_compression=120, max_recent_entries_to_keep=3, cutoff_when_token_reach=50)
    _fill(tl, 12)

    release = asyncio.Event()
    entered = asyncio.Event()

    async def slow_summarizer(prompt: str) -> str:
        entered.set()
        await release.wait()
        return json.dumps({"summary": "summarized before the appended turns"})

    async def scenario() -> int:
        tl.set_llm_call_async_fn(slow_summarizer)
        task = asyncio.ensure_future(tl.compress_async())
        await entered.wait()

        # Interleave a real user turn plus its answer while the summarizer awaits.
        tl.add_entry(_user("interleaved-question " + "i" * 50))
        tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content="interleaved-answer"))

        release.set()
        return await task

    compressed = asyncio.run(scenario())

    assert compressed > 0
    contents = [e.content for e in tl.timeline]
    assert "interleaved-question " + "i" * 50 in contents
    assert "interleaved-answer" in contents
    assert tl.get_native_token_estimate() == _recount(tl)

    # The summary lands on the oldest *retained* entry and the provider sees it.
    assert tl.has_compressed_context()
    assert any(m.content.startswith("[SUMMARY]") for m in tl.to_llm_messages())


def test_async_compress_discards_commit_when_projection_was_replaced():
    """A hydration/load that lands during the await must not be overwritten."""
    tl = CompressedTimeline(max_tokens_until_compression=120, max_recent_entries_to_keep=3, cutoff_when_token_reach=50)
    _fill(tl, 12)

    release = asyncio.Event()
    entered = asyncio.Event()

    async def slow_summarizer(prompt: str) -> str:
        entered.set()
        await release.wait()
        return json.dumps({"summary": "summary of a projection that no longer exists"})

    async def scenario() -> int:
        tl.set_llm_call_async_fn(slow_summarizer)
        task = asyncio.ensure_future(tl.compress_async())
        await entered.wait()

        # Wholesale projection replacement, e.g. journal restore into this instance.
        tl.load_from_entries([_user("restored-1"), _user("restored-2")])

        release.set()
        return await task

    compressed = asyncio.run(scenario())

    assert compressed == 0, "a commit planned against a replaced projection must be discarded"
    assert [e.content for e in tl.timeline] == ["restored-1", "restored-2"]
    assert [m.content for m in tl.native_messages] == ["restored-1", "restored-2"]
    assert not tl.has_compressed_context()
    assert tl.get_native_token_estimate() == _recount(tl)


def test_compress_old_entries_commits_the_partition_the_prompt_described():
    """prompt → host LLM → commit must not summarize entries the prompt never had."""
    tl = CompressedTimeline(max_tokens_until_compression=120, max_recent_entries_to_keep=3, cutoff_when_token_reach=50)
    _fill(tl, 12)

    prompt = tl.build_compression_prompt()
    assert prompt is not None
    planned = [e.content for e in tl.timeline if e.content in prompt]
    assert planned, "the prompt must quote the entries it will replace"

    # A turn that arrives between prompt construction and commit.
    late = _user("late-arrival " + "l" * 60)
    tl.add_entry(late)

    assert tl.compress_old_entries("committed summary") > 0

    assert late in tl.timeline, "the late entry was never in the summary and must survive"


def test_compress_old_entries_without_prompt_uses_current_partition():
    tl = CompressedTimeline(max_tokens_until_compression=120, max_recent_entries_to_keep=3, cutoff_when_token_reach=50)
    _fill(tl, 12)

    compressed = tl.compress_old_entries("direct-path summary")

    assert compressed > 0
    assert tl.has_compressed_context()
    assert tl.get_native_token_estimate() == _recount(tl)


def test_compaction_after_a_prior_summary_keeps_entry_native_alignment():
    """After a compaction the native list carries a summary message, so the two
    lists are legitimately unequal in length. A second commit that retains an
    entry inserted at the front (``set_context``) must map entries to their own
    mirrors by identity — not by tail position — or it resurrects a
    compressed-away message and drops the retained context."""
    tl = CompressedTimeline(max_tokens_until_compression=1000, max_recent_entries_to_keep=2, cutoff_when_token_reach=1000)
    for index in range(6):
        tl.add_entry(_user(f"first-{index} " + "x" * 1500))
    tl.set_llm_call_fn(lambda prompt: json.dumps({"summary": "first summary"}))
    assert tl.compress() > 0

    for index in range(6):
        tl.add_entry(_user(f"second-{index} " + "y" * 1500))
    assert tl.build_compression_prompt() is not None

    # A fresh runtime context arrives between prompt construction and commit.
    tl.set_context({"user": "newcontext"})

    assert tl.compress_old_entries("second summary") > 0

    # The retained context keeps its mirror; no compressed-away history leaks back.
    assert any("newcontext" in str(message.content) for message in tl.native_messages)
    assert not any(str(message.content).startswith("first-") for message in tl.native_messages)
    assert [str(message.content) for message in tl.native_messages if message.role != "system"] == [
        str(entry.content) for entry in tl.timeline if entry.entry_type != TimelineEntryType.CONTEXT
    ]
    assert tl.get_native_token_estimate() == _recount(tl)


def test_compaction_keeps_a_retained_entries_reasoning_metadata():
    """A retained entry's mirror keeps its own replay metadata, so the identity
    mapping may not substitute a different message."""
    tl = CompressedTimeline(max_tokens_until_compression=1000, max_recent_entries_to_keep=2, cutoff_when_token_reach=1000)
    for index in range(5):
        tl.add_entry(_user(f"old-{index} " + "x" * 1500))
    tl.add_entry(
        TimelineEntry(
            entry_type=TimelineEntryType.AGENT_THOUGHTS,
            content="kept thought",
            metadata={"reasoning_items": [{"id": "rs_1"}], "fingerprint": "fp-1", "response_id": "resp-1"},
        )
    )

    assert tl.compress_old_entries("summary") > 0

    kept = [message for message in tl.native_messages if message.metadata.get("reasoning_items")]
    assert len(kept) == 1
    assert kept[0].metadata["response_id"] == "resp-1"
    assert kept[0].metadata["reasoning_items"] == [{"id": "rs_1"}]


def test_reactive_compact_mirrors_by_identity_not_tail_slice():
    """`reactive_compact` must mirror the entries it actually kept — including the
    backward expansion of ``_ensure_tool_pair_integrity`` — rather than slicing a
    same-length tail off the native list, which is wrong whenever a prior
    compaction has prepended a summary message."""
    tl = CompressedTimeline(max_tokens_until_compression=1000, max_recent_entries_to_keep=2, cutoff_when_token_reach=1000)
    tl.add_entry(_user("u-0 " + "q" * 1200))
    tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.TOOL_CALL, content="", tool_calls=[{"id": "c1", "name": "f", "arguments": {}}]))
    tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.RESOURCE_RESULT, content="res-1", tool_call_id="c1"))
    for index in range(12):
        tl.add_entry(_user(f"w-{index} " + "w" * 400))

    tl.set_llm_call_fn(lambda prompt: json.dumps({"summary": "prior summary"}))
    assert tl.compress() > 0
    assert tl.has_compressed_context()

    for index in range(20):
        tl.add_entry(_user(f"x-{index} " + "x" * 400))

    tl.set_llm_call_fn(lambda prompt: json.dumps({"summary": "reactive summary"}))
    tl.reactive_compact(1)

    assert [str(message.content) for message in tl.native_messages if message.role != "system"] == [
        str(entry.content) for entry in tl.timeline if entry.entry_type != TimelineEntryType.CONTEXT
    ]
    assert tl.get_native_token_estimate() == _recount(tl)
    assert tl.has_compressed_context()
    # No orphaned tool result: the pair the integrity pass re-expanded is mirrored.
    live_tool_use_ids = {call.id for entry in tl.timeline if entry.tool_calls for call in entry.tool_calls}
    assert all(entry.tool_call_id in live_tool_use_ids for entry in tl.timeline if entry.tool_call_id)


def test_repeated_compaction_keeps_exactly_one_summary_message():
    """Each commit replaces the previous summary; it must not accumulate."""
    tl = CompressedTimeline(max_tokens_until_compression=1000, max_recent_entries_to_keep=2, cutoff_when_token_reach=1000)
    for index in range(6):
        tl.add_entry(_user(f"u-{index} " + "q" * 1500))

    tl.set_llm_call_fn(lambda prompt: json.dumps({"summary": "first"}))
    assert tl.compress() > 0
    for index in range(6):
        tl.add_entry(_user(f"v-{index} " + "v" * 1500))

    tl.set_llm_call_fn(lambda prompt: json.dumps({"summary": "second"}))
    tl.reactive_compact(1)

    summaries = [message for message in tl.native_messages if str(message.content).startswith("[SUMMARY]")]
    assert len(summaries) == 1
    assert summaries[0].content == "[SUMMARY] second"
    assert [str(message.content) for message in tl.native_messages if message.role != "system"] == [
        str(entry.content) for entry in tl.timeline
    ]
    assert tl.get_native_token_estimate() == _recount(tl)


def test_snapshot_round_trip_neither_persists_nor_keeps_the_internal_backlink():
    tl = CompressedTimeline(max_tokens_until_compression=100000, max_recent_entries_to_keep=2, cutoff_when_token_reach=100000)
    _fill(tl, 4, prefix="round")

    serialized = [message.to_dict() for message in tl.native_messages]

    assert all(NATIVE_SOURCE_ENTRY_KEY not in payload.get("metadata", {}) for payload in serialized)
    assert json.dumps(serialized), "the internal backlink must not break JSON persistence"

    reopened = CompressedTimeline(max_tokens_until_compression=100000, max_recent_entries_to_keep=2, cutoff_when_token_reach=100000)
    reopened.load_from_entries([], native_messages=serialized)

    assert [str(message.content) for message in reopened.native_messages] == [str(message.content) for message in tl.native_messages]
    assert all(NATIVE_SOURCE_ENTRY_KEY not in entry.metadata for entry in reopened.timeline)
    assert reopened.get_native_token_estimate() == _recount(reopened)


def test_compaction_after_decoded_native_load_keeps_the_loaded_messages():
    """A decoded native load must tag its messages as the mirrors of the entries it
    reconstructed, or the next compaction rebuilds fresh projections and drops the
    edits, replay metadata and object identity a caller holds."""
    tl = CompressedTimeline(max_tokens_until_compression=1000, max_recent_entries_to_keep=3, cutoff_when_token_reach=1000)
    tl.load_from_entries(
        [
            {"role": "user", "content": "oldest-plain", "timestamp": "2026-10-18T00:00:00"},
            {
                "role": "assistant",
                "content": "think",
                "timestamp": "2026-10-18T00:00:01",
                "metadata": {"reasoning_items": [{"id": "rs"}], "fingerprint": "fp", "response_id": "original"},
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "f", "arguments": {"a": 1}}],
                "timestamp": "2026-10-18T00:00:02",
            },
            {"role": "tool", "content": "res", "tool_call_id": "c1", "timestamp": "2026-10-18T00:00:03"},
        ]
    )

    held = [message for message in tl.native_messages if message.metadata.get("reasoning_items")][0]
    held.content = "edited think"
    held.metadata["response_id"] = "edited"

    # Commits a real partition: only the oldest entry is compressed away, so the
    # reasoning, tool-call and tool-result messages stay in the retained window.
    tl.set_llm_call_fn(lambda prompt: json.dumps({"summary": "summary"}))
    assert tl.compress_old_entries("summary") > 0

    kept_thought = [message for message in tl.native_messages if message.metadata.get("reasoning_items")]
    assert len(kept_thought) == 1
    assert kept_thought[0] is held, "the decoded message object must survive compaction"
    assert kept_thought[0].content == "edited think"
    assert kept_thought[0].metadata["response_id"] == "edited"
    # Tool call / result pairing and arguments survive as their own objects.
    tool_call = [message for message in tl.native_messages if message.tool_calls][0]
    tool_result = [message for message in tl.native_messages if message.role == "tool"][0]
    assert tool_call.tool_calls[0].arguments == {"a": 1}
    assert tool_result.tool_call_id == "c1" and tool_result.content == "res"
    assert not any(str(message.content) == "oldest-plain" for message in tl.native_messages)


def test_retention_flags_are_occurrence_aware_for_duplicate_entries():
    """The same entry object live twice (a caller re-adds it) must not have its
    earlier occurrence marked retained, or that occurrence consumes the mirror of
    the occurrence that is actually kept."""
    duplicate = _user("duplicate")
    live = [duplicate, _user("a"), _user("b"), duplicate]
    retained = [live[2], live[3]]

    flags = CompressedTimeline._suffix_retention_flags(live, retained)

    assert flags == [False, False, True, True]
    assert [entry for entry, keep in zip(live, flags, strict=True) if keep] == retained


def test_suffix_retention_flags_mark_every_retained_occurrence():
    duplicate = _user("duplicate")
    live = [duplicate, _user("a"), duplicate]

    flags = CompressedTimeline._suffix_retention_flags(live, [duplicate, duplicate])

    assert flags == [True, False, True]


def test_compaction_after_separate_native_storage_load_keeps_the_supplied_messages():
    """The separate-storage leg receives the compact snapshot shape
    ``messages=[summary, retained...]`` against ``entries=[retained...]``: the
    extra leading summary must not shift the pairing, or every retained message is
    left untagged and the next compaction discards the caller's edits."""
    earlier = "2026-10-17T00:00:00"
    entries = [_user(f"sep-{index}") for index in range(4)]
    entries[0].metadata = {"compressed_context": "earlier", "compression_timestamp": earlier}
    native_messages = [
        {
            "role": "system",
            "content": "[SUMMARY] earlier",
            "timestamp": earlier,
            "metadata": {"compressed_context": "earlier", "compression_timestamp": earlier},
        },
        {
            "role": "user",
            "content": "sep-0",
            "timestamp": earlier,
            "metadata": {"compressed_context": "earlier", "compression_timestamp": earlier},
        },
        *[{"role": "user", "content": f"sep-{index}", "timestamp": f"2026-10-18T00:00:0{index}"} for index in range(1, 4)],
    ]
    tl = CompressedTimeline(max_tokens_until_compression=1000, max_recent_entries_to_keep=3, cutoff_when_token_reach=1000)
    tl.load_from_entries(entries, native_messages=native_messages)

    held = tl.native_messages[-1]
    held.content = "edited separate projection"
    held.metadata["fingerprint"] = "fp-x"

    tl.set_llm_call_fn(lambda prompt: json.dumps({"summary": "summary"}))
    tl.compress_old_entries("summary")

    kept = [message for message in tl.native_messages if str(message.content) == "edited separate projection"]
    assert len(kept) == 1, "the retained message object must survive compaction"
    assert kept[0] is held
    assert kept[0].metadata["fingerprint"] == "fp-x"


def test_compaction_consumes_a_mirror_for_every_live_occurrence():
    """The same entry object can be live twice (a caller re-adds it during the
    summarizer). Both occurrences consume their own mirror in order, so the
    retained later occurrence never receives the compressed-away one's message."""
    tl = CompressedTimeline(max_tokens_until_compression=100000, max_recent_entries_to_keep=1, cutoff_when_token_reach=100000)
    duplicate = _user("before")
    tl.add_entry(duplicate)
    tl.add_entry(_user("keep-me " + "k" * 1500))

    assert tl.build_compression_prompt() is not None

    # The entry is re-added during the summarizer with new content: two live
    # occurrences of one object, the first of which is being compressed away.
    duplicate.content = "late duplicate"
    tl.add_entry(duplicate)

    assert tl.compress_old_entries("summary") > 0

    assert [str(entry.content) for entry in tl.timeline] == ["keep-me " + "k" * 1500, "late duplicate"]
    assert [str(message.content) for message in tl.native_messages if message.role != "system"] == [
        "keep-me " + "k" * 1500,
        "late duplicate",
    ], "the retained occurrence must not inherit the compressed-away occurrence's mirror"
    assert not any(str(message.content) == "before" for message in tl.native_messages)
    assert tl.get_native_token_estimate() == _recount(tl)


def test_compaction_metadata_matches_the_retained_and_removed_counts():
    tl = CompressedTimeline(max_tokens_until_compression=120, max_recent_entries_to_keep=3, cutoff_when_token_reach=50)
    _fill(tl, 14)
    before = len(tl.timeline)

    compressed = tl.compress_old_entries("accounted summary")

    assert 0 < compressed < before
    stamped = tl.timeline[0]
    assert stamped.metadata[COMPRESSED_ENTRIES_COUNT_KEY] == compressed
    assert len(tl.timeline) == before - compressed
    # One summary message + one mirror per retained entry.
    assert len(tl.native_messages) == len(tl.timeline) + 1
