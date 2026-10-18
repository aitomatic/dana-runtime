"""Ownership tests for reactive compaction's two seams.

The async seam (``CompressedTimeline.reactive_compact_async``) exists because
``asyncio.to_thread(timeline.reactive_compact, ...)`` cannot be stopped: an
awaiting caller that hits its logical-request deadline (or is cancelled) drops
the await while the worker keeps running, and that worker later installs a whole
new projection and stamps the compaction marker on a timeline the caller has
already released. These tests pin the replacement contract:

* no mutation of the timeline happens on any thread other than the awaiting one;
* an abandoned run commits nothing and opens no circuit;
* unavoidable blocking sync summarizers are joined, not orphaned, so the run's
  own work is finished before the cancellation propagates (the deadline may
  overrun by that call's remaining duration - recorded here, not hidden);
* the compaction lock is released even when the run is cancelled.

The sync seam is unchanged and must keep committing fully; a parity test guards
the shared internals the async seam now reuses.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import threading

import pytest

from dana.common.llm.types import CompactCircuitOpenError, LLMResponse
from dana.core.timeline import compact_trigger as ct
from dana.core.timeline.compressed_timeline import CompressedTimeline
from dana.core.timeline.timeline import TimelineEntry, TimelineEntryType


@pytest.fixture(autouse=True)
def _reset_trigger_cache():
    ct._reset_cache_for_tests()
    yield
    ct._reset_cache_for_tests()


def _build(n_entries: int = 60) -> CompressedTimeline:
    tl = CompressedTimeline(
        max_tokens_until_compression=500,
        max_recent_entries_to_keep=5,
    )
    for i in range(n_entries):
        tl.add_entry(TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content=f"message {i}" * 20))
    return tl


class _ProjectionRecorder:
    """Records every projection install with the thread that performed it."""

    def __init__(self, timeline: CompressedTimeline) -> None:
        self.timeline = timeline
        self.installs: list[tuple[int, int, int]] = []
        self.commits: list[int] = []
        self._original_install = timeline._install_model_state
        self._original_commit = timeline._apply_compression
        timeline._install_model_state = self._install  # type: ignore[method-assign]
        timeline._apply_compression = self._commit  # type: ignore[method-assign]

    def _install(self, entries, messages):
        self.installs.append((len(entries), len(messages), threading.get_ident()))
        return self._original_install(entries, messages)

    def _commit(self, *args, **kwargs):
        self.commits.append(threading.get_ident())
        return self._original_commit(*args, **kwargs)


@pytest.mark.asyncio
async def test_async_provider_cancelled_leaves_no_late_write():
    """A cancelled await must not mutate the timeline afterwards.

    The async summarizer is awaited in place, so owner cancellation reaches it:
    nothing is running when this task stops, and no commit is attempted.
    """
    tl = _build()
    recorder = _ProjectionRecorder(tl)
    loop_thread = threading.get_ident()
    provider_cancelled = asyncio.Event()
    started = asyncio.Event()

    async def summarizer(prompt: str) -> str:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            provider_cancelled.set()
            raise
        return '{"summary": "never"}'

    tl._llm_call_async_fn = summarizer

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(tl.reactive_compact_async(1), timeout=0.2)

    assert provider_cancelled.is_set(), "cancellation must reach the awaited provider"
    assert all(thread == loop_thread for _e, _m, thread in recorder.installs), "only the awaiting thread may install"
    assert recorder.commits == [], "an abandoned run must not commit"
    assert tl._consecutive_compact_failures == 0, "owner cancellation is not a compaction failure"
    assert tl._compaction_disabled is False, "owner cancellation must not open the circuit"
    assert tl.get_compressed_context() is None

    # Honest partial semantics: the bounded drop prefix is applied by the
    # awaiting thread before the summarizer runs, and the summary is not.
    await asyncio.sleep(0)
    assert len(tl.timeline) == 55, "the drop prefix the owner thread applied stays applied"


@pytest.mark.asyncio
async def test_async_provider_is_awaited_in_place_not_offloaded():
    """Cancellation can only reach the provider if the await is in place."""
    tl = _build()
    loop_thread = threading.get_ident()
    seen: list[int] = []

    async def summarizer(prompt: str) -> str:
        seen.append(threading.get_ident())
        return '{"summary": "in place"}'

    tl._llm_call_async_fn = summarizer
    await tl.reactive_compact_async(1)

    assert seen == [loop_thread], "the async provider must run on the awaiting thread"
    assert tl.get_compressed_context() == "in place"


@pytest.mark.asyncio
async def test_async_provider_cancelled_releases_the_compaction_lock():
    """A cancelled run must not wedge the lock for the next caller."""
    tl = _build()
    started = asyncio.Event()

    async def blocking(prompt: str) -> str:
        started.set()
        await asyncio.sleep(30)
        return ""

    tl._llm_call_async_fn = blocking
    task = asyncio.ensure_future(tl.reactive_compact_async(1))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async def ok(prompt: str) -> str:
        return '{"summary": "after cancel"}'

    tl._llm_call_async_fn = ok
    await asyncio.wait_for(tl.reactive_compact_async(1), timeout=1.0)
    assert tl.get_compressed_context() == "after cancel"


@pytest.mark.asyncio
async def test_sync_summarizer_is_joined_before_cancellation_propagates():
    """Unavoidable blocking work is finished, never orphaned.

    The worker only produces text, and the await is re-joined on cancellation: the
    caller's deadline may overrun by the call's remaining duration, which is
    honest, while no thread of this run is still working after it returns.
    """
    tl = _build()
    recorder = _ProjectionRecorder(tl)
    release = threading.Event()
    finished = threading.Event()
    started = threading.Event()

    def summarizer(prompt: str) -> str:
        started.set()
        release.wait(10)
        finished.set()
        return '{"summary": "sync late"}'

    tl._llm_call_fn = summarizer
    task = asyncio.ensure_future(tl.reactive_compact_async(1))
    for _ in range(500):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set(), "the blocking summarizer must start"
    assert recorder.commits == []

    task.cancel()
    await asyncio.sleep(0.2)
    assert not task.done(), "the run must keep waiting for its own blocking work"
    assert not finished.is_set()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set(), "the blocking call finished on a thread the run owned"
    await asyncio.sleep(0)
    assert recorder.commits == [], "a cancelled run commits nothing, even after its worker returned"
    assert tl._compaction_disabled is False
    assert tl._consecutive_compact_failures == 0
    assert len(tl.timeline) == 55


@pytest.mark.asyncio
async def test_worker_failure_during_the_cancel_join_does_not_replace_owner_cancellation():
    """Failing while being joined must not become the owner's outcome.

    The worker cannot be aborted, so the run waits it out. If its own exception
    (here ``ValueError``) escaped the join it would replace the owner's
    ``CancelledError`` and be mistaken for a compaction failure, wrongly charging
    the circuit for a run the owner abandoned.
    """
    tl = _build()
    recorder = _ProjectionRecorder(tl)
    release = threading.Event()
    started = threading.Event()
    finished = threading.Event()

    def summarizer(prompt: str) -> str:
        started.set()
        release.wait(10)
        finished.set()
        raise ValueError("worker exploded after the owner cancelled")

    tl._llm_call_fn = summarizer
    task = asyncio.ensure_future(tl.reactive_compact_async(1))
    for _ in range(500):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set()

    task.cancel()
    await asyncio.sleep(0.2)
    assert not task.done(), "the worker is still being joined"
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set()
    assert recorder.commits == []
    assert tl._compaction_disabled is False, "a run the owner abandoned must not charge the circuit"
    assert tl._consecutive_compact_failures == 0


@pytest.mark.asyncio
async def test_sync_summarizer_completed_run_commits_on_the_awaiting_thread():
    """The sync-callable path commits on the awaiting thread, not the worker."""
    tl = _build()
    recorder = _ProjectionRecorder(tl)
    loop_thread = threading.get_ident()
    worker_threads: list[int] = []

    def summarizer(prompt: str) -> str:
        worker_threads.append(threading.get_ident())
        return '{"summary": "offloaded ok"}'

    tl._llm_call_fn = summarizer
    await asyncio.wait_for(tl.reactive_compact_async(1), timeout=5.0)

    assert worker_threads and worker_threads[0] != loop_thread, "the blocking call runs off-loop"
    assert recorder.commits == [loop_thread], "the commit runs on the awaiting thread"
    assert all(thread == loop_thread for _e, _m, thread in recorder.installs)
    assert tl.get_compressed_context() == "offloaded ok"


@pytest.mark.asyncio
async def test_async_seam_commits_nothing_when_the_async_summarizer_fails():
    """A failing summarizer is a failure of the run, not a late commit."""
    tl = _build()
    recorder = _ProjectionRecorder(tl)

    async def boom(prompt: str) -> str:
        raise RuntimeError("summary boom")

    tl._llm_call_async_fn = boom
    with pytest.raises(RuntimeError, match="summary boom"):
        await tl.reactive_compact_async(1)

    assert recorder.commits == []
    assert tl._consecutive_compact_failures == 1


@pytest.mark.asyncio
async def test_async_seam_empty_summary_counts_as_a_failure():
    tl = _build()
    recorder = _ProjectionRecorder(tl)

    async def empty(prompt: str) -> str:
        return ""

    tl._llm_call_async_fn = empty
    with pytest.raises(RuntimeError, match="empty summary"):
        await tl.reactive_compact_async(1)

    assert recorder.commits == []
    assert tl._consecutive_compact_failures == 1


@pytest.mark.asyncio
async def test_async_seam_circuit_open_raises_before_any_mutation():
    tl = _build()
    recorder = _ProjectionRecorder(tl)
    tl._compaction_disabled = True
    tl._consecutive_compact_failures = 3
    tl._circuit_opened_at = datetime.now()
    before = len(tl.timeline)

    with pytest.raises(CompactCircuitOpenError):
        await tl.reactive_compact_async(1)

    assert recorder.installs == []
    assert recorder.commits == []
    assert len(tl.timeline) == before


@pytest.mark.asyncio
async def test_async_seam_circuit_opens_after_three_consecutive_failures():
    tl = _build()
    tl._llm_call_fn = _always_raise

    for _ in range(2):
        with pytest.raises(RuntimeError, match="summary boom"):
            await tl.reactive_compact_async(1)
    with pytest.raises(CompactCircuitOpenError):
        await tl.reactive_compact_async(1)

    assert tl._compaction_disabled is True


def _always_raise(prompt: str) -> str:
    raise RuntimeError("summary boom")


@pytest.mark.asyncio
async def test_async_seam_uses_the_runtime_async_provider_and_reads_response_text():
    """No explicit callable → the runtime's async call, whose object reduces to text."""
    tl = _build()
    calls: list[str] = []
    sync_calls: list[str] = []

    class _Runtime:
        async def call_llm_async(self, messages, messages_fn=None):
            calls.append(messages[0].content)
            return LLMResponse(content='{"summary": "from runtime"}', model="m", usage=None)

        def call_llm(self, messages, messages_fn=None):  # pragma: no cover - must not be used
            sync_calls.append("sync")
            return "sync"

    class _Agent:
        _runtime = _Runtime()

    tl._agent = _Agent()
    await asyncio.wait_for(tl.reactive_compact_async(1), timeout=5.0)

    assert calls, "the runtime's async call must be used"
    assert sync_calls == [], "the sync runtime call must not be used by the async seam"
    assert tl.get_compressed_context() == "from runtime"


@pytest.mark.asyncio
async def test_sync_seam_still_commits_fully_through_the_shared_internals():
    """Parity guard: the refactor must not weaken ``reactive_compact``."""
    tl = _build()
    tl._llm_call_fn = lambda prompt: '{"summary": "sync ok"}'
    before = len(tl.timeline)

    tl.reactive_compact(1)

    assert len(tl.timeline) < before
    assert tl.get_compressed_context() == "sync ok"
    assert tl._consecutive_compact_failures == 0
    assert tl._compaction_disabled is False


@pytest.mark.asyncio
async def test_both_seams_produce_the_same_retained_shape():
    """The two seams differ in ownership, not in what a completed run commits."""
    sync_tl = _build()
    sync_tl._llm_call_fn = lambda prompt: '{"summary": "same"}'
    sync_tl.reactive_compact(1)

    async_tl = _build()

    async def summarizer(prompt: str) -> str:
        return '{"summary": "same"}'

    async_tl._llm_call_async_fn = summarizer
    await asyncio.wait_for(async_tl.reactive_compact_async(1), timeout=5.0)

    assert len(async_tl.timeline) == len(sync_tl.timeline)
    assert async_tl.get_compressed_context() == sync_tl.get_compressed_context() == "same"
    assert [e.content for e in async_tl.timeline] == [e.content for e in sync_tl.timeline]
