"""Per-path edit-lock lifetime: bounded weak registry, real mutual exclusion.

PR32 re-review repair on ``FileEditResource``. The resource serializes
read-modify-write of one resolved path with a per-path ``threading.Lock``; the
registry that hands those locks out used to be a plain ``dict``, so it kept one
lock per distinct path edited for the whole process lifetime. It is now a
``weakref.WeakValueDictionary``, and these tests pin the two properties that
makes safe, plus the offload identity the repair depends on:

* **Bounded**: an entry lives only as long as some caller that obtained it still
  holds the strong reference ``_path_lock`` returns. Editing many distinct paths
  does not grow the map, and the entry disappears once the last holder/waiter
  exits.
* **Still mutually exclusive**: a waiter holds that same strong reference for its
  whole ``with`` block, so the entry cannot be collected (and a second lock
  cannot be minted) while anyone holds or queues for the path. Same-path callers
  therefore still get *one* lock object and never overlap; different paths stay
  concurrent.
* **Offloaded**: the critical section really runs on the offload pool's worker
  thread. These resource methods are ``async`` and offload explicitly through
  ``run_blocking`` (dispatch's ``@blocking_io`` opt-in governs *synchronous*
  callables only), which is why a plain thread lock is the right primitive.

Real resources and real events; no network, no provider calls.
"""

from __future__ import annotations

import asyncio
import gc
import pathlib
import threading
import time

import pytest

from dana.core.resource.file_edit_resource import FileEditResource
from dana.core.resource.file_io_resource import FileIOResource
from dana.core.resource.search_resource import GREPMode, SearchResource


#: Prefix of every worker thread a ``BlockingIOPool`` starts (session pool
#: ``dana-io`` and process default ``dana-io-default`` both match).
IO_THREAD_PREFIX = "dana-io"


def _registry_entries(resource: FileEditResource) -> int:
    """Live entries of the per-path lock registry (private unit under test)."""
    return len(resource._path_locks)


def _entries_settle(resource: FileEditResource, at_most: int, timeout: float = 2.0) -> int:
    """Bounded wait for dead weak entries to clear, then the live count."""
    deadline = time.monotonic() + timeout
    count = _registry_entries(resource)
    while count > at_most and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.01)
        count = _registry_entries(resource)
    return count


class _RecordingEditResource(FileEditResource):
    """FileEditResource that observes concurrency and lock identity.

    ``_edit_locked`` runs *inside* the per-path lock, so the counters and the
    entry-time registry length observed here are exactly the state under it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._probe_guard = threading.Lock()
        self.handed_lock_ids: list[int] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self.registry_len_at_entry: list[int] = []
        self.entered_threads: list[tuple[int, str]] = []
        self.hold_s = 0.0
        self.entry_barrier: threading.Barrier | None = None

    def _path_lock(self, resolved_path: pathlib.Path) -> threading.Lock:
        lock = super()._path_lock(resolved_path)
        with self._probe_guard:
            self.handed_lock_ids.append(id(lock))
        return lock

    def _edit_locked(self, resolved_path, old_string, new_string, replace_all, fuzzy_match) -> str:
        with self._probe_guard:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            self.registry_len_at_entry.append(len(self._path_locks))
            self.entered_threads.append((threading.get_ident(), threading.current_thread().name))
        try:
            if self.entry_barrier is not None:
                # Only reachable concurrently when the paths differ: same-path
                # callers cannot be here together, so this really proves parallel
                # critical sections rather than a serialized registry.
                self.entry_barrier.wait(timeout=5.0)
            if self.hold_s:
                time.sleep(self.hold_s)
            return super()._edit_locked(resolved_path, old_string, new_string, replace_all, fuzzy_match)
        finally:
            with self._probe_guard:
                self.concurrent -= 1


# ---------------------------------------------------------------------------
# Registry lifetime
# ---------------------------------------------------------------------------


def test_entry_lives_exactly_while_a_strong_handle_is_held(tmp_path):
    """The map entry is weak: gone once the handle ``_path_lock`` returned is."""
    resource = FileEditResource(resource_id="lock-lifetime", base_path=str(tmp_path))
    lock = resource._path_lock(tmp_path / "a.txt")
    assert isinstance(lock, type(threading.Lock())), "the handle must be the real lock, not a weak proxy"
    assert _registry_entries(resource) == 1, "holding the handle must keep the entry alive"

    del lock
    assert _entries_settle(resource, 0) == 0, "the entry must not outlive its last holder"


@pytest.mark.asyncio
async def test_registry_stays_bounded_across_many_distinct_paths(tmp_path):
    """Editing hundreds of distinct paths must not grow the map monotonically."""
    resource = FileEditResource(resource_id="lock-bounded", base_path=str(tmp_path))
    paths = []
    for i in range(200):
        target = tmp_path / f"file_{i}.txt"
        target.write_text(f"alpha {i}\n")
        paths.append(str(target))

    for i, target in enumerate(paths):
        result = await resource.edit(target, f"alpha {i}", f"ALPHA {i}")
        assert f"ALPHA {i}" in result, result

    assert (tmp_path / "file_199.txt").read_text().startswith("ALPHA 199")
    assert _entries_settle(resource, 0) == 0, (
        f"the registry kept {_registry_entries(resource)} entries for 200 distinct paths; it must be weak, not a growing dict"
    )


# ---------------------------------------------------------------------------
# Mutual exclusion under the weak registry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_path_edits_serialize_on_one_lock_object(tmp_path):
    """Concurrent edits of one file get one lock, never overlap, lose no update."""
    resource = _RecordingEditResource(resource_id="lock-same-path", base_path=str(tmp_path))
    resource.hold_s = 0.05
    target = tmp_path / "shared.txt"
    target.write_text("alpha\n")

    results = await asyncio.gather(*[resource.edit(str(target), "alpha", f"ALPHA{i}") for i in range(4)])

    assert resource.max_concurrent == 1, "two same-path edits were inside the critical section at once"
    assert len(set(resource.handed_lock_ids)) == 1, f"callers of one path were handed {len(set(resource.handed_lock_ids))} different locks"
    assert set(resource.registry_len_at_entry) == {1}, (
        f"the path's entry must stay single and alive while held/awaited; saw {resource.registry_len_at_entry}"
    )
    succeeded = [r for r in results if "has been updated" in r]
    assert len(succeeded) == 1, f"exactly one edit may win, got {len(succeeded)}: {results}"
    assert target.read_text().count("ALPHA") == 1, "the file must carry exactly one winning update, not a lost/duplicated one"
    assert _entries_settle(resource, 0) == 0


@pytest.mark.asyncio
async def test_waiting_caller_keeps_the_entry_alive_until_it_enters(tmp_path):
    """A queued caller holds the handle while blocked, so the lock is never reminted."""
    resource = _RecordingEditResource(resource_id="lock-waiter", base_path=str(tmp_path))
    resource.hold_s = 0.15
    target = tmp_path / "queued.txt"
    target.write_text("alpha\n")

    results = await asyncio.gather(*[resource.edit(str(target), "alpha", f"ALPHA{i}") for i in range(3)])

    assert set(resource.registry_len_at_entry) == {1}, "a blocked waiter must still see (and hold) the path's single entry"
    assert len(resource.handed_lock_ids) == 3, "each caller asks for the path lock once"
    assert len(set(resource.handed_lock_ids)) == 1, (
        f"the queued callers must all receive the same lock object, got {len(set(resource.handed_lock_ids))}"
    )
    assert len(resource.entered_threads) == 3, "all three critical sections ran"
    assert resource.max_concurrent == 1
    assert sum("has been updated" in r for r in results) == 1
    assert _entries_settle(resource, 0) == 0


@pytest.mark.asyncio
async def test_different_paths_hold_their_critical_sections_concurrently(tmp_path):
    """Separate paths must stay parallel: no global serialization through the map."""
    resource = _RecordingEditResource(resource_id="lock-parallel", base_path=str(tmp_path))
    resource.entry_barrier = threading.Barrier(4)
    targets = []
    for i in range(4):
        target = tmp_path / f"parallel_{i}.txt"
        target.write_text(f"alpha {i}\n")
        targets.append((str(target), i))

    results = await asyncio.gather(*[resource.edit(path, f"alpha {i}", f"ALPHA {i}") for path, i in targets])

    assert not any("BrokenBarrierError" in r for r in results), (
        f"distinct paths did not reach their critical sections together; something serialized them: {results}"
    )
    assert resource.max_concurrent == 4, f"expected 4 parallel critical sections, saw {resource.max_concurrent}"
    assert all("has been updated" in r for r in results), results
    assert max(resource.registry_len_at_entry) == 4, (
        f"all four held paths must have live entries at once (a held lock is never collected); saw {resource.registry_len_at_entry}"
    )
    assert _entries_settle(resource, 0) == 0


@pytest.mark.asyncio
async def test_write_and_edit_share_one_path_lock(tmp_path):
    """``write``'s existence check and ``edit``'s read-modify-write use one lock."""
    resource = FileEditResource(resource_id="lock-write-edit", base_path=str(tmp_path))
    target = str(tmp_path / "created.txt")

    write_alpha, write_beta, edit_result = await asyncio.gather(
        resource.write(target, "alpha\n"),
        resource.write(target, "beta\n"),
        resource.edit(target, "alpha", "ALPHA"),
    )

    wrote = [r for r in (write_alpha, write_beta) if "Successfully wrote" in r]
    refused = [r for r in (write_alpha, write_beta) if "already exists" in r]
    assert len(wrote) == 1, f"one create wins, got {write_alpha!r} / {write_beta!r}"
    assert len(refused) == 1, f"the second create is refused under the lock, got {write_alpha!r} / {write_beta!r}"

    body = (tmp_path / "created.txt").read_text()
    assert body in ("alpha\n", "beta\n", "ALPHA\n"), f"the body must be one whole write, not an interleaving: {body!r}"
    if "has been updated" in edit_result:
        # The edit's write ran under the same lock and is the last effect: no
        # concurrent create may have clobbered it afterwards.
        assert body == "ALPHA\n", f"a successful edit was overwritten: {body!r}"
    assert _entries_settle(resource, 0) == 0


# ---------------------------------------------------------------------------
# Offload identity: the critical section is not on the event loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_and_write_critical_sections_run_on_pool_workers(tmp_path):
    """The lock is held on a ``dana-io`` worker thread, never on the loop thread."""
    loop_ident = threading.get_ident()
    seen: dict[str, tuple[int, str]] = {}

    class _Probe(FileEditResource):
        def _edit_locked(self, resolved_path, old_string, new_string, replace_all, fuzzy_match) -> str:
            seen["edit"] = (threading.get_ident(), threading.current_thread().name)
            return super()._edit_locked(resolved_path, old_string, new_string, replace_all, fuzzy_match)

        def _write_blocking(self, resolved_path, content) -> str:
            seen["write"] = (threading.get_ident(), threading.current_thread().name)
            return super()._write_blocking(resolved_path, content)

    resource = _Probe(resource_id="offload-edit", base_path=str(tmp_path))
    target = tmp_path / "offload.txt"
    target.write_text("alpha\n")

    assert "has been updated" in await resource.edit(str(target), "alpha", "ALPHA")
    assert "Successfully wrote" in await resource.write(str(tmp_path / "new.txt"), "hi\n")

    for name in ("edit", "write"):
        ident, thread_name = seen[name]
        assert ident != loop_ident, f"{name} ran its critical section on the event-loop thread"
        assert thread_name.startswith(IO_THREAD_PREFIX), f"{name} ran on {thread_name!r}, not on an offload worker"


@pytest.mark.asyncio
async def test_read_and_grep_blocking_cores_run_on_pool_workers(tmp_path):
    """Read/Grep offload their blocking cores too: async signature != inline disk I/O."""
    loop_ident = threading.get_ident()
    seen: dict[str, tuple[int, str]] = {}

    class _ProbeIO(FileIOResource):
        def _read_text(self, resolved_path, offset, limit) -> str:
            seen["read"] = (threading.get_ident(), threading.current_thread().name)
            return super()._read_text(resolved_path, offset, limit)

    class _ProbeSearch(SearchResource):
        def _grep_python_native_blocking(self, *args, **kwargs) -> str:
            seen["grep"] = (threading.get_ident(), threading.current_thread().name)
            return super()._grep_python_native_blocking(*args, **kwargs)

    (tmp_path / "sample.txt").write_text("alpha\nneedle here\n")
    io_resource = _ProbeIO(resource_id="offload-read", base_path=str(tmp_path))
    search_resource = _ProbeSearch(resource_id="offload-grep", base_path=str(tmp_path), mode=GREPMode.PYTHON_NATIVE)

    read_result = await io_resource.read(str(tmp_path / "sample.txt"))
    assert "needle here" in read_result
    grep_result = await search_resource.grep(pattern="needle", path=str(tmp_path), output_mode="content")
    assert "needle" in grep_result

    for name in ("read", "grep"):
        ident, thread_name = seen[name]
        assert ident != loop_ident, f"{name} performed its filesystem work on the event-loop thread"
        assert thread_name.startswith(IO_THREAD_PREFIX), f"{name} ran on {thread_name!r}, not on an offload worker"


@pytest.mark.asyncio
async def test_loop_keeps_servicing_while_an_edit_holds_the_path_lock(tmp_path):
    """The offloaded critical section does not stall the loop (the reason to offload at all)."""
    resource = _RecordingEditResource(resource_id="offload-loop", base_path=str(tmp_path))
    resource.hold_s = 0.1
    target = tmp_path / "busy.txt"
    target.write_text("alpha\n")

    ticks = 0

    async def spinner() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0)
            ticks += 1

    spin_task = asyncio.create_task(spinner())
    try:
        result = await resource.edit(str(target), "alpha", "ALPHA")
    finally:
        spin_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await spin_task

    assert "has been updated" in result
    assert ticks > 10, f"the loop only ticked {ticks} times while the edit was offloaded"
    assert _entries_settle(resource, 0) == 0


def test_held_lock_is_never_collected_even_under_gc_pressure(tmp_path):
    """A held handle survives collection, so one path can never get two locks.

    This is the safety half of the weak registry: the entry is allowed to die
    *only* when nobody holds it. A design that dropped or evicted the entry at
    hand-out time would satisfy the bounded-count test and break this one.
    """
    resource = FileEditResource(resource_id="lock-held", base_path=str(tmp_path))
    held = resource._path_lock(tmp_path / "held.txt")

    for _ in range(5):
        gc.collect()
    assert resource._path_lock(tmp_path / "held.txt") is held, "the registry minted a second lock for a path whose first lock is still held"
    assert _registry_entries(resource) == 1


def test_registry_does_not_grow_across_real_edit_and_write_churn(tmp_path):
    """Public-API churn on distinct paths leaves no lock behind (bounded registry)."""
    resource = FileEditResource(resource_id="lock-churn", base_path=str(tmp_path))
    for i in range(50):
        target = tmp_path / f"churn_{i}.txt"
        target.write_text("alpha\n")
        result = resource._edit_blocking(target, "alpha", "ALPHA", False, True)
        assert "has been updated" in result, result
        assert "Successfully wrote" in resource._write_blocking(tmp_path / f"made_{i}.txt", "hi\n")

    assert _entries_settle(resource, 0) == 0, f"{_registry_entries(resource)} per-path locks survived 50 completed edits/writes"
