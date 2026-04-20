"""Unit tests for Phase 3 PTL catch + reactive_compact retry in llm_caller."""

from __future__ import annotations

import asyncio

import pytest

from dana.common.llm.types import (
    CompactCircuitOpenError,
    LLMResponse,
    PromptTooLongError,
)
from dana.core.llm.llm_caller import LLMCaller


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat_response_sync(self, messages, **kwargs):
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    async def chat_response(self, messages, **kwargs):
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class _FakeTimeline:
    def __init__(self):
        self.reactive_calls: list[int] = []
        self._consecutive_compact_failures = 0
        self._compaction_disabled = False
        self._circuit_opened_at = None

        class _Cfg:
            enable_reactive_compact = True

        self._compressed_config = _Cfg()

    def reactive_compact(self, attempt: int) -> None:
        self.reactive_calls.append(attempt)


class _FakeAgent:
    def __init__(self, timeline):
        self._timeline = timeline
        self.object_id = "a1"
        self.agent_type = "test"


def _make_caller(llm, agent):
    caller = LLMCaller(
        llm=llm,
        agent_getter=lambda: agent,
        native_tools_getter=lambda: None,
    )
    return caller


def test_ptl_single_failure_then_success_triggers_one_reactive_compact():
    ok = LLMResponse(content="ok", model="m", usage=None)
    llm = _FakeLLM([PromptTooLongError("too long"), ok])
    tl = _FakeTimeline()
    agent = _FakeAgent(tl)
    caller = _make_caller(llm, agent)

    # Skip backoff sleeps to speed up.
    import time as _time

    _time.sleep = lambda s: None  # type: ignore[assignment]

    resp = caller._invoke_llm_sync(llm, [])
    assert resp.content == "ok"
    assert tl.reactive_calls == [1]


def test_ptl_three_failures_raise_circuit_open():
    llm = _FakeLLM([PromptTooLongError("x")] * 3 + [LLMResponse(content="", model="m")])
    tl = _FakeTimeline()
    agent = _FakeAgent(tl)
    caller = _make_caller(llm, agent)

    import time as _time

    _time.sleep = lambda s: None  # type: ignore[assignment]

    with pytest.raises(CompactCircuitOpenError):
        caller._invoke_llm_sync(llm, [])
    assert tl.reactive_calls == [1, 2, 3]
    assert tl._compaction_disabled is True


def test_ptl_kill_switch_bubbles_unchanged(monkeypatch):
    monkeypatch.setenv("DANA_DISABLE_REACTIVE_COMPACT", "1")
    llm = _FakeLLM([PromptTooLongError("bubble")])
    tl = _FakeTimeline()
    agent = _FakeAgent(tl)
    caller = _make_caller(llm, agent)

    with pytest.raises(PromptTooLongError):
        caller._invoke_llm_sync(llm, [])
    assert tl.reactive_calls == []  # kill switch — no reactive_compact


def test_ptl_config_flag_off_bubbles_unchanged():
    llm = _FakeLLM([PromptTooLongError("bubble")])
    tl = _FakeTimeline()
    tl._compressed_config.enable_reactive_compact = False
    agent = _FakeAgent(tl)
    caller = _make_caller(llm, agent)

    with pytest.raises(PromptTooLongError):
        caller._invoke_llm_sync(llm, [])
    assert tl.reactive_calls == []


def test_async_ptl_recovery():
    ok = LLMResponse(content="ok", model="m")
    llm = _FakeLLM([PromptTooLongError("x"), ok])
    tl = _FakeTimeline()
    agent = _FakeAgent(tl)
    caller = _make_caller(llm, agent)

    # Patch asyncio.sleep to skip.
    async def _nosleep(s):
        return None

    asyncio.sleep = _nosleep  # type: ignore[assignment]

    resp = asyncio.run(caller._invoke_llm_async(llm, []))
    assert resp.content == "ok"
    assert tl.reactive_calls == [1]


def test_ptl_not_retried_by_failover():
    """PTL must not be classified transient (so _call_with_failover won't double-retry)."""
    assert LLMCaller._is_transient_error(PromptTooLongError("x")) is False


# ---------------------------------------------------------------------------
# CRITICAL-1 regression — messages must be rebuilt after reactive_compact so
# the retry observes the compacted timeline rather than a stale snapshot.
# ---------------------------------------------------------------------------


class _MessageSizeAwareLLM:
    """Fake LLM that raises PTL unless the message list length is below a cap.

    Lets tests assert that the retry sees a different (smaller) message list
    than the first attempt — i.e. ``messages_fn`` actually ran.
    """

    def __init__(self, max_len: int, ok_response: LLMResponse):
        self._max_len = max_len
        self._ok = ok_response
        self.seen_lengths: list[int] = []

    def chat_response_sync(self, messages, **kwargs):
        self.seen_lengths.append(len(messages))
        if len(messages) > self._max_len:
            raise PromptTooLongError(f"msgs={len(messages)} > cap={self._max_len}")
        return self._ok

    async def chat_response(self, messages, **kwargs):
        return self.chat_response_sync(messages, **kwargs)


class _CompactingTimeline(_FakeTimeline):
    """Timeline whose reactive_compact mutates a shared state so rebuild_fn
    produces shorter messages on each call."""

    def __init__(self, state: dict):
        super().__init__()
        self._state = state

    def reactive_compact(self, attempt: int) -> None:
        super().reactive_compact(attempt)
        # Simulate dropping 2 entries per attempt.
        self._state["msg_count"] = max(1, self._state["msg_count"] - 2)


def test_ptl_retry_rebuilds_messages_via_factory():
    """After reactive_compact, messages_fn must be invoked so retry uses
    the compacted payload (CRITICAL-1)."""
    ok = LLMResponse(content="ok", model="m", usage=None)
    # Cap = 3; first call sends 5 (PTL), retry sends 3 after 1 compact (ok).
    llm = _MessageSizeAwareLLM(max_len=3, ok_response=ok)
    state = {"msg_count": 5}
    tl = _CompactingTimeline(state)
    agent = _FakeAgent(tl)
    caller = _make_caller(llm, agent)

    def _rebuild():
        return [object()] * state["msg_count"]

    import time as _time

    _time.sleep = lambda s: None  # type: ignore[assignment]

    initial = _rebuild()
    resp = caller._invoke_llm_sync(llm, initial, messages_fn=_rebuild)
    assert resp.content == "ok"
    assert tl.reactive_calls == [1]
    # Proof that rebuild happened — retry sent fewer messages than the first call.
    assert llm.seen_lengths == [5, 3]


def test_ptl_retry_without_factory_keeps_stale_messages():
    """When messages_fn is None, legacy behavior preserved — messages stay stale
    across retries. Kept as a guardrail so callers who opt out are explicit."""
    llm = _MessageSizeAwareLLM(max_len=3, ok_response=LLMResponse(content="never", model="m"))
    state = {"msg_count": 5}
    tl = _CompactingTimeline(state)
    agent = _FakeAgent(tl)
    caller = _make_caller(llm, agent)

    import time as _time

    _time.sleep = lambda s: None  # type: ignore[assignment]

    with pytest.raises(CompactCircuitOpenError):
        caller._invoke_llm_sync(llm, [object()] * 5)
    # Every attempt saw the same 5-message payload — no rebuild happened.
    assert llm.seen_lengths == [5, 5, 5]


def test_async_ptl_retry_rebuilds_messages_via_factory():
    """Async counterpart of the rebuild assertion (CRITICAL-1)."""
    ok = LLMResponse(content="ok", model="m")
    llm = _MessageSizeAwareLLM(max_len=3, ok_response=ok)
    state = {"msg_count": 5}
    tl = _CompactingTimeline(state)
    agent = _FakeAgent(tl)
    caller = _make_caller(llm, agent)

    def _rebuild():
        return [object()] * state["msg_count"]

    async def _nosleep(s):
        return None

    asyncio.sleep = _nosleep  # type: ignore[assignment]

    initial = _rebuild()
    resp = asyncio.run(caller._invoke_llm_async(llm, initial, messages_fn=_rebuild))
    assert resp.content == "ok"
    assert tl.reactive_calls == [1]
    assert llm.seen_lengths == [5, 3]


def test_ptl_retry_tolerates_messages_fn_raise():
    """If messages_fn raises, retry proceeds with the current (stale) list
    rather than crashing."""
    ok = LLMResponse(content="ok", model="m", usage=None)
    tl = _FakeTimeline()
    agent = _FakeAgent(tl)

    def _bad_rebuild():
        raise RuntimeError("simulated rebuild failure")

    import time as _time

    _time.sleep = lambda s: None  # type: ignore[assignment]

    # Use a tight cap so rebuild failure → stale retry → all 3 attempts fail.
    llm2 = _MessageSizeAwareLLM(max_len=1, ok_response=ok)
    caller2 = _make_caller(llm2, agent)
    with pytest.raises(CompactCircuitOpenError):
        caller2._invoke_llm_sync(llm2, [object(), object()], messages_fn=_bad_rebuild)
    # All 3 attempts saw the same list — rebuild failed but didn't crash.
    assert llm2.seen_lengths == [2, 2, 2]
