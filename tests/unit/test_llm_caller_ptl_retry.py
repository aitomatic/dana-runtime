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
