"""Regression tests for chat_response_sync sync→async bridging.

When a caller invokes the sync LLM API from inside a running asyncio loop
(e.g. a framework that mixes sync/async), ``run_until_complete`` cannot
nest inside that loop. ``chat_response_sync`` must bridge via a dedicated
background daemon-thread loop instead of raising
"Cannot run the event loop while another loop is running".
"""

import asyncio

import pytest

from dana.common.llm.llm import LLM
from dana.common.llm.types import LLMMessage, LLMResponse


def _llm_returning(response: LLMResponse, *, raise_exc: Exception | None = None) -> LLM:
    """Build an LLM whose async ``chat_response`` returns ``response`` (or raises)."""
    llm = LLM()

    async def fake_chat_response(messages, **kwargs):
        if raise_exc is not None:
            raise raise_exc
        return response

    llm.chat_response = fake_chat_response  # type: ignore[assignment,method-assign]
    return llm


def _user_msg() -> list[LLMMessage]:
    return [LLMMessage(role="user", content="hi")]


def test_sync_works_without_running_loop():
    """Baseline: plain sync call uses the run_until_complete path."""
    expected = LLMResponse(content="ok", model="test-model")
    llm = _llm_returning(expected)

    result = llm.chat_response_sync(_user_msg())
    assert result is expected


def test_sync_works_inside_running_loop():
    """Regression: sync API called from within a running loop must not crash.

    Previously raised:
      RuntimeError: Cannot run the event loop while another loop is running
    """
    expected = LLMResponse(content="ok", model="test-model")
    llm = _llm_returning(expected)

    async def caller():
        # We are now inside a running loop on this thread.
        return llm.chat_response_sync(_user_msg())

    result = asyncio.run(caller())
    assert result is expected


def test_sync_propagates_exception_inside_running_loop():
    """Errors raised by the coroutine surface to the sync caller via the bridge."""
    llm = _llm_returning(LLMResponse(content="x", model="m"), raise_exc=RuntimeError("boom"))

    async def caller():
        return llm.chat_response_sync(_user_msg())

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(caller())
