"""Regression tests for chat_response_sync loop handling.

``chat_response_sync`` runs the async ``chat_response`` via
``loop.run_until_complete``, which cannot nest inside a running event loop.
Rather than silently bridging (which would mask misuse and block the caller's
loop), the sync API must FAIL LOUD with guidance toward the async path
(``await llm.chat_response()`` / ``await agent.aquery()``) when called from
within a running loop.

The previous guard here was dead — its explicit ``raise RuntimeError`` was
caught by the surrounding ``except RuntimeError: pass``, so callers got the
cryptic "Cannot run the event loop while another loop is running" instead of
the actionable message.
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
    """Baseline: plain sync call (no running loop) uses run_until_complete."""
    expected = LLMResponse(content="ok", model="test-model")
    llm = _llm_returning(expected)

    result = llm.chat_response_sync(_user_msg())
    assert result is expected


def test_sync_raises_loud_inside_running_loop():
    """Sync API called from within a running loop must raise a clear, actionable error.

    Previously: the dead guard let it fall through to run_until_complete →
    "Cannot run the event loop while another loop is running" (cryptic).
    """
    llm = _llm_returning(LLMResponse(content="ok", model="m"))

    async def caller():
        return llm.chat_response_sync(_user_msg())

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(caller())
    msg = str(exc_info.value)
    assert "chat_response_sync" in msg
    assert "aquery" in msg or "chat_response" in msg


def test_sync_propagates_exception_without_running_loop():
    """Coroutine errors surface to the sync caller via run_until_complete."""
    llm = _llm_returning(LLMResponse(content="x", model="m"), raise_exc=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        llm.chat_response_sync(_user_msg())
