"""Consumer-visible retry, deadline, typed-error and lifecycle boundary tests.

Scope: the behaviour a caller or the host can observe — wire attempt counts,
which exception type escapes, whether already-delivered output is re-requested,
whether a wedged call can hold the caller forever, and whether a retired client
is released. No network, no live provider.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dana.common.llm.llm import LLM
from dana.common.llm.types import (
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    LLMTimeoutError,
    PromptTooLongError,
    ProviderError,
)
from dana.core.llm.llm_caller import DEFAULT_TEMPERATURE, LLMCaller, ProviderConfig
from dana.core.runtime.default import DefaultRuntime


def _response(content: str = "ok") -> LLMResponse:
    return LLMResponse(content=content, model="test-model")


class _StreamingLLM:
    """LLM stand-in whose stream fails at a chosen point."""

    def __init__(self, *, fail_after: int, exc: BaseException, succeed_on_attempt: int | None = None):
        self.fail_after = fail_after
        self.exc = exc
        self.succeed_on_attempt = succeed_on_attempt
        self.attempts = 0
        self.sent_kwargs: list[dict] = []
        self.sent_tools: list = []

    async def stream(self, messages, tools=None, **kwargs):
        self.attempts += 1
        self.sent_kwargs.append(kwargs)
        self.sent_tools.append(tools)
        if self.succeed_on_attempt is not None and self.attempts >= self.succeed_on_attempt:
            yield LLMStreamChunk(type="text_delta", content="late-ok")
            yield LLMStreamChunk(type="response", response=_response("late-ok"))
            return
        for i in range(self.fail_after):
            yield LLMStreamChunk(type="text_delta", content=f"part{i}")
        raise self.exc


class TestStreamRetrySafety:
    """A stream must never re-request output the consumer already saw."""

    @pytest.mark.asyncio
    async def test_no_retry_after_first_emitted_chunk(self):
        llm = _StreamingLLM(fail_after=2, exc=ProviderError("Connection error."))
        caller = LLMCaller(llm=llm, stream_max_retries=2)

        seen: list[LLMStreamChunk] = []
        with pytest.raises(ProviderError):
            async for chunk in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
                seen.append(chunk)

        assert [c.content for c in seen] == ["part0", "part1"]
        assert llm.attempts == 1, "partial output must not be replayed"

    @pytest.mark.asyncio
    async def test_retries_when_nothing_was_emitted(self):
        """A transient failure before any chunk is safe to re-issue."""
        llm = _StreamingLLM(fail_after=0, exc=ProviderError("rate limit exceeded"), succeed_on_attempt=2)
        caller = LLMCaller(llm=llm, stream_max_retries=2, base_delay=0.0)

        chunks = [c async for c in caller.call_llm_stream([LLMMessage(role="user", content="hi")])]

        assert llm.attempts == 2
        assert [c.content for c in chunks if c.type == "text_delta"] == ["late-ok"]
        assert chunks[-1].type == "response"

    @pytest.mark.asyncio
    async def test_permanent_error_is_not_retried(self):
        llm = _StreamingLLM(fail_after=0, exc=ProviderError("invalid model name"))
        caller = LLMCaller(llm=llm, stream_max_retries=3, base_delay=0.0)

        with pytest.raises(ProviderError, match="invalid model name"):
            async for _ in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
                pass

        assert llm.attempts == 1

    @pytest.mark.asyncio
    async def test_provider_failure_finish_during_stream_is_retried_not_reported_success(self):
        """A typed error raised after the stream begins but before output is retried."""
        llm = _StreamingLLM(fail_after=0, exc=LLMTimeoutError("stream timeout"), succeed_on_attempt=2)
        caller = LLMCaller(llm=llm, stream_max_retries=1, base_delay=0.0)

        chunks = [c async for c in caller.call_llm_stream([LLMMessage(role="user", content="hi")])]

        assert llm.attempts == 2
        assert chunks[-1].type == "response"


class TestStreamRequestControls:
    """Streaming must send the same caller policy the non-streaming path sends.
    The one deliberate difference is the *unset* temperature: streaming omits it so
    the provider's own sampling default applies."""

    @pytest.mark.asyncio
    async def test_stream_forwards_explicit_temperature_and_max_tokens(self):
        llm = _StreamingLLM(fail_after=0, exc=ProviderError("x"), succeed_on_attempt=1)
        caller = LLMCaller(
            llm=llm,
            temperature=0.35,
            max_tokens=777,
            json_mode=False,
            agent_getter=lambda: None,
            native_tools_getter=lambda: None,
        )

        async for _ in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
            pass

        sent = llm.sent_kwargs[0]
        assert sent["temperature"] == 0.35, "an explicit non-default temperature must reach the wire"
        assert sent["max_tokens"] == 777
        assert sent["json_mode"] is False
        assert llm.sent_tools[0] is None

    @pytest.mark.asyncio
    async def test_explicit_none_temperature_is_omitted(self):
        """An explicit ``None`` asks for no sampling parameter on either path."""
        llm = _StreamingLLM(fail_after=0, exc=ProviderError("x"), succeed_on_attempt=1)
        caller = LLMCaller(llm=llm, temperature=None, max_tokens=None)

        async for _ in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
            pass

        sent = llm.sent_kwargs[0]
        assert "temperature" not in sent
        assert "max_tokens" not in sent

    @pytest.mark.asyncio
    async def test_unset_temperature_is_omitted_on_the_stream_path(self):
        """I11: a caller that never expressed a sampling policy sends none.

        ``UNSET`` is the constructor default, so the default-constructed caller
        (and the runtime that forwards its own unresolved default) leaves the
        provider's sampling default in force. o3/o4 on Responses and Azure gpt-5
        deployments not named gpt-5* reject a sampling parameter with a 400, and
        Anthropic/Gemini silently lose their provider default.
        """
        llm = _StreamingLLM(fail_after=0, exc=ProviderError("x"), succeed_on_attempt=1)
        caller = LLMCaller(llm=llm)  # temperature defaults to UNSET

        async for _ in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
            pass

        assert "temperature" not in llm.sent_kwargs[0]

    @pytest.mark.asyncio
    async def test_explicit_zero_temperature_reaches_the_stream_wire(self):
        """I11 regression: an explicit 0 is a decision, not "unset".

        The value-only rule this replaces compared against the default 0 and
        silently dropped exactly this case. Determinism is a real request: a
        caller that asks for greedy decoding must get it on the streaming path,
        the same way the non-streaming path already forwarded it.
        """
        llm = _StreamingLLM(fail_after=0, exc=ProviderError("x"), succeed_on_attempt=1)
        caller = LLMCaller(llm=llm, temperature=0)

        async for _ in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
            pass

        assert llm.sent_kwargs[0]["temperature"] == 0, "an explicit 0 must not be treated as the runtime default"

    @pytest.mark.asyncio
    async def test_explicit_zero_temperature_reaches_the_nonstream_wire(self):
        llm = MagicMock()
        llm.chat_response_sync.return_value = _response()
        caller = LLMCaller(llm=llm, temperature=0)

        caller.call_llm([])

        assert llm.chat_response_sync.call_args.kwargs["temperature"] == 0

    @pytest.mark.asyncio
    async def test_unset_temperature_keeps_the_nonstream_default_wire_shape(self):
        """The non-streaming wire shape is unchanged by this PR: an unset control
        still resolves to the runtime default ``0`` there."""
        llm = MagicMock()
        llm.chat_response_sync.return_value = _response()
        caller = LLMCaller(llm=llm)

        caller.call_llm([])

        assert llm.chat_response_sync.call_args.kwargs["temperature"] == DEFAULT_TEMPERATURE

    def test_build_request_kwargs_intent_matrix(self):
        """The full intent matrix, both routes, with no provider involved.

        Four caller intents x two routes: this is the table the value-only
        comparison could not express, and where an explicit 0 was lost.
        """
        cases = [
            # (constructor kwarg, expected stream value / omitted, expected sync value / omitted)
            ({}, "omit", DEFAULT_TEMPERATURE),
            ({"temperature": None}, "omit", "omit"),
            ({"temperature": 0}, 0, 0),
            ({"temperature": 0.0}, 0.0, 0.0),
            ({"temperature": 0.7}, 0.7, 0.7),
        ]
        for kwargs, stream_expected, sync_expected in cases:
            caller = LLMCaller(llm=MagicMock(), **kwargs)

            stream_kwargs = caller._build_request_kwargs(streaming=True)
            if stream_expected == "omit":
                assert "temperature" not in stream_kwargs, f"{kwargs}: stream must omit"
            else:
                assert stream_kwargs["temperature"] == stream_expected, f"{kwargs}: stream value"

            sync_kwargs = caller._build_request_kwargs(streaming=False)
            if sync_expected == "omit":
                assert "temperature" not in sync_kwargs, f"{kwargs}: sync must omit"
            else:
                assert sync_kwargs["temperature"] == sync_expected, f"{kwargs}: sync value"


class TestRuntimeTemperatureProvenance:
    """The runtime forwards *intent*, not just a number: its own default must not
    become a wire policy while a caller-supplied 0 must reach the wire."""

    def test_agent_runtime_default_temperature_is_unset_intent(self):
        runtime = DefaultRuntime()
        assert runtime._temperature == DEFAULT_TEMPERATURE, "attribute keeps its historic resolved value"
        assert runtime._temperature_configured is False, "no sampling policy was expressed"
        assert "temperature" not in runtime._llm_caller._build_request_kwargs(streaming=True)

    def test_agent_runtime_explicit_zero_is_configured_intent(self):
        runtime = DefaultRuntime(temperature=0)
        assert runtime._temperature == 0
        assert runtime._temperature_configured is True
        assert runtime._llm_caller._build_request_kwargs(streaming=True)["temperature"] == 0

    def test_agent_runtime_nonzero_temperature_reaches_stream_and_sync(self):
        runtime = DefaultRuntime(temperature=0.5)
        assert runtime._temperature == 0.5
        assert runtime._llm_caller._build_request_kwargs(streaming=True)["temperature"] == 0.5
        assert runtime._llm_caller._build_request_kwargs(streaming=False)["temperature"] == 0.5

    def test_codec_runtime_preserves_unset_intent(self):
        """Every runtime layer must forward the sentinel, not resolve it to 0."""
        from dana.core.runtime.codec import CodecRuntimeWithNativeToolUse

        runtime = CodecRuntimeWithNativeToolUse()
        assert runtime._temperature_configured is False
        assert "temperature" not in runtime._llm_caller._build_request_kwargs(streaming=True)

    def test_star_agent_default_runtime_omits_temperature_on_stream(self):
        """The production construction path: STARAgent builds its runtime with no
        temperature, so streaming must not send one."""
        from dana.core.agent.star_agent import STARAgent

        agent = STARAgent(
            agent_type="i11-provenance",
            auto_register=False,
            enable_web_search=False,
            enable_skills=False,
            enable_code_execution=False,
        )
        runtime = agent._runtime
        assert runtime._temperature_configured is False
        assert "temperature" not in runtime._llm_caller._build_request_kwargs(streaming=True)

    def test_runtime_default_temperature_is_not_sent_on_the_stream_wire(self):
        """End to end through the runtime's own caller: the I11 user-visible case.

        A default-constructed runtime streams through its LLMCaller; nothing may
        put the runtime's internal default on the wire.
        """
        sent: list[dict] = []

        class _StreamingLLM:
            async def stream(self, messages, tools=None, **kwargs):
                sent.append(kwargs)
                yield LLMStreamChunk(type="text_delta", content="hi")
                yield LLMStreamChunk(type="response", response=_response("hi"))

        runtime = DefaultRuntime(llm=_StreamingLLM())

        async def _drive():
            async for _ in runtime._llm_caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
                pass

        asyncio.run(_drive())

        assert sent and "temperature" not in sent[0], "the runtime default must not become a wire policy"

    def test_runtime_explicit_zero_temperature_reaches_the_stream_wire(self):
        """The same end-to-end path, with the caller actually asking for 0."""
        sent: list[dict] = []

        class _StreamingLLM:
            async def stream(self, messages, tools=None, **kwargs):
                sent.append(kwargs)
                yield LLMStreamChunk(type="text_delta", content="hi")
                yield LLMStreamChunk(type="response", response=_response("hi"))

        runtime = DefaultRuntime(llm=_StreamingLLM(), temperature=0)

        async def _drive():
            async for _ in runtime._llm_caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
                pass

        asyncio.run(_drive())

        assert sent[0]["temperature"] == 0, "an explicit 0 through the runtime must reach the wire"


class TestBoundedRetryAndDeadline:
    @patch("time.sleep")
    def test_attempt_count_is_bounded_and_capped_backoff_is_used(self, mock_sleep):
        caller = LLMCaller(llm=MagicMock(), max_retries=2, base_delay=1.0)
        caller._llm.chat_response_sync.side_effect = ProviderError("rate limit exceeded")

        with pytest.raises(ProviderError):
            caller.call_llm([])

        assert caller._llm.chat_response_sync.call_count == 3
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0], "exponential backoff must be used"

    @patch("time.sleep")
    def test_backoff_is_capped_by_retry_max_delay(self, mock_sleep):
        caller = LLMCaller(llm=MagicMock(), max_retries=4, base_delay=1.0, retry_max_delay=3.0)
        caller._llm.chat_response_sync.side_effect = ProviderError("503 service unavailable")

        with pytest.raises(ProviderError):
            caller.call_llm([])

        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0, 3.0, 3.0], "growth must cap at retry_max_delay"

    @patch("time.sleep")
    def test_deadline_stops_the_retry_sequence(self, mock_sleep):
        """A pathological retry sequence cannot outlive its wall-clock budget."""
        caller = LLMCaller(llm=MagicMock(), max_retries=5, base_delay=10.0, deadline_seconds=0.2)
        caller._llm.chat_response_sync.side_effect = ProviderError("rate limit exceeded")

        start = time.monotonic()
        with pytest.raises(LLMTimeoutError, match="deadline"):
            caller.call_llm([])
        elapsed = time.monotonic() - start

        assert elapsed < 1.0
        assert caller._llm.chat_response_sync.call_count == 1, "must not start an attempt it cannot pay for"

    @patch("time.sleep")
    def test_deadline_spans_failover_providers(self, mock_sleep):
        """The budget is per logical request, so failover cannot reset it."""
        primary = MagicMock()
        primary.chat_response_sync.side_effect = ProviderError("503 service unavailable")
        caller = LLMCaller(
            llm=primary,
            fallback_providers=[ProviderConfig(provider="openai")],
            max_retries=1,
            base_delay=5.0,
            deadline_seconds=0.05,
        )

        with pytest.raises(LLMTimeoutError):
            caller.call_llm([])

        assert primary.chat_response_sync.call_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancellation_during_backoff_raises_cancelled_error(self):
        """A cancelled session must not wait out the delay."""
        cancel_event = asyncio.Event()
        llm = MagicMock()
        llm.chat_response = AsyncMock(side_effect=ProviderError("rate limit exceeded"))
        caller = LLMCaller(
            llm=llm,
            max_retries=3,
            base_delay=30.0,
            cancel_event_getter=lambda: cancel_event,
            deadline_seconds=0,
        )

        async def _cancel_soon():
            await asyncio.sleep(0.01)
            cancel_event.set()

        task = asyncio.create_task(_cancel_soon())
        with pytest.raises(asyncio.CancelledError):
            await caller.call_llm_async([])
        await task

        assert llm.chat_response.call_count == 1, "backoff sleep must be interrupted, not completed"


class TestActualOperationIsBudgetBound:
    """The deadline must bound the awaited provider call, not just the backoff."""

    @pytest.mark.asyncio
    async def test_never_returning_async_chat_hits_the_deadline(self):
        """A provider that accepts the request and never answers must not hang the turn."""

        class _HangingLLM:
            def __init__(self):
                self.calls = 0

            async def chat_response(self, messages, **kwargs):
                self.calls += 1
                await asyncio.Event().wait()  # never completes

        llm = _HangingLLM()
        caller = LLMCaller(llm=llm, max_retries=0, deadline_seconds=0.2)

        start = time.monotonic()
        with pytest.raises(LLMTimeoutError, match="deadline"):
            await caller.call_llm_async([LLMMessage(role="user", content="hi")])
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"call must be cut off by the budget, took {elapsed:.1f}s"
        assert llm.calls == 1, "a timed-out attempt must not be retried after consuming the budget"

    @pytest.mark.asyncio
    async def test_async_attempt_receives_the_remaining_budget(self):
        """The provider is told the actual remaining budget, not the class default."""
        seen: list[float | None] = []

        class _RecordingLLM:
            async def chat_response(self, messages, **kwargs):
                seen.append(kwargs.get("timeout"))
                return _response("ok")

        caller = LLMCaller(llm=_RecordingLLM(), deadline_seconds=42.0)
        await caller.call_llm_async([LLMMessage(role="user", content="hi")])

        assert seen and seen[0] is not None
        assert 0 < seen[0] <= 42.0, "provider must receive the remaining budget"

    @patch("time.sleep")
    def test_sync_attempt_receives_the_remaining_budget(self, mock_sleep):
        """Sync path must honour the budget via the supported per-attempt timeout."""
        seen: list[float | None] = []

        class _RecordingLLM:
            def chat_response_sync(self, messages, **kwargs):
                seen.append(kwargs.get("timeout"))
                return _response("ok")

        caller = LLMCaller(llm=_RecordingLLM(), deadline_seconds=42.0)
        caller.call_llm([LLMMessage(role="user", content="hi")])

        assert seen and seen[0] is not None
        assert 0 < seen[0] <= 42.0

    def test_never_returning_sync_chat_hits_the_deadline(self):
        """A stalled sync provider must not hold the calling thread past the budget."""
        from dana.common.llm.llm import LLM
        from dana.common.llm.types import LLMProvider

        class _HangingProvider(LLMProvider):
            model = "hang"

            async def chat(self, messages, **kwargs):
                await asyncio.Event().wait()  # never completes
                return _response()  # pragma: no cover

        llm = LLM(provider=_HangingProvider())
        caller = LLMCaller(llm=llm, max_retries=0, deadline_seconds=0.3)

        start = time.monotonic()
        with pytest.raises(LLMTimeoutError):
            caller.call_llm([LLMMessage(role="user", content="hi")])
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"sync call must be cut off by the budget, took {elapsed:.1f}s"

    @pytest.mark.asyncio
    async def test_deadline_is_shared_across_retry_attempts(self):
        """A hanging attempt consumes the shared budget; the retry cannot restart it."""
        budget = 0.3
        llm = MagicMock()
        llm.chat_response = AsyncMock(side_effect=ProviderError("rate limit exceeded"))
        caller = LLMCaller(
            llm=llm,
            max_retries=5,
            base_delay=10.0,
            deadline_seconds=budget,
        )

        start = time.monotonic()
        with pytest.raises(LLMTimeoutError):
            await caller.call_llm_async([LLMMessage(role="user", content="hi")])
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, "the whole retry sequence shares one budget"
        assert llm.chat_response.call_count == 1, "backoff cannot be paid for, so no second attempt"

    @pytest.mark.asyncio
    async def test_stalled_stream_after_a_delta_times_out_without_replay(self):
        """A stream that stops delivering mid-turn must time out and never be re-issued.

        The wait after the first frame is the idle bound, not the request deadline
        (a healthy but slow stream may legitimately run past it), so the bound is
        pinned here instead of waiting out the default.
        """

        class _StallingLLM:
            def __init__(self):
                self.attempts = 0

            async def stream(self, messages, tools=None, **kwargs):
                self.attempts += 1
                yield LLMStreamChunk(type="text_delta", content="partial")
                await asyncio.Event().wait()  # stalls forever

        llm = _StallingLLM()
        caller = LLMCaller(llm=llm, stream_max_retries=3, deadline_seconds=0.25)

        seen: list[str] = []
        with patch.object(LLMCaller, "_stream_idle_timeout", return_value=0.1):
            with pytest.raises(LLMTimeoutError):
                async for chunk in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
                    seen.append(chunk.content)

        assert seen == ["partial"]
        assert llm.attempts == 1, "output was already delivered, so it must not be replayed"

    @pytest.mark.asyncio
    async def test_stalled_stream_before_the_first_frame_times_out(self):
        """A provider that never sends the first byte must not hang the turn."""

        class _NeverStartsLLM:
            def __init__(self):
                self.attempts = 0

            async def stream(self, messages, tools=None, **kwargs):
                self.attempts += 1
                await asyncio.Event().wait()
                yield  # pragma: no cover

        llm = _NeverStartsLLM()
        caller = LLMCaller(llm=llm, stream_max_retries=0, deadline_seconds=0.2)

        start = time.monotonic()
        with pytest.raises(LLMTimeoutError):
            async for _ in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
                pass
        assert time.monotonic() - start < 2.0


class TestStreamBounds:
    """I10/I12: one stalled attempt must stay retriable, a healthy stream must not
    be cut off mid-turn."""

    @pytest.mark.asyncio
    async def test_stalled_attempt_is_capped_by_the_provider_timeout(self):
        """A hung connection must not eat the whole 900s budget.

        The per-attempt wait is min(remaining budget, provider timeout), so the
        configured stream retry still has budget to run once the stalled attempt
        is cut off.
        """

        class _HangingThenSucceedingLLM:
            def __init__(self):
                self.attempts = 0

            async def stream(self, messages, tools=None, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    await asyncio.Event().wait()  # never yields a frame
                yield LLMStreamChunk(type="text_delta", content="recovered")

        llm = _HangingThenSucceedingLLM()
        caller = LLMCaller(llm=llm, stream_max_retries=1, base_delay=0.0, deadline_seconds=5.0)

        seen: list[str] = []
        with patch.object(LLMCaller, "_provider_timeout", return_value=0.3):
            async for chunk in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
                seen.append(chunk.content)

        assert seen == ["recovered"], "the stalled attempt must be cut off, not consume the budget"
        assert llm.attempts == 2, "the retry must run inside the remaining budget"

    def test_attempt_timeout_never_exceeds_the_provider_timeout(self):
        """min(remaining, provider timeout); an exhausted budget is not inflated."""
        caller = LLMCaller(llm=MagicMock(), deadline_seconds=900.0)

        assert caller._provider_timeout(caller._llm) == float(LLMProvider.DEFAULT_TIMEOUT_SECONDS)
        assert caller._attempt_timeout(0.0, caller._llm) is None, "disabled budget keeps the provider timeout"

        deadline = time.monotonic() + 10_000.0
        assert caller._attempt_timeout(deadline, caller._llm) == float(LLMProvider.DEFAULT_TIMEOUT_SECONDS)

        soon = time.monotonic() + 5.0
        remaining = caller._attempt_timeout(soon, caller._llm)
        assert remaining is not None and 0 < remaining <= 5.0, "a short budget must not be rounded up"

    @pytest.mark.asyncio
    async def test_healthy_stream_outlives_the_request_deadline(self):
        """After the first frame the deadline no longer applies: only idleness does."""
        frames = 3

        class _SlowButHealthyLLM:
            async def stream(self, messages, tools=None, **kwargs):
                # First frame lands inside the deadline, then each later gap is
                # shorter than the idle bound while their sum far exceeds the
                # logical-request deadline, which must not cut the stream off.
                yield LLMStreamChunk(type="text_delta", content="f0")
                for i in range(1, frames):
                    await asyncio.sleep(0.05)
                    yield LLMStreamChunk(type="text_delta", content=f"f{i}")
                yield LLMStreamChunk(type="response", response=_response("".join(f"f{i}" for i in range(frames))))

        caller = LLMCaller(llm=_SlowButHealthyLLM(), deadline_seconds=0.05)
        start = time.monotonic()
        chunks = [c async for c in caller.call_llm_stream([LLMMessage(role="user", content="hi")])]
        elapsed = time.monotonic() - start

        assert [c.content for c in chunks[:-1]] == ["f0", "f1", "f2"]
        assert chunks[-1].type == "response"
        assert elapsed > 0.05, "the stream legitimately ran past the deadline while making progress"

    @pytest.mark.asyncio
    async def test_stream_that_stops_delivering_after_the_deadline_still_times_out(self):
        """Idle frames are bounded even when the request deadline has long passed."""

        class _StallingLLM:
            async def stream(self, messages, tools=None, **kwargs):
                yield LLMStreamChunk(type="text_delta", content="first")
                await asyncio.Event().wait()

        caller = LLMCaller(llm=_StallingLLM(), stream_max_retries=0, deadline_seconds=0.05)
        with patch.object(LLMCaller, "_stream_idle_timeout", return_value=0.1):
            start = time.monotonic()
            with pytest.raises(LLMTimeoutError):
                async for _ in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
                    pass
        assert time.monotonic() - start < 2.0

    @pytest.mark.asyncio
    async def test_first_frame_is_bounded_even_with_the_budget_disabled(self):
        """No deadline does not mean no bound: the first frame still cannot hang."""

        class _NeverStartsLLM:
            async def stream(self, messages, tools=None, **kwargs):
                await asyncio.Event().wait()
                yield  # pragma: no cover

        caller = LLMCaller(llm=_NeverStartsLLM(), stream_max_retries=0, deadline_seconds=0)
        start = time.monotonic()
        with patch.object(LLMCaller, "_stream_idle_timeout", return_value=0.1):
            with pytest.raises(LLMTimeoutError):
                async for _ in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
                    pass
        assert time.monotonic() - start < 2.0

    @pytest.mark.asyncio
    async def test_closing_the_consumer_releases_the_provider_stream(self):
        """A consumer that stops early must not leave the provider stream open."""
        closed = {"n": 0}

        class _LLM:
            async def stream(self, messages, tools=None, **kwargs):
                try:
                    yield LLMStreamChunk(type="text_delta", content="one")
                    yield LLMStreamChunk(type="text_delta", content="two")
                finally:
                    # GeneratorExit/close reaches here, which is what releases the
                    # real SDK response in the provider implementation.
                    closed["n"] += 1

        caller = LLMCaller(llm=_LLM())
        agen = caller.call_llm_stream([LLMMessage(role="user", content="hi")])
        assert (await agen.__anext__()).content == "one"
        await agen.aclose()

        assert closed["n"] == 1, "the abandoned provider stream must be released"


class TestTypedErrorsAtTheConsumerBoundary:
    @pytest.mark.asyncio
    async def test_ptl_survives_the_facade_and_reaches_reactive_compaction(self):
        """The real path: provider raises PTL → facade must not wrap it away."""

        class _Provider(LLMProvider):
            model = "test-model"

            async def chat(self, messages, **kwargs):
                raise PromptTooLongError("prompt is too long")

            async def stream(self, messages, tools=None, **kwargs):
                raise PromptTooLongError("prompt is too long")
                yield  # pragma: no cover - makes this an async generator

        llm = LLM(provider=_Provider())

        with pytest.raises(PromptTooLongError):
            await llm.chat_response([LLMMessage(role="user", content="hi")])

        with pytest.raises(PromptTooLongError):
            async for _ in llm.stream([LLMMessage(role="user", content="hi")]):
                pass

    @pytest.mark.asyncio
    async def test_unknown_provider_exception_still_wrapped_as_provider_error(self):
        class _Provider(LLMProvider):
            model = "test-model"

            async def chat(self, messages, **kwargs):
                raise RuntimeError("boom")

            async def stream(self, messages, tools=None, **kwargs):
                raise RuntimeError("boom")
                yield  # pragma: no cover

        llm = LLM(provider=_Provider())

        with pytest.raises(ProviderError, match="Chat failed with custom: boom"):
            await llm.chat_response([LLMMessage(role="user", content="hi")])

        with pytest.raises(ProviderError, match="Stream failed with custom: boom"):
            async for _ in llm.stream([LLMMessage(role="user", content="hi")]):
                pass

    def test_terminal_response_is_attached_to_provider_error(self):
        """A failed turn's accounting must survive with the exception."""
        resp = _response("partial")
        err = ProviderError("stream failed", terminal_response=resp)
        assert err.terminal_response is resp
        # Existing construction and subclassing stay compatible.
        assert ProviderError("plain").terminal_response is None
        assert PromptTooLongError("too long").terminal_response is None
        assert isinstance(ProviderError("x"), LLMError)

    def test_ptl_is_never_classified_transient(self):
        """Retrying an identical oversized payload cannot succeed."""
        assert LLMCaller._is_transient_error(PromptTooLongError("prompt is too long")) is False

    @pytest.mark.asyncio
    async def test_foreign_timeline_compaction_runs_off_the_event_loop(self):
        """A timeline with only the sync seam is offloaded, not run inline.

        Foreign/legacy timelines expose ``reactive_compact`` alone; blocking the
        event loop on it would stall every other task, so it runs in a worker and
        the caller joins that worker rather than orphaning it.
        """
        calls: list[int] = []

        class _Timeline:
            _consecutive_compact_failures = 0
            _compaction_disabled = False
            _circuit_opened_at = None

            class _Cfg:
                enable_reactive_compact = True

            _compressed_config = _Cfg()

            def reactive_compact(self, attempt: int) -> None:
                calls.append(attempt)

        class _Agent:
            object_id = "a1"
            agent_type = "test"
            _timeline = _Timeline()

        class _LLM:
            def __init__(self):
                self.n = 0

            async def chat_response(self, messages, **kwargs):
                self.n += 1
                if self.n == 1:
                    raise PromptTooLongError("too long")
                return _response("compacted")

        llm = _LLM()
        caller = LLMCaller(llm=llm, agent_getter=lambda: _Agent(), native_tools_getter=lambda: None)

        import threading

        loop_thread = threading.get_ident()
        compact_threads: list[int] = []
        original = _Timeline.reactive_compact

        def _recording_compact(self, attempt):
            compact_threads.append(threading.get_ident())
            return original(self, attempt)

        with patch.object(_Timeline, "reactive_compact", _recording_compact), patch("asyncio.sleep", new=AsyncMock()):
            result = await caller._invoke_llm_async(llm, [LLMMessage(role="user", content="x")])

        assert result.content == "compacted"
        assert calls == [1]
        assert compact_threads and compact_threads[0] != loop_thread, "compaction must run in a worker thread"

    @pytest.mark.asyncio
    async def test_async_compaction_seam_is_preferred_and_runs_on_the_caller_thread(self):
        """A timeline exposing ``reactive_compact_async`` is awaited, not offloaded.

        This is the ownership fix: ``asyncio.to_thread`` cannot be stopped, so an
        abandoned await would leave a worker mutating a released timeline. The
        async seam keeps every mutation on this awaiting thread, so the caller's
        deadline reaches the provider and no late writer survives.
        """
        seen: list[tuple[str, int]] = []

        class _Timeline:
            _consecutive_compact_failures = 0
            _compaction_disabled = False
            _circuit_opened_at = None

            class _Cfg:
                enable_reactive_compact = True

            _compressed_config = _Cfg()

            def reactive_compact(self, attempt: int) -> None:  # pragma: no cover - must not be used
                seen.append(("sync", attempt))

            async def reactive_compact_async(self, attempt: int) -> None:
                import threading

                seen.append((f"async:{threading.get_ident()}", attempt))

        timeline = _Timeline()

        class _Agent:
            object_id = "a1"
            agent_type = "test"
            _timeline = timeline

        class _LLM:
            def __init__(self):
                self.n = 0

            async def chat_response(self, messages, **kwargs):
                self.n += 1
                if self.n == 1:
                    raise PromptTooLongError("too long")
                return _response("compacted")

        import threading

        loop_thread = threading.get_ident()
        caller = LLMCaller(llm=_LLM(), agent_getter=lambda: _Agent(), native_tools_getter=lambda: None)
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await caller._invoke_llm_async(caller._llm, [LLMMessage(role="user", content="x")])

        assert result.content == "compacted"
        assert len(seen) == 1, "exactly one compaction ran"
        marker, attempt = seen[0]
        assert marker == f"async:{loop_thread}", "the async seam must be awaited on the caller's own thread"
        assert attempt == 1

    @pytest.mark.asyncio
    async def test_cancelled_await_joins_a_foreign_sync_compaction(self):
        """Cancellation must not orphan an un-stoppable foreign compaction.

        The worker owns its mutation to completion; the caller waits for it and
        only then re-raises, so nothing keeps writing to a released timeline.
        """
        import threading

        started = threading.Event()
        finished = threading.Event()

        class _Timeline:
            _consecutive_compact_failures = 0
            _compaction_disabled = False
            _circuit_opened_at = None

            class _Cfg:
                enable_reactive_compact = True

            _compressed_config = _Cfg()

            def reactive_compact(self, attempt: int) -> None:
                started.set()
                time.sleep(0.05)
                finished.set()

        class _Agent:
            object_id = "a1"
            agent_type = "test"
            _timeline = _Timeline()

        class _LLM:
            async def chat_response(self, messages, **kwargs):
                raise PromptTooLongError("too long")

        caller = LLMCaller(llm=_LLM(), agent_getter=lambda: _Agent(), native_tools_getter=lambda: None)
        task = asyncio.ensure_future(caller._invoke_llm_async(caller._llm, [LLMMessage(role="user", content="x")]))
        deadline = time.monotonic() + 2.0
        while not started.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert started.is_set(), "the foreign compaction must have started on a worker thread"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert finished.is_set(), "the foreign compaction worker must be joined, not orphaned"

    @pytest.mark.asyncio
    async def test_cancelled_await_preserves_owner_cancellation_when_the_worker_fails(self):
        """A worker failure during the join must not replace owner cancellation.

        The worker is joined after cancellation, so its exception surfaces while
        the join is in flight. That failure is the worker's, not this call's
        result: the owner's ``CancelledError`` must still be what escapes, and the
        worker's exception must be retrieved rather than left unretrieved.
        """
        import gc
        import threading

        started = threading.Event()
        released = threading.Event()

        class _Timeline:
            _consecutive_compact_failures = 0
            _compaction_disabled = False
            _circuit_opened_at = None

            class _Cfg:
                enable_reactive_compact = True

            _compressed_config = _Cfg()

            def reactive_compact(self, attempt: int) -> None:
                started.set()
                # Hold the worker in flight so cancellation lands before it fails.
                released.wait(2.0)
                raise ValueError("worker blew up after the owner left")

        class _Agent:
            object_id = "a1"
            agent_type = "test"
            _timeline = _Timeline()

        class _LLM:
            async def chat_response(self, messages, **kwargs):
                raise PromptTooLongError("too long")

        loop = asyncio.get_running_loop()
        unretrieved: list[str] = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, ctx: unretrieved.append(str(ctx.get("message", ""))))
        try:
            caller = LLMCaller(llm=_LLM(), agent_getter=lambda: _Agent(), native_tools_getter=lambda: None)
            task = asyncio.ensure_future(caller._invoke_llm_async(caller._llm, [LLMMessage(role="user", content="x")]))
            deadline = time.monotonic() + 2.0
            while not started.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            assert started.is_set(), "the foreign compaction must have started on a worker thread"

            task.cancel()
            # Let the CancelledError be delivered into the coroutine (entering the
            # join) before the worker is allowed to fail, so the case under test is
            # "failure observed while joining", not "failure racing the cancel".
            for _ in range(5):
                await asyncio.sleep(0)
            released.set()
            with pytest.raises(asyncio.CancelledError):
                await task

            await asyncio.sleep(0)  # let any late worker teardown run
            gc.collect()
        finally:
            loop.set_exception_handler(previous)

        assert not any("never retrieved" in m for m in unretrieved), unretrieved

    @pytest.mark.asyncio
    async def test_foreign_sync_compaction_failure_propagates_without_cancellation(self):
        """Without an owner cancellation the worker's own failure is the result.

        The join only exists to keep a cancelled caller from orphaning the worker;
        the direct path must keep propagating whatever the compaction raises.
        """

        class _Timeline:
            _consecutive_compact_failures = 0
            _compaction_disabled = False
            _circuit_opened_at = None

            class _Cfg:
                enable_reactive_compact = True

            _compressed_config = _Cfg()

            def reactive_compact(self, attempt: int) -> None:
                raise ValueError("direct compaction failure")

        class _Agent:
            object_id = "a1"
            agent_type = "test"
            _timeline = _Timeline()

        class _LLM:
            async def chat_response(self, messages, **kwargs):
                raise PromptTooLongError("too long")

        caller = LLMCaller(llm=_LLM(), agent_getter=lambda: _Agent(), native_tools_getter=lambda: None)
        with pytest.raises(ValueError, match="direct compaction failure"):
            await caller._invoke_llm_async(caller._llm, [LLMMessage(role="user", content="x")])


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_llm_aclose_closes_the_provider_client(self):
        closed = {"n": 0}

        class _Provider(LLMProvider):
            model = "test-model"

            async def chat(self, messages, **kwargs):  # pragma: no cover
                return _response()

            async def stream(self, messages, tools=None, **kwargs):  # pragma: no cover
                yield LLMStreamChunk(type="response", response=_response())

            async def aclose(self) -> None:
                closed["n"] += 1

        llm = LLM(provider=_Provider())
        await llm.aclose()
        assert closed["n"] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["llm", "embedder"])
    async def test_shutdown_waits_for_retired_provider_cleanup(self, kind):
        from dana.common.llm.embedder import Embedder

        started = asyncio.Event()
        release = asyncio.Event()
        closed = asyncio.Event()

        class OldProvider(LLMProvider):
            model = "old"
            supports_embeddings = True

            async def aclose(self):
                started.set()
                await release.wait()
                closed.set()

        class NewProvider(LLMProvider):
            model = "new"
            supports_embeddings = True

            async def aclose(self):
                await started.wait()
                release.set()

        facade = (LLM if kind == "llm" else Embedder)(provider=OldProvider())
        try:
            with patch(f"dana.common.llm.{kind}.create_provider", return_value=NewProvider()):
                facade.switch_provider("test-provider")
            await asyncio.wait_for(facade.aclose(), 1)
            assert closed.is_set(), "shutdown returned with a retired connection pool still open"
        finally:
            release.set()
            await asyncio.wait_for(closed.wait(), 1)

    def test_switch_provider_closes_the_retired_client(self):
        """Switching must not leak the abandoned connection pool."""
        events: list[str] = []

        class _OldProvider(LLMProvider):
            model = "old"

            async def aclose(self) -> None:
                events.append("old-closed")

        class _NewProvider(LLMProvider):
            model = "new"

            async def aclose(self) -> None:  # pragma: no cover - not retired yet
                events.append("new-closed")

        llm = LLM(provider=_OldProvider())
        with patch("dana.common.llm.llm.create_provider", return_value=_NewProvider()):
            llm.switch_provider("anthropic", model="claude-3")

        assert events == ["old-closed"]
        assert llm.provider_name == "anthropic"

    def test_switch_provider_installs_new_provider_even_when_close_fails(self):
        class _BadClose(LLMProvider):
            model = "old"

            async def aclose(self) -> None:
                raise RuntimeError("close exploded")

        class _NewProvider(LLMProvider):
            model = "new"

        llm = LLM(provider=_BadClose())
        with patch("dana.common.llm.llm.create_provider", return_value=_NewProvider()):
            llm.switch_provider("anthropic", model="claude-3")

        assert isinstance(llm.provider, _NewProvider)

    @pytest.mark.asyncio
    async def test_provider_aclose_is_a_noop_without_a_closable_client(self):
        class _Bare(LLMProvider):
            model = "bare"

        await _Bare().aclose()

    @pytest.mark.asyncio
    async def test_provider_aclose_swallows_client_close_failure(self):
        class _Client:
            async def aclose(self):
                raise RuntimeError("httpx gone")

        class _Provider(LLMProvider):
            model = "p"

            def __init__(self):
                self.client = _Client()

        await _Provider().aclose()


class TestResolveLlmOwnership:
    def test_existing_agent_client_is_reused_not_overwritten(self):
        """Reading the agent's cached client must not construct a second LLM."""
        existing = object()

        class _Agent:
            _llm_client = existing
            object_id = "a1"
            agent_type = "t"

        caller = LLMCaller(agent_getter=lambda: _Agent(), provider="openai", model="gpt-4o")
        with patch("dana.core.llm.llm_caller.LLM") as mock_llm:
            resolved = caller._resolve_llm()

        assert resolved is existing
        mock_llm.assert_not_called()

    def test_lazy_client_is_cached_on_the_agent(self):
        """When the agent has no client yet, the resolved one is adopted by both sides."""

        class _Agent:
            _llm_client = None
            object_id = "a1"
            agent_type = "t"

            @property
            def llm_client(self):
                return self._llm_client

            @llm_client.setter
            def llm_client(self, value):
                self._llm_client = value

        agent = _Agent()
        caller = LLMCaller(agent_getter=lambda: agent, provider="openai", model="gpt-4o")
        sentinel = object()
        with patch("dana.core.llm.llm_caller.LLM", return_value=sentinel) as mock_llm:
            resolved = caller._resolve_llm()

        assert resolved is sentinel
        assert agent._llm_client is sentinel
        assert caller._llm is sentinel
        mock_llm.assert_called_once_with(provider="openai", model="gpt-4o")
