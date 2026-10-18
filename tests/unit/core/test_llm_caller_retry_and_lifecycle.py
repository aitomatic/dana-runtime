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
from dana.core.llm.llm_caller import LLMCaller, ProviderConfig


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
    """Streaming must send the same caller policy the non-streaming path sends."""

    @pytest.mark.asyncio
    async def test_stream_forwards_temperature_and_max_tokens(self):
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
        assert sent["temperature"] == 0.35
        assert sent["max_tokens"] == 777
        assert sent["json_mode"] is False
        assert llm.sent_tools[0] is None

    @pytest.mark.asyncio
    async def test_unset_controls_are_omitted_not_sent_as_null(self):
        llm = _StreamingLLM(fail_after=0, exc=ProviderError("x"), succeed_on_attempt=1)
        caller = LLMCaller(llm=llm, temperature=None, max_tokens=None)

        async for _ in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
            pass

        sent = llm.sent_kwargs[0]
        assert "temperature" not in sent
        assert "max_tokens" not in sent


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
        """A stream that stalls mid-turn must time out, not hang or re-issue."""

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
    async def test_reactive_compaction_runs_off_the_event_loop(self):
        """Async PTL recovery must not block the loop with synchronous compaction."""
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
