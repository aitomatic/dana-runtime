"""Tests for OpenAI provider streaming and API detection."""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import AsyncOpenAI
import pytest
from structlog.testing import capture_logs

from dana.common.llm.types import LLMMessage, LLMStreamChunk


# --- Helpers for building mock objects ---


def _make_chat_chunk(content=None, tool_calls=None, finish_reason=None):
    """Build a mock Chat Completions streaming chunk.

    ``usage`` is explicitly None because the real SDK only populates it on the
    dedicated usage frame; a bare ``MagicMock`` would auto-create a truthy
    attribute and the provider would treat every delta as terminal.
    """
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _make_empty_chunk():
    """Build a mock chunk with empty choices (the usage-frame shape)."""
    chunk = MagicMock()
    chunk.choices = []
    # A real usage frame carries usage; a bare heartbeat frame does not. Tests that
    # mean "nothing was reported" must say so explicitly.
    chunk.usage = None
    return chunk


def _make_tool_call_delta(index, tc_id=None, name=None, arguments=None):
    """Build a mock incremental tool call delta."""
    tc = MagicMock()
    tc.index = index
    tc.id = tc_id

    if name or arguments:
        tc.function = MagicMock()
        tc.function.name = name
        tc.function.arguments = arguments
    else:
        tc.function = None

    return tc


def _make_responses_event(event_type, **kwargs):
    """Build a mock Responses API streaming event."""
    event = MagicMock()
    event.type = event_type
    for key, value in kwargs.items():
        setattr(event, key, value)
    return event


def _make_message_item(text):
    """Build a Responses API ``message`` output item carrying output_text."""
    item = MagicMock()
    item.type = "message"
    content = MagicMock()
    content.type = "output_text"
    content.text = text
    item.content = [content]
    return item


async def _async_iter(items):
    """Wrap a list as an async iterator."""
    for item in items:
        yield item


def _create_provider(model="gpt-4o", use_responses_api=None):
    """Create an OpenAICompatibleProvider with mocked client."""
    from dana.common.llm.providers.openai_compatible_base import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.model = model
    provider.client = MagicMock()
    provider._use_responses_api = use_responses_api
    return provider


# --- Chat Completions streaming tests ---


class TestStreamChatCompletions:
    """Test _stream_chat_completions() with mocked API."""

    @pytest.mark.asyncio
    async def test_text_stream_yields_text_deltas(self):
        provider = _create_provider()
        chunks = [
            _make_chat_chunk(content="Hello"),
            _make_chat_chunk(content=" world"),
            _make_chat_chunk(finish_reason="stop"),
        ]
        provider.client.chat.completions.create = AsyncMock(return_value=_async_iter(chunks))

        results = []
        async for chunk in provider._stream_chat_completions([LLMMessage(content="Hi", role="user")]):
            results.append(chunk)

        deltas = [c for c in results if c.type != "response"]
        assert deltas == [
            LLMStreamChunk(type="text_delta", content="Hello"),
            LLMStreamChunk(type="text_delta", content=" world"),
        ]
        # The reported finish_reason must reach the consumer, not be dropped.
        carriers = [c for c in results if c.type == "response"]
        assert len(carriers) == 1
        assert carriers[0].response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_tool_call_stream_accumulates_deltas(self):
        provider = _create_provider()
        chunks = [
            _make_chat_chunk(tool_calls=[_make_tool_call_delta(0, tc_id="call_1", name="get_weather", arguments="")]),
            _make_chat_chunk(tool_calls=[_make_tool_call_delta(0, arguments='{"city":')]),
            _make_chat_chunk(tool_calls=[_make_tool_call_delta(0, arguments=' "Paris"}')]),
            _make_chat_chunk(finish_reason="tool_calls"),
        ]
        provider.client.chat.completions.create = AsyncMock(return_value=_async_iter(chunks))

        results = []
        async for chunk in provider._stream_chat_completions([LLMMessage(content="Weather?", role="user")]):
            results.append(chunk)

        tool_chunks = [c for c in results if c.type == "tool_use"]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_call["id"] == "call_1"
        assert tool_chunks[0].tool_call["name"] == "get_weather"
        assert tool_chunks[0].tool_call["input"] == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_mixed_text_and_tool_calls(self):
        provider = _create_provider()
        chunks = [
            _make_chat_chunk(content="Let me check"),
            _make_chat_chunk(tool_calls=[_make_tool_call_delta(0, tc_id="call_1", name="search", arguments='{"q": "test"}')]),
            _make_chat_chunk(finish_reason="tool_calls"),
        ]
        provider.client.chat.completions.create = AsyncMock(return_value=_async_iter(chunks))

        results = []
        async for chunk in provider._stream_chat_completions([LLMMessage(content="Search", role="user")]):
            results.append(chunk)

        assert [c.type for c in results if c.type != "response"] == ["text_delta", "tool_use"]

    @pytest.mark.asyncio
    async def test_empty_choices_skipped(self):
        provider = _create_provider()
        chunks = [_make_empty_chunk(), _make_chat_chunk(content="Hi"), _make_empty_chunk()]
        provider.client.chat.completions.create = AsyncMock(return_value=_async_iter(chunks))

        results = []
        async for chunk in provider._stream_chat_completions([LLMMessage(content="Hi", role="user")]):
            results.append(chunk)

        assert [c.content for c in results if c.type == "text_delta"] == ["Hi"]
        # No finish reason or usage frame was ever sent, so no terminal is invented.
        assert not [c for c in results if c.type == "response"]

    @pytest.mark.asyncio
    async def test_invalid_tool_args_json_yields_empty_dict(self):
        provider = _create_provider()
        chunks = [
            _make_chat_chunk(tool_calls=[_make_tool_call_delta(0, tc_id="call_1", name="bad", arguments="not json{")]),
            _make_chat_chunk(finish_reason="tool_calls"),
        ]
        provider.client.chat.completions.create = AsyncMock(return_value=_async_iter(chunks))

        results = []
        async for chunk in provider._stream_chat_completions([LLMMessage(content="x", role="user")]):
            results.append(chunk)

        assert results[0].tool_call["input"] == {}

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        provider = _create_provider()
        chunks = [
            _make_chat_chunk(
                tool_calls=[
                    _make_tool_call_delta(0, tc_id="call_1", name="tool_a", arguments='{"a": 1}'),
                    _make_tool_call_delta(1, tc_id="call_2", name="tool_b", arguments='{"b": 2}'),
                ]
            ),
            _make_chat_chunk(finish_reason="tool_calls"),
        ]
        provider.client.chat.completions.create = AsyncMock(return_value=_async_iter(chunks))

        results = []
        async for chunk in provider._stream_chat_completions([LLMMessage(content="x", role="user")]):
            results.append(chunk)

        names = {c.tool_call["name"] for c in results if c.type == "tool_use"}
        assert names == {"tool_a", "tool_b"}

    @pytest.mark.asyncio
    async def test_post_loop_flush_on_stop_finish_reason(self):
        """Tool calls flushed even if finish_reason is 'stop' instead of 'tool_calls'."""
        provider = _create_provider()
        chunks = [
            _make_chat_chunk(tool_calls=[_make_tool_call_delta(0, tc_id="call_1", name="fn", arguments='{"x": 1}')]),
            _make_chat_chunk(finish_reason="stop"),
        ]
        provider.client.chat.completions.create = AsyncMock(return_value=_async_iter(chunks))

        results = []
        async for chunk in provider._stream_chat_completions([LLMMessage(content="x", role="user")]):
            results.append(chunk)

        tool_chunks = [c for c in results if c.type == "tool_use"]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_call["input"] == {"x": 1}

    @pytest.mark.asyncio
    async def test_stream_close_releases_the_http_response(self):
        """Closing the stream generator must release the connection, not leak it.

        A plain ``async for`` over an OpenAI SDK stream never calls the stream's
        ``__aexit__``, so the provider closes it explicitly. This is the path taken
        when the consuming task is cancelled or the generator is torn down.
        """
        provider = _create_provider()
        closed = {"n": 0}

        class _Stream:
            def __init__(self):
                self._items = iter([_make_chat_chunk(content="a"), _make_chat_chunk(content="b")])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._items)
                except StopIteration:
                    raise StopAsyncIteration from None

            async def close(self):
                closed["n"] += 1

        provider.client.chat.completions.create = AsyncMock(return_value=_Stream())

        agen = provider._stream_chat_completions([LLMMessage(content="Hi", role="user")])
        await agen.__anext__()
        await agen.aclose()

        assert closed["n"] == 1, "the abandoned response must be closed"

    @pytest.mark.asyncio
    async def test_stream_close_failure_does_not_mask_the_turn(self):
        provider = _create_provider()

        class _Stream:
            def __init__(self):
                self._items = iter([_make_chat_chunk(content="a")])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._items)
                except StopIteration:
                    raise StopAsyncIteration from None

            async def close(self):
                raise RuntimeError("socket already gone")

        provider.client.chat.completions.create = AsyncMock(return_value=_Stream())

        results = [c async for c in provider._stream_chat_completions([LLMMessage(content="Hi", role="user")])]

        assert [c.content for c in results if c.type == "text_delta"] == ["a"]


class TestProviderSdkCleanup:
    @pytest.mark.asyncio
    async def test_sdk_pool_close_failure_is_redacted(self):
        from dana.common.llm.llm import LLM

        class FailingPoolTransport(httpx.MockTransport):
            async def aclose(self):
                raise RuntimeError("SECRET_SENTINEL pool")

        http_client = httpx.AsyncClient(transport=FailingPoolTransport(lambda request: httpx.Response(200, json={})))
        provider = _create_provider()
        provider.client = AsyncOpenAI(api_key="offline", max_retries=0, http_client=http_client)

        with capture_logs() as logs:
            await LLM(provider=provider).aclose()

        warnings = [log for log in logs if log["event"] == "provider_client_close_failed"]
        assert len(warnings) == 1
        assert warnings[0]["error_type"] == "RuntimeError"
        assert "error" not in warnings[0]
        assert "SECRET_SENTINEL" not in json.dumps(logs)
        assert http_client.is_closed

    @pytest.mark.asyncio
    async def test_sdk_response_close_failure_is_redacted(self):
        class FailingResponseBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                event = {
                    "type": "response.output_text.delta",
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "partial",
                    "sequence_number": 1,
                    "logprobs": [],
                }
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()

            async def aclose(self):
                raise RuntimeError("SECRET_SENTINEL response")

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=FailingResponseBody(), headers={"content-type": "text/event-stream"})
            )
        )
        client = AsyncOpenAI(api_key="offline", base_url="https://offline.invalid/v1", max_retries=0, http_client=http_client)
        try:
            stream = await client.responses.create(model="gpt-5-nano", input="hi", stream=True)
            delta = await anext(stream)
            assert delta.type == "response.output_text.delta"
            assert delta.delta == "partial"
            with capture_logs() as logs:
                await _create_provider()._close_stream(stream)
            warnings = [log for log in logs if log["event"] == "stream_close_failed"]
            assert len(warnings) == 1
            assert warnings[0]["error_type"] == "RuntimeError"
            assert "error" not in warnings[0]
            assert "SECRET_SENTINEL" not in json.dumps(logs)
            assert stream.response.is_closed
        finally:
            await client.close()


# --- Responses API streaming tests ---


class TestStreamResponses:
    """Test _stream_responses() with mocked API."""

    @pytest.mark.asyncio
    async def test_text_delta_events(self):
        provider = _create_provider(model="gpt-5")
        completion = MagicMock()
        completion.output = [
            _make_message_item("Hello world"),
        ]
        completion.id = "resp_streaming_1"
        completion.status = "completed"
        completion.model = "gpt-5"
        completion.usage = None
        completion.incomplete_details = None

        events = [
            _make_responses_event("response.output_text.delta", delta="Hello"),
            _make_responses_event("response.output_text.delta", delta=" world"),
            _make_responses_event("response.completed", response=completion),
        ]
        provider.client.responses.create = AsyncMock(return_value=_async_iter(events))

        results = []
        async for chunk in provider._stream_responses([LLMMessage(content="Hi", role="user")]):
            results.append(chunk)

        # Two text deltas plus the terminal metadata carrier.
        assert len(results) == 3
        assert results[0] == LLMStreamChunk(type="text_delta", content="Hello")
        assert results[1] == LLMStreamChunk(type="text_delta", content=" world")
        assert results[2].type == "response"
        assert results[2].response.response_id == "resp_streaming_1"
        assert results[2].response.content == "Hello world"

    @pytest.mark.asyncio
    async def test_tool_call_from_output_item_done(self):
        provider = _create_provider(model="gpt-5")
        item = MagicMock()
        item.type = "function_call"
        item.call_id = "call_resp_1"
        item.name = "get_weather"
        item.arguments = '{"city": "Tokyo"}'

        events = [_make_responses_event("response.output_item.done", item=item)]
        provider.client.responses.create = AsyncMock(return_value=_async_iter(events))

        results = []
        async for chunk in provider._stream_responses([LLMMessage(content="Weather?", role="user")]):
            results.append(chunk)

        assert len(results) == 1
        assert results[0].type == "tool_use"
        assert results[0].tool_call["id"] == "call_resp_1"
        assert results[0].tool_call["name"] == "get_weather"
        assert results[0].tool_call["input"] == {"city": "Tokyo"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "event_type",
        # Real openai SDK event names. Old wrapper listened for "response.reasoning.delta"
        # which never fires; the SDK emits these two instead.
        ["response.reasoning_summary_text.delta", "response.reasoning_text.delta"],
    )
    async def test_reasoning_delta_yields_thinking(self, event_type):
        provider = _create_provider(model="o3")
        events = [_make_responses_event(event_type, delta="Let me think...")]
        provider.client.responses.create = AsyncMock(return_value=_async_iter(events))

        results = []
        async for chunk in provider._stream_responses([LLMMessage(content="Think", role="user")]):
            results.append(chunk)

        assert len(results) == 1
        assert results[0] == LLMStreamChunk(type="thinking", content="Let me think...")


# --- API detection tests ---


class TestAPIDetection:
    """Test _should_use_responses_api() routing logic."""

    def test_config_flag_true(self):
        provider = _create_provider(model="gpt-4o", use_responses_api=True)
        assert provider._should_use_responses_api() is True

    def test_config_flag_false(self):
        provider = _create_provider(model="gpt-5", use_responses_api=False)
        assert provider._should_use_responses_api() is False

    def test_gpt5_prefix_detection(self):
        provider = _create_provider(model="gpt-5-mini")
        assert provider._should_use_responses_api() is True

    def test_o3_prefix_detection(self):
        provider = _create_provider(model="o3-mini")
        assert provider._should_use_responses_api() is True

    def test_o4_prefix_detection(self):
        provider = _create_provider(model="o4-preview")
        assert provider._should_use_responses_api() is True

    def test_gpt4o_defaults_chat_completions(self):
        provider = _create_provider(model="gpt-4o")
        assert provider._should_use_responses_api() is False

    def test_unknown_model_defaults_chat_completions(self):
        provider = _create_provider(model="some-custom-model")
        assert provider._should_use_responses_api() is False

    def test_gpt35_defaults_chat_completions(self):
        provider = _create_provider(model="gpt-3.5-turbo")
        assert provider._should_use_responses_api() is False
