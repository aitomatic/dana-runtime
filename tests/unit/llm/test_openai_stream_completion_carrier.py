"""Wire-contract tests for the Responses API streaming terminal carrier.

Covers the B1 bridge: ``_stream_responses`` must surface the
``response.completed`` payload as a terminal ``LLMStreamChunk(type="response")``
so reasoning items, the response id, real usage, and finish status survive a
streamed turn, none of which are expressible as deltas.

Also covers the matching request-side control: ``include=["reasoning.encrypted_content"]``
is requested only when replay is enabled and not already known unsupported, with
the same one-shot rejection fallback the non-streaming path uses.

The fault/status regressions use the installed SDK with HTTP MockTransport; no network.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import AsyncOpenAI, BadRequestError
from openai.types.responses import ResponseErrorEvent
import pytest

from dana.common.llm.providers.openai_compatible_base import (
    _replay_enabled,
)
from dana.common.llm.types import LLMMessage, PromptTooLongError, ProviderError


_REASONING_ITEM = {
    "type": "reasoning",
    "id": "rs_stream_1",
    "summary": [{"type": "summary_text", "text": "Considered the options."}],
    "encrypted_content": "enc_stream_blob",
}


async def _async_iter(items):
    for item in items:
        yield item


def _make_provider(model="gpt-5", include_unsupported=False):
    from dana.common.llm.providers.openai_compatible_base import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.model = model
    provider.client = MagicMock()
    provider._use_responses_api = True
    provider._include_unsupported = include_unsupported
    return provider


def _make_wire_provider(*, events=(), response=None):
    def handler(request):
        if response is not None:
            return httpx.Response(200, json=response)
        body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    provider = _make_provider(model="gpt-5-nano")
    provider.base_url = "https://offline.invalid/v1"
    provider.client = AsyncOpenAI(
        api_key="offline",
        base_url=provider.base_url,
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return provider


def _make_summary_part(text):
    part = MagicMock()
    part.type = "summary_text"
    part.text = text
    part.model_dump = MagicMock(return_value={"type": "summary_text", "text": text})
    return part


def _make_reasoning_item(item_id="rs_stream_1", summary_texts=("Considered the options.",), encrypted="enc_stream_blob"):
    item = MagicMock()
    item.type = "reasoning"
    item.id = item_id
    item.summary = [_make_summary_part(t) for t in summary_texts]
    item.encrypted_content = encrypted
    item.model_dump = MagicMock(
        return_value={
            "type": "reasoning",
            "id": item_id,
            "summary": [{"type": "summary_text", "text": t} for t in summary_texts],
            "encrypted_content": encrypted,
        }
    )
    return item


def _make_message_item(text="done"):
    item = MagicMock()
    item.type = "message"
    content = MagicMock()
    content.type = "output_text"
    content.text = text
    item.content = [content]
    return item


def _make_function_call_item(call_id="call_1", name="get_weather", arguments='{"city": "Tokyo"}'):
    item = MagicMock()
    item.type = "function_call"
    item.call_id = call_id
    item.name = name
    item.arguments = arguments
    return item


def _make_completed_response(
    output,
    response_id="resp_stream_1",
    status="completed",
    usage=None,
    model="gpt-5-nano",
    incomplete_details=None,
):
    response = MagicMock()
    response.output = output
    response.id = response_id
    response.status = status
    response.model = model
    response.usage = usage
    response.incomplete_details = incomplete_details
    return response


def _make_completed_event(response):
    event = MagicMock()
    event.type = "response.completed"
    event.response = response
    return event


def _make_delta_event(event_type, **kwargs):
    event = MagicMock()
    event.type = event_type
    for key, value in kwargs.items():
        setattr(event, key, value)
    return event


def _make_usage(prompt=120, completion=40, total=160, cached=64, reasoning=12):
    usage = MagicMock()
    usage.input_tokens = prompt
    usage.output_tokens = completion
    usage.total_tokens = total
    details_in = MagicMock()
    details_in.cached_tokens = cached
    usage.input_tokens_details = details_in
    details_out = MagicMock()
    details_out.reasoning_tokens = reasoning
    usage.output_tokens_details = details_out
    return usage


async def _collect(provider, events):
    provider.client.responses.create = AsyncMock(return_value=_async_iter(events))
    return [chunk async for chunk in provider._stream_responses([LLMMessage(content="hi", role="user")])]


class TestTerminalCarrierFromCompletedEvent:
    """`response.completed` becomes a terminal metadata chunk."""

    @pytest.mark.asyncio
    async def test_reasoning_items_and_response_id_reach_the_carrier(self):
        provider = _make_provider()
        reasoning = _make_reasoning_item()
        completion = _make_completed_response([reasoning, _make_message_item("done")])

        chunks = await _collect(
            provider,
            [
                _make_delta_event("response.output_text.delta", delta="done"),
                _make_completed_event(completion),
            ],
        )

        assert [chunk.type for chunk in chunks] == ["text_delta", "response"]
        terminal = chunks[-1]
        assert terminal.response is not None
        assert terminal.response.response_id == "resp_stream_1"
        assert terminal.response.reasoning_items == [_REASONING_ITEM]
        # Delta content is repeated on the carrier for consumers that read only it.
        assert terminal.response.content == "done"

    @pytest.mark.asyncio
    async def test_usage_and_model_are_reported_not_invented(self):
        provider = _make_provider()
        completion = _make_completed_response([_make_message_item("ok")], usage=_make_usage())

        chunks = await _collect(provider, [_make_completed_event(completion)])

        response = chunks[-1].response
        assert response.model == "gpt-5-nano"
        assert response.usage == {
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "total_tokens": 160,
            "cached_tokens": 64,
        }
        assert response.reasoning_tokens == 12
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_incomplete_status_maps_finish_reason(self):
        provider = _make_provider()
        details = MagicMock()
        details.reason = "max_output_tokens"
        completion = _make_completed_response(
            [_make_message_item("truncated")],
            status="incomplete",
            incomplete_details=details,
        )

        chunks = await _collect(provider, [_make_completed_event(completion)])

        assert chunks[-1].response.finish_reason == "length"

    @pytest.mark.asyncio
    async def test_tool_call_completion_reports_tool_calls_finish_reason(self):
        provider = _make_provider()
        completion = _make_completed_response([_make_function_call_item()])

        chunks = await _collect(provider, [_make_completed_event(completion)])

        response = chunks[-1].response
        assert response.finish_reason == "tool_calls"
        assert response.tool_calls is not None
        assert response.tool_calls[0].function.name == "get_weather"
        assert response.tool_calls[0].id == "call_1"

    @pytest.mark.asyncio
    async def test_no_reasoning_item_leaves_field_none(self):
        provider = _make_provider()
        completion = _make_completed_response([_make_message_item("plain")])

        chunks = await _collect(provider, [_make_completed_event(completion)])

        assert chunks[-1].response.reasoning_items is None
        assert chunks[-1].response.reasoning_content is None

    @pytest.mark.asyncio
    async def test_summary_text_flows_to_reasoning_content(self):
        provider = _make_provider()
        completion = _make_completed_response([_make_reasoning_item(summary_texts=("first", "second")), _make_message_item("x")])

        chunks = await _collect(provider, [_make_completed_event(completion)])

        assert chunks[-1].response.reasoning_content == "firstsecond"

    @pytest.mark.asyncio
    async def test_missing_completed_payload_yields_no_carrier(self):
        """A completed event without a payload must not fabricate a response."""
        provider = _make_provider()
        event = MagicMock()
        event.type = "response.completed"
        event.response = None

        chunks = await _collect(provider, [event])

        assert chunks == []

    @pytest.mark.asyncio
    async def test_text_and_tool_deltas_still_emitted_before_carrier(self):
        """The carrier is additive: existing delta behavior is unchanged."""
        provider = _make_provider()
        call_item = _make_function_call_item()
        completion = _make_completed_response([call_item])

        chunks = await _collect(
            provider,
            [
                _make_delta_event("response.output_text.delta", delta="thinking out loud"),
                _make_delta_event("response.output_item.done", item=call_item),
                _make_delta_event("response.reasoning_summary_text.delta", delta="hm"),
                _make_completed_event(completion),
            ],
        )

        assert [chunk.type for chunk in chunks] == ["text_delta", "tool_use", "thinking", "response"]
        assert chunks[0].content == "thinking out loud"
        assert chunks[1].tool_call == {"id": "call_1", "name": "get_weather", "input": {"city": "Tokyo"}}
        assert chunks[2].content == "hm"


class TestResponsesStreamFaultEvents:
    """`response.failed` / `error` must not read as a clean short answer.

    The installed OpenAI SDK does not raise for these events — iterating the
    stream simply stops — so a failed turn previously ended with
    ``terminal_response is None`` and the consumer synthesized success.
    """

    @pytest.mark.asyncio
    async def test_response_failed_raises_typed_error_with_terminal_evidence(self):
        from dana.common.llm.types import ProviderError

        provider = _make_provider()
        error = MagicMock()
        error.code = "server_error"
        error.message = "upstream exploded"
        failed = _make_completed_response([], response_id="resp_failed_1", status="failed", usage=_make_usage())
        failed.error = error
        event = _make_delta_event("response.failed", response=failed)

        with pytest.raises(ProviderError) as excinfo:
            await _collect(provider, [event])

        # A failed turn still has real accounting worth journaling.
        assert excinfo.value.terminal_response is not None
        assert excinfo.value.terminal_response.response_id == "resp_failed_1"
        assert excinfo.value.terminal_response.usage["prompt_tokens"] == 120

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("code", "error_type"),
        [("internal_error", ProviderError), ("context_length_exceeded", PromptTooLongError)],
    )
    async def test_sdk_error_frame_raises_typed_error(self, code, error_type):
        event = ResponseErrorEvent(type="error", message="stream broke", code=code, sequence_number=2)
        provider = _make_wire_provider(
            events=[
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "partial",
                    "sequence_number": 1,
                    "logprobs": [],
                },
                event.model_dump(),
            ]
        )
        observed = []
        try:
            with pytest.raises(error_type, match="stream broke") as excinfo:
                async for chunk in provider._stream_responses([LLMMessage(content="hi", role="user")]):
                    observed.append(chunk)
            assert [chunk.type for chunk in observed] == ["text_delta"]
            assert observed[0].content == "partial"
            assert excinfo.value.terminal_response is None
        finally:
            await provider.aclose()

    @pytest.mark.asyncio
    async def test_context_overflow_failure_maps_to_prompt_too_long(self):
        """A failed turn caused by an oversized prompt must still trigger compaction."""
        from dana.common.llm.types import PromptTooLongError

        provider = _make_provider()
        error = MagicMock()
        error.code = "context_length_exceeded"
        error.message = "prompt is too long"
        failed = _make_completed_response([], status="failed")
        failed.error = error

        with pytest.raises(PromptTooLongError):
            await _collect(provider, [_make_delta_event("response.failed", response=failed)])

    @pytest.mark.asyncio
    async def test_incomplete_emits_honest_carrier_instead_of_raising(self):
        """Truncation is a real non-success status, not an infra failure to retry."""
        provider = _make_provider()
        details = MagicMock()
        details.reason = "max_output_tokens"
        truncated = _make_completed_response(
            [_make_message_item("partial")], status="incomplete", incomplete_details=details, usage=_make_usage()
        )

        chunks = await _collect(provider, [_make_delta_event("response.incomplete", response=truncated)])

        assert len(chunks) == 1
        carrier = chunks[0].response
        # Must not be reported as a clean stop, and must carry the usage evidence.
        assert carrier.finish_reason == "length"
        assert carrier.usage["prompt_tokens"] == 120
        assert carrier.response_id == "resp_stream_1"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("streaming", [False, True])
    @pytest.mark.parametrize(
        ("status", "reason", "expected"),
        [
            ("incomplete", "max_output_tokens", "length"),
            ("incomplete", "content_filter", "incomplete"),
            ("completed", None, "tool_calls"),
        ],
    )
    async def test_sdk_terminal_status_precedes_tool_calls(self, monkeypatch, streaming, status, reason, expected):
        monkeypatch.setenv("LLM_REASONING_REPLAY", "1")
        response = {
            "id": "resp_wire_1",
            "object": "response",
            "created_at": 0,
            "status": status,
            "error": None,
            "incomplete_details": {"reason": reason} if reason else None,
            "model": "gpt-5-nano",
            "output": [
                _REASONING_ITEM,
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Tokyo"}',
                    "status": status,
                },
            ],
            "usage": {
                "input_tokens": 120,
                "output_tokens": 40,
                "total_tokens": 160,
                "input_tokens_details": {"cached_tokens": 64},
                "output_tokens_details": {"reasoning_tokens": 12},
            },
            "parallel_tool_calls": True,
            "tools": [],
            "tool_choice": "auto",
        }
        provider = _make_wire_provider(
            events=[{"type": f"response.{status}", "response": response, "sequence_number": 1}],
            response=None if streaming else response,
        )
        try:
            messages = [LLMMessage(content="hi", role="user")]
            if streaming:
                chunks = [chunk async for chunk in provider._stream_responses(messages)]
                assert [chunk.type for chunk in chunks] == ["response"]
                terminal = chunks[0].response
            else:
                terminal = await provider._chat_via_responses(messages)
            assert terminal.finish_reason == expected
            assert terminal.response_id == "resp_wire_1"
            assert terminal.tool_calls[0].id == "call_1"
            assert terminal.tool_calls[0].function.name == "get_weather"
            assert terminal.usage == {
                "prompt_tokens": 120,
                "completion_tokens": 40,
                "total_tokens": 160,
                "cached_tokens": 64,
            }
            assert terminal.reasoning_tokens == 12
            assert terminal.reasoning_items == [_REASONING_ITEM]
        finally:
            await provider.aclose()

    @pytest.mark.asyncio
    async def test_stream_ending_without_terminal_event_yields_no_carrier(self):
        """Silent truncation must stay unknown, never become a fabricated stop."""
        provider = _make_provider()
        chunks = await _collect(provider, [_make_delta_event("response.output_text.delta", delta="hi")])

        assert [c.type for c in chunks] == ["text_delta"]


class TestStreamingIncludeControl:
    """`include` is requested only when replay can use it."""

    @pytest.mark.asyncio
    async def test_include_requested_when_replay_enabled(self, monkeypatch):
        monkeypatch.delenv("LLM_REASONING_REPLAY", raising=False)
        provider = _make_provider()

        await _collect(provider, [_make_completed_event(_make_completed_response([_make_message_item()]))])

        kwargs = provider.client.responses.create.await_args.kwargs
        assert kwargs.get("include") == ["reasoning.encrypted_content"]

    @pytest.mark.asyncio
    async def test_include_skipped_when_replay_disabled(self, monkeypatch):
        monkeypatch.setenv("LLM_REASONING_REPLAY", "0")
        provider = _make_provider()

        await _collect(provider, [_make_completed_event(_make_completed_response([_make_message_item()]))])

        kwargs = provider.client.responses.create.await_args.kwargs
        assert "include" not in kwargs

    @pytest.mark.asyncio
    async def test_include_skipped_after_prior_rejection(self, monkeypatch):
        monkeypatch.delenv("LLM_REASONING_REPLAY", raising=False)
        provider = _make_provider(include_unsupported=True)

        await _collect(provider, [_make_completed_event(_make_completed_response([_make_message_item()]))])

        kwargs = provider.client.responses.create.await_args.kwargs
        assert "include" not in kwargs

    @pytest.mark.asyncio
    async def test_caller_supplied_include_entries_are_preserved(self, monkeypatch):
        monkeypatch.delenv("LLM_REASONING_REPLAY", raising=False)
        provider = _make_provider()
        provider.client.responses.create = AsyncMock(return_value=_async_iter([]))

        async for _ in provider._stream_responses(
            [LLMMessage(content="hi", role="user")],
            include=["message.output_text.logprobs"],
        ):
            pass

        kwargs = provider.client.responses.create.await_args.kwargs
        assert kwargs["include"] == ["message.output_text.logprobs", "reasoning.encrypted_content"]

    @pytest.mark.asyncio
    async def test_rejected_include_degrades_once_and_sticks(self, monkeypatch):
        """A rejecting account must not fail the turn; later calls skip the flag."""
        monkeypatch.delenv("LLM_REASONING_REPLAY", raising=False)
        provider = _make_provider()

        request = httpx.Request("POST", "https://example.com/responses")
        bad = BadRequestError(
            message="Unknown parameter: include[reasoning.encrypted_content].",
            response=httpx.Response(400, request=request),
            body={"error": {"message": "Unknown parameter: include[reasoning.encrypted_content]."}},
        )
        ok_events = [_make_completed_event(_make_completed_response([_make_message_item("fine")]))]
        provider.client.responses.create = AsyncMock(side_effect=[bad, _async_iter(ok_events)])

        chunks = [chunk async for chunk in provider._stream_responses([LLMMessage(content="hi", role="user")])]

        assert provider._include_unsupported is True
        assert chunks[-1].response.content == "fine"
        assert provider.client.responses.create.await_count == 2
        first = provider.client.responses.create.call_args_list[0].kwargs
        second = provider.client.responses.create.call_args_list[1].kwargs
        assert first.get("include") == ["reasoning.encrypted_content"]
        assert "include" not in second

    @pytest.mark.asyncio
    async def test_unrelated_400_still_propagates(self, monkeypatch):
        monkeypatch.delenv("LLM_REASONING_REPLAY", raising=False)
        provider = _make_provider()

        request = httpx.Request("POST", "https://example.com/responses")
        bad = BadRequestError(
            message="invalid model parameter",
            response=httpx.Response(400, request=request),
            body={"error": {"message": "invalid model parameter"}},
        )
        provider.client.responses.create = AsyncMock(side_effect=bad)

        with pytest.raises(BadRequestError):
            async for _ in provider._stream_responses([LLMMessage(content="hi", role="user")]):
                pass

        assert provider._include_unsupported is False


class TestReplayGateSharedWithInclude:
    """The include decision reads the same gate the splice path uses."""

    @pytest.mark.parametrize("falsy", ["0", "false", "FALSE", "off", "no", " 0 "])
    def test_falsy_values_disable(self, monkeypatch, falsy):
        monkeypatch.setenv("LLM_REASONING_REPLAY", falsy)
        assert _replay_enabled() is False

    @pytest.mark.parametrize("truthy", ["1", "true", "on", "yes", ""])
    def test_truthy_and_empty_values_enable(self, monkeypatch, truthy):
        monkeypatch.setenv("LLM_REASONING_REPLAY", truthy)
        assert _replay_enabled() is True

    def test_unset_defaults_to_enabled(self, monkeypatch):
        monkeypatch.delenv("LLM_REASONING_REPLAY", raising=False)
        assert _replay_enabled() is True
