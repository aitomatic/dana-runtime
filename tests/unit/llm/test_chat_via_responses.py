"""Tests for non-streaming chat() routed through the Responses API.

Covers:
- Reasoning summary text → LLMResponse.reasoning_content
- Output message text → LLMResponse.content
- Function-call items → tool_calls (Pydantic shape, downstream parser compatible)
- finish_reason mapping (stop / tool_calls / length / incomplete)
- Usage shape (input/output_tokens → prompt/completion_tokens)
- Routing: _should_use_responses_api() decides which path chat() takes
- json_mode → text.format passthrough
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dana.common.llm.types import LLMMessage


def _make_provider(model="gpt-5", use_responses_api=None):
    from dana.common.llm.providers.openai_compatible_base import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.model = model
    provider.client = MagicMock()
    provider._use_responses_api = use_responses_api
    return provider


def _make_output_item(item_type, **fields):
    item = MagicMock()
    item.type = item_type
    for k, v in fields.items():
        setattr(item, k, v)
    return item


def _make_summary_part(text):
    s = MagicMock()
    s.text = text
    return s


def _make_message_content(text, content_type="output_text"):
    c = MagicMock()
    c.type = content_type
    c.text = text
    return c


def _make_response(output, status="completed", usage=None, model="gpt-5", incomplete_details=None):
    resp = MagicMock()
    resp.output = output
    resp.status = status
    resp.model = model
    resp.usage = usage
    resp.incomplete_details = incomplete_details
    return resp


def _make_usage(input_tokens=10, output_tokens=20, total_tokens=30, reasoning_tokens=None, cached_tokens=None):
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.total_tokens = total_tokens
    if reasoning_tokens is not None:
        details = MagicMock()
        details.reasoning_tokens = reasoning_tokens
        usage.output_tokens_details = details
    else:
        usage.output_tokens_details = None
    if cached_tokens is not None:
        in_details = MagicMock()
        in_details.cached_tokens = cached_tokens
        usage.input_tokens_details = in_details
    else:
        usage.input_tokens_details = None
    return usage


class TestReasoningContent:
    @pytest.mark.asyncio
    async def test_reasoning_summary_populates_reasoning_content(self):
        provider = _make_provider()
        reasoning_item = _make_output_item(
            "reasoning",
            summary=[_make_summary_part("First step. "), _make_summary_part("Second step.")],
        )
        message_item = _make_output_item("message", content=[_make_message_content("answer")])
        provider.client.responses.create = AsyncMock(return_value=_make_response([reasoning_item, message_item]))

        resp = await provider._chat_via_responses([LLMMessage(role="user", content="hi")])

        assert resp.reasoning_content == "First step. Second step."
        assert resp.content == "answer"

    @pytest.mark.asyncio
    async def test_no_reasoning_item_keeps_reasoning_content_none(self):
        provider = _make_provider()
        message_item = _make_output_item("message", content=[_make_message_content("answer")])
        provider.client.responses.create = AsyncMock(return_value=_make_response([message_item]))

        resp = await provider._chat_via_responses([LLMMessage(role="user", content="hi")])

        assert resp.reasoning_content is None
        assert resp.content == "answer"


class TestContentExtraction:
    @pytest.mark.asyncio
    async def test_skips_non_output_text_blocks(self):
        provider = _make_provider()
        # Mix output_text with a refusal-like block (different type) — only output_text counts.
        message_item = _make_output_item(
            "message",
            content=[
                _make_message_content("real answer"),
                _make_message_content("ignored", content_type="refusal"),
            ],
        )
        provider.client.responses.create = AsyncMock(return_value=_make_response([message_item]))

        resp = await provider._chat_via_responses([LLMMessage(role="user", content="hi")])
        assert resp.content == "real answer"


class TestToolCalls:
    @pytest.mark.asyncio
    async def test_function_call_item_becomes_chat_completions_tool_call(self):
        provider = _make_provider()
        fc = _make_output_item(
            "function_call",
            call_id="call_abc",
            name="get_weather",
            arguments='{"city":"Tokyo"}',
        )
        provider.client.responses.create = AsyncMock(return_value=_make_response([fc]))

        resp = await provider._chat_via_responses([LLMMessage(role="user", content="hi")])

        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        # Downstream parser uses attribute access — must work without modification.
        tc = resp.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"city":"Tokyo"}'
        assert resp.finish_reason == "tool_calls"


class TestFinishReason:
    @pytest.mark.asyncio
    async def test_completed_no_tools_is_stop(self):
        provider = _make_provider()
        msg = _make_output_item("message", content=[_make_message_content("done")])
        provider.client.responses.create = AsyncMock(return_value=_make_response([msg], status="completed"))
        resp = await provider._chat_via_responses([LLMMessage(role="user", content="hi")])
        assert resp.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_incomplete_max_output_tokens_is_length(self):
        provider = _make_provider()
        msg = _make_output_item("message", content=[_make_message_content("partial")])
        details = MagicMock()
        details.reason = "max_output_tokens"
        provider.client.responses.create = AsyncMock(return_value=_make_response([msg], status="incomplete", incomplete_details=details))
        resp = await provider._chat_via_responses([LLMMessage(role="user", content="hi")])
        assert resp.finish_reason == "length"

    @pytest.mark.asyncio
    async def test_incomplete_other_reason_is_incomplete(self):
        provider = _make_provider()
        msg = _make_output_item("message", content=[_make_message_content("partial")])
        details = MagicMock()
        details.reason = "content_filter"
        provider.client.responses.create = AsyncMock(return_value=_make_response([msg], status="incomplete", incomplete_details=details))
        resp = await provider._chat_via_responses([LLMMessage(role="user", content="hi")])
        assert resp.finish_reason == "incomplete"


class TestUsageMapping:
    @pytest.mark.asyncio
    async def test_input_output_tokens_become_prompt_completion(self):
        provider = _make_provider()
        msg = _make_output_item("message", content=[_make_message_content("done")])
        usage = _make_usage(input_tokens=42, output_tokens=58, total_tokens=100, reasoning_tokens=15, cached_tokens=7)
        provider.client.responses.create = AsyncMock(return_value=_make_response([msg], usage=usage))

        resp = await provider._chat_via_responses([LLMMessage(role="user", content="hi")])

        assert resp.usage == {
            "prompt_tokens": 42,
            "completion_tokens": 58,
            "total_tokens": 100,
            "cached_tokens": 7,
        }
        assert resp.reasoning_tokens == 15


class TestRouting:
    """Verify chat() actually dispatches to _chat_via_responses for reasoning models."""

    @pytest.mark.asyncio
    async def test_chat_routes_to_responses_when_supported(self):
        provider = _make_provider(model="gpt-5", use_responses_api=True)
        msg = _make_output_item("message", content=[_make_message_content("via responses")])
        provider.client.responses.create = AsyncMock(return_value=_make_response([msg]))
        # Sentinel: chat completions must NOT be called.
        provider.client.chat.completions.create = AsyncMock(side_effect=AssertionError("wrong path"))

        resp = await provider.chat([LLMMessage(role="user", content="hi")])

        assert resp.content == "via responses"
        provider.client.responses.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_chat_routes_to_chat_completions_when_disabled(self):
        provider = _make_provider(model="gpt-5", use_responses_api=False)
        # Sentinel: responses must NOT be called.
        provider.client.responses.create = AsyncMock(side_effect=AssertionError("wrong path"))
        # Build a Chat Completions response.
        message = MagicMock()
        message.content = "via chat completions"
        message.tool_calls = None
        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop"
        cc_response = MagicMock()
        cc_response.choices = [choice]
        cc_response.model = "gpt-5"
        cc_response.usage = None
        provider.client.chat.completions.create = AsyncMock(return_value=cc_response)

        resp = await provider.chat([LLMMessage(role="user", content="hi")])

        assert resp.content == "via chat completions"
        provider.client.chat.completions.create.assert_awaited_once()


class TestJsonMode:
    @pytest.mark.asyncio
    async def test_json_mode_maps_to_text_format(self):
        provider = _make_provider()
        msg = _make_output_item("message", content=[_make_message_content("{}")])
        provider.client.responses.create = AsyncMock(return_value=_make_response([msg]))

        await provider._chat_via_responses([LLMMessage(role="user", content="give me json")], json_mode=True)

        assert provider.client.responses.create.await_args is not None
        call_kwargs = provider.client.responses.create.await_args.kwargs
        assert call_kwargs["text"] == {"format": {"type": "json_object"}}
        # json_mode must NOT leak through as an unrecognized API kwarg.
        assert "json_mode" not in call_kwargs
