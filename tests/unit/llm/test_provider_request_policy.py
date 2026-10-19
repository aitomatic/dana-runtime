"""Consumer-visible request-policy tests for the provider stack.

These assert what a caller observes on the wire, not that a value was merely
forwarded: streaming and non-streaming must send the same caller policy, and the
adapter must still translate that policy per capability.

No network: the outbound body is captured from a stub client so the assertions are
about the request that would have been sent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from dana.common.llm.types import LLMMessage


async def _async_iter(items):
    for item in items:
        yield item


class _RecordingTransport(httpx.AsyncBaseTransport):
    """Capture outbound request bodies so the test can assert on the wire payload."""

    def __init__(self, responses: list[httpx.Response] | None = None):
        self.bodies: list[dict] = []
        self.requests: list[httpx.Request] = []
        self._responses = list(responses or [])

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json

        self.requests.append(request)
        if request.content:
            try:
                self.bodies.append(json.loads(request.content))
            except json.JSONDecodeError:
                self.bodies.append({"__raw__": request.content.decode("utf-8", "replace")})
        if self._responses:
            return self._responses.pop(0)
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "choices": [], "model": "m"})


def _sse_response(*payloads: dict) -> httpx.Response:
    """Build a streamed SSE response body from JSON payloads."""
    import json

    body = "".join(f"data: {json.dumps(p)}\n\n" for p in payloads) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _make_openai_provider(model="gpt-4o", transport=None):
    """Real OpenAIProvider against a stub transport, so the request body is real."""
    import openai

    from dana.common.llm.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key", model=model)
    provider.client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        http_client=httpx.AsyncClient(transport=transport or _RecordingTransport()),
        max_retries=0,
    )
    return provider


class TestStreamingRequestControlsReachTheWire:
    """The streaming request must carry what the non-streaming request carries."""

    @pytest.mark.asyncio
    async def test_chat_completions_stream_carries_temperature_max_tokens_and_json_mode(self):
        transport = _RecordingTransport([_sse_response({"id": "c1", "model": "gpt-4o", "choices": []})])
        provider = _make_openai_provider("gpt-4o", transport)

        async for _ in provider._stream_chat_completions(
            [LLMMessage(role="user", content="hi")],
            temperature=0.25,
            max_tokens=512,
            json_mode=True,
        ):
            pass

        body = transport.bodies[0]
        assert body["temperature"] == 0.25
        assert body["max_tokens"] == 512
        assert body["response_format"] == {"type": "json_object"}
        assert body["stream"] is True

    @pytest.mark.asyncio
    async def test_chat_completions_stream_matches_nonstream_controls(self):
        """Same caller policy → same control keys on both paths."""
        stream_transport = _RecordingTransport([_sse_response({"id": "c1", "model": "gpt-4o", "choices": []})])
        provider = _make_openai_provider("gpt-4o", stream_transport)
        async for _ in provider._stream_chat_completions(
            [LLMMessage(role="user", content="hi")], temperature=0.7, max_tokens=64, json_mode=True
        ):
            pass
        stream_body = stream_transport.bodies[0]

        chat_transport = _RecordingTransport(
            [
                httpx.Response(
                    200,
                    json={
                        "id": "c2",
                        "object": "chat.completion",
                        "model": "gpt-4o",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                )
            ]
        )
        provider.client = _make_openai_provider("gpt-4o", chat_transport).client
        await provider._chat_via_chat_completions([LLMMessage(role="user", content="hi")], temperature=0.7, max_tokens=64, json_mode=True)
        chat_body = chat_transport.bodies[0]

        for key in ("temperature", "max_tokens", "response_format"):
            assert stream_body[key] == chat_body[key], f"{key} diverged between stream and non-stream"

    @pytest.mark.asyncio
    async def test_chat_completions_stream_requests_include_usage(self):
        """Without include_usage the API sends no usage frame at all."""
        transport = _RecordingTransport([_sse_response({"id": "c1", "model": "gpt-4o", "choices": []})])
        provider = _make_openai_provider("gpt-4o", transport)

        async for _ in provider._stream_chat_completions([LLMMessage(role="user", content="hi")]):
            pass

        assert transport.bodies[0]["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_model_restriction_rename_fires_on_stream_path(self):
        """gpt-5* renames max_tokens→max_completion_tokens on the stream path too."""
        transport = _RecordingTransport([_sse_response({"id": "c1", "model": "gpt-5", "choices": []})])
        provider = _make_openai_provider("gpt-5", transport)

        async for _ in provider._stream_chat_completions([LLMMessage(role="user", content="hi")], temperature=0.5, max_tokens=256):
            pass

        body = transport.bodies[0]
        assert body["max_completion_tokens"] == 256
        assert "max_tokens" not in body
        # temperature is restricted to {1} for gpt-5 and must be dropped, not sent as 0.5
        assert "temperature" not in body

    @pytest.mark.asyncio
    async def test_responses_stream_carries_controls(self):
        transport = _RecordingTransport(
            [_sse_response({"type": "response.completed", "response": {"id": "r1", "model": "gpt-5", "output": [], "status": "completed"}})]
        )
        provider = _make_openai_provider("gpt-5", transport)

        async for _ in provider._stream_responses([LLMMessage(role="user", content="hi")], temperature=1, max_tokens=128):
            pass

        body = transport.bodies[0]
        # The Responses API does not accept max_completion_tokens (the SDK raises
        # TypeError); its token ceiling is max_output_tokens.
        assert body["max_output_tokens"] == 128
        assert "max_completion_tokens" not in body
        assert body["temperature"] == 1
        assert body["stream"] is True


class TestAnthropicStreamControls:
    """Anthropic stream must honour the caller's controls, not its own defaults."""

    @pytest.mark.asyncio
    async def test_anthropic_stream_uses_caller_max_tokens_and_temperature(self):
        from dana.common.llm.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.model = "claude-3-sonnet"
        captured: dict = {}

        class _FakeStreamCtx:
            current_message_snapshot = None

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def __aiter__(self):
                return _async_iter([])

        def _capture(**kwargs):
            captured.update(kwargs)
            return _FakeStreamCtx()

        provider.client = MagicMock()
        provider.client.messages.stream = _capture

        async for _ in provider.stream([LLMMessage(role="user", content="hi")], temperature=0.3, max_tokens=512):
            pass

        assert captured["max_tokens"] == 512, "stream must not silently default to 32000"
        assert captured["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_anthropic_stream_emits_terminal_carrier_with_usage_and_stop_reason(self):
        """A completed Anthropic stream must report real usage/cache/stop_reason."""
        from dana.common.llm.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.model = "claude-3-sonnet"

        final = MagicMock()
        final.id = "msg_1"
        final.model = "claude-3-sonnet"
        final.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "hello"
        final.content = [text_block]

        class _Usage:
            input_tokens = 30
            output_tokens = 7
            cache_read_input_tokens = 12

        final.usage = _Usage()

        class _FakeStreamCtx:
            current_message_snapshot = final

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def __aiter__(self):
                return _async_iter([])

        provider.client = MagicMock()
        provider.client.messages.stream = lambda **kwargs: _FakeStreamCtx()

        chunks = [c async for c in provider.stream([LLMMessage(role="user", content="hi")])]

        terminal = [c for c in chunks if c.type == "response"]
        assert terminal, "a completed stream must emit a terminal carrier"
        carrier = terminal[-1].response
        assert carrier.finish_reason == "end_turn"
        assert carrier.usage["prompt_tokens"] == 30
        assert carrier.usage["cached_tokens"] == 12
        assert carrier.response_id == "msg_1"


class TestGeminiStreamControls:
    @pytest.mark.asyncio
    async def test_gemini_stream_carries_output_controls_and_terminal_carrier(self):
        from dana.common.llm.providers.gemini import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider.model = "gemini-2.5-flash"
        captured: dict = {}

        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        part = MagicMock()
        part.text = "hi"
        part.function_call = None
        candidate.content.parts = [part]

        chunk = MagicMock()
        chunk.candidates = [candidate]
        chunk.model_version = "gemini-2.5-flash-001"
        chunk.usage_metadata.prompt_token_count = 11
        chunk.usage_metadata.candidates_token_count = 4
        chunk.usage_metadata.total_token_count = 15

        async def _stream(**kwargs):
            captured.update(kwargs)
            return _async_iter([chunk])

        provider.client = MagicMock()
        provider.client.aio.models.generate_content_stream = _stream

        chunks = [c async for c in provider.stream([LLMMessage(role="user", content="hi")], temperature=0.4, max_tokens=77)]

        config = captured["config"]
        assert config.temperature == 0.4
        assert config.max_output_tokens == 77

        terminal = [c for c in chunks if c.type == "response"]
        assert terminal, "gemini stream must emit a terminal carrier"
        carrier = terminal[-1].response
        assert carrier.usage == {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
        assert carrier.finish_reason == "STOP"
