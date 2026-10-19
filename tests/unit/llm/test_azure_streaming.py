"""Tests for Azure provider streaming inheritance and request wire policy."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from dana.common.llm.providers.openai_compatible_base import OpenAICompatibleProvider
from dana.common.llm.types import LLMMessage


class _RecordingTransport(httpx.AsyncBaseTransport):
    """Capture outbound request bodies so a test can assert on the wire payload."""

    def __init__(self, responses: list[httpx.Response] | None = None):
        self.bodies: list[dict] = []
        self._responses = list(responses or [])

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
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
    body = "".join(f"data: {json.dumps(p)}\n\n" for p in payloads) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


class TestAzureStreamInheritsBase:
    """Verify AzureProvider inherits streaming from base."""

    def _make_provider(self, use_responses_api=None):
        """Create AzureProvider with mocked config_manager."""
        config_mock = MagicMock()
        config_mock.get_provider_api_key.return_value = "fake-key"
        config_mock.get_provider_base_url.return_value = "https://test.openai.azure.com"
        config_mock.get_provider_api_version.return_value = "2024-02-15-preview"
        config_mock.get_provider_config.return_value = {"use_responses_api": use_responses_api} if use_responses_api is not None else {}

        with patch("dana.common.llm.providers.azure.config_manager", config_mock):
            from dana.common.llm.providers.azure import AzureProvider

            return AzureProvider(api_key="fake-key", base_url="https://test.openai.azure.com")

    def test_inherits_from_openai_compatible(self):
        provider = self._make_provider()
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_has_stream_method(self):
        provider = self._make_provider()
        assert hasattr(provider, "stream")
        assert callable(provider.stream)

    def test_has_chat_completions_stream(self):
        provider = self._make_provider()
        assert hasattr(provider, "_stream_chat_completions")

    def test_has_responses_stream(self):
        provider = self._make_provider()
        assert hasattr(provider, "_stream_responses")

    def test_config_flag_set(self):
        provider = self._make_provider(use_responses_api=True)
        assert provider._use_responses_api is True

    def test_config_flag_none_by_default(self):
        provider = self._make_provider()
        assert provider._use_responses_api is None


class TestAzureStreamUsageGate:
    """C2: `stream_options.include_usage` is version-gated, not probed with a retry.

    Azure's default `api_version` here (`2024-02-15-preview`) predates the field, so
    sending it is an HTTP 400 that is never retried. Old deployments must stream
    without it; new ones still get usage accounting.
    """

    def _provider_with_version(self, api_version):
        config_mock = MagicMock()
        config_mock.get_provider_api_key.return_value = "fake-key"
        config_mock.get_provider_base_url.return_value = "https://test.openai.azure.com"
        config_mock.get_provider_api_version.return_value = api_version
        config_mock.get_provider_config.return_value = {}

        with patch("dana.common.llm.providers.azure.config_manager", config_mock):
            from dana.common.llm.providers.azure import AzureProvider

            return AzureProvider(api_key="fake-key", base_url="https://test.openai.azure.com", api_version=api_version)

    def test_default_version_does_not_support_stream_usage(self):
        provider = self._provider_with_version("2024-02-15-preview")
        assert provider._supports_stream_usage() is False

    @pytest.mark.parametrize("api_version", ["2024-08-01-preview", "2024-10-21", "2025-03-01-preview"])
    def test_new_versions_support_stream_usage(self, api_version):
        provider = self._provider_with_version(api_version)
        assert provider._supports_stream_usage() is True

    def test_missing_or_malformed_version_is_treated_as_unsupported(self):
        provider = self._provider_with_version("")
        assert provider._supports_stream_usage() is False
        provider = self._provider_with_version("2024")
        assert provider._supports_stream_usage() is False

    def test_base_provider_still_sends_include_usage(self):
        """OpenAI-compatible default stays opt-in, so usage accounting is unchanged."""
        from dana.common.llm.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        assert provider._supports_stream_usage() is True


class TestAzureChatStreamWire:
    """The Azure chat stream body must match the version's wire contract."""

    def _azure_provider(self, api_version):
        config_mock = MagicMock()
        config_mock.get_provider_api_key.return_value = "fake-key"
        config_mock.get_provider_base_url.return_value = "https://test.openai.azure.com"
        config_mock.get_provider_api_version.return_value = api_version
        config_mock.get_provider_config.return_value = {}

        with patch("dana.common.llm.providers.azure.config_manager", config_mock):
            from dana.common.llm.providers.azure import AzureProvider

            return AzureProvider(api_key="fake-key", base_url="https://test.openai.azure.com", api_version=api_version)

    @pytest.mark.asyncio
    async def test_old_api_version_omits_stream_options(self):
        import httpx
        from openai import AsyncAzureOpenAI

        provider = self._azure_provider("2024-02-15-preview")
        transport = _RecordingTransport([_sse_response({"id": "c1", "model": "gpt-4o", "choices": []})])
        provider.client = AsyncAzureOpenAI(
            api_key="fake-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=transport),
        )

        async for _ in provider._stream_chat_completions([LLMMessage(role="user", content="hi")]):
            pass

        body = transport.bodies[0]
        assert "stream_options" not in body, "an old api-version would 400 on this field"
        assert body["stream"] is True
        assert "temperature" not in body, "Azure gpt-5 deployments reject sampling; the caller sent none"

    @pytest.mark.asyncio
    async def test_new_api_version_sends_include_usage(self):
        import httpx
        from openai import AsyncAzureOpenAI

        provider = self._azure_provider("2024-10-21")
        transport = _RecordingTransport([_sse_response({"id": "c1", "model": "gpt-4o", "choices": []})])
        provider.client = AsyncAzureOpenAI(
            api_key="fake-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-10-21",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=transport),
        )

        async for _ in provider._stream_chat_completions([LLMMessage(role="user", content="hi")]):
            pass

        assert transport.bodies[0]["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_caller_supplied_stream_options_win_even_when_gated_off(self):
        """An explicit caller value is a caller decision, not a version default."""
        import httpx
        from openai import AsyncAzureOpenAI

        provider = self._azure_provider("2024-02-15-preview")
        transport = _RecordingTransport([_sse_response({"id": "c1", "model": "gpt-4o", "choices": []})])
        provider.client = AsyncAzureOpenAI(
            api_key="fake-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=transport),
        )

        async for _ in provider._stream_chat_completions([LLMMessage(role="user", content="hi")], stream_options={"include_usage": False}):
            pass

        assert transport.bodies[0]["stream_options"] == {"include_usage": False}
