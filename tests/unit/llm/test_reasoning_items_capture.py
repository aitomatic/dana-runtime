"""Tests for raw reasoning-item capture from the OpenAI Responses API.

Covers Phase 1 of plans/260507-1829-reasoning-state-replay:
- LLMResponse.reasoning_items populated verbatim from response.output[]
- response_id captured from response.id
- include=["reasoning.encrypted_content"] sent by default
- One-shot fallback when API rejects the include flag (sticky per-instance)
- encrypted_content captured when present, None when absent
- endpoint_hash deterministic and provider-distinct
- Provider fingerprint format: "<provider>:<family>:<endpoint_hash>"
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import BadRequestError
import pytest

from dana.common.llm.types import LLMMessage


def _make_provider(model="gpt-5", use_responses_api=None):
    from dana.common.llm.providers.openai_compatible_base import OpenAICompatibleProvider

    p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    p.model = model
    p.client = MagicMock()
    p._use_responses_api = use_responses_api
    p._include_unsupported = False
    return p


def _make_summary_part(text):
    s = MagicMock()
    s.type = "summary_text"
    s.text = text
    s.model_dump = MagicMock(return_value={"type": "summary_text", "text": text})
    return s


def _make_reasoning_item(item_id, summary_texts, encrypted=None):
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


def _make_message_item(text):
    item = MagicMock()
    item.type = "message"
    content = MagicMock()
    content.type = "output_text"
    content.text = text
    item.content = [content]
    return item


def _make_response(output, response_id="resp_test_123", status="completed", usage=None):
    resp = MagicMock()
    resp.output = output
    resp.id = response_id
    resp.status = status
    resp.model = "gpt-5"
    resp.usage = usage
    resp.incomplete_details = None
    return resp


def _bad_request(msg):
    """Build a real BadRequestError matching the SDK shape."""
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(400, request=request)
    return BadRequestError(message=msg, response=response, body={"error": {"message": msg}})


class TestReasoningItemsCapture:
    @pytest.mark.asyncio
    async def test_items_captured_with_encrypted_content(self):
        provider = _make_provider()
        item = _make_reasoning_item("rs_abc", ["First step.", "Second step."], encrypted="enc_blob_xyz")
        msg = _make_message_item("done")
        provider.client.responses.create = AsyncMock(return_value=_make_response([item, msg]))

        resp = await provider._chat_via_responses([LLMMessage(role="user", content="hi")])

        assert resp.reasoning_items is not None
        assert len(resp.reasoning_items) == 1
        captured = resp.reasoning_items[0]
        assert captured["type"] == "reasoning"
        assert captured["id"] == "rs_abc"
        assert captured["encrypted_content"] == "enc_blob_xyz"
        assert len(captured["summary"]) == 2
        # Summary text still flows into reasoning_content for backward compat
        assert resp.reasoning_content == "First step.Second step."

    @pytest.mark.asyncio
    async def test_items_captured_without_encrypted_content(self):
        provider = _make_provider()
        item = _make_reasoning_item("rs_def", ["Reasoning."], encrypted=None)
        msg = _make_message_item("done")
        provider.client.responses.create = AsyncMock(return_value=_make_response([item, msg]))

        resp = await provider._chat_via_responses([LLMMessage(role="user", content="hi")])

        assert resp.reasoning_items is not None
        assert resp.reasoning_items[0]["encrypted_content"] is None

    @pytest.mark.asyncio
    async def test_no_reasoning_means_none_field(self):
        provider = _make_provider()
        msg = _make_message_item("done")
        provider.client.responses.create = AsyncMock(return_value=_make_response([msg]))

        resp = await provider._chat_via_responses([LLMMessage(role="user", content="hi")])

        assert resp.reasoning_items is None

    @pytest.mark.asyncio
    async def test_response_id_populated(self):
        provider = _make_provider()
        msg = _make_message_item("done")
        provider.client.responses.create = AsyncMock(return_value=_make_response([msg], response_id="resp_xyz"))

        resp = await provider._chat_via_responses([LLMMessage(role="user", content="hi")])

        assert resp.response_id == "resp_xyz"


class TestIncludeFlagBehavior:
    @pytest.mark.asyncio
    async def test_include_flag_sent_by_default(self):
        provider = _make_provider()
        msg = _make_message_item("done")
        provider.client.responses.create = AsyncMock(return_value=_make_response([msg]))

        await provider._chat_via_responses([LLMMessage(role="user", content="hi")])

        kwargs = provider.client.responses.create.await_args.kwargs
        assert kwargs.get("include") == ["reasoning.encrypted_content"]

    @pytest.mark.asyncio
    async def test_400_on_include_triggers_one_shot_fallback(self):
        provider = _make_provider()
        msg = _make_message_item("done")
        # First call rejects with include-related message; second succeeds.
        bad = _bad_request("Unknown parameter: include[reasoning.encrypted_content].")
        ok_response = _make_response([msg])
        provider.client.responses.create = AsyncMock(side_effect=[bad, ok_response])

        resp = await provider._chat_via_responses([LLMMessage(role="user", content="hi")])

        assert resp.content == "done"
        assert provider._include_unsupported is True
        # 2 calls — first with include, second without.
        assert provider.client.responses.create.await_count == 2
        first_kwargs = provider.client.responses.create.call_args_list[0].kwargs
        second_kwargs = provider.client.responses.create.call_args_list[1].kwargs
        assert first_kwargs.get("include") == ["reasoning.encrypted_content"]
        assert "include" not in second_kwargs

    @pytest.mark.asyncio
    async def test_subsequent_calls_skip_include_after_first_rejection(self):
        provider = _make_provider()
        provider._include_unsupported = True  # simulate post-rejection state
        msg = _make_message_item("done")
        provider.client.responses.create = AsyncMock(return_value=_make_response([msg]))

        await provider._chat_via_responses([LLMMessage(role="user", content="hi")])

        kwargs = provider.client.responses.create.await_args.kwargs
        assert "include" not in kwargs

    @pytest.mark.asyncio
    async def test_unrelated_400_propagates(self):
        provider = _make_provider()
        bad = _bad_request("invalid model parameter")
        provider.client.responses.create = AsyncMock(side_effect=bad)

        with pytest.raises(BadRequestError):
            await provider._chat_via_responses([LLMMessage(role="user", content="hi")])

        assert provider._include_unsupported is False  # not flipped on unrelated errors


class TestEndpointHashAndFingerprint:
    def test_azure_endpoint_hash_deterministic(self):
        from dana.common.llm.providers.azure import AzureProvider

        p1 = AzureProvider.__new__(AzureProvider)
        p1.azure_endpoint = "https://example.openai.azure.com"
        p1.model = "gpt-5"
        p2 = AzureProvider.__new__(AzureProvider)
        p2.azure_endpoint = "https://example.openai.azure.com"
        p2.model = "gpt-5"

        assert p1.endpoint_hash == p2.endpoint_hash
        assert len(p1.endpoint_hash) == 8

    def test_different_endpoints_produce_different_hashes(self):
        from dana.common.llm.providers.azure import AzureProvider

        p1 = AzureProvider.__new__(AzureProvider)
        p1.azure_endpoint = "https://east.openai.azure.com"
        p1.model = "gpt-5"
        p2 = AzureProvider.__new__(AzureProvider)
        p2.azure_endpoint = "https://west.openai.azure.com"
        p2.model = "gpt-5"

        assert p1.endpoint_hash != p2.endpoint_hash

    def test_openai_default_endpoint_used_when_none_set(self):
        from dana.common.llm.providers.openai import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        p.base_url = None
        p.model = "gpt-5"
        # Should hash the default URL, not crash on None
        assert len(p.endpoint_hash) == 8

    def test_fingerprint_format_azure(self):
        from dana.common.llm.providers.azure import AzureProvider

        p = AzureProvider.__new__(AzureProvider)
        p.azure_endpoint = "https://example.openai.azure.com"
        p.model = "gpt-5"

        fp = p.fingerprint
        parts = fp.split(":")
        assert len(parts) == 3
        assert parts[0] == "azure"
        assert parts[1] == "gpt-5"
        assert len(parts[2]) == 8

    def test_fingerprint_format_openai(self):
        from dana.common.llm.providers.openai import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        p.base_url = "https://api.openai.com/v1"
        p.model = "gpt-5.2"

        fp = p.fingerprint
        assert fp.startswith("openai:gpt-5:")

    def test_fingerprint_falls_back_to_model_for_unknown_family(self):
        from dana.common.llm.providers.openai import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        p.base_url = None
        p.model = "some-future-model"

        # Unknown family → full model name keeps fingerprints distinct
        assert "some-future-model" in p.fingerprint
