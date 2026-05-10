#!/usr/bin/env python3
"""Verify that dana.common.llm surfaces thinking/reasoning for Azure gpt-5.2.

Checks all three surfaces:
  - streaming: at least one LLMStreamChunk(type="thinking") yielded
  - non-streaming reasoning_tokens count returned in usage details
  - non-streaming reasoning_content text populated (Responses API path)

Both stream() and chat() now route to the Responses API for gpt-5/o3/o4
when api-version supports it.

Run:
    uv run python scripts/verify-azure-thinking.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

from dana.common.llm.providers.azure import AzureProvider  # noqa: E402
from dana.common.llm.types import LLMMessage  # noqa: E402


MODEL = os.getenv("AZURE_MODEL", "gpt-5.2")

PROMPT = (
    "You have 3 boxes labeled A, B, C. One holds gold, two are empty. "
    "B's label says 'gold is in A'. C's label says 'gold is not here'. "
    "Exactly one label is true. Where is the gold? Reason step by step, then answer."
)


def _hr(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


async def check_streaming(provider: AzureProvider) -> dict:
    _hr(f"STREAM: {MODEL} (expect Responses API + thinking chunks)")
    use_responses = provider._should_use_responses_api()
    print(f"_should_use_responses_api()={use_responses}")

    thinking_chunks: list[str] = []
    text_chunks: list[str] = []
    chunk_types: dict[str, int] = {}

    stream_kwargs = {"reasoning": {"effort": "medium"}}  # wrapper now adds summary="auto"
    print(f"stream kwargs: {stream_kwargs} (wrapper merges summary='auto')")
    async for chunk in provider.stream(messages=[LLMMessage(role="user", content=PROMPT)], **stream_kwargs):
        chunk_types[chunk.type] = chunk_types.get(chunk.type, 0) + 1
        if chunk.type == "thinking":
            thinking_chunks.append(chunk.content or "")
        elif chunk.type == "text_delta":
            text_chunks.append(chunk.content or "")

    print(f"chunk type counts: {chunk_types}")
    if thinking_chunks:
        preview = "".join(thinking_chunks)[:300].replace("\n", " ")
        print(f"thinking preview ({len(''.join(thinking_chunks))} chars): {preview!r}")
    print(f"final text ({len(''.join(text_chunks))} chars): {''.join(text_chunks)[:200]!r}...")

    return {
        "uses_responses_api": use_responses,
        "thinking_chunk_count": len(thinking_chunks),
        "thinking_chars": sum(len(c) for c in thinking_chunks),
        "text_chars": sum(len(c) for c in text_chunks),
    }


async def check_nonstreaming(provider: AzureProvider) -> dict:
    _hr(f"CHAT: {MODEL} (now Responses API — expect reasoning_content populated)")
    # gpt-5* + supported api-version → wrapper routes chat() to Responses API.
    # Pass reasoning so the model actually reasons; summary auto-defaults inside the wrapper.
    chat_kwargs = {"reasoning": {"effort": "medium"}}
    print(f"chat kwargs: {chat_kwargs}")
    resp = await provider.chat(messages=[LLMMessage(role="user", content=PROMPT)], **chat_kwargs)
    print(f"finish_reason={resp.finish_reason}")
    print(f"usage={resp.usage}")
    print(f"reasoning_tokens={resp.reasoning_tokens}")
    if resp.reasoning_content:
        preview = resp.reasoning_content[:300].replace("\n", " ")
        print(f"reasoning_content ({len(resp.reasoning_content)} chars): {preview!r}...")
    else:
        print("reasoning_content=None")
    print(f"content[:200]={(resp.content or '')[:200]!r}")
    return {
        "reasoning_tokens": resp.reasoning_tokens,
        "reasoning_content_present": bool(resp.reasoning_content),
        "reasoning_content_chars": len(resp.reasoning_content or ""),
        "content_chars": len(resp.content or ""),
    }


async def main() -> int:
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("ERROR: AZURE_OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    api_version_override = os.getenv("AZURE_RESPONSES_API_VERSION", "2025-04-01-preview")
    provider = AzureProvider(model=MODEL, api_version=api_version_override)
    print(f"deployment={provider.deployment_name}  api_version={provider.api_version}")

    stream_result = await check_streaming(provider)
    chat_result = await check_nonstreaming(provider)

    _hr("VERDICT")
    stream_ok = stream_result["uses_responses_api"] and stream_result["thinking_chunk_count"] > 0
    chat_tokens_ok = (chat_result["reasoning_tokens"] or 0) > 0
    chat_text_ok = chat_result["reasoning_content_present"]

    print(
        f"streaming thinking blocks ............. {'PASS' if stream_ok else 'FAIL'}  "
        f"({stream_result['thinking_chunk_count']} chunks, "
        f"{stream_result['thinking_chars']} chars)"
    )
    print(f"non-streaming reasoning_tokens > 0 .... {'PASS' if chat_tokens_ok else 'FAIL'}  (tokens={chat_result['reasoning_tokens']})")
    print(f"non-streaming reasoning_content text .. {'PASS' if chat_text_ok else 'FAIL'}  ({chat_result['reasoning_content_chars']} chars)")

    return 0 if (stream_ok and chat_tokens_ok and chat_text_ok) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
