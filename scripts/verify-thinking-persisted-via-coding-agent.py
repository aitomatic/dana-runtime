#!/usr/bin/env python3
"""End-to-end check: does DanaCodingAgent persist gpt-5 reasoning to timeline.json?

Spins up DanaCodingAgent against Azure gpt-5.2, asks a reasoning-heavy question,
then loads the persisted timeline.json and inspects AGENT_THOUGHTS entries to
verify the model's internal reasoning (LLMResponse.reasoning_content) made it
into durable storage.

Path layout (per LocalTimelineRepository):
    <CWD>/.dana/dana_agent/<agent.object_id>/sessions/<session_id>/timeline.json

Run:
    uv run python scripts/verify-thinking-persisted-via-coding-agent.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

# Force an api-version that supports the Responses API (>= 2025-03-01-preview).
# Without this the wrapper falls back to Chat Completions and reasoning text
# is never returned, even on gpt-5.
os.environ["AZURE_OPENAI_API_VERSION"] = os.environ.get("AZURE_RESPONSES_API_VERSION", "2025-04-01-preview")

# Imports must come AFTER env override.
from dana.common.llm.providers.openai_compatible_base import OpenAICompatibleProvider  # noqa: E402
from dana.core.agent.builtin_agents.dana_coding_agent import DanaCodingAgent  # noqa: E402


# Instrumentation: log reasoning_content size on every call so it's easy to see
# what the wrapper produced vs what the agent persisted.
_orig_chat_via_responses = OpenAICompatibleProvider._chat_via_responses


async def _logged_chat_via_responses(self, messages, tools=None, **kwargs):
    resp = await _orig_chat_via_responses(self, messages, tools, **kwargs)
    print(
        f"[INSTR] _chat_via_responses → "
        f"reasoning_content={len(resp.reasoning_content or '')} chars, "
        f"reasoning_tokens={resp.reasoning_tokens or 0}, "
        f"content={len(resp.content or '')} chars"
    )
    return resp


OpenAICompatibleProvider._chat_via_responses = _logged_chat_via_responses


PROMPT_DIRECT = (
    "You have 3 boxes labeled A, B, C. One holds gold, two are empty. "
    "B's label says 'gold is in A'. C's label says 'gold is not here'. "
    "Exactly one label is true. Where is the gold? Reason step by step, then answer. "
    "Do NOT use any tools — answer directly from reasoning."
)
PROMPT_TOOL = (
    "Reason carefully about which file in the current directory is the most recent, "
    "then use the bash tool exactly once to list files (`ls -lt`) to confirm. "
    "Then state the answer."
)
SCENARIO = os.getenv("SCENARIO", "direct")  # "direct" or "tool"
PROMPT = PROMPT_TOOL if SCENARIO == "tool" else PROMPT_DIRECT

AGENT_ID = "dana-coding-agent-thinking-test"


def _hr(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def _find_timeline(session_id: str) -> Path | None:
    workspace = Path.cwd() / ".dana" / "dana_agent"
    candidates = list(workspace.glob(f"*/sessions/{session_id}/timeline.json"))
    return candidates[0] if candidates else None


def _print_thought_summary(entries: list[dict]) -> tuple[int, int]:
    """Return (count, total_chars) of substantive AGENT_THOUGHTS entries."""
    thoughts = [e for e in entries if e.get("type") == "agent_thoughts"]
    print(f"\nAGENT_THOUGHTS entries: {len(thoughts)}")
    total_chars = 0
    for i, t in enumerate(thoughts):
        content = t.get("content", "")
        if not isinstance(content, str):
            print(f"  [{i}] non-string content: {type(content).__name__}")
            continue
        total_chars += len(content)
        preview = content[:200].replace("\n", " ")
        print(f"  [{i}] {len(content)} chars: {preview!r}")
    return len(thoughts), total_chars


async def main() -> int:
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("ERROR: AZURE_OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    cwd = tempfile.mkdtemp(prefix="dana_thinking_test_")
    print(f"agent cwd: {cwd}")
    print(f"api-version: {os.environ['AZURE_OPENAI_API_VERSION']}")
    print(f"scenario: {SCENARIO}")

    agent = DanaCodingAgent(
        agent_id=AGENT_ID,
        agent_type="dana_coding_agent",
        llm_provider="azure",
        model=os.getenv("AZURE_MODEL", "gpt-5.2"),
        cwd=cwd,
    )
    print(f"session_id: {agent._session_id}")

    _hr("RUNNING aquery")
    answer = await agent.aquery(message=PROMPT)
    print(f"answer[:200]: {str(answer or '')[:200]!r}")

    _hr("LOADING TIMELINE")
    timeline_path = _find_timeline(agent._session_id)
    if timeline_path is None:
        print(f"ERROR: no timeline.json found for session {agent._session_id}")
        return 1
    print(f"timeline: {timeline_path}")

    data = json.loads(timeline_path.read_text())
    entries = data.get("entries", [])
    print(f"total entries: {len(entries)}")
    entry_types = {}
    for e in entries:
        t = e.get("type", "?")
        entry_types[t] = entry_types.get(t, 0) + 1
    print(f"entry type counts: {entry_types}")

    thought_count, thought_chars = _print_thought_summary(entries)

    _hr("VERDICT")
    if thought_count > 0 and thought_chars > 50:
        print(f"PASS — reasoning persisted as AGENT_THOUGHTS ({thought_count} entries, {thought_chars} chars)")
        print(f"\nInspect: cat {timeline_path}")
        return 0

    print("FAIL — no substantive AGENT_THOUGHTS entry in timeline")
    print("\nLikely causes:")
    print("  1. gpt-5.2 chose not to reason on this turn (nondeterministic without explicit effort)")
    print("  2. Wrapper routed to Chat Completions (api-version too old?)")
    print("  3. Codec doesn't read response.reasoning_content (only codec_with_native_tool_use does)")
    print(f"\nInspect: cat {timeline_path}")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
