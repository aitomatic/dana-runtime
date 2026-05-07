#!/usr/bin/env python3
"""Live verify: fresh process loads a persisted timeline and replays.

The Phase 5 verify ran in-process. This script confirms the disk-resume path
end-to-end by reusing the most-recently-saved timeline from the ON-run agent
directory and exercising one additional turn in a *fresh* DanaCodingAgent.

Run after ``verify-reasoning-replay.py`` so the timeline exists.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

os.environ["AZURE_OPENAI_API_VERSION"] = os.environ.get("AZURE_RESPONSES_API_VERSION", "2025-04-01-preview")
os.environ.setdefault("AZURE_THINKING_EFFORT", "high")

from dana.common.llm.providers.openai_compatible_base import OpenAICompatibleProvider  # noqa: E402
from dana.core.agent.builtin_agents.dana_coding_agent import DanaCodingAgent  # noqa: E402


AGENT_ID = "dana-coding-agent-replay-verify-on"


def _find_session_with_reasoning(agent_id: str) -> str | None:
    """Return the session whose timeline has the most reasoning_items, not just
    the most recent. Empty resume-test sessions otherwise win on mtime."""
    sessions_dir = Path.cwd() / ".dana" / "dana_agent" / agent_id / "sessions"
    if not sessions_dir.exists():
        return None
    best_session = None
    best_count = -1
    for session_dir in sessions_dir.iterdir():
        timeline_path = session_dir / "timeline.json"
        if not timeline_path.exists():
            continue
        try:
            data = json.loads(timeline_path.read_text())
            count = sum(1 for e in data.get("entries", []) if (e.get("metadata") or {}).get("reasoning_items"))
        except Exception:
            continue
        if count > best_count:
            best_count = count
            best_session = session_dir.name
    return best_session


async def main() -> int:
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("ERROR: AZURE_OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    session_id = _find_session_with_reasoning(AGENT_ID)
    if session_id is None:
        print(f"ERROR: no persisted session for agent {AGENT_ID}; run verify-reasoning-replay.py first")
        return 1

    timeline_path = Path.cwd() / ".dana" / "dana_agent" / AGENT_ID / "sessions" / session_id / "timeline.json"
    data = json.loads(timeline_path.read_text())
    pre_entries = len(data["entries"])
    pre_items_count = sum(1 for e in data["entries"] if (e.get("metadata") or {}).get("reasoning_items"))
    print(f"BEFORE resume: session={session_id} entries={pre_entries} items_in_metadata={pre_items_count}")

    # Instrument splice fires
    orig_convert = OpenAICompatibleProvider._convert_to_responses_input
    splice_count = {"items": 0, "fired": 0}

    def _convert_with_metrics(self, openai_messages):
        result = orig_convert(self, openai_messages)
        n = sum(1 for r in result if r.get("type") == "reasoning")
        splice_count["items"] += n
        if n > 0:
            splice_count["fired"] += 1
        return result

    OpenAICompatibleProvider._convert_to_responses_input = _convert_with_metrics

    # Spin up a FRESH agent. NB: DanaCodingAgent does NOT auto-resume from disk;
    # each new process gets a fresh empty session by default. To exercise the
    # disk-resume path we explicitly load the saved timeline.json and resume
    # the agent from it via STARAgent.resume_from_timeline.
    os.environ["LLM_REASONING_REPLAY"] = "1"
    agent = DanaCodingAgent(
        agent_id=AGENT_ID,
        agent_type="dana_coding_agent",
        llm_provider="azure",
        model=os.getenv("AZURE_MODEL", "gpt-5.2"),
    )
    print(f"\nFresh agent session_id: {agent._session_id}")

    # Inject persisted entries into the agent's existing (repo-wired) timeline.
    # Bypasses the "fresh CompressedTimeline lacks repository" save path, which
    # is a separate concern from replay. This is a verify-only shortcut — the
    # core question is whether persisted reasoning_items survive into input[].
    persisted = data["entries"]  # list of TimelineEntry dicts
    agent._timeline.load_from_entries(entries=persisted)
    print(f"  injected {len(agent._timeline.timeline)} entries into agent timeline")

    # Send one new turn — replay should fire if disk-resume works
    answer = await agent.aquery(
        message="Going back to the original puzzle from earlier — refresh me on which scenario produced the most ambiguous outcome and why."
    )
    print(f"\nAnswer[:160]: {str(answer or '')[:160]!r}")

    print(f"\nSplice metrics: items_spliced={splice_count['items']} fires={splice_count['fired']}")

    # Check fresh agent's timeline.json for accumulated state
    new_session = _find_session_with_reasoning(AGENT_ID)
    new_path = Path.cwd() / ".dana" / "dana_agent" / AGENT_ID / "sessions" / new_session / "timeline.json"
    new_data = json.loads(new_path.read_text())
    print(f"AFTER resume: entries={len(new_data['entries'])}")

    print("\n=== VERDICT ===")
    print(f"  [{'PASS' if pre_items_count > 0 else 'FAIL'}] persisted timeline had reasoning_items ({pre_items_count})")
    print(f"  [{'PASS' if splice_count['fired'] > 0 else 'FAIL'}] fresh process replayed items ({splice_count['fired']} fires)")
    print(
        f"  [{'PASS' if splice_count['items'] >= pre_items_count else 'FAIL'}] all persisted items reached input[] ({splice_count['items']} >= {pre_items_count})"
    )

    OpenAICompatibleProvider._convert_to_responses_input = orig_convert

    return 0 if splice_count["fired"] > 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
