"""Measure ordinary warm preparation with fixed external-writer work, not reopen cost.

Synthetic SQLite journals only. Real AgentSession, STARAgent, runtime, catalog and
CompressedTimeline with compression disabled; controlled provider is never called.
Generation and optional skills/assistant resources are excluded. Empty B0 context
is a correctness failure, not evidence of cheap equivalent work.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import tempfile
import time

from benchmark_tools_history import (
    AgentSession,
    ChildAgentFactory,
    OwnerScope,
    SQLiteJournalRepository,
    _seed_session,
    _turn_facts,
)


async def measure(turns: int, repetitions: int, root: Path) -> dict:
    path = root / f"history-{turns}.db"
    repo = await SQLiteJournalRepository.open(str(path))
    writer = await SQLiteJournalRepository.open(str(path))
    scope = OwnerScope("benchmark-warm-history", "synthetic")
    sid = f"history-{turns}"
    children = root / f"children-{turns}"
    children.mkdir()
    factory = ChildAgentFactory(children, delay_ms=0.0)
    reads: list[dict] = []
    read_facts = repo.read_facts

    async def recorded_read(*args, **kwargs):
        facts = await read_facts(*args, **kwargs)
        reads.append({"after_sequence": kwargs.get("after_sequence", 0), "rows": len(facts)})
        return facts

    repo.read_facts = recorded_read
    try:
        seed = await _seed_session(repo, scope, sid, turns, 512)
        session = AgentSession(owner_scope=scope, session_id=sid, repository=repo, agent_factory=factory.factory)
        await session.load()
        samples = []
        read_samples = []
        fidelity = []
        version = seed["final_version"]
        for repetition in range(repetitions + 1):
            turn = turns + repetition
            additions = _turn_facts(turn, 512)
            appended = await writer.append(scope, sid, version, additions)
            version = appended.new_version
            reads.clear()
            start = time.perf_counter_ns()
            await session._prepare_agent()
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            messages = session._agent._timeline.to_llm_messages()
            expected = [additions[1].payload["text"], additions[2].payload["text"]]
            correct = [message.content for message in messages[-2:]] == expected
            if repetition:
                samples.append(elapsed)
                read_samples.append(list(reads))
                fidelity.append(correct)
        before_reopen = session._agent._timeline.to_llm_messages()
        reopened = AgentSession(owner_scope=scope, session_id=sid, repository=repo, agent_factory=factory.factory)
        await reopened.load()
        after_reopen = reopened._agent._timeline.to_llm_messages()
        return {
            "turns": turns,
            "warmups": 1,
            "repetitions": repetitions,
            "new_facts_per_sample": 4,
            "external_writer": "independent SQLite connection",
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.95))] if repetitions >= 100 else None,
            "journal_reads": read_samples,
            "new_turn_context_correct": all(fidelity),
            "reopen_context_equal": before_reopen == after_reopen,
            "provider_entries": factory.children_entered_provider,
        }
    finally:
        await writer.close()
        await repo.close()


async def run(args):
    with tempfile.TemporaryDirectory(prefix="dana-warm-history-") as directory:
        results = []
        for turns in (10, 100, 1000, 5000):
            results.append(await measure(turns, args.repetitions, Path(directory)))
    return {"schema": "dana.warm-history/1", "clock": "perf_counter_ns", "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": [{k: v for k, v in row.items() if k not in {"samples_ms", "journal_reads"}} for row in result["cases"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
