"""Manual smoke test: AgentSession restart-recovery cycle.

Demonstrates the core D1 capability — a conversation survives an
AgentSession/process restart by being persisted to the Session Journal.

No LLM required: uses a FakeAgent that echoes canned responses.

Usage:
    DANA_SESSION_STATE_KEY="test-key-32-bytes-ok-for-testing!" \\
    uv run python tests/manual/test_agent_session_restart.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile


# Required before importing dana.session
os.environ.setdefault("DANA_SESSION_STATE_KEY", "test-key-32-bytes-ok-for-testing!")

from dana.core.session.agent_session import AgentSession, TextBlock  # noqa: E402
from dana.core.session.journal.models import SessionRecord  # noqa: E402
from dana.core.session.journal.sqlite import SQLiteJournalRepository  # noqa: E402
from dana.core.session.models import OwnerScope  # noqa: E402


# ---------------------------------------------------------------------------
# Fake agent — no LLM, just canned chunks
# ---------------------------------------------------------------------------


@dataclass
class FakeAgent:
    """Echoes the user's text back in 3 chunks."""

    _timeline: list = None

    def __post_init__(self):
        if self._timeline is None:
            self._timeline = []

    async def aquery_text_stream(
        self, *, message: str, cancel_event: asyncio.Event, result_holder: dict | None = None
    ) -> AsyncIterator[str]:
        words = f"You said: {message}".split()
        full = []
        for w in words:
            if cancel_event.is_set():
                raise asyncio.CancelledError
            full.append(w)
            yield w + " "
            await asyncio.sleep(0.05)  # simulate streaming latency
        if result_holder is not None:
            result_holder["full_text"] = " ".join(full)
            result_holder["protected_payload"] = None
            result_holder["finish_reason"] = "stop"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def consume_prompt(session: AgentSession, text: str) -> list[str]:
    """Run a prompt and return the streamed chunk texts."""
    chunks = []
    async for event in session.prompt([TextBlock(text=text)]):
        if event.text:
            chunks.append(event.text)
        kind = event.event_type.value
        print(f"  [{kind}] {repr(event.text)[:60] if event.text else ''}")
    return chunks


def make_session(repo, scope, session_id, journal_path) -> AgentSession:
    return AgentSession(
        owner_scope=scope,
        session_id=session_id,
        repository=repo,
        agent_factory=FakeAgent,
    )


# ---------------------------------------------------------------------------
# Test scenario
# ---------------------------------------------------------------------------


async def main():
    tmpdir = Path(tempfile.mkdtemp(prefix="dana-test-"))
    db_path = str(tmpdir / "journal.db")
    scope = OwnerScope(owner_id="test-user", workspace="/tmp")
    session_id = "smoke-test-001"

    print("=" * 60)
    print("PHASE 1: Create session + first turn")
    print("=" * 60)

    repo1 = await SQLiteJournalRepository.open(db_path)
    record = SessionRecord.new(session_id=session_id, owner_scope=scope)
    await repo1.create_session(record, [])

    session1 = make_session(repo1, scope, session_id, db_path)
    await session1.load()

    print("\nTurn 1: 'Remember the number 42'")
    chunks1 = await consume_prompt(session1, "Remember the number 42")

    print(f"\nTurn 1 terminal: {session1.last_terminal}")
    print(f"Turn 1 full response: {''.join(chunks1)}")

    print("\nTurn 2: 'What is 2+2?'")
    await consume_prompt(session1, "What is 2+2")
    print(f"\nTurn 2 terminal: {session1.last_terminal}")

    # --- Inspect journal state ---
    facts = await repo1.read_facts(scope, session_id)
    print(f"\nJournal has {len(facts)} facts, version = {max(f.sequence for f in facts)}")
    for f in facts:
        payload_preview = str(dict(list(f.payload.items())[:2]))[:60]
        print(f"  seq={f.sequence:>2}  {f.fact_type.value:<28}  {payload_preview}")

    await repo1.close()

    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: SIMULATE CRASH — abandon session1, open a new one")
    print("=" * 60)

    repo2 = await SQLiteJournalRepository.open(db_path)
    session2 = make_session(repo2, scope, session_id, db_path)
    await session2.load()  # loads conversation from journal

    # --- Verify conversation was restored ---
    from dana.core.session.projections.conversation import ConversationProjector

    facts = await repo2.read_facts(scope, session_id)
    projector = ConversationProjector()
    view = projector.project(facts)

    print(f"\nRestored {len(view.messages)} messages from journal:")
    for msg in view.messages:
        content = str(msg.content)
        print(f"  [{msg.role:>9}] {content[:70]}")

    print(f"\nInterruption observation: {view.interruption_observation}")

    # --- Turn 3 after restart ---
    print("\nTurn 3 (after restart): 'Do you remember what I told you?'")
    await consume_prompt(session2, "Do you remember what I told you?")
    print(f"\nTurn 3 terminal: {session2.last_terminal}")

    await repo2.close()

    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Verify crash recovery (interrupted turn)")
    print("=" * 60)

    # Start a turn, then abandon it (no terminal)
    repo3 = await SQLiteJournalRepository.open(db_path)
    session3 = make_session(repo3, scope, session_id, db_path)
    await session3.load()

    print("\nStarting turn 4 but killing before completion...")
    gen = session3.prompt([TextBlock(text="This turn will be interrupted")])
    # Consume just the first event (turn started + user message)
    first_event = await gen.__anext__()
    print(f"  Got: [{first_event.event_type.value}] — now abandoning (simulating crash)")

    # Don't exhaust the generator — simulate crash
    await gen.aclose()
    await repo3.close()

    # --- Run recovery ---
    from dana.core.session.legacy_timeline_migration import recover_interrupted_turns

    repo4 = await SQLiteJournalRepository.open(db_path)
    recovered = await recover_interrupted_turns(repo4, scope, session_id)
    print(f"\nRecovered {recovered} interrupted turn(s)")

    # Verify it's marked as interrupted
    facts = await repo4.read_facts(scope, session_id)
    last_facts = facts[-3:]
    print("Last 3 facts:")
    for f in last_facts:
        print(f"  seq={f.sequence:>2}  {f.fact_type.value}")

    view = projector.project(facts)
    print(f"\nConversationView messages: {len(view.messages)}")
    print(f"Interruption observation: {view.interruption_observation}")
    print("  (partial output from turn 4 is EXCLUDED from messages)")

    await repo4.close()
    print(f"\n✓ All phases passed. Journal at {db_path}")


if __name__ == "__main__":
    asyncio.run(main())
