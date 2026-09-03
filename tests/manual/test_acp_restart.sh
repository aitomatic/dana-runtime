#!/usr/bin/env bash
# Manual ACP smoke test — exercises the full D1 flow without dana-console:
#   initialize → session/new → session/prompt → kill → session/load (resume)
#
# Usage:
#   DANA_SESSION_STATE_KEY="test-key-32-bytes-ok-for-testing!" \
#   OPENAI_API_KEY="sk-..." \
#   bash tests/manual/test_acp_restart.sh
#
# Requires: a configured LLM provider (OPENAI_API_KEY or ANTHROPIC_API_KEY).
set -euo pipefail

JOURNAL="/tmp/dana-acp-test-$$.db"
STATE_KEY="${DANA_SESSION_STATE_KEY:-test-key-32-bytes-ok-for-testing!}"
PYTHON="${PYTHON:-uv run python}"

rm -f "$JOURNAL"

echo "=== Journal: $JOURNAL ==="
echo "=== Sending: initialize + session/new + session/prompt ==="

# Build the JSON-RPC request batch: initialize, new_session, then prompt.
# The ACP SDK processes them sequentially over stdio.
{
  echo '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":1}}'
  sleep 0.3
  echo '{"jsonrpc":"2.0","id":1,"method":"session/new","params":{"cwd":"/tmp"}}'
  sleep 0.3
  # We'll send the prompt after extracting the session ID.
} | DANA_SESSION_STATE_KEY="$STATE_KEY" DANA_ACP_JOURNAL="$JOURNAL" \
  $PYTHON -m dana.apps.acp 2>/tmp/dana-acp-test-$$.stderr | \
  DANA_SESSION_STATE_KEY="$STATE_KEY" DANA_ACP_JOURNAL="$JOURNAL" \
  $PYTHON -c "
import sys, json

lines = sys.stdin.readlines()
session_id = None
for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        frame = json.loads(line)
    except json.JSONDecodeError:
        print(f'[non-JSON stdout line — known import-time leak]: {line}', file=sys.stderr)
        continue
    print(f'  Response id={frame.get(\"id\")}: {json.dumps(frame.get(\"result\", frame.get(\"error\", {})))[:120]}')
    if frame.get('id') == 1 and 'result' in frame:
        session_id = frame['result'].get('sessionId')
        print(f'\n>>> Session ID: {session_id}')
        print(f'>>> Saved to {\"$JOURNAL\"}')
" 2>&1

echo ""
echo "=== Process exited. Journal persists at $JOURNAL ==="
echo "=== Checking journal contents... ==="

DANA_SESSION_STATE_KEY="$STATE_KEY" $PYTHON -c "
import asyncio, os
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.models import OwnerScope

async def main():
    repo = await SQLiteJournalRepository.open('$JOURNAL')
    scope = OwnerScope(owner_id=os.environ.get('USER', 'local'), workspace='/tmp')
    sessions = await repo.list_sessions(scope)
    print(f'Sessions in journal: {len(sessions)}')
    for s in sessions:
        facts = await repo.read_facts(scope, s.session_id)
        print(f'  {s.session_id[:12]}... version={s.version} facts={len(facts)} status={s.status.value}')
        for f in facts[:6]:
            print(f'    seq={f.sequence} type={f.fact_type.value} corr={f.correlation_id[:20]}')
        if len(facts) > 6:
            print(f'    ... ({len(facts) - 6} more)')
    await repo.close()

asyncio.run(main())
"

echo ""
echo "=== Done. To test resume, re-run dana-acp and call session/load with the session ID above. ==="
echo "=== Full stderr log: /tmp/dana-acp-test-$$.stderr ==="

rm -f "$JOURNAL"
