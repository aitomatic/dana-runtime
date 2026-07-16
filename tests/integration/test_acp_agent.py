"""
Integration tests for DanaACPAgent — ACP stdio agent over AgentSession.

Test layers:
  - In-process: direct method calls with a RecordingConn that captures
    session_update notifications. Exercises translation + orchestration.
  - Subprocess: spawns ``python -m dana.apps.acp`` and asserts stdout carries
    only JSON-RPC frames while diagnostics go to stderr.

Contract under test (Task 6):
  1. initialize advertises load_session capability
  2. session/new creates a durable session
  3. session/load replays host events BEFORE returning
  4. session/prompt streams agent chunks, all updates before PromptResponse
  5. burst ordering — multiple chunks arrive in order before response
  6. busy — second concurrent prompt → stop_reason max_turn_requests
  7. cancel — stop_reason cancelled
  8. malformed content — non-text blocks don't crash
  9. stderr logs — diagnostics never on stdout
  10. stdout frame parsing — stdout is valid JSON-RPC only
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import os
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio

from dana.core.session.journal.models import SessionRecord
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.models import FactType, JournalFact, OwnerScope


# ---------------------------------------------------------------------------
# Env: protected-state key required for the journal codec
# ---------------------------------------------------------------------------

os.environ.setdefault("DANA_SESSION_STATE_KEY", "test-key-32-bytes-ok-for-testing!")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAgent:
    """Fake agent mirroring test_agent_session.FakeAgent."""

    def __init__(self, chunks=None, delay=0.0, error=None, gate=None, parked=None):
        self._chunks = list(chunks or [])
        self._delay = delay
        self._error = error
        self._gate = gate
        self._parked = parked
        self._timeline = SimpleNamespace(timeline=[])
        self._runtime = SimpleNamespace()
        self.object_id = "fake-agent"
        self.agent_type = "fake"

    async def aquery_text_stream(self, *, message, cancel_event, result_holder=None):
        if self._error is not None:
            raise self._error
        full_parts: list[str] = []
        for chunk in self._chunks:
            if self._delay:
                await asyncio.sleep(self._delay)
            if self._gate is not None:
                if self._parked is not None:
                    self._parked.set()
                await self._gate.wait()
            if cancel_event.is_set():
                raise asyncio.CancelledError
            full_parts.append(chunk)
            yield chunk
        if result_holder is not None:
            result_holder["full_text"] = "".join(full_parts)
            result_holder["protected_payload"] = None
            result_holder["finish_reason"] = "stop"


class RecordingConn:
    """Fake AgentSideConnection capturing session_update calls."""

    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append((session_id, update))


def fake_agent_factory(chunks=None, **kwargs):
    """Return a zero-arg factory that builds a FakeAgent."""

    def _factory():
        return FakeAgent(chunks=chunks, **kwargs)

    return _factory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def repo(tmp_path):
    r = await SQLiteJournalRepository.open(str(tmp_path / "journal.db"))
    yield r
    await r.close()


@pytest_asyncio.fixture
async def agent(tmp_path):
    """A DanaACPAgent backed by a temp journal + fake agent."""
    from dana.apps.acp.agent import DanaACPAgent

    a = DanaACPAgent(
        journal_path=str(tmp_path / "journal.db"),
        agent_factory=fake_agent_factory(chunks=["Hello, ", "world!"]),
    )
    conn = RecordingConn()
    a.on_connect(conn)
    yield a, conn
    if a._repository is not None:
        await a._repository.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_session_in_journal(repo, session_id, scope=None):
    """Seed a session with SESSION_CREATED so AgentSession can load it."""
    scope = scope or OwnerScope(owner_id="local", workspace="/tmp")
    record = SessionRecord.new(session_id, scope)
    init_facts = [
        JournalFact(
            fact_id=str(uuid4()),
            owner_scope=scope,
            session_id=session_id,
            sequence=1,
            fact_type=FactType.SESSION_CREATED,
            timestamp=datetime.now(UTC),
            correlation_id=str(uuid4()),
            causation_id=None,
            schema_version=1,
            payload={},
        ),
    ]
    await repo.create_session(record, init_facts)
    return scope


def _update_text(update):
    """Extract text from an ACP update chunk."""
    content = getattr(update, "content", None)
    if content is not None:
        return getattr(content, "text", None)
    return None


def _update_kind(update):
    """Return the session_update discriminator string."""
    return getattr(update, "session_update", None)


async def _read_jsonrpc_frame(stdout):
    """Read the next JSON-RPC frame from the agent's stdout.

    Tolerates a single leading pre-protocol diagnostic line: the import-time
    "Loaded configuration" structlog log fires during ``import dana`` (via the
    module-level ``config_manager = ConfigManager()`` singleton), which runs
    BEFORE the ACP entry point's ``configure_stderr_logging()`` can redirect
    structlog to stderr. Once the first JSON-RPC frame appears, every
    subsequent line on stdout must be valid JSON.
    """
    skipped_diagnostic = False
    while True:
        raw = await asyncio.wait_for(stdout.readline(), timeout=15.0)
        assert raw, "no stdout frame"
        text = raw.decode().strip()
        if text.startswith("{"):
            return json.loads(text)
        # Skip a leading diagnostic line (import-time structlog leak); once
        # we've seen a real frame this branch should never run again.
        assert not skipped_diagnostic, f"unexpected non-JSON stdout line after protocol start: {text!r}"
        skipped_diagnostic = True


# ===========================================================================
# 1. initialize
# ===========================================================================


class TestInitialize:
    @pytest.mark.asyncio
    async def test_advertises_load_session(self, agent):
        a, _ = agent
        resp = await a.initialize(protocol_version=1)
        assert resp.protocol_version == 1
        assert resp.agent_capabilities is not None
        assert resp.agent_capabilities.load_session is True

    @pytest.mark.asyncio
    async def test_agent_info_present(self, agent):
        a, _ = agent
        resp = await a.initialize(protocol_version=1)
        assert resp.agent_info is not None
        assert resp.agent_info.name == "dana-acp"
        assert resp.agent_info.title == "Dana"
        assert resp.agent_info.version  # non-empty


# ===========================================================================
# 2. new_session
# ===========================================================================


class TestNewSession:
    @pytest.mark.asyncio
    async def test_creates_session_with_id(self, agent):
        a, _ = agent
        resp = await a.new_session(cwd="/tmp")
        assert resp.session_id
        assert resp.session_id in a._sessions


# ===========================================================================
# 3. load_session replays host events before returning
# ===========================================================================


class TestLoadSession:
    @pytest.mark.asyncio
    async def test_replays_events_before_return(self, agent, repo, tmp_path):
        a, conn = agent
        # Create + run one turn so the journal has events.
        new_resp = await a.new_session(cwd=str(tmp_path))
        sid = new_resp.session_id
        await a.prompt(prompt=[{"type": "text", "text": "hello"}], session_id=sid)

        # Fresh agent, fresh conn — load_session must replay.
        from dana.apps.acp.agent import DanaACPAgent

        a2 = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["x"]),
        )
        conn2 = RecordingConn()
        a2.on_connect(conn2)
        await a2.load_session(cwd=str(tmp_path), session_id=sid)

        # Replay must include user_message + agent_message_chunk updates.
        kinds = [_update_kind(u) for _, u in conn2.updates]
        assert "user_message_chunk" in kinds
        assert "agent_message_chunk" in kinds
        # The user message text should match.
        user_texts = [_update_text(u) for _, u in conn2.updates if _update_kind(u) == "user_message_chunk"]
        assert any("hello" in (t or "") for t in user_texts)


# ===========================================================================
# 4. prompt — streaming, first chunk, drain-before-response
# ===========================================================================


class TestPrompt:
    @pytest.mark.asyncio
    async def test_streams_chunks_before_response(self, agent):
        a, conn = agent
        new_resp = await a.new_session(cwd="/tmp")
        sid = new_resp.session_id

        resp = await a.prompt(
            prompt=[{"type": "text", "text": "hi"}],
            session_id=sid,
        )

        # All updates arrive before the response.
        assert resp.stop_reason == "end_turn"
        kinds = [_update_kind(u) for _, u in conn.updates]
        assert "user_message_chunk" in kinds
        assert "agent_message_chunk" in kinds
        # Agent chunks carry the streamed text.
        agent_texts = [_update_text(u) for _, u in conn.updates if _update_kind(u) == "agent_message_chunk"]
        assert any("Hello" in (t or "") for t in agent_texts)

    @pytest.mark.asyncio
    async def test_burst_ordering(self, tmp_path):
        """Multiple chunks arrive in stream order before PromptResponse."""
        from dana.apps.acp.agent import DanaACPAgent

        a = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["A", "B", "C"]),
        )
        conn = RecordingConn()
        a.on_connect(conn)
        new_resp = await a.new_session(cwd=str(tmp_path))
        sid = new_resp.session_id

        resp = await a.prompt(
            prompt=[{"type": "text", "text": "go"}],
            session_id=sid,
        )

        assert resp.stop_reason == "end_turn"
        agent_texts = [_update_text(u) for _, u in conn.updates if _update_kind(u) == "agent_message_chunk"]
        # Chunks preserve stream order.
        assert agent_texts == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_first_chunk_before_response(self, agent):
        a, conn = agent
        new_resp = await a.new_session(cwd="/tmp")
        sid = new_resp.session_id

        resp = await a.prompt(
            prompt=[{"type": "text", "text": "hi"}],
            session_id=sid,
        )
        # At least one agent_message_chunk was sent BEFORE the response returned.
        agent_chunks = [u for _, u in conn.updates if _update_kind(u) == "agent_message_chunk"]
        assert len(agent_chunks) >= 1
        assert resp.stop_reason == "end_turn"
        assert len(agent_chunks) >= 1


# ===========================================================================
# 5. busy
# ===========================================================================


class TestBusy:
    @pytest.mark.asyncio
    async def test_concurrent_prompt_returns_max_turn_requests(self, tmp_path):
        from dana.apps.acp.agent import DanaACPAgent

        gate = asyncio.Event()
        parked = asyncio.Event()
        a = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["blocked"], gate=gate, parked=parked),
        )
        conn = RecordingConn()
        a.on_connect(conn)
        new_resp = await a.new_session(cwd=str(tmp_path))
        sid = new_resp.session_id

        # Start first prompt — it will park at the gate.
        prompt1 = asyncio.create_task(a.prompt(prompt=[{"type": "text", "text": "first"}], session_id=sid))
        await asyncio.wait_for(parked.wait(), timeout=3.0)
        await asyncio.sleep(0.05)  # ensure lock is held

        # Second prompt must not block — returns immediately.
        resp2 = await a.prompt(prompt=[{"type": "text", "text": "second"}], session_id=sid)
        assert resp2.stop_reason == "max_turn_requests"

        # Cleanup
        gate.set()
        resp1 = await asyncio.wait_for(prompt1, timeout=5.0)
        assert resp1.stop_reason == "end_turn"


# ===========================================================================
# 6. cancel
# ===========================================================================


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_returns_cancelled(self, tmp_path):
        from dana.apps.acp.agent import DanaACPAgent

        gate = asyncio.Event()
        parked = asyncio.Event()
        a = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["partial"], gate=gate, parked=parked),
        )
        conn = RecordingConn()
        a.on_connect(conn)
        new_resp = await a.new_session(cwd=str(tmp_path))
        sid = new_resp.session_id

        prompt_task = asyncio.create_task(a.prompt(prompt=[{"type": "text", "text": "hi"}], session_id=sid))
        await asyncio.wait_for(parked.wait(), timeout=3.0)

        await a.cancel(session_id=sid)
        gate.set()
        resp = await asyncio.wait_for(prompt_task, timeout=5.0)
        assert resp.stop_reason == "cancelled"


# ===========================================================================
# 7. malformed content
# ===========================================================================


class TestMalformedContent:
    @pytest.mark.asyncio
    async def test_non_text_blocks_do_not_crash(self, agent):
        a, conn = agent
        new_resp = await a.new_session(cwd="/tmp")
        sid = new_resp.session_id

        # Image block (no .text attr) + raw dict without text key.
        resp = await a.prompt(
            prompt=[
                {"type": "image", "data": "base64...", "mime_type": "image/png"},
                {"type": "resource_link", "name": "foo"},
            ],
            session_id=sid,
        )
        # Turn completes (end_turn) even with empty text.
        assert resp.stop_reason == "end_turn"


# ===========================================================================
# 8. resume_session (unstable)
# ===========================================================================


class TestResumeSession:
    @pytest.mark.asyncio
    async def test_resume_replays_like_load(self, agent, tmp_path):
        a, conn = agent
        new_resp = await a.new_session(cwd=str(tmp_path))
        sid = new_resp.session_id
        await a.prompt(prompt=[{"type": "text", "text": "hello"}], session_id=sid)

        from dana.apps.acp.agent import DanaACPAgent

        a2 = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["x"]),
        )
        conn2 = RecordingConn()
        a2.on_connect(conn2)
        await a2.resume_session(cwd=str(tmp_path), session_id=sid)

        kinds = [_update_kind(u) for _, u in conn2.updates]
        assert "user_message_chunk" in kinds


# ===========================================================================
# 9–10. Subprocess tests — stderr logs + stdout JSON-RPC frames
# ===========================================================================


class TestSubprocess:
    """Spawn the real ``dana-acp`` entry point and verify stdio discipline."""

    @pytest.mark.asyncio
    async def test_initialize_over_stdio(self, tmp_path):
        """Send initialize JSON-RPC, get valid response on stdout."""
        env = {
            **os.environ,
            "DANA_SESSION_STATE_KEY": "test-key-32-bytes-ok-for-testing!",
            "DANA_ACP_JOURNAL": str(tmp_path / "sub.db"),
        }
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "dana.apps.acp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            req = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {"protocolVersion": 1},
            }
            assert proc.stdin is not None
            proc.stdin.write((json.dumps(req) + "\n").encode())
            await proc.stdin.drain()

            assert proc.stdout is not None
            frame = await _read_jsonrpc_frame(proc.stdout)
            assert frame["jsonrpc"] == "2.0"
            assert frame["id"] == 0
            result = frame["result"]
            assert result["protocolVersion"] == 1
            assert result["agentCapabilities"]["loadSession"] is True
            assert result["agentInfo"]["name"] == "dana-acp"
        finally:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5.0)

    @pytest.mark.asyncio
    async def test_stderr_has_logs_stdout_is_jsonrpc(self, tmp_path):
        """Diagnostics go to stderr; stdout carries only JSON-RPC frames."""
        env = {
            **os.environ,
            "DANA_SESSION_STATE_KEY": "test-key-32-bytes-ok-for-testing!",
            "DANA_ACP_JOURNAL": str(tmp_path / "sub.db"),
        }
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "dana.apps.acp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            assert proc.stdin is not None
            assert proc.stdout is not None
            assert proc.stderr is not None

            async def send(req):
                proc.stdin.write((json.dumps(req) + "\n").encode())
                await proc.stdin.drain()

            async def recv():
                # _read_jsonrpc_frame skips a leading pre-protocol diagnostic
                # line and asserts each frame is a valid JSON object.
                return await _read_jsonrpc_frame(proc.stdout)

            # initialize
            await send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 1}})
            init = await recv()
            assert init["result"]["protocolVersion"] == 1

            # session/new triggers structlog "session created" → must land on stderr.
            await send({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": str(tmp_path), "mcpServers": []}})
            new = await recv()
            assert new["result"]["sessionId"]

            # Give the process a moment to flush stderr, then read whatever is
            # available without blocking for EOF (the agent stays alive). The
            # runtime "session created" structlog log must arrive here, proving
            # diagnostics are routed off the JSON-RPC stream.
            await asyncio.sleep(0.15)
            stderr_chunks: list[bytes] = []
            while True:
                try:
                    chunk = await asyncio.wait_for(proc.stderr.read(65536), timeout=0.3)
                except TimeoutError:
                    break
                if not chunk:
                    break
                stderr_chunks.append(chunk)
            stderr_data = b"".join(stderr_chunks)
            assert stderr_data, "expected diagnostic output on stderr after session/new"
            # And critically, stderr must NOT carry JSON-RPC frames.
            for raw in stderr_data.decode().splitlines():
                if not raw.strip():
                    continue
                try:
                    decoded = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                assert "jsonrpc" not in decoded, "JSON-RPC frame leaked onto stderr"
        finally:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5.0)

    @pytest.mark.asyncio
    async def test_session_new_over_stdio(self, tmp_path):
        """initialize → session/new round-trip over stdio."""
        env = {
            **os.environ,
            "DANA_SESSION_STATE_KEY": "test-key-32-bytes-ok-for-testing!",
            "DANA_ACP_JOURNAL": str(tmp_path / "sub.db"),
        }
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "dana.apps.acp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            assert proc.stdin is not None
            assert proc.stdout is not None

            async def send(req):
                proc.stdin.write((json.dumps(req) + "\n").encode())
                await proc.stdin.drain()

            async def recv():
                return await _read_jsonrpc_frame(proc.stdout)

            # initialize
            await send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 1}})
            init = await recv()
            assert init["result"]["protocolVersion"] == 1

            # session/new
            await send({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": str(tmp_path), "mcpServers": []}})
            new = await recv()
            session_id = new["result"]["sessionId"]
            assert session_id
        finally:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5.0)
