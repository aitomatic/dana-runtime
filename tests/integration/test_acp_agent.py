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
  11. failed turn: exactly one visible transcript notification, no
      duplicate RuntimeError/-32603, protocol response completes
  12. genuine RPC errors: unknown session / missing params still surface
      as JSON-RPC error frames
  13. load_session policy failure: previously published session and grant
      store stay owned; the failed replacement is closed
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import os
import sys
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio

from dana.core.runtime.protocols import StreamEvent, StreamEventType
from dana.core.session.journal.models import SessionRecord
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.models import FactType, JournalFact, OwnerScope


# ---------------------------------------------------------------------------
# Env: protected-state key required for the journal codec
# ---------------------------------------------------------------------------

os.environ.setdefault("DANA_SESSION_STATE_KEY", "test-key-32-bytes-ok-for-testing!")
os.environ.setdefault("DANA_POLICY_GRANTS_ENABLED", "0")


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

    async def aquery_stream(self, *, message=None, **kwargs):
        """Streaming STAR-loop stand-in: yields TEXT_DELTA per chunk, then DONE."""
        if self._error is not None:
            raise self._error
        for chunk in self._chunks:
            if self._delay:
                await asyncio.sleep(self._delay)
            if self._gate is not None:
                if self._parked is not None:
                    self._parked.set()
                await self._gate.wait()
            yield StreamEvent(event_type=StreamEventType.TEXT_DELTA, data=chunk, iteration=0)
        yield StreamEvent(event_type=StreamEventType.DONE, data=None, iteration=0)


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
    await a.aclose()


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


# ---------------------------------------------------------------------------
# In-process JSON-RPC harness
#
# Drives the real ``acp.agent.connection.AgentSideConnection`` over in-memory
# streams, so tests exercise the actual wire behaviour: JSON-RPC frame
# framing, request/notification routing, and the transport's own error mapping
# framing and error mapping, without spawning a subprocess or touching the network.
# ---------------------------------------------------------------------------


class _MemoryTransport(asyncio.BaseTransport):
    """Collect newline-delimited frames written by the ACP sender loop."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self._buffer = b""
        self._closing = False

    def write(self, data: bytes) -> None:
        self._buffer += data
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            if line.strip():
                self.frames.append(json.loads(line))

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        self._closing = True

    def abort(self) -> None:
        self.close()

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        return default


class _MemoryProtocol(asyncio.streams.FlowControlMixin):
    """Minimal protocol stand-in required by ``asyncio.StreamWriter``."""


class AcpWire:
    """In-process ACP peer: real connection, in-memory streams, frame capture."""

    def __init__(self, agent) -> None:
        from acp.agent.connection import AgentSideConnection

        self._loop = asyncio.get_running_loop()
        self.reader = asyncio.StreamReader()
        self.transport = _MemoryTransport()
        self.writer = asyncio.StreamWriter(self.transport, _MemoryProtocol(loop=self._loop), self.reader, self._loop)
        self.conn = AgentSideConnection(agent, self.writer, self.reader, use_unstable_protocol=True)

    @property
    def frames(self) -> list[dict]:
        return self.transport.frames

    def send(self, request_id: int, method: str, params: dict) -> None:
        """Feed one JSON-RPC request frame to the agent."""
        frame = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        self.reader.feed_data((json.dumps(frame) + "\n").encode())

    async def response(self, request_id: int, timeout: float = 15.0) -> dict:
        """Await the response frame carrying ``request_id``."""

        async def _poll() -> dict:
            while True:
                for frame in self.frames:
                    if frame.get("id") == request_id:
                        return frame
                await asyncio.sleep(0.005)

        return await asyncio.wait_for(_poll(), timeout=timeout)

    def updates(self) -> list[dict]:
        """Return every ``session/update`` notification payload in order."""
        return [f["params"]["update"] for f in self.frames if f.get("method") == "session/update"]

    async def close(self) -> None:
        await self.conn.close()


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

        await a2.aclose()


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

        await a.aclose()

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

        await a.aclose()


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

        await a.aclose()


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

        await a2.aclose()


# ===========================================================================
# 8b. DANA_SESSION_JOURNAL_AUTHORITY feature flag (Task 8 cutover switch)
# ===========================================================================


class TestJournalAuthorityFlag:
    """Rollback switch wiring — DANA_SESSION_JOURNAL_AUTHORITY.

    The default is journal-backed (D1 cutover). Setting the env var to "0"
    flips the flag off without otherwise changing behavior; full legacy
    fallback is documented in docs/session-journal-storage.md and deferred.
    """

    @pytest.mark.asyncio
    async def test_default_is_journal_backed(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DANA_SESSION_JOURNAL_AUTHORITY", raising=False)
        from dana.apps.acp.agent import DanaACPAgent

        a = DanaACPAgent(journal_path=str(tmp_path / "j.db"), agent_factory=fake_agent_factory())
        assert a.journal_authority_enabled is True

    @pytest.mark.asyncio
    async def test_flag_zero_disables_authority(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DANA_SESSION_JOURNAL_AUTHORITY", "0")
        from dana.apps.acp.agent import DanaACPAgent

        a = DanaACPAgent(journal_path=str(tmp_path / "j.db"), agent_factory=fake_agent_factory())
        assert a.journal_authority_enabled is False

    @pytest.mark.asyncio
    async def test_flag_explicit_one_enables_authority(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DANA_SESSION_JOURNAL_AUTHORITY", "1")
        from dana.apps.acp.agent import DanaACPAgent

        a = DanaACPAgent(journal_path=str(tmp_path / "j.db"), agent_factory=fake_agent_factory())
        assert a.journal_authority_enabled is True


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


# ===========================================================================
# D4: Model switching (ADR-007)
# ===========================================================================


class TestModelSwitching:
    """D4 model switching — session/set_model, model state, cross-provider continuation."""

    @pytest.mark.asyncio
    async def test_new_session_includes_model_state(self, tmp_path):
        """session/new returns model state with available models."""
        from dana.apps.acp.agent import DanaACPAgent
        from dana.core.model.catalog import ModelCatalog, ModelTarget

        catalog = ModelCatalog(
            [
                ModelTarget(provider="anthropic", model="claude-sonnet-4"),
                ModelTarget(provider="openai", model="gpt-4o"),
            ]
        )
        a = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["ok"]),
            model_catalog=catalog,
        )
        conn = RecordingConn()
        a.on_connect(conn)
        resp = await a.new_session(cwd=str(tmp_path))
        assert resp.models is not None
        assert len(resp.models.available_models) == 2
        assert resp.models.current_model_id == "anthropic/claude-sonnet-4"

        await a.aclose()

    @pytest.mark.asyncio
    async def test_set_session_model_switches_model(self, tmp_path):
        """session/set_model switches to a configured target and journals MODEL_CHANGED."""
        from dana.apps.acp.agent import DanaACPAgent
        from dana.core.model.catalog import ModelCatalog, ModelTarget

        catalog = ModelCatalog(
            [
                ModelTarget(provider="anthropic", model="claude-sonnet-4"),
                ModelTarget(provider="openai", model="gpt-4o"),
            ]
        )
        a = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["ok"]),
            model_catalog=catalog,
        )
        conn = RecordingConn()
        a.on_connect(conn)
        new_resp = await a.new_session(cwd=str(tmp_path))
        sid = new_resp.session_id

        # Switch to openai/gpt-4o
        switch_resp = await a.set_session_model(model_id="openai/gpt-4o", session_id=sid)
        assert switch_resp is not None

        # Session should have updated provider/model
        session = a._sessions[sid]
        assert session.current_provider == "openai"
        assert session.current_model == "gpt-4o"

        # Journal should contain exactly one MODEL_CHANGED fact
        repo = await a._get_repository()
        facts = await repo.read_facts(session._owner_scope, session._session_id)
        model_changed_facts = [f for f in facts if f.fact_type == FactType.MODEL_CHANGED]
        assert len(model_changed_facts) == 1
        assert model_changed_facts[0].payload["provider"] == "openai"
        assert model_changed_facts[0].payload["model"] == "gpt-4o"

        await a.aclose()

    @pytest.mark.asyncio
    async def test_history_survives_switch(self, tmp_path):
        """Conversation history from both pre- and post-switch providers is preserved."""
        from dana.apps.acp.agent import DanaACPAgent
        from dana.core.model.catalog import ModelCatalog, ModelTarget

        catalog = ModelCatalog(
            [
                ModelTarget(provider="anthropic", model="claude-sonnet-4"),
                ModelTarget(provider="openai", model="gpt-4o"),
            ]
        )
        a = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["pre-switch response"]),
            model_catalog=catalog,
        )
        conn = RecordingConn()
        a.on_connect(conn)
        new_resp = await a.new_session(cwd=str(tmp_path))
        sid = new_resp.session_id

        # Run a turn on the startup model
        await a.prompt(prompt=[{"type": "text", "text": "hello from anthropic"}], session_id=sid)

        # Switch to openai
        await a.set_session_model(model_id="openai/gpt-4o", session_id=sid)

        # Load the session fresh and verify history is intact
        from dana.apps.acp.agent import DanaACPAgent as DanaACPAgent2

        a2 = DanaACPAgent2(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["post-switch response"]),
            model_catalog=catalog,
        )
        conn2 = RecordingConn()
        a2.on_connect(conn2)
        load_resp = await a2.load_session(cwd=str(tmp_path), session_id=sid)

        # Model state should reflect the switch
        assert load_resp.models is not None
        assert load_resp.models.current_model_id == "openai/gpt-4o"

        # Replay should include pre-switch messages
        kinds = [getattr(u, "session_update", None) for _, u in conn2.updates]
        assert "user_message_chunk" in kinds
        assert "agent_message_chunk" in kinds

        await a2.aclose()
        await a.aclose()

    @pytest.mark.asyncio
    async def test_unknown_model_target_raises(self, tmp_path):
        """Switching to an unconfigured model raises ValueError."""
        from dana.apps.acp.agent import DanaACPAgent
        from dana.core.model.catalog import ModelCatalog, ModelTarget

        catalog = ModelCatalog(
            [
                ModelTarget(provider="anthropic", model="claude-sonnet-4"),
            ]
        )
        a = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["ok"]),
            model_catalog=catalog,
        )
        conn = RecordingConn()
        a.on_connect(conn)
        new_resp = await a.new_session(cwd=str(tmp_path))
        sid = new_resp.session_id

        with pytest.raises(ValueError, match="Unknown model target"):
            await a.set_session_model(model_id="openai/gpt-4o", session_id=sid)

        await a.aclose()

    @pytest.mark.asyncio
    async def test_rollback_disables_model_switching(self, tmp_path, monkeypatch):
        """When DANA_MODEL_SWITCHING_ENABLED=0, set_session_model raises and model state is None."""
        monkeypatch.setenv("DANA_MODEL_SWITCHING_ENABLED", "0")
        from dana.apps.acp.agent import DanaACPAgent
        from dana.core.model.catalog import ModelCatalog, ModelTarget

        catalog = ModelCatalog(
            [
                ModelTarget(provider="anthropic", model="claude-sonnet-4"),
                ModelTarget(provider="openai", model="gpt-4o"),
            ]
        )
        a = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["ok"]),
            model_catalog=catalog,
        )
        conn = RecordingConn()
        a.on_connect(conn)
        new_resp = await a.new_session(cwd=str(tmp_path))
        sid = new_resp.session_id

        # Model state should be None (selector hidden)
        assert new_resp.models is None

        # set_session_model should raise
        with pytest.raises(RuntimeError, match="Model switching is disabled"):
            await a.set_session_model(model_id="openai/gpt-4o", session_id=sid)

        await a.aclose()

    @pytest.mark.asyncio
    async def test_model_change_journaled_once(self, tmp_path):
        """Model change is journaled as exactly one MODEL_CHANGED fact per switch."""
        from dana.apps.acp.agent import DanaACPAgent
        from dana.core.model.catalog import ModelCatalog, ModelTarget

        catalog = ModelCatalog(
            [
                ModelTarget(provider="anthropic", model="claude-sonnet-4"),
                ModelTarget(provider="openai", model="gpt-4o"),
                ModelTarget(provider="anthropic", model="claude-haiku-3"),
            ]
        )
        a = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["ok"]),
            model_catalog=catalog,
        )
        conn = RecordingConn()
        a.on_connect(conn)
        new_resp = await a.new_session(cwd=str(tmp_path))
        sid = new_resp.session_id

        # Two switches
        await a.set_session_model(model_id="openai/gpt-4o", session_id=sid)
        await a.set_session_model(model_id="anthropic/claude-haiku-3", session_id=sid)

        repo = await a._get_repository()
        session = a._sessions[sid]
        facts = await repo.read_facts(session._owner_scope, session._session_id)
        model_changed_facts = [f for f in facts if f.fact_type == FactType.MODEL_CHANGED]
        assert len(model_changed_facts) == 2
        assert model_changed_facts[0].payload["model"] == "gpt-4o"
        assert model_changed_facts[1].payload["model"] == "claude-haiku-3"

        await a.aclose()


# ===========================================================================
# 11-12. Prompt failure contract over the real JSON-RPC connection
# ===========================================================================


class _FailOnceAgent(FakeAgent):
    """Fails the first turn, then streams normally (session-still-usable check)."""

    def __init__(self) -> None:
        super().__init__(chunks=["recovered"])
        self._failed = False

    async def aquery_stream(self, *, message=None, **kwargs):
        if not self._failed:
            self._failed = True
            raise RuntimeError("boom")
        async for event in super().aquery_stream(message=message, **kwargs):
            yield event


class TestPromptFailureContract:
    """A failed turn must be recoverable, not a failed JSON-RPC request.

    Prior contract: one visible transcript notification (the ``TURN_ERROR``
    translation), no duplicate ``RuntimeError``/``-32603`` response, and a
    successful ``PromptResponse``. Genuine RPC validation errors keep surfacing
    as JSON-RPC error frames.
    """

    @pytest.mark.asyncio
    async def test_failed_turn_sends_one_notification_and_completes(self, tmp_path):
        from dana.apps.acp.agent import DanaACPAgent

        a = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["x"], error=RuntimeError("boom")),
        )
        wire = AcpWire(a)
        try:
            wire.send(1, "session/new", {"cwd": str(tmp_path), "mcpServers": []})
            sid = (await wire.response(1))["result"]["sessionId"]

            wire.send(2, "session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "hi"}]})
            frame = await wire.response(2)

            # The request completes successfully: no second failure surface.
            assert "error" not in frame, frame
            assert frame["result"]["stopReason"] == "end_turn"
            assert not [f for f in wire.frames if "error" in f], "failure turn produced a JSON-RPC error frame"

            # Exactly one visible transcript notification carries the failure.
            chunks = [u for u in wire.updates() if u.get("sessionUpdate") == "agent_message_chunk"]
            assert len(chunks) == 1, chunks
            assert "boom" in chunks[0]["content"]["text"]

            # Non-duplication across the whole stream: one chunk, one response.
            assert len([f for f in wire.frames if f.get("method") == "session/update"]) == 2
        finally:
            await wire.close()
            await a.aclose()

    @pytest.mark.asyncio
    async def test_session_usable_after_failed_turn(self, tmp_path):
        """A recoverable failure leaves the session able to run the next turn."""
        from dana.apps.acp.agent import DanaACPAgent

        a = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=lambda: _FailOnceAgent(),
        )
        wire = AcpWire(a)
        try:
            wire.send(1, "session/new", {"cwd": str(tmp_path), "mcpServers": []})
            sid = (await wire.response(1))["result"]["sessionId"]

            wire.send(2, "session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "first"}]})
            first = await wire.response(2)
            assert first["result"]["stopReason"] == "end_turn"

            wire.send(3, "session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "second"}]})
            second = await wire.response(3)
            assert second["result"]["stopReason"] == "end_turn"

            texts = [u["content"]["text"] for u in wire.updates() if u.get("sessionUpdate") == "agent_message_chunk"]
            assert any("recovered" in t for t in texts), texts
            assert len([t for t in texts if "Error: boom" in t]) == 1
        finally:
            await wire.close()
            await a.aclose()

    @pytest.mark.asyncio
    async def test_unknown_session_still_returns_internal_error(self, tmp_path):
        """Genuine handler failures are not swallowed by the recovery contract."""
        from dana.apps.acp.agent import DanaACPAgent

        a = DanaACPAgent(journal_path=str(tmp_path / "journal.db"), agent_factory=fake_agent_factory(chunks=["x"]))
        wire = AcpWire(a)
        try:
            wire.send(1, "session/prompt", {"sessionId": "no-such-session", "prompt": [{"type": "text", "text": "hi"}]})
            frame = await wire.response(1)
            assert frame["error"]["code"] == -32603
            assert "no-such-session" in json.dumps(frame["error"]["data"])
        finally:
            await wire.close()

    @pytest.mark.asyncio
    async def test_missing_params_still_returns_invalid_params(self, tmp_path):
        """Schema validation failures still surface as -32602, unsuppressed."""
        from dana.apps.acp.agent import DanaACPAgent

        a = DanaACPAgent(journal_path=str(tmp_path / "journal.db"), agent_factory=fake_agent_factory(chunks=["x"]))
        wire = AcpWire(a)
        try:
            wire.send(1, "session/prompt", {"prompt": [{"type": "text", "text": "hi"}]})
            frame = await wire.response(1)
            assert frame["error"]["code"] == -32602
        finally:
            await wire.close()


# ===========================================================================
# 13. load_session policy failure ownership (S1)
# ===========================================================================


class TestLoadSessionFailureOwnership:
    """A failed replacement load must not strand a closed published session."""

    @pytest.mark.asyncio
    async def test_policy_failure_keeps_previous_session_and_store(self, tmp_path, monkeypatch):
        """Policy init failure: prior session/store stay owned, replacement closed."""
        monkeypatch.setenv("DANA_POLICY_GRANTS_ENABLED", "1")
        import dana.apps.acp.agent as acp_agent

        a = acp_agent.DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["ok"]),
        )
        a.on_connect(RecordingConn())
        sid = (await a.new_session(cwd=str(tmp_path))).session_id
        previous = a._sessions[sid]
        previous_grants = a._grant_dbs[sid]

        created: list = []
        real_session = acp_agent.AgentSession

        def _recording_session(*args, **kwargs):
            session = real_session(*args, **kwargs)
            created.append(session)
            return session

        def _failing_evaluator(*args, **kwargs):
            raise RuntimeError("policy init failed")

        monkeypatch.setattr(acp_agent, "AgentSession", _recording_session)
        monkeypatch.setattr(acp_agent, "PolicyEvaluator", _failing_evaluator)

        with pytest.raises(RuntimeError, match="policy init failed"):
            await a.load_session(cwd=str(tmp_path), session_id=sid)

        # The previously published session stays registered and open...
        assert a._sessions[sid] is previous
        assert previous._closed is False
        # ...with its grant store still owned and still usable.
        assert a._grant_dbs[sid] is previous_grants
        assert len(a._grant_dbs) == 1
        await previous_grants.execute("SELECT 1")

        # The failed replacement released its resources and took no ownership.
        assert len(created) == 1
        assert created[0]._closed is True

        await a.aclose()

    @pytest.mark.asyncio
    async def test_successful_load_replaces_previous_session_and_store(self, tmp_path, monkeypatch):
        """The replacement publishes and the superseded pair is released."""
        monkeypatch.setenv("DANA_POLICY_GRANTS_ENABLED", "1")
        import dana.apps.acp.agent as acp_agent

        a = acp_agent.DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["ok"]),
        )
        a.on_connect(RecordingConn())
        sid = (await a.new_session(cwd=str(tmp_path))).session_id
        previous = a._sessions[sid]
        previous_grants = a._grant_dbs[sid]

        await a.load_session(cwd=str(tmp_path), session_id=sid)

        assert a._sessions[sid] is not previous
        assert a._sessions[sid]._closed is False
        assert previous._closed is True
        assert a._grant_dbs[sid] is not previous_grants
        assert len(a._grant_dbs) == 1
        # The superseded store is closed, not merely dereferenced.
        with pytest.raises(ValueError):
            await previous_grants.execute("SELECT 1")

        await a.aclose()


# ===========================================================================
# 14. teardown under registry mutation (S2)
# ===========================================================================


class TestAcloseRegistryMutation:
    """``aclose`` snapshots both registries before the first await."""

    @pytest.mark.asyncio
    async def test_close_tolerates_registry_mutation_during_teardown(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DANA_POLICY_GRANTS_ENABLED", "1")
        from dana.apps.acp.agent import DanaACPAgent

        a = DanaACPAgent(
            journal_path=str(tmp_path / "journal.db"),
            agent_factory=fake_agent_factory(chunks=["ok"]),
        )
        a.on_connect(RecordingConn())
        sid_a = (await a.new_session(cwd=str(tmp_path))).session_id
        sid_b = (await a.new_session(cwd=str(tmp_path))).session_id

        session_a = a._sessions[sid_a]
        original_aclose = session_a.aclose
        removed_connection = a._grant_dbs[sid_b]

        async def _mutating_aclose() -> None:
            # Teardown of one session dropping sibling registrations mirrors a
            # session releasing shared resources while the agent closes.
            a._sessions.pop(sid_b, None)
            a._grant_dbs.pop(sid_b, None)
            await original_aclose()

        monkeypatch.setattr(session_a, "aclose", _mutating_aclose)

        try:
            await a.aclose()
            assert a._sessions == {}
            assert a._grant_dbs == {}
            with pytest.raises(ValueError, match="no active connection"):
                await removed_connection.execute("SELECT 1")
        finally:
            await removed_connection.close()
