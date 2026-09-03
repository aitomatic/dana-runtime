"""D7.1 — DanaCodeApp AgentSession construction + async REPL bridge tests.

No live LLM. The AgentSession path is exercised with a fake session that
yields canned HostEvents; construction uses a real temporary journal. Also
asserts the rollback flag dispatch and the ADR-001 import boundary (AC #3).
"""

from __future__ import annotations

from datetime import UTC, datetime
import io
from pathlib import Path

import pytest
from rich.console import Console

from dana.apps.code import code_app as code_app_module
from dana.apps.code.code_app import DanaCodeApp, _agentsession_enabled
from dana.core.session.agent_session import AgentSession, SessionBusy
from dana.core.session.projections.host_events import HostEvent, HostEventType


CODE_APP_SRC = Path(code_app_module.__file__).read_text()


# ---------------------------------------------------------------------------
# Rollback flag (AC #5)
# ---------------------------------------------------------------------------


def test_agentsession_enabled_by_default(monkeypatch):
    monkeypatch.delenv("DANA_CODE_AGENTSESSION_ENABLED", raising=False)
    assert _agentsession_enabled() is True


def test_agentsession_disabled_when_flag_zero(monkeypatch):
    monkeypatch.setenv("DANA_CODE_AGENTSESSION_ENABLED", "0")
    assert _agentsession_enabled() is False


def test_run_dispatches_to_agentsession_by_default(monkeypatch):
    monkeypatch.delenv("DANA_CODE_AGENTSESSION_ENABLED", raising=False)
    app = DanaCodeApp()
    called = {}

    async def fake_run(self):
        called["agentsession"] = True

    monkeypatch.setattr(DanaCodeApp, "_run_agentsession", fake_run)
    app.run()
    assert called.get("agentsession") is True


def test_run_dispatches_to_legacy_when_disabled(monkeypatch):
    monkeypatch.setenv("DANA_CODE_AGENTSESSION_ENABLED", "0")
    app = DanaCodeApp()
    called = {}

    def fake_legacy(self):
        called["legacy"] = True

    monkeypatch.setattr(DanaCodeApp, "_run_legacy", fake_legacy)
    app.run()
    assert called.get("legacy") is True


# ---------------------------------------------------------------------------
# Construction (AC #1, #3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_session_constructs_agent_session(monkeypatch, tmp_path):
    """AC #1: under the default flag, DanaCodeApp constructs an AgentSession."""
    monkeypatch.setenv("DANA_CODE_JOURNAL", str(tmp_path / "journal.db"))
    monkeypatch.setenv("DANA_LLM_PROVIDER", "openai")
    monkeypatch.setenv("DANA_MODEL", "gpt-test")

    app = DanaCodeApp()
    await app._initialize_session()

    assert isinstance(app.agent_session, AgentSession)
    assert app.agent_session is not None
    # A real session id was minted and the journal was created on disk.
    assert app.agent_session._session_id
    assert (tmp_path / "journal.db").exists()
    # Renderer wired up.
    assert app.renderer is not None


def test_no_star_core_imports_outside_legacy_path():
    """AC #3: `from dana.core.agent` may appear ONLY in the legacy rollback path."""
    star_lines = [ln for ln in CODE_APP_SRC.splitlines() if "from dana.core.agent" in ln]
    # Exactly one STAR import — the DanaCodingAgent rollback import.
    assert len(star_lines) == 1, f"expected exactly one legacy STAR import, got: {star_lines}"
    assert "dana_coding_agent" in star_lines[0], "the sole STAR import must be the legacy DanaCodingAgent"
    # It must be indented (function-scope), not module-top-level.
    assert star_lines[0].startswith("        "), "STAR import must live inside the legacy function, not at module scope"


# ---------------------------------------------------------------------------
# Async bridge (AC #2) + busy semantics
# ---------------------------------------------------------------------------


class _FakeAgentSession:
    """Fake AgentSession whose prompt() yields canned events or raises busy."""

    def __init__(self, events: list[HostEvent] | None = None, busy: bool = False) -> None:
        self._events = events or []
        self._busy = busy

    async def prompt(self, blocks, content_blocks=None):
        if self._busy:
            raise SessionBusy("fake-session")
        for event in self._events:
            yield event


def _make_events() -> list[HostEvent]:
    now = datetime.now(UTC)
    return [
        HostEvent(HostEventType.TURN_STARTED, 1, "c", now),
        HostEvent(HostEventType.ASSISTANT_CONTENT_CHUNK, 0, "c", now, text="Hello "),
        HostEvent(HostEventType.ASSISTANT_CONTENT_FINAL, 2, "c", now, text="Hello world"),
        HostEvent(HostEventType.TURN_COMPLETED, 3, "c", now),
    ]


@pytest.mark.asyncio
async def test_consume_hostevent_stream_drives_renderer(monkeypatch, tmp_path):
    """AC #2: the async loop drives AgentSession.prompt and renders each event."""
    from dana.cli.rich_cli_renderer import RichCLIRenderer

    buf = io.StringIO()
    console = Console(file=buf, width=100, highlight=False, soft_wrap=True)
    app = DanaCodeApp()
    app.agent_session = _FakeAgentSession(_make_events())
    app.renderer = RichCLIRenderer(console=console, verbose=True)

    # Should consume all 4 events without raising.
    await app._converse_async("hi")

    out = buf.getvalue()
    assert "Hello world" in out  # final response rendered


@pytest.mark.asyncio
async def test_concurrent_prompt_surfaces_busy(monkeypatch, tmp_path, capsys):
    """Busy semantics: a SessionBusy is caught and surfaced, not crashed."""
    from dana.cli.rich_cli_renderer import RichCLIRenderer

    buf = io.StringIO()
    console = Console(file=buf, width=100, highlight=False, soft_wrap=True)
    app = DanaCodeApp()
    app.agent_session = _FakeAgentSession(busy=True)
    app.renderer = RichCLIRenderer(console=console, verbose=True)

    await app._converse_async("hi")  # must not raise
    captured = capsys.readouterr()
    assert "turn is already in progress" in captured.out.lower()
