"""DanaACPAgent — ACP Agent protocol adapter over Dana AgentSession.

Translates ACP protocol calls (``initialize``, ``session/new``, ``session/load``,
``session/resume``, ``session/prompt``, ``session/cancel``) to
:class:`~dana.core.session.agent_session.AgentSession` operations and streams
:class:`~dana.core.session.projections.host_events.HostEvent` values back as ACP
``session_update`` notifications.

Stdout is reserved for JSON-RPC frames; all diagnostics go to stderr via the
logging configured in :mod:`dana.apps.acp.__main__`.
"""

from __future__ import annotations

from datetime import UTC, datetime
import os
from typing import Any
from uuid import uuid4

from acp import PROTOCOL_VERSION
from acp.schema import (
    AgentCapabilities,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
)
import structlog

from dana.apps.acp.translation import host_event_to_acp_update
from dana.core.session.agent_session import AgentSession, SessionBusy, TextBlock
from dana.core.session.journal.models import SessionRecord
from dana.core.session.journal.protocol import JournalRepository
from dana.core.session.journal.sqlite import SQLiteJournalRepository
from dana.core.session.legacy_timeline_migration import recover_interrupted_turns
from dana.core.session.models import FactType, JournalFact, OwnerScope


logger = structlog.get_logger()


def _dana_version() -> str:
    """Return the installed dana package version, or a fallback."""
    try:
        from importlib.metadata import version

        return version("dana")
    except Exception:
        return "0.0.0+unknown"


def _default_agent_factory() -> Any:
    """Build a minimal STARAgent for production use (text-only, no tools)."""
    from dana.core.agent.star_agent import STARAgent

    return STARAgent(
        agent_type="dana-acp",
        auto_register=False,
        enable_skills=False,
        enable_web_search=False,
        enable_code_execution=False,
        enable_assistant=False,
        compress_timeline=False,
    )


class DanaACPAgent:
    """ACP agent that exposes Dana AgentSession over the Agent Client Protocol.

    Translates ACP protocol calls to AgentSession operations and HostEvent
    streams back to ACP ``session_update`` notifications. ACP types never enter
    STAR core — translation happens entirely in :mod:`dana.apps.acp`.
    """

    def __init__(
        self,
        journal_path: str | None = None,
        agent_factory: Any | None = None,
        owner_id: str | None = None,
    ) -> None:
        self._journal_path = os.path.expanduser(journal_path or os.environ.get("DANA_ACP_JOURNAL", "~/.dana/journal.db"))
        self._agent_factory = agent_factory or _default_agent_factory
        self._owner_id = owner_id or os.environ.get("USER", "local")
        self._sessions: dict[str, AgentSession] = {}
        self._repository: JournalRepository | None = None
        self._conn: Any = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def on_connect(self, conn: Any) -> None:
        """Store the client connection for sending session_update notifications."""
        self._conn = conn

    async def _get_repository(self) -> JournalRepository:
        if self._repository is None:
            os.makedirs(os.path.dirname(self._journal_path) or ".", exist_ok=True)
            self._repository = await SQLiteJournalRepository.open(self._journal_path)
        return self._repository

    # ------------------------------------------------------------------
    # ACP protocol: initialize
    # ------------------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any | None = None,
        client_info: Any | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(load_session=True),
            agent_info=Implementation(
                name="dana-acp",
                title="Dana",
                version=_dana_version(),
            ),
        )

    # ------------------------------------------------------------------
    # ACP protocol: session/new
    # ------------------------------------------------------------------

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: Any | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        scope = OwnerScope(owner_id=self._owner_id, workspace=cwd)
        session_id = str(uuid4())
        repo = await self._get_repository()

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

        session = AgentSession(
            owner_scope=scope,
            session_id=session_id,
            repository=repo,
            agent_factory=self._agent_factory,
        )
        self._sessions[session_id] = session
        logger.info("session created", session_id=session_id, cwd=cwd)
        return NewSessionResponse(session_id=session_id)

    # ------------------------------------------------------------------
    # ACP protocol: session/load
    # ------------------------------------------------------------------

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: Any | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        scope = OwnerScope(owner_id=self._owner_id, workspace=cwd)
        repo = await self._get_repository()

        await recover_interrupted_turns(repo, scope, session_id)

        session = AgentSession(
            owner_scope=scope,
            session_id=session_id,
            repository=repo,
            agent_factory=self._agent_factory,
        )
        await session.load()
        self._sessions[session_id] = session

        # Replay host events as session_update notifications BEFORE returning.
        async for event in session.replay_host_events(0):
            update = host_event_to_acp_update(event)
            if update is not None:
                await self._notify(session_id, update)

        logger.info("session loaded", session_id=session_id, cwd=cwd)
        return LoadSessionResponse()

    # ------------------------------------------------------------------
    # ACP protocol: session/resume (unstable)
    # ------------------------------------------------------------------

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: Any | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        await self.load_session(cwd=cwd, session_id=session_id, **kwargs)
        return ResumeSessionResponse()

    # ------------------------------------------------------------------
    # ACP protocol: session/prompt
    # ------------------------------------------------------------------

    async def prompt(
        self,
        prompt: list,
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")

        blocks = _content_blocks_to_text_blocks(prompt)
        stop_reason = "end_turn"

        try:
            async for event in session.prompt(blocks):
                update = host_event_to_acp_update(event)
                if update is not None:
                    await self._notify(session_id, update)
        except SessionBusy:
            stop_reason = "max_turn_requests"
            return PromptResponse(stop_reason=stop_reason)

        terminal = session.last_terminal
        if terminal is not None:
            if terminal.fact_type is FactType.TURN_CANCELLED:
                stop_reason = "cancelled"
            elif terminal.fact_type is FactType.TURN_ERROR:
                # ACP has no "error" stop_reason; end_turn signals the turn ended.
                # The error details are visible in host events (TURN_ERROR update).
                stop_reason = "end_turn"

        return PromptResponse(stop_reason=stop_reason)

    # ------------------------------------------------------------------
    # ACP protocol: session/cancel
    # ------------------------------------------------------------------

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            await session.cancel()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _notify(self, session_id: str, update: Any) -> None:
        """Send a session_update notification if a connection is available."""
        if self._conn is not None:
            await self._conn.session_update(session_id=session_id, update=update)


# ---------------------------------------------------------------------------
# Content translation: ACP blocks → TextBlock
# ---------------------------------------------------------------------------


def _content_blocks_to_text_blocks(blocks: list) -> list[TextBlock]:
    """Extract text from ACP content blocks into a single TextBlock.

    Non-text blocks (images, audio, resources) are silently ignored in D1.
    Multiple text blocks are joined with a space, matching AgentSession's
    convention.
    """
    text_parts: list[str] = []
    for block in blocks:
        text = _extract_text(block)
        if text:
            text_parts.append(text)
    if not text_parts:
        return [TextBlock(text="")]
    return [TextBlock(text=" ".join(text_parts))]


def _extract_text(block: Any) -> str | None:
    """Extract text from a single ACP content block (Pydantic model or dict)."""
    # Pydantic model with .text attribute
    text = getattr(block, "text", None)
    if isinstance(text, str):
        return text
    # Dict with type="text"
    if isinstance(block, dict):
        if block.get("type") == "text":
            return block.get("text", "")
        return None
    return None
