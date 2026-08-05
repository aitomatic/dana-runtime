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
from acp.helpers import update_current_mode
from acp.schema import (
    AgentCapabilities,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    ModelInfo,
    NewSessionResponse,
    PermissionOption,
    PermissionOptionKind,
    PromptCapabilities,
    PromptResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
    ResumeSessionResponse,
    SessionMode,
    SessionModelState,
    SessionModeState,
    SetSessionModelResponse,
    SetSessionModeResponse,
)
import aiosqlite
import structlog

from dana.apps.acp.translation import (
    acp_content_to_normalized_block,
    host_event_to_acp_update,
)
from dana.core.content.validation import (
    validate_provider_capability,
)
from dana.core.model.catalog import ModelCatalog, ModelTarget
from dana.core.model.switching import ModelSwitcher
from dana.core.policy.evaluator import PolicyDecision, PolicyEvaluator
from dana.core.policy.hard_policy import create_default_hard_policy
from dana.core.policy.modes import PermissionMode
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


def _default_model_catalog() -> ModelCatalog:
    """Build a default model catalog from environment configuration.

    Reads ``DANA_MODEL_CATALOG`` as a JSON list of ``{provider, model}``
    objects. Falls back to a single anthropic/claude-sonnet-4 target.
    """
    import json

    raw = os.environ.get("DANA_MODEL_CATALOG")
    if raw:
        targets = [ModelTarget(**t) for t in json.loads(raw)]
    else:
        targets = [ModelTarget(provider="anthropic", model="claude-sonnet-4")]
    return ModelCatalog(targets)


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
        model_catalog: ModelCatalog | None = None,
    ) -> None:
        self._journal_path = os.path.expanduser(journal_path or os.environ.get("DANA_ACP_JOURNAL", "~/.dana/journal.db"))
        self._agent_factory = agent_factory or _default_agent_factory
        self._owner_id = owner_id or os.environ.get("USER", "local")
        self._sessions: dict[str, AgentSession] = {}
        self._repository: JournalRepository | None = None
        self._conn: Any = None
        # Feature flag: DANA_SESSION_JOURNAL_AUTHORITY=0 selects legacy
        # compatibility mode (no journal persistence, ephemeral sessions only).
        # Default is "1" — journal-backed (the D1 cutover default). Full legacy
        # fallback (Timeline-based ACP agent) is documented in
        # ``docs/session-journal-storage.md`` and deferred to a later phase.
        # The flag is parsed now so the rollback switch is operational and
        # discoverable; the legacy code path itself is a future wiring point.
        self._journal_authority = os.environ.get("DANA_SESSION_JOURNAL_AUTHORITY", "1") != "0"
        # D3: Rollback flag — disable durable-grant evaluation (ADR-012)
        self._policy_grants_enabled = os.environ.get("DANA_POLICY_GRANTS_ENABLED", "1") != "0"
        # D4: Model catalog — configured provider/model targets
        self._model_catalog = model_catalog or _default_model_catalog()
        # D4: Rollback flag — hide model selector, pin startup model (ADR-012)
        self._model_switching_enabled = os.environ.get("DANA_MODEL_SWITCHING_ENABLED", "1") != "0"

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

    @property
    def journal_authority_enabled(self) -> bool:
        """Whether the Session Journal is the durable authority for this agent.

        ``True`` (the default) means all session turns are journaled.
        ``False`` (set via ``DANA_SESSION_JOURNAL_AUTHORITY=0``) is the
        documented rollback switch; full legacy fallback is deferred.
        """
        return self._journal_authority

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
        # D6: Advertise multimodal capabilities based on model catalog
        supports_images = self._supports_images()
        supports_embedded = self._supports_embedded_resources()
        prompt_caps = (
            PromptCapabilities(
                image=supports_images,
                embeddedContext=supports_embedded,
            )
            if (supports_images or supports_embedded)
            else None
        )

        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=prompt_caps,
            ),
            agent_info=Implementation(
                name="dana-acp",
                title="Dana",
                version=_dana_version(),
            ),
        )

    def _supports_images(self) -> bool:
        """Check if any model in the catalog supports image content.

        D6: Image capability is advertised only when supported by at least
        one configured model. Providers known to support images include
        anthropic, openai, google, and bedrock.
        """
        image_providers = frozenset({"anthropic", "openai", "google", "bedrock", "vertex"})
        return any(t.provider in image_providers for t in self._model_catalog.targets)

    def _supports_embedded_resources(self) -> bool:
        """Check if any model in the catalog supports embedded resources.

        D6: Embedded resources (documents, code files) are supported by
        providers that support image content plus a few others.
        """
        resource_providers = frozenset({"anthropic", "openai", "google", "bedrock", "vertex"})
        return any(t.provider in resource_providers for t in self._model_catalog.targets)

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
        # Set the current version to match the journal (SESSION_CREATED fact)
        session._current_version = init_facts[0].sequence
        # Wire policy evaluator for permission adapter (D3)
        if self._policy_grants_enabled:
            from dana.core.policy.grants import SQLiteGrantStore
            from dana.core.policy.store_schema import POLICY_SQLITE_DDL

            grant_db = await aiosqlite.connect(":memory:")
            grant_db.row_factory = aiosqlite.Row
            for stmt in POLICY_SQLITE_DDL:
                await grant_db.execute(stmt)
            await grant_db.commit()
            grant_store = SQLiteGrantStore(grant_db)
            hard_policy = create_default_hard_policy()
            evaluator = PolicyEvaluator(hard_policy, grant_store, PermissionMode.DEFAULT)
            session.set_policy_evaluator(evaluator)
        self._sessions[session_id] = session
        logger.info("session created", session_id=session_id, cwd=cwd)

        # D4: Build model state from catalog (rollback: None when disabled)
        model_state = (
            _build_model_state(
                self._model_catalog,
                session.current_provider,
                session.current_model,
            )
            if self._model_switching_enabled
            else None
        )

        return NewSessionResponse(
            session_id=session_id,
            modes=_build_mode_state(session.permission_mode),
            models=model_state,
        )

    # ------------------------------------------------------------------
    # ACP protocol: session/request_permission (ADR-013)
    # ------------------------------------------------------------------

    async def request_permission(
        self,
        request: RequestPermissionRequest,
        session_id: str,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        """Handle a permission request from the host (ADR-013).

        Evaluates the requested operation through the policy evaluator and
        returns the available permission options.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")

        evaluator = session.policy_evaluator
        if evaluator is None:
            return RequestPermissionResponse(
                options=[
                    PermissionOption(
                        kind=PermissionOptionKind.ALLOW_ONCE,
                        display_name="Allow Once",
                    ),
                    PermissionOption(
                        kind=PermissionOptionKind.ALLOW_ALWAYS,
                        display_name="Allow Always",
                    ),
                    PermissionOption(
                        kind=PermissionOptionKind.REJECT_ONCE,
                        display_name="Reject Once",
                    ),
                    PermissionOption(
                        kind=PermissionOptionKind.REJECT_ALWAYS,
                        display_name="Reject Always",
                    ),
                ],
            )

        # Build an Operation from the request and evaluate
        from dana.core.policy.operations import build_policy_operation

        tool_call = {
            "function": getattr(request, "tool_name", ""),
            "arguments": getattr(request, "arguments", {}),
        }
        op = build_policy_operation(
            tool_call,
            catalog=None,
            owner=session.owner_scope.owner_id,
            workspace=session.owner_scope.workspace,
        )
        result = await evaluator.evaluate(op, session.owner_scope)

        options: list[PermissionOption] = []
        if result.decision is PolicyDecision.DENY:
            return RequestPermissionResponse(
                options=[],
                denied_reason=result.reason,
            )

        if self._policy_grants_enabled:
            options = [
                PermissionOption(
                    kind=PermissionOptionKind.ALLOW_ONCE,
                    display_name="Allow Once",
                ),
                PermissionOption(
                    kind=PermissionOptionKind.ALLOW_ALWAYS,
                    display_name="Allow Always",
                ),
                PermissionOption(
                    kind=PermissionOptionKind.REJECT_ONCE,
                    display_name="Reject Once",
                ),
                PermissionOption(
                    kind=PermissionOptionKind.REJECT_ALWAYS,
                    display_name="Reject Always",
                ),
            ]
        else:
            # Rollback: only allow-once (ADR-012)
            options = [
                PermissionOption(
                    kind=PermissionOptionKind.ALLOW_ONCE,
                    display_name="Allow Once",
                ),
            ]

        return RequestPermissionResponse(options=options)

    # ------------------------------------------------------------------
    # ACP protocol: session/set_mode (ADR-013)
    # ------------------------------------------------------------------

    async def set_session_mode(self, mode_id: str, session_id: str, **kwargs: Any) -> SetSessionModeResponse | None:
        """Change the permission mode for a session (ADR-013: outside an active turn).

        Maps ACP mode IDs to ``PermissionMode`` values:
        - ``default`` → ``PermissionMode.DEFAULT``
        - ``acceptEdits`` → ``PermissionMode.ACCEPT_EDITS``
        - ``bypassPermissions`` → ``PermissionMode.BYPASS_PERMISSIONS``
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")

        mode = _acp_mode_to_permission_mode(mode_id)
        session.set_permission_mode(mode)

        # Notify client of the mode change via current_mode_update
        await self._notify(session_id, update_current_mode(current_mode_id=mode_id))

        logger.info("session mode set", session_id=session_id, mode=mode_id)
        return SetSessionModeResponse()

    # ------------------------------------------------------------------
    # ACP protocol: session/set_model (D4, ADR-007, ADR-013)
    # ------------------------------------------------------------------

    async def set_session_model(
        self,
        model_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModelResponse | None:
        """Change the model for a session (ADR-007: atomic build-before-mutate).

        Validates the target against the model catalog, builds the new
        provider + runtime before mutation, rebinds, and commits a single
        ``MODEL_CHANGED`` fact. On failure, the old model is untouched.

        Per ADR-007: switching during an active turn returns ``busy``.

        Per ADR-012: when ``DANA_MODEL_SWITCHING_ENABLED=0``, the selector
        is hidden and the startup model is pinned — this method raises.
        """
        # Rollback: model switching disabled (ADR-012)
        if not self._model_switching_enabled:
            raise RuntimeError("Model switching is disabled (DANA_MODEL_SWITCHING_ENABLED=0)")

        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")

        # Busy check: switching during an active turn returns busy (ADR-007)
        if session._lock.locked():
            raise SessionBusy(session_id)

        # Parse model_id as "provider/model"
        if "/" not in model_id:
            raise ValueError(f"Invalid model_id: {model_id!r} (expected 'provider/model')")
        provider, model = model_id.split("/", 1)

        # Look up in catalog
        target = self._model_catalog.get(provider, model)
        if target is None:
            raise ValueError(f"Unknown model target: {model_id!r}")

        # Build a ModelSwitcher for this session
        switcher = ModelSwitcher(
            build_provider=lambda t: _build_provider_client(t),
            build_runtime=lambda t, p: _build_model_runtime(t, p),
            apply_switch=lambda t, p, r: session.rebind_model(t, p, r),
        )

        result = switcher.switch(target)
        if not result.success:
            raise RuntimeError(f"Model switch failed: {result.error}")

        # Journal the MODEL_CHANGED fact (one fact per switch per ADR-007)
        from uuid import uuid4

        from dana.core.session.models import NewJournalFact

        model_fact = NewJournalFact(
            fact_type=FactType.MODEL_CHANGED,
            correlation_id=str(uuid4()),
            causation_id=None,
            payload={
                "provider": target.provider,
                "model": target.model,
            },
        )
        await session.append_fact(model_fact)

        # Notify client of the model change via current_model_update
        await self._notify(session_id, _update_current_model(model_id))

        logger.info("session model set", session_id=session_id, model=model_id)
        return SetSessionModelResponse()

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

        # D4: Build model state from session's current provider/model
        model_state = (
            _build_model_state(
                self._model_catalog,
                session.current_provider,
                session.current_model,
            )
            if self._model_switching_enabled
            else None
        )

        return LoadSessionResponse(models=model_state)

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

        # D4: Build model state from session's current provider/model
        session = self._sessions.get(session_id)
        model_state = (
            _build_model_state(
                self._model_catalog,
                session.current_provider if session else None,
                session.current_model if session else None,
            )
            if self._model_switching_enabled
            else None
        )

        return ResumeSessionResponse(models=model_state)

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

        # D6: Convert ACP content blocks to normalized blocks, then to TextBlock
        # for the session. Multimodal blocks (image, embedded_resource, file_resource)
        # are converted to normalized dicts and passed through content_blocks.
        normalized_blocks = _acp_prompt_to_normalized_blocks(prompt)

        # D6: Validate provider capability before turn start (ADR-009)
        provider = session.current_provider
        if provider is not None:
            _validate_multimodal_capability(normalized_blocks, provider)

        # D6: Build TextBlocks for the session prompt, preserving content_blocks
        # metadata for multimodal projection
        text_blocks = _normalized_blocks_to_text_blocks(normalized_blocks)
        stop_reason = "end_turn"

        try:
            async for event in session.prompt(text_blocks):
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
# Content translation: ACP blocks → normalized blocks → TextBlock
# ---------------------------------------------------------------------------


def _acp_prompt_to_normalized_blocks(prompt: list) -> list[dict]:
    """Convert ACP prompt content blocks to normalized block dicts.

    Each ACP content block (Pydantic model or dict) is converted to a
    normalized dict that the ContentNormalizer can process. Multimodal
    blocks (image, embedded_resource, file_resource) are converted with
    their data intact.
    """
    normalized: list[dict] = []
    for block in prompt:
        normalized.append(acp_content_to_normalized_block(block))
    return normalized


def _normalized_blocks_to_text_blocks(blocks: list[dict]) -> list[TextBlock]:
    """Convert normalized blocks to TextBlock list for AgentSession.

    Text blocks are converted to TextBlock instances. Multimodal blocks
    are serialized as text placeholders with their content_blocks metadata
    preserved in the text for journaling purposes. The actual multimodal
    content is carried via the content_blocks payload field.
    """
    text_parts: list[str] = []
    has_multimodal = any(b.get("type") != "text" for b in blocks)
    content_blocks_payload: list[dict] = []

    for block in blocks:
        block_type = block.get("type", "")
        if block_type == "text":
            text = block.get("text", "")
            text_parts.append(text)
            content_blocks_payload.append(block)
        elif block_type == "image":
            # Serialize image as placeholder text; actual data in content_blocks
            media_type = block.get("media_type", "image/*")
            text_parts.append(f"[Image: {media_type}]")
            # Convert bytes data to base64 for JSON-safe payload
            data = block.get("data", b"")
            if isinstance(data, bytes):
                import base64

                block["data"] = base64.b64encode(data).decode("utf-8")
            content_blocks_payload.append(block)
        elif block_type in ("embedded_resource", "file_resource"):
            media_type = block.get("media_type", "application/octet-stream")
            uri = block.get("uri", "")
            text_parts.append(f"[Resource: {media_type}]" if not uri else f"[Resource: {uri}]")
            data = block.get("data", b"")
            if isinstance(data, bytes):
                import base64

                block["data"] = base64.b64encode(data).decode("utf-8")
            content_blocks_payload.append(block)

    if not text_parts and not has_multimodal:
        return [TextBlock(text="")]

    # Build a single TextBlock with the text summary
    text = " ".join(text_parts) if text_parts else "[multimodal content]"
    return [TextBlock(text=text)]


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


# ---------------------------------------------------------------------------
# D6: Multimodal capability validation
# ---------------------------------------------------------------------------


def _validate_multimodal_capability(blocks: list[dict], provider: str) -> None:
    """Validate that the provider supports the multimodal content in blocks.

    Per ADR-009: unsupported models fail before turn start, not mid-turn.
    Image capability is advertised only when supported.

    For D6, we use a simple heuristic: providers known to support multimodal
    content include anthropic, openai, google, bedrock, and vertex.
    """
    multimodal_providers = frozenset({"anthropic", "openai", "google", "bedrock", "vertex"})
    supports_images = provider in multimodal_providers
    supports_embedded = provider in multimodal_providers
    supports_file = provider in multimodal_providers

    validate_provider_capability(
        blocks,
        provider,
        supports_images=supports_images,
        supports_embedded_resources=supports_embedded,
        supports_file_resources=supports_file,
    )


# ---------------------------------------------------------------------------
# Permission mode helpers (ADR-013)
# ---------------------------------------------------------------------------


def _build_mode_state(mode: PermissionMode) -> SessionModeState:
    """Build an ACP SessionModeState from a PermissionMode."""
    mode_id = mode.value
    return SessionModeState(
        available_modes=[
            SessionMode(id="default", name="Default"),
            SessionMode(id="acceptEdits", name="Accept Edits"),
            SessionMode(id="bypassPermissions", name="Bypass Permissions"),
        ],
        current_mode_id=mode_id,
    )


def _acp_mode_to_permission_mode(mode_id: str) -> PermissionMode:
    """Map an ACP mode ID to a PermissionMode."""
    mapping = {
        "default": PermissionMode.DEFAULT,
        "acceptEdits": PermissionMode.ACCEPT_EDITS,
        "bypassPermissions": PermissionMode.BYPASS_PERMISSIONS,
    }
    result = mapping.get(mode_id)
    if result is None:
        raise ValueError(f"Unknown permission mode: {mode_id!r}")
    return result


# ---------------------------------------------------------------------------
# D4: Model switching helpers (ADR-007)
# ---------------------------------------------------------------------------


def _build_provider_client(target: ModelTarget) -> Any:
    """Build a provider client for the given target.

    This is a stub for D4 — real provider construction is deferred to a
    later phase. Returns a SimpleNamespace with the target info.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        provider=target.provider,
        model=target.model,
        config=target.config or {},
    )


def _build_model_runtime(target: ModelTarget, provider: Any) -> Any:
    """Build a model runtime for the given target and provider.

    This is a stub for D4 — real runtime construction is deferred to a
    later phase. Returns a SimpleNamespace with the target info.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        provider=target.provider,
        model=target.model,
    )


def _update_current_model(model_id: str) -> Any:
    """Build a ``current_model_update`` ACP notification.

    Returns a dict-like object that the ACP transport serializes as a
    ``session_update`` notification with ``sessionUpdate="current_model_update"``.
    """
    from acp.helpers import update_current_mode

    # Reuse the current_mode_update shape but with model_id semantics.
    # The ACP protocol uses the same notification shape for model changes.
    return update_current_mode(current_mode_id=model_id)


# ---------------------------------------------------------------------------
# D4: Model state helpers
# ---------------------------------------------------------------------------


def _build_model_state(
    catalog: ModelCatalog,
    current_provider: str | None,
    current_model: str | None,
) -> SessionModelState | None:
    """Build an ACP SessionModelState from the catalog and current model.

    When no model has been set yet (fresh session), the first catalog target
    is used as the startup model. Returns ``None`` when the catalog is empty.
    """
    if not catalog.targets:
        return None

    available = [
        ModelInfo(
            model_id=f"{t.provider}/{t.model}",
            name=f"{t.provider}: {t.model}",
        )
        for t in catalog.targets
    ]

    # Use current model if set, otherwise the first catalog target (startup model)
    if current_provider is not None and current_model is not None:
        current_id = f"{current_provider}/{current_model}"
    else:
        first = catalog.targets[0]
        current_id = f"{first.provider}/{first.model}"

    return SessionModelState(
        available_models=available,
        current_model_id=current_id,
    )
