#!/usr/bin/env python3
"""Deterministic AgentSession + STARAgent harness benchmark (measurement only).

Runs the real ``AgentSession`` orchestrator and the real ``STARAgent`` streaming
STAR loop against a deterministic injected ``LLMProvider`` that records every
request it receives. The injected provider is the only LLM the session's agent
holds, and two independent fail-closed guards make unmetered live model traffic
impossible: (1) every real provider construction path (``create_provider``) is
patched to raise, and (2) outbound network connections to hosts outside a
declared allowlist raise.

Usage
-----
    uv run --no-sync python scripts/benchmark_session.py --output results.json
    uv run --no-sync python scripts/benchmark_session.py --mode isolated --output -
    uv run --no-sync python scripts/benchmark_session.py --repetitions 100 --cases request_fidelity_compressed --output -

What it measures
----------------
Session construction, first-use and warm no-tool turns, provider-text vs
host-delivery timing (thought vs assistant delta), delayed text followed by
tools, cancellation at four lifecycle boundaries, completion, and close/reopen
request fidelity (compressed and uncompressed timelines, interrupted turn,
multimodal content, tool pairs, protected replay state under a model switch),
plus the actual provider request payloads for each of those paths.

What it does NOT do
-------------------
No runtime source is modified. No pytest/ruff/build/format is run. No live model
request is possible (guards + injected provider). Every repository and journal is
a per-run temporary; every resource is constructed with ``auto_register=False``.
Nothing is written outside the temporary workspace and the ``--output`` path.
``uv.lock`` and ``.DS_Store`` are never touched.

Lanes
-----
``isolated``
    Network-isolated. On the runtime *instance*, the context field list is
    rebound to ``["timestamp", "timezone", "user"]`` so prompt construction
    performs no IP-geolocation lookup; ``DANA_MCP_SERVERS`` is cleared so no MCP
    server can be spawned; the network guard allows loopback only.

``production``
    Production-default. The runtime context fields are untouched, so the default
    ``location`` field is present and the production geolocation path may run
    (``http://ip-api.com``, 2s timeout - the only allowlisted external host);
    ``DANA_MCP_SERVERS`` is left as configured; ``enable_skills`` and
    ``enable_assistant`` keep their production defaults (``True``), which is why
    the fail-closed provider guard matters: any subsidiary LLM must build its
    provider through ``create_provider`` and will raise instead of going live.
"""

from __future__ import annotations

# Trace backends are disabled before importing dana: `dana.common.observable`
# picks its backend at import time and the Langfuse backend performs a
# synchronous network flush after every decorated operation. Declared deviation,
# applied to both lanes.
import os


os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["DANA_LANGSMITH_ENABLED"] = "0"

import argparse
import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import platform
import socket
import statistics
import sys
import tempfile
import time
import types
from typing import Any
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dana import LLM  # noqa: E402
from dana.common.llm.types import LLMMessage, LLMProvider, LLMResponse, LLMStreamChunk  # noqa: E402
from dana.common.protocols import named_tool  # noqa: E402
from dana.config.storage_config import FileStorageConfig  # noqa: E402
from dana.core.agent.star_agent import STARAgent  # noqa: E402
from dana.core.knowledge.prompts.codecs import NativeToolsCodec  # noqa: E402
from dana.core.resource.base_resource import BaseResource  # noqa: E402
from dana.core.session.agent_session import AgentSession, TextBlock  # noqa: E402
from dana.core.session.journal.models import SessionRecord  # noqa: E402
from dana.core.session.journal.sqlite import SQLiteJournalRepository  # noqa: E402
from dana.core.session.legacy_timeline_migration import recover_interrupted_turns  # noqa: E402
from dana.core.session.models import FactType, JournalFact, NewJournalFact, OwnerScope  # noqa: E402
from dana.core.session.projections.conversation import ConversationProjector  # noqa: E402
from dana.core.session.protected_state import ProtectedStateCodec  # noqa: E402
from dana.repositories.local_file_repository import (  # noqa: E402
    LocalEventRepository,
    LocalLearningRepository,
    LocalPromptRepository,
    LocalTimelineRepository,
)
from dana.repositories.repository_factory import RepositoryFactory, RepositoryType  # noqa: E402


SCRIPT_VERSION = 2

# Host events that terminate a turn.
TERMINAL_EVENT_VALUES = ("turn_completed", "turn_cancelled", "turn_error", "turn_interrupted")

# The single external host the production lane may reach (production geolocation).
GEOLOCATION_HOST = "ip-api.com"

# Canonical synthetic provider-replay item (shape documented on LLMResponse):
# {"type":"reasoning","id":...,"summary":[...]} plus the optional
# "encrypted_content" that only trusted accounts receive. Base64 below decodes to
# "deterministic-encrypted-content-bench" - synthetic, no secret, no PII.
SYNTHETIC_REASONING_ITEM: dict[str, Any] = {
    "type": "reasoning",
    "id": "rs_bench_deterministic_0001",
    "summary": [{"type": "summary_text", "text": "Synthetic benchmark reasoning summary."}],
    "encrypted_content": "ZGV0ZXJtaW5pc3RpYy1lbmNyeXB0ZWQtY29udGVudC1iZW5jaA==",
}
SYNTHETIC_RESPONSE_ID = "resp_bench_deterministic_0001"

# Underscore-prefixed carrier keys threaded LLMMessage -> wire dict -> Responses
# input[]; the splicer reads and strips them. Mirrors the provider module's
# private `_REPLAY_CARRIER_KEYS` (asserted equal at runtime in the fixture case).
REPLAY_CARRIER_KEYS = ("_reasoning_items", "_reasoning_fingerprint", "_response_id")

# Frozen fixture identity: the provider/model this replay state belongs to, so
# compatibility is explicit even when no MODEL_CHANGED fact precedes it.
FIXTURE_PROVIDER = "openai"
FIXTURE_MODEL = "gpt-5"
FIXTURE_FINGERPRINT = "openai:gpt-5:00000000"

# The protected-state envelope schema for a completed native turn.
NATIVE_TURN_SCHEMA = "dana.native-turn/1"


def build_native_turn_envelope(
    user_text: str,
    assistant_text: str,
    *,
    reasoning_item: dict[str, Any] | None = None,
    response_id: str | None = None,
    tool_calls: Sequence[tuple[str, str, dict[str, Any]]] = (),
) -> bytes:
    """Frozen native-turn envelope for ``ASSISTANT_CONTENT_FINAL.protected_payload``.

    Built from existing :class:`NativeMessage` records via ``to_dict()`` - the
    completed turn's user message plus its assistant message, carrying the native
    replay metadata under the canonical keys ``reasoning_items`` /
    ``fingerprint`` / ``response_id``. The JSON envelope is
    ``{"schema": "dana.native-turn/1", "messages": [...]}``.
    """
    from dana.core.timeline.native_message import NativeMessage, NativeToolCall

    messages: list[dict[str, Any]] = [
        NativeMessage(role="user", content=user_text).to_dict(),
    ]
    assistant_metadata: dict[str, Any] = {}
    if reasoning_item is not None:
        assistant_metadata = {
            "reasoning_items": [dict(reasoning_item)],
            "fingerprint": FIXTURE_FINGERPRINT,
            "response_id": response_id,
        }
    assistant = NativeMessage(
        role="assistant",
        content=assistant_text,
        tool_calls=[NativeToolCall(id=cid, name=name, arguments=args) for cid, name, args in tool_calls] or None,
        metadata=assistant_metadata,
    )
    messages.append(assistant.to_dict())
    return json.dumps({"schema": NATIVE_TURN_SCHEMA, "messages": messages}).encode("utf-8")


# ---------------------------------------------------------------------------
# Fail-closed guards
# ---------------------------------------------------------------------------


class LiveProviderBlocked(RuntimeError):
    """Raised when something tries to construct a real (network) LLM provider."""


class LiveNetworkBlocked(RuntimeError):
    """Raised when something tries to reach a host outside the allowlist."""


class FailClosedGuard:
    """Two independent fail-closed guards against unmetered live model traffic.

    1. **Provider construction.** ``create_provider`` is the single choke point
       that every concrete provider (openai/anthropic/azure/gemini/moonshot/...)
       is built through, including the lazy path used by subsidiary agents
       (assistant agent, sub-agents), the embedder, and long-term memory. Both
       importing modules (``dana.common.llm.llm``, ``dana.common.llm.embedder``)
       have their bound name replaced, so an injected provider instance is the
       only way an LLM can exist. Any attempt raises and is recorded.

    2. **Outbound network.** ``socket.getaddrinfo`` and ``socket.socket.connect``
       are patched. Loopback (and AF_UNIX) always pass; any other host must be on
       the declared allowlist or it raises and is recorded. ``getaddrinfo`` is
       patched as well as ``connect`` because ``create_connection``/urllib
       resolve the hostname first, so a name allowlist has to be enforced before
       resolution.
    """

    def __init__(self) -> None:
        self.allowed_hosts: set[str] = set()
        self.provider_attempts: list[dict[str, Any]] = []
        self.network_attempts: list[dict[str, Any]] = []
        self.allowed_resolutions: set[str] = set()
        self._installed = False
        self._orig_getaddrinfo: Any = None
        self._orig_connect: Any = None
        self._orig_connect_ex: Any = None
        self._provider_patches: list[tuple[Any, str, Any]] = []

    # -- configuration ------------------------------------------------------

    def configure(self, allowed_hosts: Iterable[str]) -> None:
        self.allowed_hosts = {h.strip().lower() for h in allowed_hosts if h and h.strip()}
        self.allowed_resolutions.clear()

    # -- install / uninstall ------------------------------------------------

    def install(self) -> None:
        if self._installed:
            return
        self._orig_getaddrinfo = socket.getaddrinfo
        self._orig_connect = socket.socket.connect
        self._orig_connect_ex = socket.socket.connect_ex
        guard = self

        def guarded_getaddrinfo(host, *args, **kwargs):  # type: ignore[no-untyped-def]
            hostname = str(host).lower() if host is not None else ""
            if not hostname or hostname in ("localhost", "localhost.localdomain") or hostname.startswith("127."):
                return guard._orig_getaddrinfo(host, *args, **kwargs)
            if guard._host_allowed(hostname):
                results = guard._orig_getaddrinfo(host, *args, **kwargs)
                for entry in results:
                    try:
                        guard.allowed_resolutions.add(str(entry[4][0]))
                    except (IndexError, TypeError):
                        pass
                return results
            guard.network_attempts.append({"stage": "getaddrinfo", "host": hostname, "monotonic_s": time.perf_counter()})
            raise LiveNetworkBlocked(f"outbound resolution blocked for host {hostname!r}")

        def guarded_connect(sock, address, *args, **kwargs):  # type: ignore[no-untyped-def]
            guard._check_address("connect", address)
            return guard._orig_connect(sock, address, *args, **kwargs)

        def guarded_connect_ex(sock, address, *args, **kwargs):  # type: ignore[no-untyped-def]
            guard._check_address("connect_ex", address)
            return guard._orig_connect_ex(sock, address, *args, **kwargs)

        socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
        socket.socket.connect = guarded_connect  # type: ignore[assignment]
        socket.socket.connect_ex = guarded_connect_ex  # type: ignore[assignment]

        import dana.common.llm.embedder as embedder_module
        import dana.common.llm.llm as llm_module

        def blocked_create_provider(provider=None, **kwargs):  # type: ignore[no-untyped-def]
            guard.provider_attempts.append({"provider": None if provider is None else str(provider), "monotonic_s": time.perf_counter()})
            raise LiveProviderBlocked(f"refusing to construct a live LLM provider ({provider!r}); the deterministic provider is injected")

        for module in (llm_module, embedder_module):
            if hasattr(module, "create_provider"):
                self._provider_patches.append((module, "create_provider", module.create_provider))
                module.create_provider = blocked_create_provider  # type: ignore[attr-defined]
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        socket.getaddrinfo = self._orig_getaddrinfo  # type: ignore[assignment]
        socket.socket.connect = self._orig_connect  # type: ignore[assignment]
        socket.socket.connect_ex = self._orig_connect_ex  # type: ignore[assignment]
        for module, name, original in self._provider_patches:
            setattr(module, name, original)
        self._provider_patches.clear()
        self._installed = False

    # -- internals ----------------------------------------------------------

    def _host_allowed(self, hostname: str) -> bool:
        if hostname in self.allowed_hosts:
            return True
        return any(hostname.endswith("." + allowed) for allowed in self.allowed_hosts)

    def _check_address(self, stage: str, address: Any) -> None:
        if isinstance(address, str):  # AF_UNIX path
            return
        if not isinstance(address, tuple) or not address:
            return
        host = str(address[0])
        if host.startswith("127.") or host in ("::1", "localhost", "0.0.0.0", ""):
            return
        if host in self.allowed_resolutions or self._host_allowed(host.lower()):
            return
        self.network_attempts.append({"stage": stage, "host": host, "monotonic_s": time.perf_counter()})
        raise LiveNetworkBlocked(f"outbound connection blocked to {host!r}")

    def report(self) -> dict[str, Any]:
        return {
            "installed": self._installed,
            "allowed_hosts": sorted(self.allowed_hosts),
            "loopback_allowed": True,
            "provider_construction_attempts": self.provider_attempts,
            "network_attempts": self.network_attempts,
            "blocked_provider_constructions": len(self.provider_attempts),
            "blocked_network_attempts": len(self.network_attempts),
            "provider_patched_modules": sorted(f"{module.__name__}.{name}" for module, name, _ in self._provider_patches),
            "verdict": ("fail-closed: no live provider could be constructed and no non-allowlisted host was reachable" if True else ""),
        }


GUARD = FailClosedGuard()


# ---------------------------------------------------------------------------
# Deterministic provider
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One synthetic provider response (one ``stream()`` call)."""

    text_chunks: tuple[str, ...] = ()
    thinking_chunks: tuple[str, ...] = ()
    chunk_delay_s: float = 0.0
    tool_calls: tuple[dict[str, Any], ...] = ()
    hold_before_first_chunk: asyncio.Event | None = None
    hold_before_done: asyncio.Event | None = None
    hold_timeout_s: float = 3.0


@dataclass
class StepOutcome:
    """What one synthetic step actually yielded (observed, not assumed)."""

    index: int
    text_chunks: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    first_text_monotonic: float | None = None
    last_text_monotonic: float | None = None
    entered_monotonic: float | None = None
    exited_monotonic: float | None = None


def _safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


class DeterministicProvider(LLMProvider):
    """Injected provider: scripted chunks + full request recording.

    ``stream()`` serves one scripted step per STAR-loop iteration. ``chat()``
    serves compaction and any other non-streaming call. Both record the request
    they received, so the benchmark can prove what actually reached the model.
    """

    def __init__(
        self,
        script: Sequence[Step] | None = None,
        *,
        default_text: str = "ok",
        replay_metadata: bool = False,
    ) -> None:
        self.name = "deterministic-bench"
        self.model = "bench-deterministic"
        # Provider-style replay fingerprint, exposed so the agent's
        # ``_provider_fingerprint()`` lookup has something to read (the real
        # providers expose ``provider:model_family:endpoint_hash``).
        self.fingerprint = "deterministic-bench:bench-deterministic:00000000"
        self.script: list[Step] = list(script or [])
        self.default_text = default_text
        # Exercise a terminal replay carrier even on revisions that do not
        # declare or consume it yet. The same fixture runs on each baseline.
        self.replay_metadata = replay_metadata
        self.requests: list[dict[str, Any]] = []
        self.step_outcomes: list[StepOutcome] = []
        self.chat_calls: list[dict[str, Any]] = []
        self.stream_calls = 0
        self.chat_call_count = 0

    @property
    def supports_native_tools(self) -> bool:
        return True

    def convert_multimodal_content(self, blocks: list[dict]) -> list:
        return list(blocks)

    def prepare_tools(self, tools: list) -> list[dict]:
        return list(tools or [])

    def _record(self, kind: str, messages: list[LLMMessage], tools: Any, kwargs: dict) -> dict[str, Any]:
        record = {
            "index": len(self.requests),
            "kind": kind,
            "monotonic_s": time.perf_counter(),
            "messages": serialize_messages(messages),
            "tool_schema_count": len(tools or []),
            "kwarg_keys": sorted(str(k) for k in kwargs),
        }
        self.requests.append(record)
        return record

    async def stream(self, messages: list[LLMMessage], tools: list | None = None, **kwargs: Any) -> AsyncIterator[LLMStreamChunk]:
        record = self._record("stream", list(messages), tools, kwargs)
        index = self.stream_calls
        self.stream_calls += 1
        step = self.script[index] if index < len(self.script) else Step(text_chunks=(self.default_text,))
        outcome = StepOutcome(index=index, entered_monotonic=time.perf_counter())
        record["step_index"] = index
        self.step_outcomes.append(outcome)
        try:
            if step.hold_before_first_chunk is not None:
                await asyncio.wait_for(step.hold_before_first_chunk.wait(), timeout=step.hold_timeout_s)

            for chunk in step.thinking_chunks:
                if step.chunk_delay_s:
                    await asyncio.sleep(step.chunk_delay_s)
                yield LLMStreamChunk(type="thinking", content=chunk)

            for chunk in step.text_chunks:
                if step.chunk_delay_s:
                    await asyncio.sleep(step.chunk_delay_s)
                if outcome.first_text_monotonic is None:
                    outcome.first_text_monotonic = time.perf_counter()
                outcome.text_chunks.append(chunk)
                yield LLMStreamChunk(type="text_delta", content=chunk)
                outcome.last_text_monotonic = time.perf_counter()

            for call in step.tool_calls:
                outcome.tool_calls.append(call)
                yield LLMStreamChunk(type="tool_use", tool_call=call)

            if step.hold_before_done is not None:
                try:
                    await asyncio.wait_for(step.hold_before_done.wait(), timeout=step.hold_timeout_s)
                except TimeoutError:
                    pass
            if self.replay_metadata:
                completion = LLMStreamChunk(type="response")
                completion.response = LLMResponse(
                    content="".join(step.text_chunks),
                    model=self.model,
                    finish_reason="stop",
                    reasoning_items=[dict(SYNTHETIC_REASONING_ITEM)],
                    response_id=SYNTHETIC_RESPONSE_ID,
                )
                yield completion
        finally:
            outcome.exited_monotonic = time.perf_counter()

    async def chat(self, messages: list[LLMMessage], tools: list | None = None, **kwargs: Any) -> LLMResponse:
        record = self._record("chat", list(messages), tools, kwargs)
        self.chat_call_count += 1
        self.chat_calls.append(record)
        return LLMResponse(
            content='{"summary": "deterministic compaction summary"}',
            model=self.model,
            finish_reason="stop",
            reasoning_items=[dict(SYNTHETIC_REASONING_ITEM)] if self.replay_metadata else None,
            response_id=SYNTHETIC_RESPONSE_ID if self.replay_metadata else None,
        )


def serialize_messages(messages: Iterable[LLMMessage]) -> list[dict[str, Any]]:
    """Serialize provider-facing messages for objective fidelity assertions."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            blocks: list[Any] = []
            for block in content:
                if isinstance(block, dict):
                    blocks.append(
                        {
                            "type": block.get("type"),
                            "text": block.get("text"),
                            "media_type": block.get("media_type") or block.get("mediaType"),
                            "path": block.get("path"),
                            "uri": block.get("uri"),
                        }
                    )
                else:
                    blocks.append(_safe(block))
            content = blocks
        record: dict[str, Any] = {"role": msg.role, "content": content}
        if getattr(msg, "tool_calls", None):
            record["tool_calls"] = [
                {
                    "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None),
                    "name": tc.get("name") if isinstance(tc, dict) else None,
                    "function": _safe(tc.get("function")) if isinstance(tc, dict) else None,
                }
                for tc in msg.tool_calls
            ]
        if getattr(msg, "tool_call_id", None):
            record["tool_call_id"] = msg.tool_call_id
        if getattr(msg, "cache_control", None):
            record["cache_control"] = msg.cache_control
        if getattr(msg, "reasoning_items", None):
            # Record the full carrier so replay fidelity can be asserted exactly
            # (including encrypted_content) rather than by presence only.
            record["reasoning_items"] = _safe(msg.reasoning_items)
            record["reasoning_fingerprint"] = msg.reasoning_fingerprint
            record["response_id"] = msg.response_id
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# Bench resource (auto_register=False - the global registry is never touched)
# ---------------------------------------------------------------------------


class BenchToolResource(BaseResource):
    """Named tools the synthetic provider can call, through the real agent loop."""

    def __init__(self, resource_id: str = "bench-tools") -> None:
        super().__init__(resource_type="bench_tool", resource_id=resource_id, auto_register=False)
        self.calls: list[dict[str, Any]] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    @named_tool(name="bench_echo")
    async def bench_echo(self, text: str = "") -> str:
        """Echo the supplied text back (deterministic, no I/O)."""
        self.calls.append({"tool": "bench_echo", "monotonic_s": time.perf_counter(), "text": text})
        return f"echo:{text}"

    @named_tool(name="bench_block")
    async def bench_block(self, seconds: float = 5.0) -> str:
        """Block until the benchmark releases the gate (or ``seconds`` elapse)."""
        self.calls.append({"tool": "bench_block", "monotonic_s": time.perf_counter()})
        self.entered.set()
        try:
            await asyncio.wait_for(self.release.wait(), timeout=seconds)
        except TimeoutError:
            return "blocked:timeout"
        return "released"


def tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"id": call_id, "name": name, "input": arguments}


# ---------------------------------------------------------------------------
# Workspace / harness plumbing
# ---------------------------------------------------------------------------


class EphemeralKeyProvider:
    """Protected-state key provider backed by a per-run random key (no env secret)."""

    def __init__(self, key: bytes) -> None:
        self._key = key

    def key(self) -> bytes:
        return self._key


def build_temp_repository_factory(workspace: Path) -> RepositoryFactory:
    """Repository factory whose every backend writes only under ``workspace``.

    Mirrors the shape of the repository factories the app adapters use, but the
    storage root is temporary, so no default store (``~/.dana``) is touched.
    """
    factory = RepositoryFactory()
    cfg = FileStorageConfig(workspace_folder=str(workspace))
    for repo_type, creator in (
        (RepositoryType.TIMELINE, LocalTimelineRepository),
        (RepositoryType.EVENT, LocalEventRepository),
        (RepositoryType.LEARNING, LocalLearningRepository),
        (RepositoryType.PROMPT, LocalPromptRepository),
    ):
        factory.register(repo_type, creator, cfg)
    return factory


@dataclass
class LaneConfig:
    name: str
    isolated_context: bool
    clear_mcp_env: bool
    allowed_hosts: tuple[str, ...]
    enable_skills: bool
    enable_assistant: bool


@dataclass
class Harness:
    """One temporary workspace + one agent + one journal-backed session."""

    tmpdir: tempfile.TemporaryDirectory
    workspace: Path
    journal_path: Path
    scope: OwnerScope
    session_id: str
    factory: RepositoryFactory
    repo: SQLiteJournalRepository
    provider: DeterministicProvider
    key: bytes
    compress_timeline: bool
    agent: Any
    session: AgentSession
    build_ms: float
    tool_resource: BenchToolResource | None = None


def make_agent(
    lane: LaneConfig,
    factory: RepositoryFactory,
    provider: DeterministicProvider,
    *,
    compress_timeline: bool,
    max_context_tokens: int,
    tool_resource: BenchToolResource | None,
) -> Any:
    agent = STARAgent(
        agent_id="bench-session-agent",
        agent_type="bench_session_agent",
        llm_provider_instance=provider,
        codec=NativeToolsCodec,
        repository_factory=factory,
        auto_register=False,
        enable_skills=lane.enable_skills,
        enable_assistant=lane.enable_assistant,
        compress_timeline=compress_timeline,
        max_context_tokens=max_context_tokens,
        identity_override="Benchmark identity for deterministic session measurements.",
    )
    if tool_resource is not None:
        agent.with_resources(tool_resource)
    if lane.isolated_context:
        # Instance-level rebinding only; the class, the module and the
        # production default (location) are untouched.
        agent._runtime.get_runtime_context_fields = types.MethodType(lambda self: ["timestamp", "timezone", "user"], agent._runtime)
    return agent


def seed_session_fact(scope: OwnerScope, session_id: str) -> JournalFact:
    return JournalFact(
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
    )


async def open_harness(
    lane: LaneConfig,
    *,
    script: Sequence[Step] | None = None,
    session_id: str | None = None,
    resume: bool = False,
    compress_timeline: bool = True,
    with_tools: bool = False,
    max_context_tokens: int = 20000,
    provider_replay_metadata: bool = False,
) -> Harness:
    """Build one isolated harness: temp workspace, repos, agent, session, load()."""
    t0 = time.perf_counter()
    tmpdir = tempfile.TemporaryDirectory(prefix="dana-bench-session-")
    workspace = Path(tmpdir.name)
    journal_path = workspace / "journal.db"
    factory = build_temp_repository_factory(workspace)
    scope = OwnerScope(owner_id="bench-session-script", workspace=str(workspace))
    sid = session_id or str(uuid4())
    provider = DeterministicProvider(script, replay_metadata=provider_replay_metadata)
    key = os.urandom(32)
    tool_resource = BenchToolResource() if with_tools else None

    repo = await SQLiteJournalRepository.open(str(journal_path))
    if resume:
        await recover_interrupted_turns(repo, scope, sid)
    else:
        await repo.create_session(SessionRecord.new(sid, scope), [seed_session_fact(scope, sid)])

    def agent_factory() -> Any:
        return make_agent(
            lane,
            factory,
            provider,
            compress_timeline=compress_timeline,
            max_context_tokens=max_context_tokens,
            tool_resource=tool_resource,
        )

    session = AgentSession(
        owner_scope=scope,
        session_id=sid,
        repository=repo,
        agent_factory=agent_factory,
        protected_state_codec=ProtectedStateCodec(EphemeralKeyProvider(key)),
    )
    await session.load()
    return Harness(
        tmpdir=tmpdir,
        workspace=workspace,
        journal_path=journal_path,
        scope=scope,
        session_id=sid,
        factory=factory,
        repo=repo,
        provider=provider,
        key=key,
        compress_timeline=compress_timeline,
        agent=session._agent,
        session=session,
        build_ms=(time.perf_counter() - t0) * 1000.0,
        tool_resource=tool_resource,
    )


async def close_harness(harness: Harness) -> None:
    """Release every resource the harness owns."""
    if harness.tool_resource is not None:
        harness.tool_resource.release.set()
    unregister = getattr(harness.session, "_unregister_policy_hook", None)
    if callable(unregister):
        try:
            unregister()
        except Exception:  # pragma: no cover - best-effort teardown
            pass
    dispose = getattr(harness.session, "dispose_mcp", None)
    if callable(dispose):
        try:
            await dispose()
        except Exception:  # pragma: no cover - best-effort teardown
            pass
    try:
        await harness.repo.close()
    except Exception:  # pragma: no cover - best-effort teardown
        pass
    await drain_pending_tasks()
    try:
        harness.tmpdir.cleanup()
    except Exception:  # pragma: no cover - best-effort teardown
        pass


async def drain_pending_tasks() -> int:
    """Cancel+await tasks left over by fire-and-forget reflection/learning."""
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    return len(pending)


# ---------------------------------------------------------------------------
# Turn driving
# ---------------------------------------------------------------------------


@dataclass
class TurnTrace:
    """Observed timeline of one turn, in monotonic seconds."""

    start: float
    events: list[dict[str, Any]] = field(default_factory=list)
    first_host_thought: float | None = None
    first_host_chunk: float | None = None
    first_host_final: float | None = None
    first_tool_requested: float | None = None
    first_tool_started: float | None = None
    first_tool_result: float | None = None
    terminal: float | None = None
    terminal_type: str | None = None
    final_text: str | None = None
    chunks_after_terminal: int = 0
    thoughts_after_terminal: int = 0

    def note_event(self, event: Any, at: float) -> None:
        value = event.event_type.value
        self.events.append({"type": value, "text": _safe(event.text), "metadata_keys": sorted(event.metadata or {})})
        if value == "thought" and self.first_host_thought is None:
            self.first_host_thought = at
        if value == "assistant_content_chunk" and self.first_host_chunk is None:
            self.first_host_chunk = at
        if value == "assistant_content_final":
            if self.first_host_final is None:
                self.first_host_final = at
            self.final_text = event.text
        if value == "tool_requested" and self.first_tool_requested is None:
            self.first_tool_requested = at
        if value == "tool_started" and self.first_tool_started is None:
            self.first_tool_started = at
        if value in ("tool_result", "tool_failure") and self.first_tool_result is None:
            self.first_tool_result = at
        if value in TERMINAL_EVENT_VALUES:
            self.terminal = at
            self.terminal_type = value
        elif self.terminal is not None:
            if value == "assistant_content_chunk":
                self.chunks_after_terminal += 1
            elif value == "thought":
                self.thoughts_after_terminal += 1

    def ms(self, value: float | None) -> float | None:
        if value is None:
            return None
        return round((value - self.start) * 1000.0, 4)

    def complete_ms(self) -> float:
        terminal = self.terminal if self.terminal is not None else time.perf_counter()
        return round((terminal - self.start) * 1000.0, 4)

    def event_types(self) -> list[str]:
        return [e["type"] for e in self.events]

    def count(self, name: str) -> int:
        return sum(1 for e in self.events if e["type"] == name)


async def consume_turn(
    session: AgentSession,
    text: str,
    *,
    content_blocks: list[dict] | None = None,
) -> TurnTrace:
    """Consume a full turn, timestamping each host event."""
    trace = TurnTrace(start=time.perf_counter())
    async for event in session.prompt([TextBlock(text=text)], content_blocks=content_blocks):
        trace.note_event(event, time.perf_counter())
    if trace.terminal is None:
        trace.terminal = time.perf_counter()
        trace.terminal_type = "missing-terminal"
    return trace


def terminal_summary(session: AgentSession) -> dict[str, Any]:
    last = session.last_terminal
    if last is None:
        return {"fact_type": None, "sequence": None}
    return {
        "fact_type": last.fact_type.name,
        "fact_type_value": last.fact_type.value,
        "sequence": last.sequence,
        "text_len": len(last.text) if last.text is not None else None,
        "error": last.error,
    }


def stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "median": round(statistics.median(ordered), 4),
        "max": round(ordered[-1], 4),
        "mean": round(statistics.fmean(ordered), 4),
        "p95": round(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)], 4),
    }


def first_text_after(provider: DeterministicProvider, request_index: int) -> float | None:
    for outcome in provider.step_outcomes:
        if outcome.index >= request_index and outcome.first_text_monotonic is not None:
            return outcome.first_text_monotonic
    return None


def fd_count() -> int | None:
    """Open file-descriptor count (macOS/Linux) for the resource-cleanup check."""
    try:
        return len(os.listdir("/dev/fd"))
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Cases - request fidelity
# ---------------------------------------------------------------------------


async def case_request_fidelity(lane: LaneConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Capture the exact provider request for a plain turn (compressed/uncompressed)."""
    compress = args.compress_timeline
    harness = await open_harness(lane, script=[Step(text_chunks=("answers",))], compress_timeline=compress)
    try:
        trace = await consume_turn(harness.session, "fidelity question")
        request = harness.provider.requests[0]
        return {
            "samples_ms": {"turn_complete_ms": {"n": 1, "value": trace.complete_ms()}},
            "raw": {"turn_complete_monotonic_samples": [trace.complete_ms()]},
            "per_sample": [
                {
                    "compress_timeline": compress,
                    "provider_request": request,
                    "request_roles": [m["role"] for m in request["messages"]],
                    "tool_schema_count": request["tool_schema_count"],
                    "terminal": terminal_summary(harness.session),
                }
            ],
            "observation": {
                "measures": "the exact messages + tools the runtime built for one plain turn",
                "compress_timeline": compress,
                "runtime_context_fields": sorted(harness.agent._runtime._get_runtime_context().keys()),
                "note": (
                    "compress_timeline=False disables compaction but STARAgent still constructs a CompressedTimeline; "
                    "the plain Timeline class is unreachable through the real agent path (coverage gap)"
                    if not compress
                    else "default production setting"
                ),
                "sample_count_disclosure": "1 sample (payload capture, not a timing case)",
            },
        }
    finally:
        await close_harness(harness)


# ---------------------------------------------------------------------------
# Cases - timing
# ---------------------------------------------------------------------------


async def case_construction(lane: LaneConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Agent + journal + session construction plus the first ``load()``."""
    raw: list[float] = []
    for _ in range(args.construction_repetitions):
        harness = await open_harness(lane)
        try:
            raw.append(round(harness.build_ms, 4))
        finally:
            await close_harness(harness)
    return {
        "samples_ms": {"session_build_ms": stats(raw)},
        "raw": {"session_build_monotonic_samples": raw},
        "observation": {
            "measures": "temp workspace + repository factory + SQLite journal + STARAgent + AgentSession + load()",
            "excludes": "process start / interpreter imports / provider SDK construction",
            "sample_count_disclosure": (
                f"{args.construction_repetitions} samples (each builds a full workspace + journal + agent; "
                "deliberately not the 100-sample cheap budget)"
            ),
        },
    }


async def case_first_use_turn(lane: LaneConfig, args: argparse.Namespace) -> dict[str, Any]:
    """First-use turn on a fresh session (fresh agent, fresh empty journal)."""
    raw: list[float] = []
    provider_text_ms: list[float] = []
    host_chunk_ms: list[float] = []
    context_keys: list[str] = []
    cache_states: list[Any] = []
    for _ in range(args.repetitions):
        harness = await open_harness(lane, script=[Step(text_chunks=("first-use answer",))])
        try:
            if not context_keys:
                context_keys = sorted(harness.agent._runtime._get_runtime_context().keys())
            trace = await consume_turn(harness.session, "first use prompt")
            raw.append(trace.complete_ms())
            first = first_text_after(harness.provider, 0)
            if first is not None:
                provider_text_ms.append(round((first - trace.start) * 1000.0, 4))
            if trace.first_host_chunk is not None:
                host_chunk_ms.append(trace.ms(trace.first_host_chunk))
            if not lane.isolated_context:
                cache_states.append(_safe(getattr(type(harness.agent._runtime), "_cached_location", "unset")))
        finally:
            await close_harness(harness)
    observation: dict[str, Any] = {
        "measures": "first turn on an empty session, one immediate provider chunk",
        "excludes": "agent construction and process imports",
        "runtime_context_keys_first_sample": context_keys,
        "sample_count_disclosure": f"{args.repetitions} samples",
    }
    if not lane.isolated_context:
        from dana.core.runtime.base import AgentRuntime

        observation["geolocation"] = {
            "class_cache_state_after_samples": cache_states[:3],
            "class_cache_reset_during_run": False,
            "note": (
                "the production location field is present and the production geolocation path may run once per "
                "process (class-level cache); sample 0 is therefore the only sample that can include the lookup, so "
                "turn_complete_ms_excluding_first_sample is reported alongside"
            ),
            "turn_complete_ms_excluding_first_sample": stats(raw[1:]),
            "class_cache_value": _safe(AgentRuntime._cached_location),
        }
    return {
        "samples_ms": {
            "turn_complete_ms": stats(raw),
            "provider_first_text_ms": stats(provider_text_ms),
            "host_first_assistant_chunk_ms": stats(host_chunk_ms),
        },
        "raw": {
            "turn_complete_monotonic_samples": raw,
            "provider_first_text_monotonic_samples": provider_text_ms,
            "host_first_chunk_monotonic_samples": host_chunk_ms,
        },
        "observation": observation,
    }


async def case_warm_turns(lane: LaneConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Warm no-tool turns on one long-lived session; declared warm-up excluded."""
    harness = await open_harness(lane, script=[Step(text_chunks=("warm-up answer",))])
    try:
        warmup = await consume_turn(harness.session, "warm-up turn")
        raw: list[float] = []
        provider_text_ms: list[float] = []
        host_chunk_ms: list[float] = []
        thought_ms: list[float] = []
        terminals: list[str] = []
        fd_before = fd_count()
        for i in range(args.repetitions):
            harness.provider.script.append(Step(text_chunks=(f"warm answer {i}",)))
            before_request = len(harness.provider.requests)
            trace = await consume_turn(harness.session, f"warm prompt {i}")
            raw.append(trace.complete_ms())
            first = first_text_after(harness.provider, before_request)
            if first is not None:
                provider_text_ms.append(round((first - trace.start) * 1000.0, 4))
            if trace.first_host_chunk is not None:
                host_chunk_ms.append(trace.ms(trace.first_host_chunk))
            if trace.first_host_thought is not None:
                thought_ms.append(trace.ms(trace.first_host_thought))
            terminals.append(str(terminal_summary(harness.session)["fact_type"]))
        return {
            "samples_ms": {
                "turn_complete_ms": stats(raw),
                "provider_first_text_ms": stats(provider_text_ms),
                "host_first_chunk_ms": stats(host_chunk_ms),
                "host_first_thought_ms": stats(thought_ms),
            },
            "raw": {
                "turn_complete_monotonic_samples": raw,
                "provider_first_text_monotonic_samples": provider_text_ms,
            },
            "observation": {
                "measures": f"{args.repetitions} consecutive no-tool turns on one session/agent",
                "warmups_declared": args.warmup_turns,
                "warmup_excluded_ms": warmup.complete_ms(),
                "context_growth": "each turn appends to the journal + timeline; later turns carry more history",
                "terminal_fact_types": sorted(set(terminals)),
                "compaction_chat_calls_observed": harness.provider.chat_call_count,
                "fd_delta_over_case": None if fd_before is None else fd_count() - fd_before,
                "sample_count_disclosure": f"{args.repetitions} samples (+1 declared warm-up)",
            },
        }
    finally:
        await close_harness(harness)


async def case_streaming_text_only(lane: LaneConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Delayed provider text, no tools: provider text vs host thought vs assistant text."""
    per_sample: list[dict[str, Any]] = []
    samples: dict[str, list[float]] = {
        "provider_first_text_ms": [],
        "last_provider_chunk_ms": [],
        "host_first_thought_ms": [],
        "host_first_assistant_chunk_ms": [],
        "turn_complete_ms": [],
    }
    for _ in range(args.delayed_repetitions):
        harness = await open_harness(lane, script=[Step(text_chunks=("Hello", " from", " Dana."), chunk_delay_s=args.stream_chunk_delay_s)])
        try:
            trace = await consume_turn(harness.session, "stream a greeting")
            outcome = harness.provider.step_outcomes[0]
            entry = {
                "provider_first_text_ms": (
                    round((outcome.first_text_monotonic - trace.start) * 1000.0, 4) if outcome.first_text_monotonic else None
                ),
                "last_provider_chunk_ms": (
                    round((outcome.last_text_monotonic - trace.start) * 1000.0, 4) if outcome.last_text_monotonic else None
                ),
                "host_first_thought_ms": trace.ms(trace.first_host_thought),
                "host_first_assistant_chunk_ms": trace.ms(trace.first_host_chunk),
                "host_terminal_ms": trace.complete_ms(),
                "thought_count": trace.count("thought"),
                "host_assistant_chunk_count": trace.count("assistant_content_chunk"),
                "event_sequence": trace.event_types(),
                "final_text": trace.final_text,
                "terminal": terminal_summary(harness.session),
            }
            per_sample.append(entry)
            if outcome.first_text_monotonic:
                samples["provider_first_text_ms"].append(round((outcome.first_text_monotonic - trace.start) * 1000.0, 4))
            if outcome.last_text_monotonic:
                samples["last_provider_chunk_ms"].append(round((outcome.last_text_monotonic - trace.start) * 1000.0, 4))
            if trace.first_host_thought is not None:
                samples["host_first_thought_ms"].append(trace.ms(trace.first_host_thought))
            if trace.first_host_chunk is not None:
                samples["host_first_assistant_chunk_ms"].append(trace.ms(trace.first_host_chunk))
            samples["turn_complete_ms"].append(trace.complete_ms())
        finally:
            await close_harness(harness)
    gap = None
    if samples["host_first_assistant_chunk_ms"] and samples["last_provider_chunk_ms"]:
        gap = round(statistics.median(samples["host_first_assistant_chunk_ms"]) - statistics.median(samples["last_provider_chunk_ms"]), 4)
    delayed = gap is not None and gap >= 0
    return {
        "samples_ms": {key: stats(value) for key, value in samples.items()},
        "raw": dict(samples),
        "per_sample": per_sample[:3],
        "observation": {
            "measures": f"3 provider text chunks {args.stream_chunk_delay_s}s apart, no tools",
            "host_text_delivery": (
                "observed: each provider text chunk is delivered as a THOUGHT while streaming; the assistant text "
                "event arrives only after the provider stream ends (end-of-generation buffering)"
                if delayed
                else "observed: host assistant text was not delayed relative to the last provider chunk"
            ),
            "end_of_generation_buffering_gap_ms": gap,
            "sample_count_disclosure": f"{args.delayed_repetitions} samples (each injects real per-chunk delays)",
        },
    }


async def case_text_then_tools(lane: LaneConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Delayed narration text followed by a tool call, then a final answer."""
    per_sample: list[dict[str, Any]] = []
    turn_ms: list[float] = []
    provider_first_ms: list[float] = []
    for i in range(args.delayed_repetitions):
        script = [
            Step(
                text_chunks=("Let me check", " that."),
                chunk_delay_s=args.stream_chunk_delay_s,
                tool_calls=(tool_call(f"call-{i}", "bench_echo", {"text": "probe"}),),
            ),
            Step(text_chunks=("All good.",)),
        ]
        harness = await open_harness(lane, script=script, with_tools=True)
        try:
            trace = await consume_turn(harness.session, "use the tool")
            outcome = harness.provider.step_outcomes[0]
            entry = {
                "provider_first_text_ms": (
                    round((outcome.first_text_monotonic - trace.start) * 1000.0, 4) if outcome.first_text_monotonic else None
                ),
                "host_first_thought_ms": trace.ms(trace.first_host_thought),
                "host_first_assistant_chunk_ms": trace.ms(trace.first_host_chunk),
                "host_first_tool_requested_ms": trace.ms(trace.first_tool_requested),
                "host_first_tool_result_ms": trace.ms(trace.first_tool_result),
                "host_first_assistant_chunk_count": trace.count("assistant_content_chunk"),
                "thought_count": trace.count("thought"),
                "turn_complete_ms": trace.complete_ms(),
                "provider_stream_requests": len(harness.provider.requests),
                "tool_resource_calls": list(harness.tool_resource.calls) if harness.tool_resource else [],
                "event_sequence": trace.event_types(),
                "terminal": terminal_summary(harness.session),
            }
            per_sample.append(entry)
            turn_ms.append(entry["turn_complete_ms"])
            if entry["provider_first_text_ms"] is not None:
                provider_first_ms.append(entry["provider_first_text_ms"])
        finally:
            await close_harness(harness)
    narration_only = all(s["host_first_assistant_chunk_count"] == 0 for s in per_sample) if per_sample else None
    return {
        "samples_ms": {"turn_complete_ms": stats(turn_ms), "provider_first_text_ms": stats(provider_first_ms)},
        "raw": {"turn_complete_monotonic_samples": turn_ms},
        "per_sample": per_sample,
        "observation": {
            "measures": "delayed narration text + one native tool call, then a final-answer iteration",
            "narration_delivery": (
                "observed: narration was delivered as THOUGHT events and the tool ran; assistant text is emitted only for the final answer"
                if narration_only
                else "recorded: see per_sample event sequences for the actual thought/assistant split"
            ),
            "sample_count_disclosure": f"{args.delayed_repetitions} samples (two provider iterations each)",
        },
    }


# ---------------------------------------------------------------------------
# Cases - cancellation
# ---------------------------------------------------------------------------


async def cancel_case(
    lane: LaneConfig,
    args: argparse.Namespace,
    *,
    case: str,
    script: Sequence[Step],
    trigger: str,
    with_tools: bool = False,
) -> dict[str, Any]:
    """Start a turn, cancel at ``trigger``, and record what the session actually did."""
    per_sample: list[dict[str, Any]] = []
    cancel_to_terminal_ms: list[float] = []
    for _ in range(args.delayed_repetitions):
        harness = await open_harness(lane, script=script, with_tools=with_tools)
        try:
            trace = TurnTrace(start=time.perf_counter())
            first_chunk = asyncio.Event()
            tool_started = asyncio.Event()

            async def consume(harness=harness, trace=trace, first_chunk=first_chunk, tool_started=tool_started) -> None:
                async for event in harness.session.prompt([TextBlock(text="cancel me")]):
                    trace.note_event(event, time.perf_counter())
                    if event.event_type.value == "assistant_content_chunk":
                        first_chunk.set()
                    if event.event_type.value == "tool_started":
                        tool_started.set()

            consumer = asyncio.create_task(consume())
            deadline = time.perf_counter() + args.timeout_seconds
            while not harness.provider.requests and time.perf_counter() < deadline:
                await asyncio.sleep(0.001)

            if trigger == "midstream":
                await asyncio.wait_for(first_chunk.wait(), timeout=args.timeout_seconds)
            elif trigger == "during_tool":
                await asyncio.wait_for(tool_started.wait(), timeout=args.timeout_seconds)
            elif trigger == "before_done":
                while (
                    harness.provider.step_outcomes
                    and harness.provider.step_outcomes[0].last_text_monotonic is None
                    and time.perf_counter() < deadline
                ):
                    await asyncio.sleep(0.001)
            elif trigger == "before_first_token":
                await asyncio.sleep(args.cancel_recognise_delay_s)  # provider parked in its pre-chunk hold

            cancel_at = time.perf_counter()
            await harness.session.cancel()
            release_at: float | None = None
            await asyncio.sleep(args.cancel_release_delay_s)
            if trigger == "during_tool" and harness.tool_resource is not None:
                harness.tool_resource.release.set()
                release_at = time.perf_counter()
            elif trigger in ("midstream", "before_done"):
                released = False
                for step in script:
                    if step.hold_before_done is not None:
                        step.hold_before_done.set()
                        released = True
                if released:
                    release_at = time.perf_counter()

            observed_completion = True
            try:
                await asyncio.wait_for(consumer, timeout=args.timeout_seconds)
            except TimeoutError:
                observed_completion = False
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)

            tool_calls = harness.tool_resource.calls if harness.tool_resource else []
            entry = {
                "cancel_trigger": trigger,
                "cancel_requested_ms": round((cancel_at - trace.start) * 1000.0, 4),
                "hold_released_after_cancel_ms": (round((release_at - cancel_at) * 1000.0, 4) if release_at is not None else None),
                "consumer_completed": observed_completion,
                "terminal_event": trace.terminal_type,
                "terminal": terminal_summary(harness.session),
                "cancel_to_terminal_ms": (round((trace.terminal - cancel_at) * 1000.0, 4) if trace.terminal is not None else None),
                "late_chunks_after_terminal": trace.chunks_after_terminal,
                "late_thoughts_after_terminal": trace.thoughts_after_terminal,
                "event_sequence": trace.event_types(),
                "tool_completed_despite_cancel": len(tool_calls) > 0,
                "tool_resource_calls": list(tool_calls),
                "journal_fact_types": [f.fact_type.value for f in await harness.repo.read_facts(harness.scope, harness.session_id)],
                "compaction_chat_calls_observed": harness.provider.chat_call_count,
            }
            if trace.terminal is not None:
                cancel_to_terminal_ms.append(round((trace.terminal - cancel_at) * 1000.0, 4))
            per_sample.append(entry)
        finally:
            await close_harness(harness)
    recognised = all(v < args.prompt_cancel_bound_ms for v in cancel_to_terminal_ms) if cancel_to_terminal_ms else None
    return {
        "samples_ms": {"cancel_to_terminal_ms": stats(cancel_to_terminal_ms)},
        "raw": {"cancel_to_terminal_ms_monotonic_samples": cancel_to_terminal_ms},
        "per_sample": per_sample[:3],
        "observation": {
            "case": case,
            "cancel_trigger": trigger,
            "terminal_fact_types": sorted({s["terminal"]["fact_type"] for s in per_sample if s}),
            "terminal_events": sorted({s["terminal_event"] for s in per_sample if s}),
            "cancel_recognised_within_bound": recognised,
            "cancel_bound_ms": args.prompt_cancel_bound_ms,
            "mechanism": (
                "AgentSession.prompt checks its cancel event only between agent stream events, and the agent loop does "
                "not check it at all, so a cancellation issued while the provider or a tool is parked is recognised at "
                "the next delivered event"
            ),
            "sample_count_disclosure": f"{args.delayed_repetitions} samples (each injects a real provider/tool hold)",
        },
    }


async def case_cancel_after_terminal(lane: LaneConfig, args: argparse.Namespace) -> dict[str, Any]:
    """``cancel()`` outside an active turn - records the actual API behaviour."""
    observations: list[dict[str, Any]] = []
    for _ in range(min(3, args.delayed_repetitions)):
        harness = await open_harness(lane, script=[Step(text_chunks=("done",))])
        try:
            trace = await consume_turn(harness.session, "complete then cancel")
            try:
                await harness.session.cancel()
                outcome = {"raised": None, "returned": True}
            except Exception as exc:  # noqa: BLE001 - the raised type is the observation
                outcome = {"raised": type(exc).__name__, "message": str(exc), "returned": False}
            observations.append(
                {
                    "terminal_event": trace.terminal_type,
                    "terminal": terminal_summary(harness.session),
                    "cancel_after_terminal": outcome,
                }
            )
        finally:
            await close_harness(harness)
    return {
        "samples_ms": {},
        "raw": {},
        "per_sample": observations,
        "observation": {
            "measures": "cancel() called after the turn already reached a terminal fact",
            "observed_outcomes": [o["cancel_after_terminal"]["raised"] for o in observations],
            "note": "recorded as observed behaviour (the API rejects cancelling a session with no active turn)",
        },
    }


# ---------------------------------------------------------------------------
# Cases - close/reopen fidelity
# ---------------------------------------------------------------------------


async def reopen_case(
    lane: LaneConfig,
    args: argparse.Namespace,
    *,
    case: str,
    first_script: Sequence[Step],
    first_prompt: str,
    second_prompt: str,
    check: str,
    with_tools: bool = False,
    content_blocks: list[dict] | None = None,
    seed_interrupted_turn: bool = False,
    compress: bool = True,
) -> dict[str, Any]:
    """Run a turn, close the journal, reopen the session, capture the reopened request."""
    per_sample: list[dict[str, Any]] = []
    for _ in range(args.reopen_repetitions):
        harness = await open_harness(lane, script=list(first_script), with_tools=with_tools, compress_timeline=compress)
        try:
            if seed_interrupted_turn:
                # The genuine interrupted-turn journal shape: a TURN_STARTED +
                # USER_CONTENT_FINAL with no terminal fact, i.e. a process crash
                # mid-turn. The cancel path always terminalizes (TURN_CANCELLED),
                # so this shape is seeded directly rather than faked.
                corr = str(uuid4())
                await harness.session.append_fact(
                    NewJournalFact(
                        fact_type=FactType.TURN_STARTED,
                        correlation_id=corr,
                        causation_id=None,
                        payload={"prompt_summary": first_prompt},
                    )
                )
                await harness.session.append_fact(
                    NewJournalFact(
                        fact_type=FactType.USER_CONTENT_FINAL,
                        correlation_id=corr,
                        causation_id=corr,
                        payload={"text": first_prompt},
                    )
                )
                first_terminal = {"fact_type": None, "sequence": None, "note": "seeded crash shape; no terminal"}
                first_events = []
                first_request_messages = []
            else:
                trace1 = await consume_turn(harness.session, first_prompt, content_blocks=content_blocks)
                first_terminal = terminal_summary(harness.session)
                first_events = trace1.event_types()
                first_request_messages = harness.provider.requests[0]["messages"] if harness.provider.requests else []
            await harness.repo.close()
            sid = harness.session_id
            scope = harness.scope
            journal_path = harness.journal_path
            factory = harness.factory
            key = harness.key
            compress = harness.compress_timeline
        finally:
            if harness.tool_resource is not None:
                harness.tool_resource.release.set()
            await drain_pending_tasks()

        recovered = 0
        reopen_provider = DeterministicProvider([Step(text_chunks=("second answer",))])
        reopen_tool = BenchToolResource() if with_tools else None

        def reopen_factory(factory=factory, reopen_provider=reopen_provider, compress=compress, reopen_tool=reopen_tool) -> Any:
            return make_agent(
                lane,
                factory,
                reopen_provider,
                compress_timeline=compress,
                max_context_tokens=20000,
                tool_resource=reopen_tool,
            )

        repo2 = await SQLiteJournalRepository.open(str(journal_path))
        try:
            recovered = await recover_interrupted_turns(repo2, scope, sid)
            session2 = AgentSession(
                owner_scope=scope,
                session_id=sid,
                repository=repo2,
                agent_factory=reopen_factory,
                protected_state_codec=ProtectedStateCodec(EphemeralKeyProvider(key)),
            )
            await session2.load()
            facts = await repo2.read_facts(scope, sid)
            view = ConversationProjector(ProtectedStateCodec(EphemeralKeyProvider(key))).project(facts)
            trace2 = await consume_turn(session2, second_prompt)
            second_request = reopen_provider.requests[0] if reopen_provider.requests else {"messages": [], "tool_schema_count": 0}
            payload = {
                "check": check,
                "recovered_interrupted_turns": recovered,
                "journal_fact_types": [f.fact_type.value for f in facts],
                "first_turn": {
                    "terminal": first_terminal,
                    "event_sequence": first_events,
                    "request_messages": first_request_messages,
                },
                "projection": {
                    "roles": [m.role for m in view.messages],
                    "interruption_observation": _safe(view.interruption_observation),
                    "replay_state_len": len(view.replay_state) if view.replay_state else 0,
                    "model_changes": [_safe(dict(c)) for c in view.model_changes],
                    "current_provider": view.current_provider,
                    "current_model": view.current_model,
                },
                "second_turn": {
                    "request_messages": second_request["messages"],
                    "tool_schema_count": second_request["tool_schema_count"],
                    "terminal": terminal_summary(session2),
                    "event_sequence": trace2.event_types(),
                },
            }
            payload["verdict"] = _verdict(check, payload)
            per_sample.append(payload)
        finally:
            if reopen_tool is not None:
                reopen_tool.release.set()
            await repo2.close()
            await drain_pending_tasks()
            try:
                harness.tmpdir.cleanup()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
    verdicts = sorted({p["verdict"]["status"] for p in per_sample})
    return {
        "samples_ms": {},
        "raw": {},
        "per_sample": per_sample,
        "observation": {
            "check": check,
            "verdicts": verdicts,
            "consistent_across_samples": len(verdicts) == 1,
            "sample_count_disclosure": f"{args.reopen_repetitions} samples (each performs a close/reopen cycle)",
        },
    }


def _verdict(check: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Objective fidelity verdict computed from the captured provider request."""
    messages = payload["second_turn"]["request_messages"]
    if check == "compressed_history":
        user_texts = [m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], str)]
        assistant_texts = [m["content"] for m in messages if m["role"] == "assistant" and isinstance(m["content"], str)]
        prior_user = any("reopen-marker" in (t or "") for t in user_texts)
        return {
            "status": "pass" if prior_user else "fail",
            "expected": "the prior user turn is present in the reopened provider request",
            "observed_prior_user_turn": prior_user,
            "observed_prior_assistant_turn": any("acknowledged" in (t or "") for t in assistant_texts),
            "projected_roles": payload["projection"]["roles"],
            "request_roles": [m["role"] for m in messages],
        }
    if check == "interrupted_turn":
        observation = payload["projection"]["interruption_observation"]
        return {
            "status": "pass" if observation else "fail",
            "expected": "an interrupted-turn observation is projected for the model",
            "observed_observation": _safe(observation),
            "observation_in_request": any("interrupt" in str(m.get("content", "")).lower() for m in messages),
            "recovered_interrupted_turns": payload["recovered_interrupted_turns"],
            "journal_fact_types": payload["journal_fact_types"],
        }
    if check == "multimodal":
        block_messages = [m for m in messages if isinstance(m["content"], list)]
        str_users = [m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], str)]
        return {
            "status": "pass" if block_messages else "fail",
            "expected": "the multimodal user message reaches the provider as content blocks",
            "observed_block_messages": len(block_messages),
            "observed_block_types": sorted({b.get("type") for m in block_messages for b in m["content"] if isinstance(b, dict)}),
            "observed_str_user_messages": str_users,
            "first_turn_request_messages": payload["first_turn"]["request_messages"],
        }
    if check == "tool_pair":
        tool_messages = [m for m in messages if m["role"] == "tool"]
        assistant_with_calls = [m for m in messages if m.get("tool_calls")]
        return {
            "status": "pass" if (tool_messages and assistant_with_calls) else "fail",
            "expected": "an assistant tool_calls message paired with its role=tool result",
            "observed_tool_role_messages": len(tool_messages),
            "observed_assistant_tool_call_messages": len(assistant_with_calls),
            "request_roles": [m["role"] for m in messages],
            "first_turn_request_messages": payload["first_turn"]["request_messages"],
        }
    if check == "protected_replay":
        return {
            "status": "pass" if payload["projection"]["replay_state_len"] else "fail",
            "expected": "protected replay state is projected for the compatible provider key",
            "observed_replay_state_len": payload["projection"]["replay_state_len"],
            "observed_replay_state_matches_written": payload.get("replay_matches_written"),
            "mismatched_provider_key_hides_state": payload.get("mismatch_hides_state"),
            "replay_state_consumers": payload.get("replay_state_consumers"),
            "model_changes": payload["projection"]["model_changes"],
        }
    return {"status": "unknown", "expected": None}


async def case_protected_replay_model_switch(lane: LaneConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Protected replay state + model switch, written through the journal.

    Two independent observations per sample:
      1. ``ConversationProjector`` recovers the plaintext under a matching
         provider key and hides it under a different key (objective).
      2. Whether that state reaches the provider: the reopened request is
         captured, and a static scan records whether any live prompt-path
         consumer of ``replay_state`` exists in this revision.
    """
    per_sample: list[dict[str, Any]] = []
    for _ in range(args.reopen_repetitions):
        harness = await open_harness(lane, script=[Step(text_chunks=("noted",))])
        try:
            await consume_turn(harness.session, "protected-state turn")
            codec = ProtectedStateCodec(EphemeralKeyProvider(harness.key))
            replay_blob = b"bench-replay-state-v1"
            correlation = str(uuid4())
            seed = [
                NewJournalFact(
                    fact_type=FactType.MODEL_CHANGED,
                    correlation_id=str(uuid4()),
                    causation_id=None,
                    payload={"provider": "azure", "model": "gpt-4o"},
                ),
                NewJournalFact(
                    fact_type=FactType.TURN_STARTED,
                    correlation_id=correlation,
                    causation_id=None,
                    payload={"prompt_summary": "protected turn"},
                ),
                NewJournalFact(
                    fact_type=FactType.USER_CONTENT_FINAL,
                    correlation_id=correlation,
                    causation_id=correlation,
                    payload={"text": "protected turn"},
                ),
                NewJournalFact(
                    fact_type=FactType.ASSISTANT_CONTENT_FINAL,
                    correlation_id=correlation,
                    causation_id=correlation,
                    payload={"text": "protected answer"},
                    protected_payload=codec.encrypt(replay_blob),
                ),
                NewJournalFact(
                    fact_type=FactType.TURN_COMPLETED,
                    correlation_id=correlation,
                    causation_id=correlation,
                    payload={},
                ),
            ]
            result = await harness.repo.append(harness.scope, harness.session_id, harness.session._current_version, seed)
            harness.session._current_version = result.new_version
            sid = harness.session_id
            scope = harness.scope
            journal_path = harness.journal_path
            factory = harness.factory
            key = harness.key
            await harness.repo.close()
        finally:
            if harness.tool_resource is not None:
                harness.tool_resource.release.set()
            await drain_pending_tasks()

        reopen_provider = DeterministicProvider([Step(text_chunks=("second answer",))])

        def reopen_factory(factory=factory, reopen_provider=reopen_provider) -> Any:
            return make_agent(
                lane,
                factory,
                reopen_provider,
                compress_timeline=True,
                max_context_tokens=20000,
                tool_resource=None,
            )

        repo2 = await SQLiteJournalRepository.open(str(journal_path))
        try:
            session2 = AgentSession(
                owner_scope=scope,
                session_id=sid,
                repository=repo2,
                agent_factory=reopen_factory,
                protected_state_codec=ProtectedStateCodec(EphemeralKeyProvider(key)),
            )
            await session2.load()
            facts = await repo2.read_facts(scope, sid)
            codec2 = ProtectedStateCodec(EphemeralKeyProvider(key))
            matching = ConversationProjector(codec2).project(facts, provider_key="deterministic-bench")
            mismatched = ConversationProjector(codec2).project(facts, provider_key="some-other-provider")
            trace2 = await consume_turn(session2, "after model switch")
            request = reopen_provider.requests[0] if reopen_provider.requests else {"messages": [], "tool_schema_count": 0}
            payload = {
                "check": "protected_replay",
                "replay_matches_written": matching.replay_state == replay_blob,
                "mismatch_hides_state": mismatched.replay_state is None,
                "replay_state_consumers": replay_state_consumers(),
                "projection": {
                    "roles": [m.role for m in matching.messages],
                    "replay_state_len": len(matching.replay_state) if matching.replay_state else 0,
                    "interruption_observation": None,
                    "model_changes": [_safe(dict(c)) for c in matching.model_changes],
                    "current_provider": matching.current_provider,
                    "current_model": matching.current_model,
                },
                "mismatch_projection": {
                    "replay_state_len": len(mismatched.replay_state) if mismatched.replay_state else 0,
                    "current_provider": mismatched.current_provider,
                },
                "second_turn": {
                    "request_messages": request["messages"],
                    "tool_schema_count": request["tool_schema_count"],
                    "terminal": terminal_summary(session2),
                    "event_sequence": trace2.event_types(),
                },
                "first_turn": {"terminal": terminal_summary(harness.session), "event_sequence": []},
            }
            payload["verdict"] = _verdict("protected_replay", payload)
            per_sample.append(payload)
        finally:
            await repo2.close()
            await drain_pending_tasks()
            try:
                harness.tmpdir.cleanup()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
    return {
        "samples_ms": {},
        "raw": {},
        "per_sample": per_sample,
        "observation": {
            "check": "protected replay state + model switch across close/reopen",
            "verdicts": sorted({p["verdict"]["status"] for p in per_sample}),
            "replay_state_consumers": replay_state_consumers(),
            "sample_count_disclosure": f"{args.reopen_repetitions} samples",
        },
    }


# ---------------------------------------------------------------------------
# Cases - native replay fixture + carrier-gap probes
# ---------------------------------------------------------------------------


async def case_native_replay_fixture(lane: LaneConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Frozen compatible native replay fixture, exercised through existing carriers.

    The frozen fixture is one completed native turn
    (``{"schema": "dana.native-turn/1", "messages": [...]}``) built from real
    :class:`NativeMessage` records, encrypted by the provided
    :class:`ProtectedStateCodec` and stored in
    ``ASSISTANT_CONTENT_FINAL.protected_payload``, with the assistant fact payload
    naming the fixture provider so compatibility is explicit.

    Probes (no runtime change):
      1. Decode the frozen envelope and rehydrate its native messages through
         :meth:`NativeMessage.to_llm_message` - the exact retained
         ``encrypted_content`` must survive.
      2. Feed the rehydrated messages through the real provider conversion; the
         fingerprint gate must splice matching state and exclude mismatched state.
      3. Record the journal boundary: replay state lives only in
         ``protected_payload``, never in the cleartext ``payload``.
    """
    from dana.common.llm.providers import openai_compatible_base as ocb
    from dana.core.timeline.native_message import NativeMessage
    from dana.core.timeline.timeline import Timeline, TimelineEntry, TimelineEntryType

    carriers_match_provider = tuple(ocb._REPLAY_CARRIER_KEYS) == REPLAY_CARRIER_KEYS

    # --- Frozen fixture payload -------------------------------------------------
    fixture_envelope = build_native_turn_envelope(
        "frozen replay fixture question",
        "frozen replay fixture answer",
        reasoning_item=SYNTHETIC_REASONING_ITEM,
        response_id=SYNTHETIC_RESPONSE_ID,
    )
    decoded = json.loads(fixture_envelope.decode("utf-8"))
    native_messages = [NativeMessage.from_dict(d) for d in decoded["messages"]]
    llm_messages = [m.to_llm_message() for m in native_messages]
    assistant_with_items = [m for m in llm_messages if m.reasoning_items]
    captured_items = assistant_with_items[0].reasoning_items if assistant_with_items else None
    exact_encrypted = bool(captured_items) and captured_items[0].get("encrypted_content") == SYNTHETIC_REASONING_ITEM["encrypted_content"]

    # --- Probe 1: timeline carrier round-trip ----------------------------------
    timeline: Any = Timeline(max_context_tokens=32000, agent=None)
    timeline.add_entry(TimelineEntry(entry_type=TimelineEntryType.USER_MESSAGE, content="replay probe question"))
    timeline.add_entry(
        TimelineEntry(
            entry_type=TimelineEntryType.AGENT_THOUGHTS,
            content="",
            metadata={
                "reasoning_items": [dict(SYNTHETIC_REASONING_ITEM)],
                "fingerprint": FIXTURE_FINGERPRINT,
                "response_id": SYNTHETIC_RESPONSE_ID,
            },
        )
    )
    timeline.add_entry(TimelineEntry(entry_type=TimelineEntryType.AGENT_RESPONSE, content="replay probe answer"))
    messages = timeline.to_llm_messages()
    timeline_items = [m.reasoning_items for m in messages if getattr(m, "reasoning_items", None)]
    timeline_exact = bool(timeline_items) and timeline_items[0] == [SYNTHETIC_REASONING_ITEM]

    # --- Probe 2: real provider conversion / fingerprint gate -------------------
    probe = object.__new__(ocb.OpenAICompatibleProvider)
    probe.model = FIXTURE_MODEL
    own_fingerprint = probe.fingerprint
    _, wire_messages = probe.prepare_messages(messages)
    wire_carriers = [
        {k: (v if k != "_reasoning_items" else [i.get("id") for i in v]) for k, v in w.items() if k in REPLAY_CARRIER_KEYS}
        for w in wire_messages
        if any(k in w for k in REPLAY_CARRIER_KEYS)
    ]

    def items_with_fingerprint(target: str) -> list[Any]:
        for msg in messages:
            if getattr(msg, "reasoning_items", None):
                msg.reasoning_fingerprint = target  # type: ignore[assignment]
        _, wire = probe.prepare_messages(messages)
        return probe._convert_to_responses_input(wire)

    spliced = items_with_fingerprint(own_fingerprint)
    spliced_items = [w for w in spliced if isinstance(w, dict) and w.get("type") == "reasoning"]
    spliced_ids = [w.get("id") for w in spliced_items]
    spliced_encrypted = [w.get("encrypted_content") for w in spliced_items]
    stripped_in_splice = all(not any(k in w for k in REPLAY_CARRIER_KEYS) for w in spliced)

    mismatched = items_with_fingerprint("other-provider:family:ffffffff")
    mismatch_splices = [w for w in mismatched if isinstance(w, dict) and w.get("type") == "reasoning"]
    mismatch_stripped = all(not any(k in w for k in REPLAY_CARRIER_KEYS) for w in mismatched)

    # --- Probe 3: journal boundary + provider-identity compatibility ------------
    harness = await open_harness(lane, script=[Step(text_chunks=("noted",))])
    try:
        await consume_turn(harness.session, "frozen fixture journal probe")
        correlation = str(uuid4())
        codec = ProtectedStateCodec(EphemeralKeyProvider(harness.key))
        seed = [
            NewJournalFact(
                fact_type=FactType.TURN_STARTED,
                correlation_id=correlation,
                causation_id=None,
                payload={"prompt_summary": "frozen replay fixture"},
            ),
            NewJournalFact(
                fact_type=FactType.USER_CONTENT_FINAL,
                correlation_id=correlation,
                causation_id=correlation,
                payload={"text": "frozen replay fixture question"},
            ),
            NewJournalFact(
                fact_type=FactType.ASSISTANT_CONTENT_FINAL,
                correlation_id=correlation,
                causation_id=correlation,
                # Provider named on the fact itself so compatibility is explicit
                # even with no preceding MODEL_CHANGED.
                payload={"text": "frozen replay fixture answer", "provider": FIXTURE_PROVIDER, "model": FIXTURE_MODEL},
                protected_payload=codec.encrypt(fixture_envelope),
            ),
            NewJournalFact(
                fact_type=FactType.TURN_COMPLETED,
                correlation_id=correlation,
                causation_id=correlation,
                payload={},
            ),
        ]
        result = await harness.repo.append(harness.scope, harness.session_id, harness.session._current_version, seed)
        harness.session._current_version = result.new_version
        facts = await harness.repo.read_facts(harness.scope, harness.session_id)
        compatible = ConversationProjector(codec).project(facts, provider_key=FIXTURE_PROVIDER)
        incompatible = ConversationProjector(codec).project(facts, provider_key="azure")
        decoded_state = json.loads(compatible.replay_state.decode("utf-8")) if compatible.replay_state else None
        cleartext_leaks = [
            {"sequence": f.sequence, "fact_type": f.fact_type.value}
            for f in facts
            if "reasoning_items" in (f.payload or {}) or "encrypted_content" in json.dumps(f.payload or {})
        ]
        journal_boundary = {
            "protected_payload_facts": [f.fact_type.value for f in facts if f.protected_payload is not None],
            "cleartext_payload_leaks": cleartext_leaks,
            "compatible_provider_recovers_envelope": decoded_state is not None and decoded_state.get("schema") == NATIVE_TURN_SCHEMA,
            "incompatible_provider_key_hides_state": incompatible.replay_state is None,
            "matching_provider_key": FIXTURE_PROVIDER,
            "incompatible_provider_key": "azure",
        }
    finally:
        await close_harness(harness)

    return {
        "samples_ms": {},
        "raw": {},
        "per_sample": [
            {
                "frozen_fixture": {
                    "schema": NATIVE_TURN_SCHEMA,
                    "provider": FIXTURE_PROVIDER,
                    "model": FIXTURE_MODEL,
                    "fingerprint": FIXTURE_FINGERPRINT,
                    "envelope_messages": decoded["messages"],
                    "reasoning_item": SYNTHETIC_REASONING_ITEM,
                    "response_id": SYNTHETIC_RESPONSE_ID,
                },
                "carrier_keys_match_provider": carriers_match_provider,
                "native_message_rehydration": {
                    "decoded_roles": [m["role"] for m in decoded["messages"]],
                    "llm_roles": [m.role for m in llm_messages],
                    "captured_items": captured_items,
                    "exact_encrypted_content_retained": exact_encrypted,
                },
                "timeline_carrier": {
                    "assistant_messages_with_items": len(timeline_items),
                    "exact_items_match": timeline_exact,
                },
                "responses_input_conversion": {
                    "synthetic_fingerprint": own_fingerprint,
                    "wire_carriers_on_mismatch": wire_carriers,
                    "matching_fingerprint_spliced_ids": spliced_ids,
                    "matching_fingerprint_spliced_encrypted_content": spliced_encrypted,
                    "matching_fingerprint_carriers_stripped": stripped_in_splice,
                    "mismatched_fingerprint_splice_count": len(mismatch_splices),
                    "mismatched_fingerprint_carriers_stripped": mismatch_stripped,
                },
                "journal_boundary": journal_boundary,
            }
        ],
        "observation": {
            "measures": "the frozen native replay fixture through existing timeline + provider carriers and the journal boundary",
            "verdict": {
                "fixture_envelope_rehydrates": "pass" if decoded["messages"] else "fail",
                "native_message_retains_encrypted_content": "pass" if exact_encrypted else "fail",
                "timeline_carrier_round_trip": "pass" if timeline_exact else "fail",
                "fingerprint_gate_match_splices": "pass" if spliced_ids else "fail",
                "fingerprint_gate_match_retains_encrypted_content": (
                    "pass" if SYNTHETIC_REASONING_ITEM["encrypted_content"] in spliced_encrypted else "fail"
                ),
                "fingerprint_gate_mismatch_excludes": "pass" if not mismatch_splices else "fail",
                "carriers_always_stripped": "pass" if (stripped_in_splice and mismatch_stripped) else "fail",
                "journal_boundary_protected_only": "pass"
                if journal_boundary["protected_payload_facts"] and not cleartext_leaks
                else "fail",
                "provider_key_compatibility_gate": "pass" if journal_boundary["incompatible_provider_key_hides_state"] else "fail",
            },
            "note": (
                "Fixture-level probe through production conversion code: it does not construct a real provider, call "
                "the Responses API, or assert that a live session path populates these carriers. Freezing this fixture "
                "does not implement the B1 projector hydration."
            ),
        },
    }


async def case_replay_carrier_gap(lane: LaneConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Where each path drops or retains the replay carrier - observed, not asserted.

    Paths checked:
      * **streaming** (the STAR loop used by ``prompt()``): do later provider
        requests carry replay items, and does an expected
        ``LLMStreamChunk(type="response", response=LLMResponse(...))`` propagate?
      * **session journal**: does the session persist a protected payload?
      * **frozen envelope round-trip**: does an encrypted native-turn envelope stay
        out of the cleartext payload and decode intact via the projector?
    """
    streaming = await open_harness(
        lane,
        script=[Step(text_chunks=("first streamed answer",)), Step(text_chunks=("second streamed answer",))],
    )
    try:
        await consume_turn(streaming.session, "carrier gap turn one")
        await consume_turn(streaming.session, "carrier gap turn two")
        requests = [r for r in streaming.provider.requests if r["kind"] == "stream"]
        carriers_seen = [
            {"request_index": r["index"], "roles_with_items": [m["role"] for m in r["messages"] if m.get("reasoning_items")]}
            for r in requests
        ]
        streaming_facts = await streaming.repo.read_facts(streaming.scope, streaming.session_id)
        streaming_protected = [f.fact_type.value for f in streaming_facts if f.protected_payload is not None]
        # Expected future carrier: a chunk carrying a full LLMResponse. The real
        # LLMStreamChunk has no `response` attribute (B0 gap); setattr matches the
        # existing chunk attributes so propagation can be exercised without
        # altering production types. Whether it survives is the observation.
        probe_chunk = LLMStreamChunk(type="response")
        setattr(
            probe_chunk,
            "response",
            LLMResponse(
                content="",
                model=FIXTURE_MODEL,
                reasoning_items=[dict(SYNTHETIC_REASONING_ITEM)],
                response_id=SYNTHETIC_RESPONSE_ID,
            ),
        )
        chunk_response_gap = {
            "chunk_dataclass_fields": sorted(LLMStreamChunk.__dataclass_fields__.keys()),
            "response_field_declared": "response" in LLMStreamChunk.__dataclass_fields__,
            "setattr_response_attached": getattr(probe_chunk, "response", None) is not None,
            "consumer_reads_chunk_response": _chunk_response_consumers(),
        }
    finally:
        await close_harness(streaming)

    # Frozen envelope round-trip through the journal (codec-only observation).
    harness = await open_harness(lane, script=[Step(text_chunks=("noted",))])
    try:
        await consume_turn(harness.session, "fixture carrier probe")
        codec = ProtectedStateCodec(EphemeralKeyProvider(harness.key))
        envelope = build_native_turn_envelope(
            "fixture carrier question",
            "fixture carrier answer",
            reasoning_item=SYNTHETIC_REASONING_ITEM,
            response_id=SYNTHETIC_RESPONSE_ID,
        )
        correlation = str(uuid4())
        seed = [
            NewJournalFact(
                fact_type=FactType.TURN_STARTED,
                correlation_id=correlation,
                causation_id=None,
                payload={"prompt_summary": "carrier probe"},
            ),
            NewJournalFact(
                fact_type=FactType.USER_CONTENT_FINAL,
                correlation_id=correlation,
                causation_id=correlation,
                payload={"text": "fixture carrier question"},
            ),
            NewJournalFact(
                fact_type=FactType.ASSISTANT_CONTENT_FINAL,
                correlation_id=correlation,
                causation_id=correlation,
                payload={"text": "fixture carrier answer", "provider": FIXTURE_PROVIDER, "model": FIXTURE_MODEL},
                protected_payload=codec.encrypt(envelope),
            ),
            NewJournalFact(
                fact_type=FactType.TURN_COMPLETED,
                correlation_id=correlation,
                causation_id=correlation,
                payload={},
            ),
        ]
        result = await harness.repo.append(harness.scope, harness.session_id, harness.session._current_version, seed)
        harness.session._current_version = result.new_version
        facts = await harness.repo.read_facts(harness.scope, harness.session_id)
        view = ConversationProjector(codec).project(facts, provider_key=FIXTURE_PROVIDER)
        recovered = json.loads(view.replay_state.decode("utf-8")) if view.replay_state else None
        payload_leaks = [
            {"sequence": f.sequence, "fact_type": f.fact_type.value}
            for f in facts
            if "reasoning_items" in (f.payload or {}) or "encrypted_content" in json.dumps(f.payload or {})
        ]
    finally:
        await close_harness(harness)

    return {
        "samples_ms": {},
        "raw": {},
        "per_sample": [],
        "observation": {
            "measures": "which existing paths retain the replay carrier",
            "streaming_path": {
                "stream_requests_observed": len(requests),
                "carriers_in_requests": carriers_seen,
                "any_carrier_in_streaming_requests": any(c["roles_with_items"] for c in carriers_seen),
                "session_journal_protected_payloads": streaming_protected,
                "compaction_chat_calls": streaming.provider.chat_call_count,
            },
            "frozen_envelope_round_trip": {
                "recovered_schema": (recovered or {}).get("schema"),
                "recovered_matches_written": recovered == json.loads(envelope.decode("utf-8")),
                "cleartext_payload_leaks": payload_leaks,
                "projector_roles": [m.role for m in view.messages],
            },
            "expected_chunk_response_carrier": chunk_response_gap,
            "unsupported_carrier_gap": {
                "streaming_chunks_cannot_carry_replay_metadata": "response" not in LLMStreamChunk.__dataclass_fields__,
                "session_prompt_writes_no_protected_payload": not streaming_protected,
                "journal_to_request_bridge_absent": not any(c["roles_with_items"] for c in carriers_seen),
                "statement": (
                    "The streaming path used by AgentSession.prompt() cannot carry replay metadata: LLMStreamChunk "
                    "declares no 'response' field, the session never writes a protected payload, and no bridge exists "
                    "from a decrypted protected payload back into the provider request. A live provider request "
                    "therefore cannot retain reasoning state across turns in this revision. Recorded as a gap; the "
                    "expected carrier shape (LLMStreamChunk(type='response', response=LLMResponse(...))) is probed with "
                    "setattr only, without altering production types."
                ),
            },
            "codec_only_observation": (
                "The frozen envelope round-trip proves the codec + projector retain the item exactly, but it is a "
                "codec/projection observation only - explicitly NOT behavioural replay acceptance."
            ),
        },
    }


def _chunk_response_consumers() -> list[str]:
    """Static scan: does any production consumer read a chunk's ``response`` attr?"""
    hits: list[str] = []
    for path in (REPO_ROOT / "dana").rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "chunk.response" in text or 'chunk, "response"' in text or 'getattr(chunk, "response"' in text:
            hits.append(str(path.relative_to(REPO_ROOT)))
    return sorted(hits)


def replay_state_consumers() -> list[str]:
    """Static call-site scan for ``replay_state`` usage outside the projector."""
    consumers: list[str] = []
    for path in (REPO_ROOT / "dana").rglob("*.py"):
        if path.name == "conversation.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "replay_state" in text:
            consumers.append(str(path.relative_to(REPO_ROOT)))
    return sorted(consumers)


async def case_session_protected_payload_persistence(lane: LaneConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Whether a provider completion carrying replay metadata reaches the journal.

    The controlled provider returns a real non-streaming completion carrying
    ``reasoning_items`` + ``response_id`` (the shape a Responses-API provider
    returns). Both the streaming path used by ``prompt()`` and the non-streaming
    ``chat_response`` path are exercised, and the journal + the next captured
    request are inspected for the carrier.
    """
    harness = await open_harness(
        lane,
        script=[Step(text_chunks=("plain answer",), thinking_chunks=("reasoning",))],
        provider_replay_metadata=True,
    )
    try:
        trace = await consume_turn(harness.session, "reasoning turn")
        facts = await harness.repo.read_facts(harness.scope, harness.session_id)
        with_payload = [f.fact_type.value for f in facts if f.protected_payload is not None]
        await consume_turn(harness.session, "continue the reasoning turn")

        # Non-streaming path: carry the same replay metadata through chat_response.
        chat_carrier: dict[str, Any]
        try:
            response = await harness.agent.llm_client.chat_response(
                [LLMMessage(role="user", content="carrier check")], agent_id="bench", agent_type="bench"
            )
            chat_carrier = {
                "reasoning_items_returned": _safe(response.reasoning_items),
                "response_id_returned": response.response_id,
                "carrier_reaches_agent_llm_client": bool(response.reasoning_items),
            }
        except Exception as exc:  # noqa: BLE001 - recorded
            chat_carrier = {"raised": type(exc).__name__, "message": str(exc)[:200]}

        # Is the carrier threaded onto the next provider request?
        next_request_carriers = [
            {"role": m["role"], "reasoning_items": m.get("reasoning_items")}
            for r in harness.provider.requests
            for m in r["messages"]
            if m.get("reasoning_items")
        ]
        return {
            "samples_ms": {},
            "raw": {},
            "per_sample": [],
            "observation": {
                "measures": "whether a supported replay-metadata completion becomes journal protected state and travels on",
                "provider_replay_metadata_enabled": True,
                "facts_with_protected_payload": with_payload,
                "journal_fact_types": [f.fact_type.value for f in facts],
                "next_request_carriers": next_request_carriers,
                "non_streaming_chat_carrier": chat_carrier,
                "terminal": terminal_summary(harness.session),
                "event_sequence": trace.event_types(),
                "result": (
                    "observed carrier gap: the provider returned supported replay metadata (reasoning_items + "
                    "response_id) and it did NOT reach any journal protected_payload, the agent timeline, or the next "
                    "provider request. No session path converts provider replay metadata into protected state."
                    if not with_payload and not next_request_carriers
                    else "observed: carrier reached the journal and/or a later request"
                ),
            },
        }
    finally:
        await close_harness(harness)


# ---------------------------------------------------------------------------
# Cases - guards / provenance
# ---------------------------------------------------------------------------


async def case_fail_closed_guard(lane: LaneConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Prove the guards fail closed, without any real request."""
    attempts: list[dict[str, Any]] = []

    def record(name: str, fn: Any) -> None:
        try:
            fn()
            attempts.append({"attempt": name, "raised": None, "blocked": False})
        except Exception as exc:  # noqa: BLE001 - the raised type is the observation
            attempts.append(
                {
                    "attempt": name,
                    "raised": type(exc).__name__,
                    "message": str(exc)[:200],
                    "blocked": isinstance(exc, (LiveProviderBlocked, LiveNetworkBlocked)),
                }
            )

    import dana.common.llm.embedder as embedder_module
    import dana.common.llm.llm as llm_module

    record("llm_module.create_provider('openai')", lambda: llm_module.create_provider("openai", model="gpt-4o"))
    record("embedder_module.create_provider('openai')", lambda: embedder_module.create_provider("openai"))
    record("LLM(provider='openai')", lambda: LLM(provider="openai"))
    record("socket.getaddrinfo('api.openai.com', 443)", lambda: socket.getaddrinfo("api.openai.com", 443))
    record("socket.getaddrinfo(GEOLOCATION_HOST, 80)", lambda: socket.getaddrinfo(GEOLOCATION_HOST, 80))
    record("loopback getaddrinfo('127.0.0.1', 1)", lambda: socket.getaddrinfo("127.0.0.1", 1))

    harness = await open_harness(lane, script=[Step(text_chunks=("guarded",))])
    try:
        resolved = None
        resolve_error = None
        caller = getattr(harness.agent._runtime, "_llm_caller", None)
        if caller is not None and hasattr(caller, "_resolve_llm"):
            try:
                resolved = caller._resolve_llm()
            except Exception as exc:  # noqa: BLE001 - recorded
                resolve_error = f"{type(exc).__name__}: {exc}"
        trace = await consume_turn(harness.session, "guard prompt")
        from dana.common.observable import LANGFUSE_ENABLED, LANGSMITH_ENABLED

        return {
            "samples_ms": {},
            "raw": {},
            "per_sample": attempts,
            "observation": {
                "measures": "fail-closed behaviour of the provider-construction and network guards",
                "all_model_attempts_blocked": all(a["blocked"] for a in attempts if "provider" in a["attempt"] or "LLM(" in a["attempt"]),
                "external_host_blocked": next((a["blocked"] for a in attempts if "api.openai.com" in a["attempt"]), None),
                "geolocation_host_blocked_in_this_lane": next((a["blocked"] for a in attempts if GEOLOCATION_HOST in a["attempt"]), None),
                "loopback_blocked": next((a["blocked"] for a in attempts if "127.0.0.1" in a["attempt"]), None),
                "agent_llm_provider_is_injected": harness.agent.llm_client.provider is harness.provider,
                "runtime_caller_provider_is_injected": getattr(resolved, "provider", None) is harness.provider,
                "runtime_caller_is_agent_llm_client": resolved is harness.agent.llm_client,
                "runtime_caller_resolve_error": resolve_error,
                "provider_requests_this_case": len(harness.provider.requests),
                "provider_stream_calls_this_case": harness.provider.stream_calls,
                "provider_chat_calls_this_case": harness.provider.chat_call_count,
                "tracing_backends_active": {"langfuse": LANGFUSE_ENABLED, "langsmith": LANGSMITH_ENABLED},
                "mcp_configured": bool(os.environ.get("DANA_MCP_SERVERS", "").strip()),
                "session_mcp_wiring": getattr(harness.session, "_mcp_wiring", None) is not None,
                "subsidiary_agents_present": len(getattr(harness.agent, "_agents", []) or []),
                "terminal": terminal_summary(harness.session),
                "event_sequence": trace.event_types(),
            },
        }
    finally:
        await close_harness(harness)


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseSpec:
    name: str
    fn: Any
    lanes: tuple[str, ...]
    note: str


def build_case_specs(args: argparse.Namespace) -> tuple[CaseSpec, ...]:
    pre_token_hold = max(args.hold_timeout_s, args.cancel_recognise_delay_s + 0.5)
    return (
        CaseSpec("fail_closed_guard", case_fail_closed_guard, ("isolated", "production"), "guard proof + provider identity"),
        CaseSpec("request_fidelity_compressed", case_request_fidelity, ("isolated", "production"), "captured provider request (default)"),
        CaseSpec("request_fidelity_uncompressed", case_request_fidelity, ("isolated",), "captured provider request (compaction off)"),
        CaseSpec("session_construction", case_construction, ("isolated", "production"), "temp workspace + agent + session build"),
        CaseSpec("first_use_turn", case_first_use_turn, ("isolated", "production"), "first turn on a fresh session"),
        CaseSpec("warm_no_tool_turns", case_warm_turns, ("isolated", "production"), "consecutive no-tool turns"),
        CaseSpec("streaming_text_only", case_streaming_text_only, ("isolated", "production"), "provider text vs host delivery"),
        CaseSpec("text_then_tools", case_text_then_tools, ("isolated",), "delayed narration then a tool call"),
        CaseSpec(
            "cancel_before_first_token",
            lambda lane, a: cancel_case(
                lane,
                a,
                case="cancel_before_first_token",
                script=(Step(text_chunks=("late answer",), hold_before_first_chunk=asyncio.Event(), hold_timeout_s=pre_token_hold),),
                trigger="before_first_token",
            ),
            ("isolated",),
            "cancel while the provider is parked before its first token",
        ),
        CaseSpec(
            "cancel_midstream",
            lambda lane, a: cancel_case(
                lane,
                a,
                case="cancel_midstream",
                script=(
                    Step(
                        text_chunks=("a", "b"),
                        chunk_delay_s=a.stream_chunk_delay_s,
                        hold_before_done=asyncio.Event(),
                        hold_timeout_s=a.hold_timeout_s,
                    ),
                ),
                trigger="midstream",
            ),
            ("isolated",),
            "cancel between tokens",
        ),
        CaseSpec(
            "cancel_during_tool_execution",
            lambda lane, a: cancel_case(
                lane,
                a,
                case="cancel_during_tool_execution",
                script=(
                    Step(tool_calls=(tool_call("cancel-tool", "bench_block", {"seconds": a.hold_timeout_s}),)),
                    Step(text_chunks=("after tool",)),
                ),
                trigger="during_tool",
                with_tools=True,
            ),
            ("isolated",),
            "cancel while a cooperative async tool is running",
        ),
        CaseSpec(
            "cancel_before_done",
            lambda lane, a: cancel_case(
                lane,
                a,
                case="cancel_before_done",
                script=(Step(text_chunks=("final text",), hold_before_done=asyncio.Event(), hold_timeout_s=a.hold_timeout_s),),
                trigger="before_done",
            ),
            ("isolated",),
            "cancel after the last chunk, before the stream completes (finalization)",
        ),
        CaseSpec("cancel_after_terminal", case_cancel_after_terminal, ("isolated",), "cancel with no active turn"),
        CaseSpec(
            "reopen_compressed_history",
            lambda lane, a: reopen_case(
                lane,
                a,
                case="reopen_compressed_history",
                first_script=(Step(text_chunks=("acknowledged reopen-marker",)),),
                first_prompt="Please note reopen-marker.",
                second_prompt="What did I ask you to remember?",
                check="compressed_history",
            ),
            ("isolated", "production"),
            "close/reopen with the default compressed timeline",
        ),
        CaseSpec(
            "reopen_uncompressed_history",
            lambda lane, a: reopen_case(
                lane,
                a,
                case="reopen_uncompressed_history",
                first_script=(Step(text_chunks=("acknowledged reopen-marker",)),),
                first_prompt="Please note reopen-marker.",
                second_prompt="What did I ask you to remember?",
                check="compressed_history",
                compress=False,
            ),
            ("isolated",),
            "close/reopen with compaction disabled",
        ),
        CaseSpec(
            "reopen_interrupted_turn",
            lambda lane, a: reopen_case(
                lane,
                a,
                case="reopen_interrupted_turn",
                first_script=(Step(text_chunks=("interrupted answer",)),),
                first_prompt="incomplete turn prompt",
                second_prompt="continue",
                check="interrupted_turn",
                seed_interrupted_turn=True,
            ),
            ("isolated",),
            "crash-interrupted turn recovery across reopen",
        ),
        CaseSpec(
            "reopen_multimodal",
            lambda lane, a: reopen_case(
                lane,
                a,
                case="reopen_multimodal",
                first_script=(Step(text_chunks=("image noted",)),),
                first_prompt="describe it",
                second_prompt="what did I send?",
                content_blocks=[
                    {"type": "text", "text": "describe it"},
                    {"type": "image", "media_type": "image/png", "data": "YmVuY2gtcG5n"},
                ],
                check="multimodal",
            ),
            ("isolated",),
            "multimodal content across close/reopen",
        ),
        CaseSpec(
            "reopen_tool_pair",
            lambda lane, a: reopen_case(
                lane,
                a,
                case="reopen_tool_pair",
                first_script=(
                    Step(tool_calls=(tool_call("pair-1", "bench_echo", {"text": "paired"}),)),
                    Step(text_chunks=("tool finished",)),
                ),
                first_prompt="call the echo tool",
                second_prompt="what did the tool return?",
                with_tools=True,
                check="tool_pair",
            ),
            ("isolated",),
            "tool-call/tool-result pairing across close/reopen",
        ),
        CaseSpec(
            "reopen_protected_replay_model_switch",
            case_protected_replay_model_switch,
            ("isolated",),
            "protected replay state + model switch",
        ),
        CaseSpec(
            "native_replay_fixture",
            case_native_replay_fixture,
            ("isolated",),
            "canonical replay metadata through existing timeline + provider carriers",
        ),
        CaseSpec(
            "replay_carrier_gap",
            case_replay_carrier_gap,
            ("isolated",),
            "which existing paths retain the replay carrier (streaming / journal)",
        ),
        CaseSpec(
            "session_protected_payload_persistence",
            case_session_protected_payload_persistence,
            ("isolated",),
            "whether the session journals provider replay state",
        ),
    )


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------


LANES: dict[str, LaneConfig] = {
    "isolated": LaneConfig(
        name="isolated",
        isolated_context=True,
        clear_mcp_env=True,
        allowed_hosts=(),
        enable_skills=False,
        enable_assistant=False,
    ),
    "production": LaneConfig(
        name="production",
        isolated_context=False,
        clear_mcp_env=False,
        allowed_hosts=(GEOLOCATION_HOST,),
        enable_skills=True,
        enable_assistant=True,
    ),
}


def apply_lane_environment(lane: LaneConfig) -> dict[str, Any]:
    """Apply lane-specific environment + guard allowlist; return the deviation record."""
    if lane.clear_mcp_env:
        os.environ["DANA_MCP_SERVERS"] = ""
    GUARD.configure(lane.allowed_hosts)
    return {
        "DANA_MCP_SERVERS": (
            "cleared (no external MCP server can spawn)" if lane.clear_mcp_env else "left as configured in the environment"
        ),
        "DANA_MCP_SERVERS_present_in_environment": bool(os.environ.get("DANA_MCP_SERVERS", "").strip()),
        "runtime_context_fields": (
            "location removed on the runtime instance (rebound to timestamp/timezone/user)"
            if lane.isolated_context
            else "production default: location present, geolocation permitted (allowlisted host only)"
        ),
        "enable_skills": lane.enable_skills,
        "enable_assistant": lane.enable_assistant,
        "network_guard_allowed_hosts": list(lane.allowed_hosts),
        "provider_construction_guard": "all create_provider paths patched to raise LiveProviderBlocked",
        "tracing_backends": "Langfuse/LangSmith disabled before import (both lanes)",
    }


def capture_environment() -> dict[str, Any]:
    try:
        import dana

        dana_version = getattr(dana, "__version__", "unknown")
    except Exception:  # pragma: no cover - introspection only
        dana_version = "unknown"
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dana_version": dana_version,
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "env_keys_present": sorted(
            k
            for k in (
                "DANA_LLM_PROVIDER",
                "DANA_MODEL",
                "DANA_CODE_JOURNAL",
                "DANA_ACP_JOURNAL",
                "DANA_COMPACT_TRIGGER_TOKENS",
                "DANA_CLAUDE_SKILLS",
                "DANA_MCP_SERVERS",
                "LANGFUSE_ENABLED",
                "LANGSMITH_TRACING",
            )
            if k in os.environ
        ),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic AgentSession/STARAgent harness benchmark (no live model traffic).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default="-", help="JSON output path, or '-' for stdout")
    parser.add_argument("--repetitions", type=int, default=100, help="cheap local timing samples")
    parser.add_argument("--construction-repetitions", type=int, default=20, help="samples for full harness construction")
    parser.add_argument("--delayed-repetitions", type=int, default=10, help="samples for delay-injected / cancellation cases")
    parser.add_argument("--reopen-repetitions", type=int, default=5, help="samples for close/reopen fidelity cases")
    parser.add_argument("--warmup-turns", type=int, default=1, help="declared warm-up turns excluded from warm-turn samples")
    parser.add_argument("--mode", choices=("isolated", "production", "both"), default="both", help="lane(s) to run")
    parser.add_argument("--cases", default="", help="comma-separated case names to run (default: all for the lane)")
    parser.add_argument("--compress-timeline", action="store_true", default=True, help="request-fidelity compression setting")
    parser.add_argument("--no-compress-timeline", dest="compress_timeline", action="store_false")
    parser.add_argument("--stream-chunk-delay-s", type=float, default=0.05, help="declared per-chunk provider delay")
    parser.add_argument("--hold-timeout-s", type=float, default=3.0, help="max provider/tool hold duration")
    parser.add_argument("--cancel-release-delay-s", type=float, default=0.2, help="hold release delay measured after cancel()")
    parser.add_argument("--cancel-recognise-delay-s", type=float, default=1.0, help="pre-first-token hold before cancelling")
    parser.add_argument("--prompt-cancel-bound-ms", type=float, default=100.0, help="cooperative local cancellation bound")
    parser.add_argument("--timeout-seconds", type=float, default=60.0, help="per-case wall-clock guard")
    parser.add_argument("--seed", type=int, default=0, help="recorded seed for reproducibility metadata")
    parser.add_argument("--quiet", action="store_true", help="suppress the stderr summary")
    return parser.parse_args(argv)


async def run_case(spec: CaseSpec, lane: LaneConfig, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        async with asyncio.timeout(args.timeout_seconds):
            result = await spec.fn(lane, args)
        result["status"] = "ok"
    except TimeoutError:
        await drain_pending_tasks()
        result = {
            "samples_ms": {},
            "raw": {},
            "observation": {"status": "timeout", "timeout_seconds": args.timeout_seconds},
        }
        result["status"] = "timeout"
    except Exception as exc:  # noqa: BLE001 - a case crash is recorded, never hidden
        await drain_pending_tasks()
        result = {
            "samples_ms": {},
            "raw": {},
            "observation": {"status": "case_error", "error_type": type(exc).__name__, "error": str(exc)[:500]},
        }
        result["status"] = "case_error"
    result["case"] = spec.name
    result["lane"] = lane.name
    result["note"] = spec.note
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 4)
    return result


def collect_known_failures(cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Observed failures recorded as failures (never repaired in this script)."""
    findings: list[dict[str, Any]] = []
    for key, result in cases.items():
        observation = result.get("observation") or {}
        if result.get("status") != "ok":
            findings.append({"id": f"{key}:{result.get('status')}", "detail": observation})
            continue
        if "verdicts" in observation and "fail" in observation["verdicts"]:
            failing = [s for s in result.get("per_sample", []) if s.get("verdict", {}).get("status") == "fail"]
            findings.append({"id": f"{key}:fidelity_fail", "detail": failing[0]["verdict"] if failing else observation})
        if observation.get("end_of_generation_buffering_gap_ms") not in (None, 0):
            findings.append(
                {
                    "id": f"{key}:end_of_generation_text_buffering",
                    "detail": {
                        "gap_ms": observation["end_of_generation_buffering_gap_ms"],
                        "text": observation.get("host_text_delivery"),
                    },
                }
            )
        if observation.get("cancel_recognised_within_bound") is False:
            findings.append(
                {
                    "id": f"{key}:cancel_not_recognised_within_bound",
                    "detail": {
                        "bound_ms": observation.get("cancel_bound_ms"),
                        "per_sample": [s.get("cancel_to_terminal_ms") for s in result.get("per_sample", [])],
                    },
                }
            )
        if key.endswith("session_protected_payload_persistence"):
            carrier = observation.get("non_streaming_chat_carrier") or {}
            if not observation.get("facts_with_protected_payload"):
                findings.append(
                    {
                        "id": f"{key}:no_protected_payload_journaled",
                        "detail": {
                            "provider_returned_replay_metadata": carrier.get("carrier_reaches_agent_llm_client"),
                            "provider_reasoning_items": carrier.get("reasoning_items_returned"),
                            "provider_response_id": carrier.get("response_id_returned"),
                            "journal_fact_types": observation.get("journal_fact_types"),
                            "next_request_carriers": observation.get("next_request_carriers"),
                            "boundary": (
                                "provider returns supported replay metadata -> agent LLM client receives it "
                                "(carrier_reaches_agent_llm_client=true) -> no path converts it into journal "
                                "protected_payload or a later provider request"
                            ),
                        },
                    }
                )
        if key.endswith("replay_carrier_gap"):
            gap = observation.get("unsupported_carrier_gap") or {}
            if gap.get("journal_to_request_bridge_absent"):
                findings.append(
                    {
                        "id": f"{key}:no_journal_to_request_replay_bridge",
                        "detail": gap.get("statement"),
                    }
                )
            if gap.get("session_prompt_writes_no_protected_payload"):
                findings.append({"id": f"{key}:session_never_writes_protected_payload", "detail": gap.get("statement")})
        if key.endswith("native_replay_fixture"):
            verdict = observation.get("verdict") or {}
            failures = {k: v for k, v in verdict.items() if v == "fail"}
            if failures:
                findings.append({"id": f"{key}:fixture_probe_failures", "detail": failures})
        if key.endswith("reopen_protected_replay_model_switch") and not observation.get("replay_state_consumers"):
            findings.append(
                {
                    "id": f"{key}:no_live_replay_state_consumer",
                    "detail": observation.get("gap") or "no live prompt-path consumer of replay_state found",
                }
            )
    return findings


COVERAGE_GAPS: tuple[dict[str, str], ...] = (
    {
        "id": "plain_timeline_unreachable",
        "detail": (
            "STARAgent._build_timeline always constructs a CompressedTimeline, so the plain Timeline class cannot be "
            "exercised through the real agent path without a runtime change. 'uncompressed' here means "
            "compress_timeline=False, which disables compaction but still uses CompressedTimeline."
        ),
    },
    {
        "id": "no_process_startup_measurement",
        "detail": "CLI/ACP --help startup and interpreter import cost are outside this script.",
    },
    {
        "id": "no_live_provider",
        "detail": "Every lane injects a deterministic provider; live model latency/quality is not measured here.",
    },
    {
        "id": "no_concurrency_or_history_scaling",
        "detail": (
            "Concurrent sessions, event-loop heartbeat under mixed sync/async tool batches, and long-history scaling "
            "(10/100/1000/5000 turns) are separate lanes."
        ),
    },
    {
        "id": "no_acp_cli_rendering",
        "detail": "Measurement stops at the AgentSession host-event boundary; ACP/CLI rendering is not exercised.",
    },
    {
        "id": "no_postgres_storage",
        "detail": "Only the SQLite journal backend is exercised; the PostgreSQL journal contract is untested here.",
    },
    {
        "id": "session_tool_engine_path_untested",
        "detail": (
            "AgentSession.execute_tool_call (session ToolExecutionEngine) is not driven; tool lifecycle events here come "
            "from the agent's own _act_async via prompt(), which is the path the session uses."
        ),
    },
    {
        "id": "compaction_trigger_not_exercised",
        "detail": (
            "Compaction is enabled with production defaults but the trigger is never crossed at this history size; only "
            "the deterministic chat() interception is recorded (compaction_chat_calls_observed=0)."
        ),
    },
    {
        "id": "multimodal_live_request_text_only",
        "detail": (
            "Content blocks are journaled and the live current turn reaches the provider as text; only the reopened "
            "request is checked for blocks."
        ),
    },
    {
        "id": "replay_carrier_absent_on_live_requests",
        "detail": (
            "The streaming path used by AgentSession.prompt() cannot carry reasoning_items, the session writes no "
            "protected payload, and no bridge exists from a decrypted protected payload back into the provider request. "
            "The native_replay_fixture case proves the timeline + provider carriers work when populated, and "
            "replay_carrier_gap records the exact point where the live path drops the carrier. Reported as a gap, not "
            "repaired."
        ),
    },
    {
        "id": "replay_fixture_is_not_end_to_end_acceptance",
        "detail": (
            "native_replay_fixture exercises production conversion code on a synthetic carrier without constructing a "
            "real provider or calling the Responses API; it is not acceptance of live/resume replay fidelity."
        ),
    },
    {
        "id": "production_lane_geolocation_network",
        "detail": (
            f"The production lane allowlists {GEOLOCATION_HOST} so the production geolocation path may make one outbound "
            "request per process; if blocked/offline the lookup resolves to None and that is recorded."
        ),
    },
)


def coverage_gaps_for(cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = [dict(g) for g in COVERAGE_GAPS]
    for key, result in cases.items():
        observation = result.get("observation") or {}
        if observation.get("verdicts") and "fail" in observation["verdicts"]:
            gaps.append({"id": f"{key}:fidelity_failure", "detail": "see known_failures for the observed request payload"})
    return gaps


async def run_lane(lane: LaneConfig, args: argparse.Namespace, specs: Sequence[CaseSpec]) -> dict[str, Any]:
    deviations = apply_lane_environment(lane)
    selected = {name.strip() for name in args.cases.split(",") if name.strip()}
    provider_before = len(GUARD.provider_attempts)
    network_before = len(GUARD.network_attempts)
    cases: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if lane.name not in spec.lanes:
            continue
        if selected and spec.name not in selected:
            continue
        cases[f"{lane.name}/{spec.name}"] = await run_case(spec, lane, args)
    return {
        "environment_deviations": deviations,
        "guard_activity_this_lane": {
            "provider_construction_attempts": GUARD.provider_attempts[provider_before:],
            "network_attempts": GUARD.network_attempts[network_before:],
        },
        "cases": cases,
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    lane_names = ["isolated", "production"] if args.mode == "both" else [args.mode]
    specs = build_case_specs(args)
    GUARD.install()
    try:
        document: dict[str, Any] = {
            "script": "scripts/benchmark_session.py",
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(UTC).isoformat(),
            "environment": capture_environment(),
            "config": {
                "modes": lane_names,
                "repetitions": args.repetitions,
                "construction_repetitions": args.construction_repetitions,
                "delayed_repetitions": args.delayed_repetitions,
                "reopen_repetitions": args.reopen_repetitions,
                "warmup_turns": args.warmup_turns,
                "compress_timeline_default": args.compress_timeline,
                "stream_chunk_delay_s": args.stream_chunk_delay_s,
                "hold_timeout_s": args.hold_timeout_s,
                "cancel_release_delay_s": args.cancel_release_delay_s,
                "cancel_recognise_delay_s": args.cancel_recognise_delay_s,
                "prompt_cancel_bound_ms": args.prompt_cancel_bound_ms,
                "timeout_seconds": args.timeout_seconds,
                "seed": args.seed,
                "clock": "time.perf_counter (monotonic seconds; derived fields in milliseconds)",
                "provider": {
                    "class": "DeterministicProvider",
                    "model_id": "bench-deterministic",
                    "supports_native_tools": True,
                    "network": "none - every response is scripted locally",
                    "compaction": "chat() served deterministically and recorded; a live call is impossible (guarded)",
                },
                "repositories": "per-run temporary FileStorageConfig workspace + temporary SQLite journal",
                "resources_auto_register": False,
                "declared_deviations_common_to_all_lanes": [
                    "deterministic injected provider instead of the configured live provider",
                    "temporary repository factory + temporary SQLite journal (no ~/.dana writes)",
                    "every agent/resource constructed with auto_register=False",
                    "Langfuse/LangSmith trace backends disabled before importing dana",
                    "fail-closed provider-construction guard + non-allowlisted network guard installed for the whole run",
                ],
            },
            "lanes": {},
            "guard": {},
            "known_failures": [],
            "coverage_gaps": [],
        }
        all_cases: dict[str, dict[str, Any]] = {}
        for name in lane_names:
            lane = LANES[name]
            lane_result = await run_lane(lane, args, specs)
            document["lanes"][name] = lane_result
            all_cases.update(lane_result["cases"])
        document["guard"] = GUARD.report()
        document["known_failures"] = collect_known_failures(all_cases)
        document["coverage_gaps"] = coverage_gaps_for(all_cases)
        document["case_index"] = [
            {"key": key, "status": value["status"], "elapsed_ms": value["elapsed_ms"]} for key, value in all_cases.items()
        ]
        return document
    finally:
        GUARD.uninstall()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1:
        print("--repetitions must be >= 1", file=sys.stderr)
        return 2
    started = time.perf_counter()
    document = asyncio.run(main_async(args))
    payload = json.dumps(document, indent=2, sort_keys=False, ensure_ascii=False)
    if args.output == "-":
        print(payload)
    else:
        out_path = Path(args.output)
        if str(out_path.parent) not in ("", "."):
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    if not args.quiet:
        statuses: dict[str, int] = {}
        for entry in document["case_index"]:
            statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
        print(
            json.dumps(
                {
                    "cases": len(document["case_index"]),
                    "statuses": statuses,
                    "known_failures": len(document["known_failures"]),
                    "coverage_gaps": len(document["coverage_gaps"]),
                    "blocked_model_constructions": document["guard"].get("blocked_provider_constructions"),
                    "blocked_network_attempts": document["guard"].get("blocked_network_attempts"),
                    "output": args.output,
                    "wall_seconds": round(time.perf_counter() - started, 3),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
