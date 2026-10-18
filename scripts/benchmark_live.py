"""Serial synthetic AgentSession evaluation. Explicit paid authorization is required.

Run with the same file and arguments against preserved B0/B1/candidate sources.
No dependency sync, production journals, shell tools, or private fixture content.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


MODEL = "gpt-5.4-nano"
AZURE_API_VERSION = "2025-04-01-preview"
INPUT_LIMIT = 24_000
OUTPUT_LIMIT = 4_000
REQUEST_LIMIT = 8
# Uncached reference pricing, with a conservative 2x Azure-price reserve.
REQUEST_RESERVE_USD = (INPUT_LIMIT * 0.20 + OUTPUT_LIMIT * 1.25) / 1_000_000 * 2
TASKS = [
    {
        "id": "discovery",
        "prompts": ["Find the Python module defining add, read it, and answer with its relative path."],
        "answer": "src/math_ops.py",
    },
    {"id": "read", "prompts": ["Read config.json. Answer with the configured port number only."], "answer": "8127"},
    {
        "id": "bounded-edit",
        "prompts": ["Read src/math_ops.py and fix add so it adds rather than subtracts. Preserve every other line."],
        "files": {"src/math_ops.py": "def add(a, b):\n    return a + b\n\n# preserve this marker\n"},
    },
    {
        "id": "independent-read",
        "prompts": ["Read a.txt and b.txt, preferably together. Answer with both codes separated by a space."],
        "answer": "ALPHA17 BETA29",
    },
    {
        "id": "independent-write",
        "prompts": ["Create left.txt containing exactly LEFT and right.txt containing exactly RIGHT. These writes are independent."],
        "files": {"left.txt": "LEFT", "right.txt": "RIGHT"},
    },
    {
        "id": "dependent-edit",
        "prompts": [
            "Read state.txt. First change phase=one to phase=two. Only after that succeeds, change phase=two to phase=three. Keep the newline."
        ],
        "files": {"state.txt": "phase=three\n"},
    },
    {
        "id": "conflicting-edit",
        "prompts": [
            "Read counter.txt. Make two ordered edits to the same file: value=0 to value=1, then value=1 to value=2. Preserve the newline."
        ],
        "files": {"counter.txt": "value=2\n"},
    },
    {
        "id": "read-recovery",
        "prompts": ["First try reading missing.txt. If missing, read fallback.txt instead and answer its code only."],
        "answer": "RECOVER73",
        "required_tools": 2,
    },
    {
        "id": "edit-recovery",
        "prompts": [
            "Try replacing absent-token with repaired in repair.txt. If that fails, read the file and replace broken with repaired instead. Preserve its newline."
        ],
        "files": {"repair.txt": "repaired\n"},
        "required_tools": 3,
    },
    {
        "id": "warm-memory",
        "prompts": [
            "Remember synthetic project release code ORBIT-631. Reply acknowledged.",
            "What release code did I give you? Answer the code only.",
        ],
        "answer": "ORBIT-631",
    },
    {
        "id": "resumed-memory",
        "prompts": [
            "Remember synthetic project release code NOVA-947. Reply acknowledged.",
            "What release code did I give you? Answer the code only.",
        ],
        "answer": "NOVA-947",
        "reopen": True,
    },
    {
        "id": "long-recall",
        "prompts": [
            "Remember launch code COMET-283. Synthetic background follows.\n"
            + "\n".join(f"Log {i:03d}: unit passed its routine check." for i in range(120))
            + "\nReply acknowledged.",
            "What was the launch code before the logs? Answer the code only.",
        ],
        "answer": "COMET-283",
        "reopen": True,
    },
]
FIXTURES = {
    "src/math_ops.py": "def add(a, b):\n    return a - b\n\n# preserve this marker\n",
    "config.json": '{"port":8127}\n',
    "a.txt": "ALPHA17\n",
    "b.txt": "BETA29\n",
    "state.txt": "phase=one\n",
    "counter.txt": "value=0\n",
    "fallback.txt": "RECOVER73\n",
    "repair.txt": "broken\n",
}
MANIFEST = {
    "tasks": TASKS,
    "fixtures": FIXTURES,
    "model": MODEL,
    "max_requests_per_task": REQUEST_LIMIT,
    "azure_api_version": AZURE_API_VERSION,
    "max_wire_input_bytes": INPUT_LIMIT,
    "max_output_tokens": OUTPUT_LIMIT,
    "task_timeout_seconds": 180,
    "context_profile": "synthetic-only explicit runtime context; no ambient geolocation",
    "concurrency": 1,
    "reasoning_effort": "provider default, recorded on wire",
    "cache": "uncontrolled; report observed usage only",
}
MANIFEST_HASH = hashlib.sha256(json.dumps(MANIFEST, sort_keys=True).encode()).hexdigest()


class Ledger:
    def __init__(self, path: Path, cap: float):
        self.path, self.cap = path, cap
        self.data = json.loads(path.read_text()) if path.exists() else {"reserved_usd": 0.0, "attempted_requests": 0}

    def reserve(self):
        if self.data["reserved_usd"] + REQUEST_RESERVE_USD > self.cap:
            raise RuntimeError("Authorized reference-price spending reserve exhausted")
        self.data["reserved_usd"] += REQUEST_RESERVE_USD
        self.data["attempted_requests"] += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")


class Meter:
    def __init__(self, provider, ledger):
        self.requests = []
        self.first_text = None
        self.tool_calls = 0
        self.task_sends = 0
        self.provider = provider
        self.ledger = ledger
        # HTTP hooks count SDK retries as actual attempts, not just outer calls.
        provider.client._client.event_hooks.setdefault("request", []).insert(0, self.on_request)
        provider.client._client.event_hooks.setdefault("response", []).append(self.on_response)
        for endpoint, output_key in (
            (provider.client.responses, "max_output_tokens"),
            (provider.client.chat.completions, "max_completion_tokens"),
        ):
            original = endpoint.create

            async def bounded(*, _original=original, _key=output_key, **kwargs):
                kwargs.pop("max_tokens", None)
                kwargs[_key] = min(kwargs.get(_key, OUTPUT_LIMIT), OUTPUT_LIMIT)
                if _key == "max_completion_tokens" and kwargs.get("stream"):
                    kwargs["stream_options"] = {"include_usage": True}
                stream = await _original(**kwargs)
                return self.observe(stream) if kwargs.get("stream") else stream

            endpoint.create = bounded

    async def on_response(self, response):
        self.requests[-1]["http_status"] = response.status_code

    async def on_request(self, request):
        body = json.loads(request.content)
        if body.get("model") != MODEL:
            raise RuntimeError("Unexpected live model")
        # Byte count is a deliberately conservative upper bound for text tokens.
        if len(request.content) > INPUT_LIMIT:
            raise RuntimeError("Frozen input byte ceiling exceeded")
        if self.task_sends >= REQUEST_LIMIT:
            raise RuntimeError("Frozen per-task request ceiling exceeded")
        self.ledger.reserve()
        self.task_sends += 1
        self.requests.append(
            {
                "sent_ns": time.monotonic_ns(),
                "route": request.url.path,
                "wire_input_bytes": len(request.content),
                "request_sha256": hashlib.sha256(request.content).hexdigest(),
                "settings": {k: body.get(k) for k in ("model", "reasoning", "temperature", "max_output_tokens", "max_completion_tokens")},
                "usage": None,
                "finish_status": None,
                "response_id": None,
            }
        )

    async def observe(self, stream):
        row = self.requests[-1]
        try:
            async for event in stream:
                if getattr(event, "type", "") == "response.output_text.delta" or any(
                    getattr(getattr(c, "delta", None), "content", None) for c in getattr(event, "choices", [])
                ):
                    self.first_text = self.first_text or time.monotonic_ns()
                response = getattr(event, "response", event)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    row["usage"] = usage.model_dump(mode="json")
                row["response_id"] = getattr(response, "id", None) or row["response_id"]
                status = getattr(response, "status", None)
                if status:
                    row["finish_status"] = status
                for choice in getattr(event, "choices", []):
                    if choice.finish_reason:
                        row["finish_status"] = choice.finish_reason
                yield event
        finally:
            await stream.close()


async def attempt(provider_name, task, repetition, ledger):
    from dana.common.llm.providers.factory import create_provider
    from dana.config.storage_config import FileStorageConfig
    from dana.core.agent.star_agent import STARAgent
    from dana.core.resource.file_edit_resource import FileEditResource
    from dana.core.resource.file_io_resource import FileIOResource
    from dana.core.resource.search_resource import SearchResource
    from dana.core.session.agent_session import AgentSession, TextBlock
    from dana.core.session.journal.models import OwnerScope
    from dana.repositories.repository_factory import RepositoryFactory

    provider = create_provider(provider_name, model=MODEL, **({"api_version": AZURE_API_VERSION} if provider_name == "azure" else {}))
    meter = Meter(provider, ledger)
    started = time.monotonic_ns()
    row = {
        "provider": provider_name,
        "model": MODEL,
        "task": task["id"],
        "repetition": repetition,
        "manifest_sha256": MANIFEST_HASH,
        "success": False,
        "error": None,
        "events": [],
        "answers": [],
        "first_host_progress_ns": None,
        "first_host_assistant_ns": None,
        "tool_events": 0,
        "api_version": getattr(provider.client, "_api_version", None),
    }
    session = None
    try:
        with tempfile.TemporaryDirectory(prefix="dana-live-synthetic-") as folder:
            root = Path(folder).resolve()
            for name, content in FIXTURES.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            scope = OwnerScope(owner_id="synthetic-benchmark", workspace=folder)
            repositories = RepositoryFactory()
            for kind, (creator, _) in list(repositories._creators.items()):
                repositories.register(kind, creator, FileStorageConfig(workspace_folder=str(root / ".storage")))

            def factory():
                agent = STARAgent(
                    llm_provider_instance=provider,
                    auto_register=False,
                    enable_skills=False,
                    enable_assistant=False,
                    max_context_tokens=20_000,
                    repository_factory=repositories,
                    identity_override=f"Complete the synthetic coding task using tools. Work only under {folder}. Never access other directories. Be concise.",
                )
                resources = [
                    FileIOResource(resource_id="files", base_path=folder, auto_register=False),
                    FileEditResource(resource_id="edit", base_path=folder, auto_register=False),
                    SearchResource(resource_id="search", base_path=folder, auto_register=False),
                ]
                for resource in resources:
                    resolve = resource._resolve_path

                    def restricted(path, _resolve=resolve):
                        resolved = _resolve(path).resolve()
                        if not resolved.is_relative_to(root):
                            raise PermissionError("Benchmark tools cannot leave the synthetic fixture")
                        return resolved

                    resource._resolve_path = restricted
                agent.with_resources(*resources)
                # Same benchmark-only isolation in every revision; not a claimed optimization.
                agent._runtime._get_runtime_context = lambda *a, **kw: {"benchmark": "synthetic"}
                return agent

            async def drive():
                nonlocal session
                session = await AgentSession.create(
                    session_id="synthetic", journal_path=str(root / "journal.db"), owner_scope=scope, agent_factory=factory
                )
                for turn, prompt in enumerate(task["prompts"]):
                    if turn and task.get("reopen"):
                        await session._repository.close()
                        session = await AgentSession.create(
                            session_id="synthetic", journal_path=str(root / "journal.db"), owner_scope=scope, agent_factory=factory
                        )
                    chunks = []
                    async for event in session.prompt([TextBlock(text=prompt)]):
                        now = time.monotonic_ns()
                        kind = event.event_type.value
                        row["events"].append({"kind": kind, "at_ns": now})
                        if "thought" in kind or "chunk" in kind:
                            row["first_host_progress_ns"] = row["first_host_progress_ns"] or now
                        if "assistant" in kind and "chunk" in kind:
                            row["first_host_assistant_ns"] = row["first_host_assistant_ns"] or now
                            chunks.append(event.text or "")
                        if "tool" in kind:
                            row["tool_events"] += 1
                    row["answers"].append("".join(chunks))
                    terminal = session.last_terminal
                    if terminal is None or terminal.fact_type.name != "TURN_COMPLETED":
                        row["error"] = {"type": "SessionTerminal", "fact_type": terminal.fact_type.value if terminal else None}
                checks = {
                    name: (root / name).exists() and (root / name).read_text() == expected
                    for name, expected in task.get("files", {}).items()
                }
                if "answer" in task:
                    checks["answer"] = task["answer"] in row["answers"][-1]
                if task.get("required_tools"):
                    checks["tool_requests"] = sum("tool_requested" in e["kind"] for e in row["events"]) >= task["required_tools"]
                row["checks"] = checks
                row["success"] = bool(checks) and all(checks.values()) and row["error"] is None

            await asyncio.wait_for(drive(), timeout=180)
    except Exception as exc:
        # Do not serialize exception bodies or headers that might contain credentials.
        row["error"] = {"type": type(exc).__name__, "status_code": getattr(exc, "status_code", None), "code": getattr(exc, "code", None)}
    finally:
        if session is not None:
            await session._repository.close()
        await provider.client.close()
    row.update(
        {
            "started_ns": started,
            "completed_ns": time.monotonic_ns(),
            "first_provider_text_ns": meter.first_text,
            "requests": meter.requests,
            "attempted_requests": meter.task_sends,
        }
    )
    return row


async def main(args):
    # Must precede imports of dana, which loads .env. Disable payload tracing in this
    # synthetic lane explicitly and consistently, never mutate production settings.
    os.environ["LANGFUSE_ENABLED"] = "false"
    os.environ["DANA_DISABLE_TOOL_RESULT_DUMP_RESOURCE"] = "1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(args.ledger, 30.0)
    selected = TASKS[:2] if args.pilot else TASKS
    rows = []
    result = {
        "manifest": MANIFEST,
        "manifest_sha256": MANIFEST_HASH,
        "label": args.label,
        "pilot": args.pilot,
        "repetitions": args.repetitions,
        "selected_providers": args.providers,
        "rows": rows,
    }
    for repetition in range(args.repetitions):
        for provider in args.providers:
            for task in selected:
                rows.append(await attempt(provider, task, repetition, ledger))
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({k: rows[-1][k] for k in ("provider", "task", "success", "error", "attempted_requests")}), flush=True)
    print(json.dumps({"manifest_sha256": MANIFEST_HASH, "reserved_usd": ledger.data["reserved_usd"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--providers", nargs="+", choices=("azure", "openai"), default=["azure", "openai"])
    parser.add_argument("--label", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--pilot", action="store_true")
    asyncio.run(main(parser.parse_args()))
