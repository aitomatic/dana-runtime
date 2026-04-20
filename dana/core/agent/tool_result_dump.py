"""Oversized tool_result dumping — CRITICAL-2 fix.

When a tool returns a very large result (e.g. full HTML scrape, huge log,
large JSON blob), keeping the content verbatim in the timeline wedges
compression: ``cheap_shrink_tool_results`` skips recent entries by design,
and ``reactive_compact`` only drops OLDEST entries, so the oversized recent
result is unreclaimable and every retry hits PromptTooLong again.

This module moves oversized content to a session-scoped file at ingest time
and leaves a compact marker in the timeline. The marker preserves the
``tool_call_id`` so any agent with a file-read tool (or the dedicated
``ToolResultDumpResource.read_tool_result``) can fetch the original bytes on
demand.

Threshold knob: ``DANA_TOOL_RESULT_DUMP_THRESHOLD_CHARS`` (default 50000).
Set to ``0`` to disable dumping entirely (YAGNI escape hatch for tests).
"""

from __future__ import annotations

import os
from pathlib import Path
import uuid


DEFAULT_THRESHOLD_CHARS = 50000
ENV_THRESHOLD = "DANA_TOOL_RESULT_DUMP_THRESHOLD_CHARS"

# Files written by ``maybe_dump_oversized_content`` land here under the
# session folder. Kept as a submodule-level constant so tests and consumers
# (e.g. ``ToolResultDumpResource``) share the same name.
DUMP_SUBFOLDER = "tool_results"


def resolve_threshold_chars() -> int:
    """Return the active dump threshold in characters.

    ``0`` disables dumping. Invalid env values fall back to the default.
    """
    raw = os.getenv(ENV_THRESHOLD)
    if raw is None:
        return DEFAULT_THRESHOLD_CHARS
    try:
        v = int(raw)
        return max(0, v)
    except ValueError:
        return DEFAULT_THRESHOLD_CHARS


def _build_marker(path: Path, size_chars: int, tool_call_id: str | None) -> str:
    """Construct the replacement content stored in the timeline entry."""
    id_part = f"tool_call_id={tool_call_id}" if tool_call_id else "tool_call_id=unavailable"
    return (
        f"[Large tool result dumped to file — {size_chars} chars. "
        f"Path: {path}. Use the read_tool_result tool with {id_part} to inspect "
        f"the original content, or request a slice via offset/limit.]"
    )


def maybe_dump_oversized_content(
    content: str,
    tool_call_id: str | None,
    session_folder: Path | None,
) -> str:
    """If ``content`` exceeds the configured threshold, write it to a file
    under ``session_folder / DUMP_SUBFOLDER`` and return a marker string.
    Otherwise return ``content`` unchanged.

    Args:
        content: Stringified tool_result body.
        tool_call_id: Stable identifier from the upstream tool_use. Used as
            the filename stem for deterministic lookup; falls back to a uuid
            when absent.
        session_folder: Per-session directory. If ``None`` (no repository,
            in-memory tests), dumping is skipped even when over threshold.

    Returns:
        Either the original ``content`` or a marker string. The timeline
        entry's ``tool_call_id`` is preserved separately by the caller so
        API pair-integrity (tool_use ↔ tool_result) is not broken.
    """
    threshold = resolve_threshold_chars()
    if threshold <= 0 or len(content) <= threshold:
        return content

    if session_folder is None:
        # No filesystem available — leave content as-is rather than
        # silently dropping it. Downstream compression will still be
        # strained but at least data isn't lost.
        return content

    dump_dir = session_folder / DUMP_SUBFOLDER
    dump_dir.mkdir(parents=True, exist_ok=True)

    stem = tool_call_id if tool_call_id else f"anon-{uuid.uuid4().hex[:12]}"
    # Sanitize to keep filesystem happy — tool_call_ids are already ASCII
    # in practice but provider IDs occasionally carry ``/`` or ``:``.
    stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)
    path = dump_dir / f"{stem}.txt"

    path.write_text(content, encoding="utf-8")
    return _build_marker(path, len(content), tool_call_id)


def resolve_session_folder_for_agent(agent) -> Path | None:
    """Best-effort resolve the per-session dump folder from an agent.

    Returns ``None`` when the agent lacks a filesystem-backed repository
    (tests with in-memory repos, unconfigured agents, etc.). Callers must
    handle ``None`` by skipping dump.
    """
    timeline = getattr(agent, "_timeline", None)
    if timeline is None:
        return None
    repository = getattr(timeline, "_repository", None)
    if repository is None or not hasattr(repository, "_events_path"):
        return None
    session_id = getattr(agent, "_session_id", None)
    if not session_id:
        return None
    return Path(repository._events_path) / session_id
