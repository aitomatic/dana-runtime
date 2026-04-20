"""Resource exposing a ``read_tool_result`` tool that reads back oversized
tool_result content that was dumped to disk at ingest time.

Works in tandem with ``dana.core.agent.tool_result_dump.maybe_dump_oversized_content``:
when a tool_result exceeds the configured threshold, the body is written
under ``{session_folder}/tool_results/{tool_call_id}.txt`` and replaced in
the timeline with a marker. This resource lets the LLM read the file back
on demand, deterministically by ``tool_call_id`` — no absolute paths needed
in the prompt.

Auto-injection: ``STARAgent.__init__`` attaches one instance per agent when
the agent has a filesystem-backed repository. Opt out via the env var
``DANA_DISABLE_TOOL_RESULT_DUMP_RESOURCE=1`` for environments where
filesystem access from tools is undesirable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dana.common.protocols.war import named_tool
from dana.core.agent.tool_result_dump import DUMP_SUBFOLDER
from dana.core.resource.base_resource import BaseResource


class ToolResultDumpResource(BaseResource):
    """Reads oversized tool_result content previously dumped to disk.

    Use the ``read_tool_result`` tool whenever a prior tool response in your
    context has been replaced by a marker like
    ``[Large tool result dumped to file — ... tool_call_id=<id>]``. Pass that
    ``tool_call_id`` to retrieve the original content, optionally sliced via
    ``offset`` and ``limit`` to stay within the token budget.
    """

    def __init__(self, agent: Any, resource_id: str = "tool_result_dump", **kwargs):
        super().__init__(resource_type="tool_result_dump", resource_id=resource_id, **kwargs)
        self._agent = agent

    def _resolve_dump_dir(self) -> Path | None:
        """Return the session's dump folder, or None if unavailable.

        Kept defensive because the agent may be constructed without a
        filesystem-backed repository (e.g. certain unit tests).
        """
        timeline = getattr(self._agent, "_timeline", None)
        if timeline is None:
            return None
        repository = getattr(timeline, "_repository", None)
        if repository is None or not hasattr(repository, "_events_path"):
            return None
        session_id = getattr(self._agent, "_session_id", None)
        if not session_id:
            return None
        return Path(repository._events_path) / session_id / DUMP_SUBFOLDER

    @named_tool(name="read_tool_result")
    def read_tool_result(self, tool_call_id: str, offset: int = 0, limit: int = 2000) -> str:
        """Read back a tool_result that was dumped to disk because it exceeded
        the context budget at ingest time.

        Args:
            tool_call_id: The ``tool_call_id`` from the timeline marker. Use
                the value shown inside the marker, e.g. the ``tc_abc123`` in
                ``[... tool_call_id=tc_abc123]``.
            offset: Character offset to start reading from (default 0). Use
                for paginated reads when the dumped content is still larger
                than fits in your budget.
            limit: Maximum number of characters to return (default 2000). The
                returned string is always truncated to this length; request
                additional slices via ``offset + limit`` to continue.

        Returns:
            The requested slice of the dumped content, or an error string if
            the dump cannot be located or read.
        """
        dump_dir = self._resolve_dump_dir()
        if dump_dir is None:
            return "Error: tool-result dump directory is not available (no filesystem repository configured)."

        # Sanitize to match the writer's filename policy.
        stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tool_call_id)
        path = dump_dir / f"{stem}.txt"

        if not path.exists():
            return f"Error: no dumped content for tool_call_id={tool_call_id} at {path}"

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading dump for tool_call_id={tool_call_id}: {e}"

        if offset < 0:
            offset = 0
        if limit <= 0:
            limit = 2000

        total = len(text)
        end = min(total, offset + limit)
        slice_ = text[offset:end]
        header = f"[read_tool_result tool_call_id={tool_call_id} offset={offset} returned={len(slice_)} total={total}]"
        if end < total:
            header += f" [more available — call again with offset={end}]"
        return f"{header}\n{slice_}"
