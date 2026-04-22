"""CRITICAL-2 tests: oversized tool_result dump + read_tool_result resource.

Covers:
- Ingest-time: content under threshold passes through unchanged.
- Ingest-time: content over threshold is written to a session-scoped file and
  replaced with a marker string that preserves the tool_call_id.
- Threshold can be disabled via env (``DANA_TOOL_RESULT_DUMP_THRESHOLD_CHARS=0``).
- read_tool_result resource slice semantics (offset, limit, more-available hint).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dana.core.agent.tool_result_dump import (
    DUMP_SUBFOLDER,
    maybe_dump_oversized_content,
    resolve_threshold_chars,
)
from dana.core.resource.tool_result_dump_resource import ToolResultDumpResource


class _StubRepo:
    def __init__(self, events_path: Path):
        self._events_path = events_path


class _StubTimeline:
    def __init__(self, repo: _StubRepo):
        self._repository = repo


class _StubAgent:
    def __init__(self, events_path: Path, session_id: str = "sess-1"):
        self._timeline = _StubTimeline(_StubRepo(events_path))
        self._session_id = session_id


# ---------------------------------------------------------------------------
# Threshold env resolver
# ---------------------------------------------------------------------------


def test_threshold_defaults_to_50000(monkeypatch):
    monkeypatch.delenv("DANA_TOOL_RESULT_DUMP_THRESHOLD_CHARS", raising=False)
    assert resolve_threshold_chars() == 50000


def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv("DANA_TOOL_RESULT_DUMP_THRESHOLD_CHARS", "1024")
    assert resolve_threshold_chars() == 1024


def test_threshold_env_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("DANA_TOOL_RESULT_DUMP_THRESHOLD_CHARS", "not-an-int")
    assert resolve_threshold_chars() == 50000


def test_threshold_zero_disables_dump(monkeypatch, tmp_path):
    monkeypatch.setenv("DANA_TOOL_RESULT_DUMP_THRESHOLD_CHARS", "0")
    huge = "x" * 1_000_000
    assert maybe_dump_oversized_content(huge, "tc_1", tmp_path) == huge


# ---------------------------------------------------------------------------
# Dump behavior
# ---------------------------------------------------------------------------


def test_under_threshold_content_passes_through(tmp_path, monkeypatch):
    monkeypatch.setenv("DANA_TOOL_RESULT_DUMP_THRESHOLD_CHARS", "1000")
    content = "hello"
    result = maybe_dump_oversized_content(content, "tc_1", tmp_path)
    assert result == content
    # No file created.
    assert not (tmp_path / DUMP_SUBFOLDER).exists()


def test_over_threshold_content_is_dumped_and_replaced_with_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("DANA_TOOL_RESULT_DUMP_THRESHOLD_CHARS", "100")
    content = "A" * 500
    result = maybe_dump_oversized_content(content, "tc_abc", tmp_path)

    assert result != content
    assert "Large tool result dumped to file" in result
    assert "tool_call_id=tc_abc" in result
    # Physical file exists with original content.
    dumped = tmp_path / DUMP_SUBFOLDER / "tc_abc.txt"
    assert dumped.exists()
    assert dumped.read_text() == content


def test_dump_sanitizes_tool_call_id_filesystem_chars(tmp_path, monkeypatch):
    """IDs sometimes carry ``/`` or ``:`` from certain providers. The file
    name must not traverse directories."""
    monkeypatch.setenv("DANA_TOOL_RESULT_DUMP_THRESHOLD_CHARS", "10")
    _ = maybe_dump_oversized_content("X" * 50, "../escape:attempt", tmp_path)
    files = list((tmp_path / DUMP_SUBFOLDER).iterdir())
    assert len(files) == 1
    assert ".." not in files[0].name and ":" not in files[0].name


def test_no_session_folder_skips_dump(monkeypatch):
    """When no filesystem is available, content is returned unchanged rather
    than silently dropped."""
    monkeypatch.setenv("DANA_TOOL_RESULT_DUMP_THRESHOLD_CHARS", "10")
    content = "Y" * 100
    result = maybe_dump_oversized_content(content, "tc_x", None)
    assert result == content


def test_missing_tool_call_id_gets_anon_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("DANA_TOOL_RESULT_DUMP_THRESHOLD_CHARS", "10")
    _ = maybe_dump_oversized_content("Z" * 50, None, tmp_path)
    files = list((tmp_path / DUMP_SUBFOLDER).iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("anon-")


# ---------------------------------------------------------------------------
# ToolResultDumpResource.read_tool_result
# ---------------------------------------------------------------------------


def _session_dir(tmp_path: Path, session_id: str = "sess-1") -> Path:
    folder = tmp_path / session_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def test_read_tool_result_returns_content(tmp_path):
    session = _session_dir(tmp_path)
    agent = _StubAgent(events_path=tmp_path)
    # Write a dump manually — decouples from the ingest-time writer.
    (session / DUMP_SUBFOLDER).mkdir(parents=True, exist_ok=True)
    (session / DUMP_SUBFOLDER / "tc_abc.txt").write_text("hello world")

    res = ToolResultDumpResource(agent=agent, auto_register=False)
    out = res.read_tool_result("tc_abc", offset=0, limit=100)
    assert "hello world" in out
    assert "tool_call_id=tc_abc" in out


def test_read_tool_result_honors_offset_and_limit(tmp_path):
    session = _session_dir(tmp_path)
    agent = _StubAgent(events_path=tmp_path)
    (session / DUMP_SUBFOLDER).mkdir(parents=True, exist_ok=True)
    (session / DUMP_SUBFOLDER / "tc_big.txt").write_text("0123456789" * 10)  # 100 chars

    res = ToolResultDumpResource(agent=agent, auto_register=False)
    out = res.read_tool_result("tc_big", offset=20, limit=10)

    # Only 10 chars of payload returned, and a "more available" hint present.
    assert "more available" in out
    body = out.split("\n", 1)[1] if "\n" in out else ""
    assert body == "0123456789"


def test_read_tool_result_missing_file_returns_error(tmp_path):
    _ = _session_dir(tmp_path)
    agent = _StubAgent(events_path=tmp_path)
    res = ToolResultDumpResource(agent=agent, auto_register=False)
    out = res.read_tool_result("does_not_exist")
    assert out.startswith("Error:")


def test_read_tool_result_no_repository_returns_error():
    class _NoRepoAgent:
        _timeline = None
        _session_id = "sess"

    res = ToolResultDumpResource(agent=_NoRepoAgent(), auto_register=False)
    out = res.read_tool_result("tc_1")
    assert out.startswith("Error:")


# ---------------------------------------------------------------------------
# End-to-end sanity: CRITICAL-2 scenario — huge recent tool_result
# ---------------------------------------------------------------------------


def test_critical_2_scenario_timeline_stays_small_after_huge_tool_result(tmp_path, monkeypatch):
    """The wedge the review described: one 500KB tool_result would keep
    PTL'ing forever because neither cheap_shrink (skips recent) nor
    reactive_compact (drops oldest) can shed it. With the dump fix, the
    timeline sees only a compact marker and stays manageable."""
    monkeypatch.setenv("DANA_TOOL_RESULT_DUMP_THRESHOLD_CHARS", "1000")
    huge = "H" * 500_000
    marker = maybe_dump_oversized_content(huge, "tc_huge", tmp_path)
    # Marker is tiny — well under 1KB, independent of the original blob size.
    assert len(marker) < 1000
    # Original content is still retrievable for audit / read_tool_result.
    dumped = tmp_path / DUMP_SUBFOLDER / "tc_huge.txt"
    assert dumped.read_text() == huge


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
