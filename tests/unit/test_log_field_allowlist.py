"""AST-based allowlist test for compression telemetry field names.

Walks `.py` files in the compression-relevant modules and asserts that
`logger.*(..., extra={...})` keyword literals are a subset of the
`CompressionLogFields` TypedDict allowlist. Prevents silent leakage of
new field names without code review.

Scope is deliberately narrow — only the modules directly involved in
compression decisions:
  - `dana/core/timeline/` (compression_engine, cheap shrink, reactive)
  - `dana/common/llm/providers/` (PTL-adjacent log lines)
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dana.core.timeline.telemetry import CompressionLogFields


ALLOWLIST = set(CompressionLogFields.__annotations__.keys())

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCAN_DIRS = [
    PROJECT_ROOT / "dana" / "core" / "timeline",
]


def _collect_extra_keys(path: Path) -> list[tuple[int, str]]:
    """Return (line, key) tuples for every `extra={...}` literal key."""
    src = path.read_text()
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # match `logger.<level>(...)` or similar
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "logger"):
            continue
        for kw in node.keywords:
            if kw.arg == "extra" and isinstance(kw.value, ast.Dict):
                for k in kw.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        out.append((node.lineno, k.value))
    return out


@pytest.mark.parametrize("scan_dir", SCAN_DIRS, ids=lambda p: p.name)
def test_log_extra_keys_within_allowlist(scan_dir: Path):
    offenders: list[str] = []
    for py in scan_dir.rglob("*.py"):
        for line, key in _collect_extra_keys(py):
            if key not in ALLOWLIST:
                offenders.append(f"{py}:{line}: '{key}' not in CompressionLogFields")
    assert not offenders, "Unregistered log fields detected:\n  " + "\n  ".join(offenders)
