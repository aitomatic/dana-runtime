"""D7.3 capability rollback flags for ``dana-code`` (CLI).

Every D2–D6 capability surfaced by the CLI is gated by an independent
``DANA_CODE_*_ENABLED`` flag (default on, "1"), so any capability can be turned
off without code changes (project constraint: a rollback flag per delivery).

Flags are read from the environment at call time (not cached) so toggling at
runtime is respected. Mirrors the inline ``os.environ.get`` pattern used by
``dana-acp`` (e.g. ``DANA_MODEL_SWITCHING_ENABLED``), centralised here for
discoverability and unit testing.
"""

from __future__ import annotations

import os


# Default-on: any value other than "0" enables the capability.
_DEFAULT_ON = "1"


def _flag(name: str, default: str = _DEFAULT_ON) -> bool:
    """Read a ``DANA_CODE_*_ENABLED`` flag (default on)."""
    return os.environ.get(name, default) != "0"


def permission_preflight_enabled() -> bool:
    """DANA_CODE_PERMISSION_PREFLIGHT_ENABLED — interactive permission prompts (D3)."""
    return _flag("DANA_CODE_PERMISSION_PREFLIGHT_ENABLED")


def model_switch_enabled() -> bool:
    """DANA_CODE_MODEL_SWITCH_ENABLED — ``/model`` switching (D4)."""
    return _flag("DANA_CODE_MODEL_SWITCH_ENABLED")


def tool_catalog_enabled() -> bool:
    """DANA_CODE_TOOL_CATALOG_ENABLED — Tool Catalog-backed tool calls (D2)."""
    return _flag("DANA_CODE_TOOL_CATALOG_ENABLED")


def mcp_enabled() -> bool:
    """DANA_CODE_MCP_ENABLED — MCP tool leases (D5)."""
    return _flag("DANA_CODE_MCP_ENABLED")


def multimodal_enabled() -> bool:
    """DANA_CODE_MULTIMODAL_ENABLED — multimodal input (D6)."""
    return _flag("DANA_CODE_MULTIMODAL_ENABLED")


ALL_FLAGS: tuple[str, ...] = (
    "DANA_CODE_PERMISSION_PREFLIGHT_ENABLED",
    "DANA_CODE_MODEL_SWITCH_ENABLED",
    "DANA_CODE_TOOL_CATALOG_ENABLED",
    "DANA_CODE_MCP_ENABLED",
    "DANA_CODE_MULTIMODAL_ENABLED",
)
