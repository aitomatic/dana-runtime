"""Agent I/O security guard (protectai/llm-guard).

Public surface:
  - GuardService           the pluggable protocol (injection seam)
  - GuardOutcome/Decision  result value types
  - build_default_guard    env-driven factory used by STARAgent
  - register_input_scanner / register_output_scanner   pluggability hooks
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

from dana.core.guard.config import GuardConfig
from dana.core.guard.noop_service import NoOpGuardService
from dana.core.guard.protocols import GuardService
from dana.core.guard.result import GuardDecision, GuardOutcome
from dana.core.guard.scanner_factory import register_input_scanner, register_output_scanner


logger = structlog.get_logger()

__all__ = [
    "GuardService",
    "GuardOutcome",
    "GuardDecision",
    "GuardConfig",
    "NoOpGuardService",
    "build_default_guard",
    "register_input_scanner",
    "register_output_scanner",
]


def build_default_guard(
    llm_getter: Callable[[], Any],
    config: GuardConfig | None = None,
) -> GuardService:
    """Build the configured guard, falling back to a no-op on disable/import error.

    Args:
        llm_getter: Zero-arg callable returning the LLM used for the scrub pass
            (typically ``lambda: agent.llm_client``).
        config: Optional explicit config; defaults to :meth:`GuardConfig.from_env`.
    """
    config = config or GuardConfig.from_env()
    if not config.enabled:
        return NoOpGuardService()
    try:
        from dana.core.guard.llm_guard_service import LLMGuardService

        return LLMGuardService(config, llm_getter)
    except Exception as exc:  # e.g. llm-guard not installed — degrade gracefully
        logger.warning("guard_init_failed_noop", error=str(exc))
        return NoOpGuardService()
