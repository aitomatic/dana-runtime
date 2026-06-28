"""Env-var driven guard configuration.

Simple env vars over a config system (project preference). All knobs have safe
defaults so the guard is on out-of-the-box with the Core Security scanner set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os


_DEFAULT_INPUT_SCANNERS = "prompt_injection,secrets,toxicity"
_DEFAULT_OUTPUT_SCANNERS = "sensitive,toxicity"
# Input scanners whose trip blocks the request outright (classifier-type threats
# like prompt injection can't be "sanitized" — only detected — so block them).
_DEFAULT_BLOCK_ON = "prompt_injection"
_DEFAULT_BLOCK_MESSAGE = "I can't help with that request — it was flagged by the security policy."


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class GuardConfig:
    """Resolved guard settings."""

    enabled: bool = True
    input_scanners: list[str] = field(default_factory=lambda: _csv(_DEFAULT_INPUT_SCANNERS))
    output_scanners: list[str] = field(default_factory=lambda: _csv(_DEFAULT_OUTPUT_SCANNERS))
    sanitize_llm_enabled: bool = True
    fail_mode: str = "open"  # "open" -> fail-open (no-op + warn) on any guard error
    block_on: list[str] = field(default_factory=lambda: _csv(_DEFAULT_BLOCK_ON))
    block_message: str = _DEFAULT_BLOCK_MESSAGE

    @classmethod
    def from_env(cls) -> GuardConfig:
        return cls(
            enabled=_bool(os.getenv("DANA_GUARD_ENABLED"), True),
            input_scanners=_csv(os.getenv("DANA_GUARD_INPUT_SCANNERS", _DEFAULT_INPUT_SCANNERS)),
            output_scanners=_csv(os.getenv("DANA_GUARD_OUTPUT_SCANNERS", _DEFAULT_OUTPUT_SCANNERS)),
            sanitize_llm_enabled=_bool(os.getenv("DANA_GUARD_SANITIZE_LLM_ENABLED"), True),
            fail_mode=os.getenv("DANA_GUARD_FAIL_MODE", "open").strip().lower(),
            block_on=_csv(os.getenv("DANA_GUARD_BLOCK_ON", _DEFAULT_BLOCK_ON)),
            block_message=os.getenv("DANA_GUARD_BLOCK_MESSAGE", _DEFAULT_BLOCK_MESSAGE),
        )
