"""No-op guard — the fail-open / disabled implementation of GuardService.

Used when guarding is disabled (``DANA_GUARD_ENABLED=false``) or when llm-guard
cannot be imported. Every method returns the text unchanged so existing runs are
never affected.
"""

from __future__ import annotations

from dana.core.guard.result import GuardOutcome


class NoOpGuardService:
    """Returns ALLOW for everything; no scanning, no LLM calls."""

    def scan_input(self, message: str) -> GuardOutcome:
        return GuardOutcome.allow(message)

    def scan_output(self, prompt: str, output: str) -> GuardOutcome:
        return GuardOutcome.allow(output)

    def sanitize_output(self, output: str) -> GuardOutcome:
        return GuardOutcome.allow(output)
