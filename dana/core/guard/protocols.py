"""GuardService protocol — the single dependency-inversion seam for I/O security.

Mirrors how :class:`LLMProvider` is the one swappable seam for providers: callers
depend on this Protocol, concrete implementations (llm-guard backed, no-op) are
injected. Keep the surface small — three operations, all synchronous (async
callers offload to an executor; see STARAgent.aquery).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dana.core.guard.result import GuardOutcome


@runtime_checkable
class GuardService(Protocol):
    """Contract for guarding agent input and output."""

    def scan_input(self, message: str) -> GuardOutcome:
        """Scan + sanitize a user message before it reaches the LLM."""
        ...

    def scan_output(self, prompt: str, output: str) -> GuardOutcome:
        """Run rule-based output scanners (PII, toxicity, ...) over a response."""
        ...

    def sanitize_output(self, output: str) -> GuardOutcome:
        """LLM-based scrub pass guaranteeing remaining sensitive data is stripped."""
        ...
