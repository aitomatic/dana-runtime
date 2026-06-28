"""Guard result types.

Pure data structures describing the outcome of a guard scan/sanitize pass.
No behavior beyond simple construction/serialization helpers — these are the
values that flow from :class:`GuardService` back to the agent and into the
audit trail (logs + ``timeline.json``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class GuardDecision(StrEnum):
    """What the guard did to a piece of text."""

    ALLOW = "allow"  # nothing flagged; text returned unchanged
    SANITIZED = "sanitized"  # text was modified/redacted by scanners and/or LLM scrub
    BLOCKED = "blocked"  # input matched a block-listed scanner; request refused (input only)


@dataclass
class GuardOutcome:
    """Result of a single guard operation.

    Attributes:
        decision: ALLOW or SANITIZED.
        text: Final (possibly sanitized) text the caller should use.
        findings: Per-scanner detail, ``{scanner_name: {"valid": bool, "score": float}}``.
        triggered: Names of scanners that flagged the text as invalid.
    """

    decision: GuardDecision
    text: str
    findings: dict = field(default_factory=dict)
    triggered: list[str] = field(default_factory=list)
    block_message: str | None = None  # refusal text to return when decision is BLOCKED

    @property
    def sanitized(self) -> bool:
        return self.decision == GuardDecision.SANITIZED

    @property
    def blocked(self) -> bool:
        return self.decision == GuardDecision.BLOCKED

    def to_audit(self) -> dict:
        """Compact dict for structured logs / timeline metadata."""
        return {
            "decision": self.decision.value,
            "triggered": self.triggered,
            "findings": self.findings,
        }

    @classmethod
    def allow(cls, text: str) -> GuardOutcome:
        """Convenience constructor for the no-change path."""
        return cls(decision=GuardDecision.ALLOW, text=text)
