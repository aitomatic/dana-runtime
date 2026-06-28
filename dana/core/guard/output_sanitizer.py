"""LLM-based output scrubber — implementation detail behind
``GuardService.sanitize_output``.

Rule-based scanners (llm-guard) catch known patterns; this pass uses the agent's
own LLM to catch sensitive data they miss, rewriting the response with sensitive
content redacted. It is a *scrubber*, not a judge: it never decides allow/block,
it only returns cleaned text. Fail-open everywhere — any error yields the original
text unchanged so a flaky LLM never drops a legitimate response.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

from dana.common.llm.types import LLMMessage
from dana.core.guard.result import GuardDecision, GuardOutcome


logger = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You are a data-loss-prevention redactor. Rewrite the assistant message so that ALL "
    "sensitive data is removed or masked: personal identifiers (names, emails, phone numbers, "
    "physical addresses), credentials, API keys, secrets/tokens, and financial data (card or "
    "account numbers). Replace each with a clear placeholder such as [REDACTED_EMAIL] or "
    "[REDACTED_SECRET]. Preserve the meaning, tone, and formatting of everything else. "
    "Return ONLY the cleaned message — no preamble, no explanation."
)


class OutputSanitizer:
    """Reuses the agent's LLM (via ``llm_getter``) to scrub remaining sensitive data."""

    def __init__(self, llm_getter: Callable[[], Any]):
        self._llm_getter = llm_getter

    def sanitize(self, output: str) -> GuardOutcome:
        if not output or not output.strip():
            return GuardOutcome.allow(output)

        try:
            llm = self._llm_getter()
            if llm is None:
                return GuardOutcome.allow(output)

            messages = [
                LLMMessage(role="system", content=_SYSTEM_PROMPT),
                LLMMessage(role="user", content=output),
            ]
            response = llm.chat_response_sync(messages, temperature=0, json_mode=False)
            cleaned = (getattr(response, "content", "") or "").strip()

            if not cleaned or cleaned == output.strip():
                return GuardOutcome.allow(output)

            logger.info("guard_output_sanitize", changed=True, original_len=len(output), cleaned_len=len(cleaned))
            return GuardOutcome(
                decision=GuardDecision.SANITIZED,
                text=cleaned,
                findings={"llm_sanitizer": {"changed": True}},
                triggered=["llm_sanitizer"],
            )
        except Exception as exc:  # never let scrubbing failure drop a response
            logger.warning("guard_output_sanitize_failed", error=str(exc))
            return GuardOutcome.allow(output)
