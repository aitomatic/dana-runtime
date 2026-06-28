"""llm-guard backed GuardService.

Orchestrates the Core Security pipeline:
  - scan_input:  llm-guard input scanners sanitize the user message.
  - scan_output: llm-guard output scanners strip/redact the response.
  - sanitize_output: LLM scrub pass (delegated to OutputSanitizer).

Scanners are loaded lazily on first use (model init is heavy) and cached. Every
operation is wrapped fail-open: any error logs a warning and returns the text
unchanged, so a broken/missing model never breaks an agent run.
"""

from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any

import structlog

from dana.core.guard.config import GuardConfig
from dana.core.guard.output_sanitizer import OutputSanitizer
from dana.core.guard.result import GuardDecision, GuardOutcome
from dana.core.guard.scanner_factory import build_input_scanners, build_output_scanners


logger = structlog.get_logger()


class LLMGuardService:
    """GuardService implementation using protectai/llm-guard."""

    def __init__(self, config: GuardConfig, llm_getter: Callable[[], Any]):
        self._config = config
        self._sanitizer = OutputSanitizer(llm_getter)
        self._input_scanners: list[Any] | None = None  # lazy
        self._output_scanners: list[Any] | None = None  # lazy
        # Guards lazy scanner init: aquery offloads scans to executor threads, so
        # concurrent calls on one agent could otherwise double-load heavy models.
        self._scanner_lock = threading.Lock()

    # ------------------------------------------------------------------ input
    def scan_input(self, message: str) -> GuardOutcome:
        if not self._config.enabled or not message or not message.strip():
            return GuardOutcome.allow(message)
        try:
            from llm_guard import scan_prompt

            scanners = self._ensure_input_scanners()
            if not scanners:
                return GuardOutcome.allow(message)
            sanitized, valid, score = scan_prompt(scanners, message)
            outcome = self._build_outcome(message, sanitized, valid, score)
            # Block-listed scanners (e.g. prompt_injection) are classifiers that
            # detect but cannot sanitize — escalate a trip to a hard block.
            if any(name in self._config.block_on for name in outcome.triggered):
                outcome.decision = GuardDecision.BLOCKED
                outcome.block_message = self._config.block_message
            logger.info("guard_input_scan", **outcome.to_audit())
            return outcome
        except Exception as exc:
            logger.warning("guard_input_scan_failed", error=str(exc))
            return GuardOutcome.allow(message)

    # ----------------------------------------------------------------- output
    def scan_output(self, prompt: str, output: str) -> GuardOutcome:
        if not self._config.enabled or not output or not output.strip():
            return GuardOutcome.allow(output)
        try:
            from llm_guard import scan_output as lg_scan_output

            scanners = self._ensure_output_scanners()
            if not scanners:
                return GuardOutcome.allow(output)
            sanitized, valid, score = lg_scan_output(scanners, prompt or "", output)
            outcome = self._build_outcome(output, sanitized, valid, score)
            logger.info("guard_output_scan", **outcome.to_audit())
            return outcome
        except Exception as exc:
            logger.warning("guard_output_scan_failed", error=str(exc))
            return GuardOutcome.allow(output)

    def sanitize_output(self, output: str) -> GuardOutcome:
        if not self._config.enabled or not self._config.sanitize_llm_enabled:
            return GuardOutcome.allow(output)
        return self._sanitizer.sanitize(output)

    # ----------------------------------------------------------------- helpers
    def _ensure_input_scanners(self) -> list[Any]:
        scanners = self._input_scanners
        if scanners is not None:
            return scanners
        with self._scanner_lock:
            scanners = self._input_scanners
            if scanners is None:
                scanners = build_input_scanners(self._config.input_scanners)
                self._input_scanners = scanners
            return scanners

    def _ensure_output_scanners(self) -> list[Any]:
        scanners = self._output_scanners
        if scanners is not None:
            return scanners
        with self._scanner_lock:
            scanners = self._output_scanners
            if scanners is None:
                scanners = build_output_scanners(self._config.output_scanners)
                self._output_scanners = scanners
            return scanners

    @staticmethod
    def _build_outcome(original: str, sanitized: str, valid: dict, score: dict) -> GuardOutcome:
        findings = {name: {"valid": bool(ok), "score": score.get(name)} for name, ok in valid.items()}
        triggered = [name for name, ok in valid.items() if not ok]
        changed = sanitized != original
        decision = GuardDecision.SANITIZED if (changed or triggered) else GuardDecision.ALLOW
        return GuardOutcome(decision=decision, text=sanitized, findings=findings, triggered=triggered)
