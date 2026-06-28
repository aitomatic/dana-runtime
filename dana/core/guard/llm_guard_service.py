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
import re
import threading
from typing import Any

import structlog

from dana.core.guard.config import GuardConfig
from dana.core.guard.output_sanitizer import OutputSanitizer
from dana.core.guard.result import GuardDecision, GuardOutcome
from dana.core.guard.scanner_factory import build_input_scanners, build_output_scanners


logger = structlog.get_logger()


def _normalize_name(name: str) -> str:
    """Canonical form for scanner-name comparison: lowercase, alphanumeric only.
    Makes ``prompt_injection``, ``PromptInjection`` and ``promptinjection`` equal."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


class LLMGuardService:
    """GuardService implementation using protectai/llm-guard."""

    def __init__(self, config: GuardConfig, llm_getter: Callable[[], Any]):
        self._config = config
        self._sanitizer = OutputSanitizer(llm_getter)
        self._input_scanners: list[Any] | None = None  # lazy
        self._output_scanners: list[Any] | None = None  # lazy
        # llm-guard keys its results by scanner CLASS name (e.g. "PromptInjection");
        # these maps translate that back to our snake_case registry keys so findings,
        # triggered, and block_on all share one vocabulary.
        self._input_name_map: dict[str, str] = {}
        self._output_name_map: dict[str, str] = {}
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
            outcome = self._build_outcome(message, sanitized, valid, score, self._input_name_map)
            # Block-listed scanners (e.g. prompt_injection) are classifiers that
            # detect but cannot sanitize — escalate a trip to a hard block.
            if self._should_block(outcome.triggered):
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
            outcome = self._build_outcome(output, sanitized, valid, score, self._output_name_map)
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
                scanners, self._input_name_map = build_input_scanners(self._config.input_scanners)
                self._input_scanners = scanners
            return scanners

    def _ensure_output_scanners(self) -> list[Any]:
        scanners = self._output_scanners
        if scanners is not None:
            return scanners
        with self._scanner_lock:
            scanners = self._output_scanners
            if scanners is None:
                scanners, self._output_name_map = build_output_scanners(self._config.output_scanners)
                self._output_scanners = scanners
            return scanners

    def _should_block(self, triggered: list[str]) -> bool:
        """Match triggered scanners against ``block_on`` tolerant of casing/underscores
        (so ``prompt_injection`` ≡ ``PromptInjection`` ≡ ``promptinjection``)."""
        block = {_normalize_name(name) for name in self._config.block_on}
        return any(_normalize_name(name) in block for name in triggered)

    @staticmethod
    def _build_outcome(original: str, sanitized: str, valid: dict, score: dict, name_map: dict[str, str]) -> GuardOutcome:
        # Translate llm-guard's class-name keys to our snake_case registry keys so
        # findings/triggered match config (input_scanners/output_scanners/block_on).
        def key(raw: str) -> str:
            return name_map.get(raw, raw)

        findings = {key(name): {"valid": bool(ok), "score": score.get(name)} for name, ok in valid.items()}
        triggered = [key(name) for name, ok in valid.items() if not ok]
        changed = sanitized != original
        decision = GuardDecision.SANITIZED if (changed or triggered) else GuardDecision.ALLOW
        return GuardOutcome(decision=decision, text=sanitized, findings=findings, triggered=triggered)
