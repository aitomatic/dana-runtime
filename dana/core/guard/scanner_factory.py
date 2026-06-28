"""Pluggable scanner registry — the mechanism that makes the guard's scanner set
swappable, the same way the provider factory makes LLM providers swappable.

A scanner is registered by name against a zero-arg *builder* callable that returns
an llm-guard scanner instance. ``llm_guard`` is imported lazily inside the builders
so importing this module never pulls in torch/transformers.

Extend at runtime:

    from dana.core.guard.scanner_factory import register_input_scanner
    register_input_scanner("ban_topics", lambda: BanTopics(topics=["violence"]))
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog


logger = structlog.get_logger()

# name -> builder() -> llm-guard scanner instance
_INPUT_REGISTRY: dict[str, Callable[[], Any]] = {}
_OUTPUT_REGISTRY: dict[str, Callable[[], Any]] = {}


def register_input_scanner(name: str, builder: Callable[[], Any]) -> None:
    """Register (or override) an input scanner builder."""
    _INPUT_REGISTRY[name] = builder


def register_output_scanner(name: str, builder: Callable[[], Any]) -> None:
    """Register (or override) an output scanner builder."""
    _OUTPUT_REGISTRY[name] = builder


def _register_defaults() -> None:
    """Idempotently register the Core Security scanner set (lazy llm-guard imports)."""
    if _INPUT_REGISTRY and _OUTPUT_REGISTRY:
        return

    def _prompt_injection() -> Any:
        from llm_guard.input_scanners import PromptInjection

        return PromptInjection()

    def _secrets() -> Any:
        from llm_guard.input_scanners import Secrets

        return Secrets()

    def _input_toxicity() -> Any:
        from llm_guard.input_scanners import Toxicity

        return Toxicity()

    def _sensitive() -> Any:
        from llm_guard.output_scanners import Sensitive

        return Sensitive()

    def _output_toxicity() -> Any:
        from llm_guard.output_scanners import Toxicity

        return Toxicity()

    # setdefault: never clobber a scanner the user registered before first build.
    _INPUT_REGISTRY.setdefault("prompt_injection", _prompt_injection)
    _INPUT_REGISTRY.setdefault("secrets", _secrets)
    _INPUT_REGISTRY.setdefault("toxicity", _input_toxicity)
    _OUTPUT_REGISTRY.setdefault("sensitive", _sensitive)
    _OUTPUT_REGISTRY.setdefault("toxicity", _output_toxicity)


def build_input_scanners(names: list[str]) -> list[Any]:
    _register_defaults()
    return _build(names, _INPUT_REGISTRY, kind="input")


def build_output_scanners(names: list[str]) -> list[Any]:
    _register_defaults()
    return _build(names, _OUTPUT_REGISTRY, kind="output")


def _build(names: list[str], registry: dict[str, Callable[[], Any]], kind: str) -> list[Any]:
    """Instantiate the named scanners; skip unknown/failed ones (fail-open)."""
    scanners: list[Any] = []
    for name in names:
        builder = registry.get(name)
        if builder is None:
            logger.warning("guard_unknown_scanner", kind=kind, name=name, available=list(registry))
            continue
        try:
            scanners.append(builder())
        except Exception as exc:  # model download / load failure must not break boot
            logger.warning("guard_scanner_build_failed", kind=kind, name=name, error=str(exc))
    return scanners
