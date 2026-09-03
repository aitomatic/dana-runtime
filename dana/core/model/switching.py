"""Atomic provider/runtime rebinding — build-before-mutate.

Per ADR-007: switch validates configured target → builds provider + model client
+ compatible runtime before mutation → rebinds → commits model-change fact.
Failure preserves the old model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dana.core.model.catalog import ModelTarget


@dataclass
class ModelSwitchResult:
    """Result of an atomic model switch.

    ``success``   — whether the switch completed.
    ``target``    — the target that was switched to (or attempted).
    ``error``     — error message if failed.
    """

    success: bool
    target: str  # "provider/model"
    error: str | None = None


class ModelSwitcher:
    """Atomic model switcher — build before mutate, failure preserves old.

    ``apply_switch`` MUST be idempotent and failure-safe: if it raises,
    the old model must still be usable. The switcher does not roll back
    partial mutations — that is the caller's responsibility.

    Usage::

        switcher = ModelSwitcher(
            build_provider=lambda target: ...,
            build_runtime=lambda target, provider: ...,
            apply_switch=lambda target, provider, runtime: ...,
        )
        result = switcher.switch(target)
    """

    def __init__(
        self,
        build_provider: Callable[[ModelTarget], Any],
        build_runtime: Callable[[ModelTarget, Any], Any],
        apply_switch: Callable[[ModelTarget, Any, Any], None],
    ) -> None:
        self._build_provider = build_provider
        self._build_runtime = build_runtime
        self._apply_switch = apply_switch

    def switch(self, target: ModelTarget) -> ModelSwitchResult:
        """Attempt an atomic switch.

        1. Build provider + runtime **before mutation**.
        2. Apply the switch (rebind).
        3. On any failure, leave the old model untouched.

        ``apply_switch`` must be idempotent and failure-safe: if it raises
        after partially mutating state, the old model is lost. The caller
        should ensure ``apply_switch`` is atomic or provides its own rollback.

        Returns:
            ModelSwitchResult with success/error.
        """
        target_str = f"{target.provider}/{target.model}"
        try:
            provider = self._build_provider(target)
            runtime = self._build_runtime(target, provider)
            self._apply_switch(target, provider, runtime)
            return ModelSwitchResult(success=True, target=target_str)
        except Exception as exc:
            return ModelSwitchResult(success=False, target=target_str, error=str(exc))
