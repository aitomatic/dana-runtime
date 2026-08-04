"""Model Catalog — configured provider/model targets only.

Per ADR-007: the Model Catalog exposes only configured provider/model
combinations — no arbitrary IDs, no automatic routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelTarget:
    """A configured provider/model combination.

    ``provider`` — the LLM provider name (e.g. "anthropic", "openai").
    ``model``    — the model identifier (e.g. "claude-sonnet-4", "gpt-4o").
    ``config``   — optional extra configuration (API keys, endpoints, etc.).
    """

    provider: str
    model: str
    config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.provider or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not self.model or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if "/" in self.provider or "/" in self.model:
            raise ValueError("provider and model must not contain '/'")


class ModelCatalog:
    """Catalog of configured model targets.

    Only targets added during construction are visible. No arbitrary model IDs.
    Duplicate provider+model combinations fail construction.
    """

    def __init__(self, targets: list[ModelTarget]) -> None:
        self._targets = list(targets)
        self._by_key: dict[tuple[str, str], ModelTarget] = {}

        for target in targets:
            key = (target.provider, target.model)
            if key in self._by_key:
                raise ValueError(f"Duplicate model target: {target.provider}/{target.model}")
            self._by_key[key] = target

    @property
    def targets(self) -> list[ModelTarget]:
        return list(self._targets)

    def get(self, provider: str, model: str) -> ModelTarget | None:
        """Look up a target by provider and model name."""
        return self._by_key.get((provider, model))
