"""
Compact trigger resolution — single env knob `DANA_COMPACT_TRIGGER_TOKENS`.

Resolves the compression trigger threshold from environment, clamping to a
sane range and falling back to a safe default on parse/validation errors.
The resolved value is cached per-process.
"""

from __future__ import annotations

import os
import threading

from structlog import get_logger


logger = get_logger()

DEFAULT_TRIGGER = 150_000
MIN_TRIGGER = 8_000
MAX_TRIGGER = 2_000_000

_ENV_VAR = "DANA_COMPACT_TRIGGER_TOKENS"

_cached_trigger: int | None = None
_cache_lock = threading.Lock()
_startup_logged = False


def _reset_cache_for_tests() -> None:
    """Clear cached resolution. For tests only — not called from production."""
    global _cached_trigger, _startup_logged
    with _cache_lock:
        _cached_trigger = None
        _startup_logged = False


def resolve_trigger_tokens() -> int:
    """Resolve the compression trigger threshold.

    Reads `DANA_COMPACT_TRIGGER_TOKENS`, parses as int, clamps to
    `[MIN_TRIGGER, MAX_TRIGGER]`. Invalid / missing / out-of-range values
    fall back to `DEFAULT_TRIGGER` with a WARNING log.

    Result is cached once per process.
    """
    global _cached_trigger, _startup_logged

    with _cache_lock:
        if _cached_trigger is not None:
            return _cached_trigger

        raw = os.getenv(_ENV_VAR)
        trigger = DEFAULT_TRIGGER

        if raw is not None and raw != "":
            try:
                parsed = int(raw)
                if parsed < MIN_TRIGGER or parsed > MAX_TRIGGER:
                    logger.warning(
                        "compact_trigger_out_of_range",
                        value=raw,
                        min=MIN_TRIGGER,
                        max=MAX_TRIGGER,
                        fallback=DEFAULT_TRIGGER,
                    )
                else:
                    trigger = parsed
            except (ValueError, TypeError):
                logger.warning(
                    "compact_trigger_invalid",
                    value=raw,
                    fallback=DEFAULT_TRIGGER,
                )

        _cached_trigger = trigger
        if not _startup_logged:
            logger.info("compression_threshold_resolved", trigger=trigger)
            _startup_logged = True

        return trigger
