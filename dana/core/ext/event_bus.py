"""EventBus substrate — intercept-capable event bus for dana v2.0 extensibility.

Backbone milestone S1. See sprint/plans/S1-eventbus-substrate.md.

Contract (locked, do not change without asking):
- Aggregation = FIRST-WINS: the first handler returning non-None wins; later
  handlers for that event are skipped.
- Handlers may be sync or async; awaitable return values are auto-awaited.
- A handler that raises is caught, logged, and treated as None (pass-through);
  later handlers still run.
- Handlers MUST return a dict (HandlerResult) or None. A non-dict non-None
  return is a contract violation; the bus logs a warning and skips it (treated
  as None) so downstream consumers never see a malformed result.
- Do NOT mutate ``event.payload``. The payload dict is SHARED across handlers
  and the caller; mutating it corrupts siblings silently. Return a result dict
  (e.g. ``{"modify": ...}``) instead — that is the only supported interception
  path.
- The bus is AGNOSTIC to the *keys* of a handler result dict. ``{"block": ...}``
  / ``{"modify": ...}`` are consumer conventions, not bus semantics — do not
  special-case those keys here.

Thread-safety (Finding A): the bus is NOT thread-safe for concurrent
subscribe/unsubscribe against emit. ``emit_sync`` delegates to
``Misc.safe_asyncio_run``, which — when called from inside a running loop —
runs ``emit`` in a worker thread on a fresh loop. Therefore: only mutate the
handler set at setup time. Do NOT subscribe/unsubscribe from inside a handler
or concurrently with a turn that uses ``emit_sync``; that races on
``self._handlers``. (A lock is intentionally omitted for v0.1 perf; if dynamic
subscription during turns becomes a real need, add a lock then.)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import logging
from typing import Any

from dana.common.utils.misc import Misc


logger = logging.getLogger(__name__)

# Handler return value: opaque to the bus. Consumer conventions (NOT bus logic):
#   {"block": True, "reason": str}  -> intercept/abort (effect defined by consumer)
#   {"modify": dict}                -> patch the payload (interpretation is consumer's)
HandlerResult = dict[str, Any]


@dataclass(frozen=True)
class Event:
    """An immutable event. Do NOT mutate ``payload`` — return a result dict."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)


# A handler receives an Event and returns a HandlerResult, None, or an awaitable
# yielding either. The bus auto-awaits awaitables so sync and async handlers
# are both accepted.
EventHandler = Callable[[Event], Awaitable[HandlerResult | None] | HandlerResult | None]

# Returned by ``subscribe``; calling it removes the handler.
Subscription = Callable[[], None]


class EventBus:
    """Per-agent intercept-capable event bus. See module docstring for the contract."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}

    def subscribe(self, event_type: str, handler: EventHandler) -> Subscription:
        """Register ``handler`` for ``event_type``. Returns an unsubscribe callable."""
        if not callable(handler):
            raise TypeError(f"handler must be callable, got {type(handler).__name__}")
        handlers = self._handlers.setdefault(event_type, [])
        handlers.append(handler)

        def _unsubscribe() -> None:
            try:
                handlers.remove(handler)
            except ValueError:
                pass

        return _unsubscribe

    async def emit(self, event: Event) -> HandlerResult | None:
        """Dispatch ``event`` to handlers in subscribe order.

        FIRST-WINS: stops at the first handler returning a non-None result and
        returns it. A handler that raises is logged and skipped (pass-through).
        Returns None when no handler produces a result.
        """
        for handler in list(self._handlers.get(event.type, [])):
            try:
                raw = handler(event)
                if inspect.isawaitable(raw):
                    raw = await raw
            except Exception:
                logger.exception("event handler error: type=%s", event.type)
                continue
            if raw is not None:
                if not isinstance(raw, dict):
                    logger.warning(
                        "event handler returned non-dict result (type=%s, got=%s); skipping",
                        event.type,
                        type(raw).__name__,
                    )
                    continue
                return raw
        return None

    def emit_sync(self, event: Event) -> HandlerResult | None:
        """Sync entry point. Delegates to ``Misc.safe_asyncio_run`` which handles
        both the no-running-loop case (``asyncio.run``) and the running-loop case
        (runs the coroutine in a worker thread — see ``misc._run_in_existing_loop``).

        T1.verify conclusion: ``safe_asyncio_run`` DOES handle nested/running
        loops, so the notification-only fallback is NOT needed.

        Thread-safety: when a loop is running, emit executes in a WORKER THREAD.
        Do NOT subscribe/unsubscribe concurrently with this call (see module
        docstring, Finding A). Use the handler set frozen at setup time.
        """
        result: HandlerResult | None = Misc.safe_asyncio_run(self.emit, event)
        return result

    def handlers(self, event_type: str) -> list[EventHandler]:
        """Snapshot copy of handlers for ``event_type`` (for tests/debug)."""
        return list(self._handlers.get(event_type, []))
