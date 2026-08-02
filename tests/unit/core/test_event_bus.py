"""S1 EventBus substrate — 13 scenarios from sprint/plans/S1-eventbus-substrate.md.

Each test maps 1:1 to a row in the plan's given/when/then table (T1.1..T1.13).
Async cases use asyncio.run to avoid any pytest-asyncio plugin dependency.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from dana.core.agent.base_star_agent import BaseSTARAgent
from dana.core.ext.event_bus import Event, EventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit(bus: EventBus, event: Event):
    """Run async emit to completion from a sync test."""
    return asyncio.run(bus.emit(event))


def _make_handler(result=None, *, raises=None, record: list | None = None):
    """Build a sync handler returning ``result`` or raising ``raises``.

    Appends a marker to ``record`` when called so tests can assert call order / count.
    """

    def _h(_event: Event):
        if record is not None:
            record.append(result if raises is None else "RAISED")
        if raises is not None:
            raise raises
        return result

    return _h


def _make_async_handler(result=None, *, record: list | None = None):
    """Build an async handler returning ``result``."""

    async def _h(_event: Event):
        if record is not None:
            record.append(result)
        return result

    return _h


# ---------------------------------------------------------------------------
# T1.1 — zero handlers -> None
# ---------------------------------------------------------------------------


def test_t11_zero_handlers_returns_none():
    bus = EventBus()
    assert _emit(bus, Event("x")) is None


# ---------------------------------------------------------------------------
# T1.2 — single handler returns None -> None, payload untouched
# ---------------------------------------------------------------------------


def test_t12_passthrough_returns_none_and_payload_untouched():
    bus = EventBus()
    payload = {"a": 1}
    bus.subscribe("x", _make_handler(result=None))
    event = Event("x", payload)
    assert _emit(bus, event) is None
    assert event.payload == {"a": 1}  # not mutated


# ---------------------------------------------------------------------------
# T1.3 — first-wins: h1 blocks -> h2 never called
# ---------------------------------------------------------------------------


def test_t13_first_wins_blocks_skips_later_handlers():
    bus = EventBus()
    calls: list = []
    block = {"block": True, "reason": "no"}
    bus.subscribe("x", _make_handler(result=block, record=calls))
    bus.subscribe("x", _make_handler(result={"modify": {"a": 2}}, record=calls))
    result = _emit(bus, Event("x"))
    assert result == block
    assert calls == [block]  # h2 never called


# ---------------------------------------------------------------------------
# T1.4 — h1 None, h2 modify -> returns h2's dict
# ---------------------------------------------------------------------------


def test_t14_first_none_second_modify_returns_second():
    bus = EventBus()
    modify = {"modify": {"a": 2}}
    bus.subscribe("x", _make_handler(result=None))
    bus.subscribe("x", _make_handler(result=modify))
    assert _emit(bus, Event("x")) == modify


# ---------------------------------------------------------------------------
# T1.5 — handler raises -> None, no propagation, logged
# ---------------------------------------------------------------------------


def test_t15_handler_raise_isolated_no_propagation(caplog):
    bus = EventBus()
    bus.subscribe("x", _make_handler(raises=RuntimeError("boom")))
    with caplog.at_level(logging.ERROR):
        result = _emit(bus, Event("x"))
    assert result is None  # raise treated as None
    assert any("event handler error" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# T1.6 — raise then block -> block wins (raise isolated, later handler runs)
# ---------------------------------------------------------------------------


def test_t16_raise_isolated_then_block_wins():
    bus = EventBus()
    block = {"block": True, "reason": "denied"}
    bus.subscribe("x", _make_handler(raises=RuntimeError("boom")))
    bus.subscribe("x", _make_handler(result=block))
    assert _emit(bus, Event("x")) == block


# ---------------------------------------------------------------------------
# T1.7 — handlers called in subscribe order
# ---------------------------------------------------------------------------


def test_t17_handlers_called_in_subscribe_order():
    bus = EventBus()
    order: list = []
    bus.subscribe("x", _make_handler(result=None, record=order))
    bus.subscribe("x", _make_handler(result=None, record=order))
    bus.subscribe("x", _make_handler(result=None, record=order))
    _emit(bus, Event("x"))
    assert order == [None, None, None]
    # Stronger: distinct markers confirm order
    bus2 = EventBus()
    seq: list = []
    bus2.subscribe("x", lambda _e: (seq.append("first") or None))
    bus2.subscribe("x", lambda _e: (seq.append("second") or None))
    bus2.subscribe("x", lambda _e: (seq.append("third") or None))
    _emit(bus2, Event("x"))
    assert seq == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# T1.8 — unsubscribe -> handler not called
# ---------------------------------------------------------------------------


def test_t18_unsubscribe_removes_handler():
    bus = EventBus()
    calls: list = []
    unsub = bus.subscribe("x", _make_handler(result={"block": True}, record=calls))
    unsub()
    assert _emit(bus, Event("x")) is None
    assert calls == []
    assert bus.handlers("x") == []


# ---------------------------------------------------------------------------
# T1.9 — sync handler (returns dict, not async) -> auto-detected, returned
# ---------------------------------------------------------------------------


def test_t19_sync_handler_returning_dict_is_used():
    bus = EventBus()
    block = {"block": True, "reason": "sync"}
    bus.subscribe("x", _make_handler(result=block))  # sync handler
    assert _emit(bus, Event("x")) == block


# ---------------------------------------------------------------------------
# T1.10 — async handler -> awaited, result returned
# ---------------------------------------------------------------------------


def test_t110_async_handler_is_awaited():
    bus = EventBus()
    modify = {"modify": {"a": 9}}
    bus.subscribe("x", _make_async_handler(result=modify))
    assert _emit(bus, Event("x")) == modify


# ---------------------------------------------------------------------------
# T1.11 — parity: emit_sync == await emit
# ---------------------------------------------------------------------------


def test_t111_emit_sync_matches_async_emit():
    bus = EventBus()
    block = {"block": True, "reason": "parity"}
    bus.subscribe("x", _make_async_handler(result=block))
    event = Event("x")
    assert bus.emit_sync(event) == _emit(bus, Event("x"))


# ---------------------------------------------------------------------------
# T1.12 — emit_sync called within a running loop -> no RuntimeError, parity
# ---------------------------------------------------------------------------


def test_t112_emit_sync_inside_running_loop_no_runtime_error():
    bus = EventBus()
    block = {"block": True, "reason": "nested"}
    bus.subscribe("x", _make_async_handler(result=block))

    async def _from_inside_loop():
        # emit_sync invoked while a loop is running (sync-called-from-async)
        return bus.emit_sync(Event("x"))

    result = asyncio.run(_from_inside_loop())
    assert result == block


# ---------------------------------------------------------------------------
# T1.13 — agent.event_bus is an EventBus, independent per agent
# ---------------------------------------------------------------------------


class _ConcreteSTAR(BaseSTARAgent):
    """Minimal concrete STAR agent to exercise the event_bus mount point."""

    def _see(self, trace_inputs):  # type: ignore[override]
        return {"trace_percepts": trace_inputs}

    def _think(self, trace_percepts):  # type: ignore[override]
        return {"trace_thoughts": trace_percepts}

    def _act(self, trace_thoughts):  # type: ignore[override]
        return {"trace_outputs": trace_thoughts}

    def _reflect(self, trace_outputs):  # type: ignore[override]
        return {"trace_learning": trace_outputs}

    async def _think_async(self, trace_percepts):  # type: ignore[override]
        return {"trace_thoughts": trace_percepts}

    async def _act_async(self, trace_thoughts):  # type: ignore[override]
        return {"trace_outputs": trace_thoughts}


def test_t113_agent_event_bus_is_per_instance():
    a1 = _ConcreteSTAR(auto_register=False)
    a2 = _ConcreteSTAR(auto_register=False)
    assert isinstance(a1.event_bus, EventBus)
    assert isinstance(a2.event_bus, EventBus)
    assert a1.event_bus is not a2.event_bus  # independent per agent
    # stable on repeated access
    assert a1.event_bus is a1.event_bus


# ---------------------------------------------------------------------------
# subscribe validation (defensive)
# ---------------------------------------------------------------------------


def test_subscribe_rejects_non_callable():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.subscribe("x", "not callable")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Finding F — extra coverage from adversarial review
# ---------------------------------------------------------------------------


def test_duplicate_subscription_runs_handler_twice():
    """Subscribing the same handler twice registers it twice (runs twice)."""
    bus = EventBus()
    calls: list = []
    handler = _make_handler(result={"block": True, "reason": "dup"}, record=calls)
    bus.subscribe("x", handler)
    bus.subscribe("x", handler)  # duplicate
    assert len(bus.handlers("x")) == 2
    _emit(bus, Event("x"))
    # First-wins returns after the first invocation; the second never runs.
    assert calls == [{"block": True, "reason": "dup"}]
    # Confirm: with a pass-through duplicate, both run.
    bus2 = EventBus()
    calls2: list = []
    passthrough = _make_handler(result=None, record=calls2)
    bus2.subscribe("x", passthrough)
    bus2.subscribe("x", passthrough)
    _emit(bus2, Event("x"))
    assert calls2 == [None, None]


def test_non_dict_return_is_warned_and_skipped(caplog):
    """A handler returning a non-dict non-None value violates the contract:
    the bus logs a warning and treats it as None (skips), so a later valid
    handler can still win, and consumers never see a malformed result."""
    bus = EventBus()
    block = {"block": True, "reason": "valid"}
    bus.subscribe("x", _make_handler(result=True))  # non-dict, non-None — bug
    bus.subscribe("x", _make_handler(result=block))
    with caplog.at_level(logging.WARNING):
        result = _emit(bus, Event("x"))
    assert result == block  # malformed skipped, valid handler won
    assert any("non-dict result" in rec.message for rec in caplog.records)


def test_non_dict_return_alone_returns_none(caplog):
    """If the only handler returns a non-dict, emit returns None (not the junk)."""
    bus = EventBus()
    bus.subscribe("x", _make_handler(result="oops"))  # str, not dict
    with caplog.at_level(logging.WARNING):
        assert _emit(bus, Event("x")) is None
