"""S2 STAR loop wiring — scenarios from sprint/plans/S2-star-loop-wiring.md.

Each test maps to a row in the plan's given/when/then table (T2.1..T2.8).
Uses a minimal concrete BaseSTARAgent subclass (no LLM/runtime) so the
orchestrator's phase-event emit is exercised directly. Async cases use
asyncio.run (no pytest-asyncio dependency).
"""

from __future__ import annotations

import asyncio

from dana.core.agent.base_star_agent import BaseSTARAgent
from dana.core.ext.event_bus import Event
from dana.core.ext.events import ACT_END, REFLECT_END, SEE_END, THINK_END


# ---------------------------------------------------------------------------
# Minimal concrete STAR agent for testing the orchestrator wiring
# ---------------------------------------------------------------------------


class _MiniSTAR(BaseSTARAgent):
    """Controllable STAR agent: records phase calls, exits after one iteration."""

    def __init__(self) -> None:
        super().__init__(agent_type="mini", auto_register=False)
        self.phase_calls: list[str] = []

    def _see(self, trace_inputs):
        self.phase_calls.append("see")
        return {"trace_percepts": dict(trace_inputs)}

    def _think(self, trace_percepts):
        self.phase_calls.append("think")
        return {"trace_thoughts": dict(trace_percepts)}

    def _act(self, trace_thoughts):
        self.phase_calls.append("act")
        # signal loop exit so query() terminates after one iteration
        return {"trace_outputs": self._mark_star_loop_exit(dict(trace_thoughts))}

    def _reflect(self, trace_outputs):
        self.phase_calls.append("reflect")
        return {"trace_learning": dict(trace_outputs)}

    async def _think_async(self, trace_percepts):
        self.phase_calls.append("think")
        return {"trace_thoughts": dict(trace_percepts)}

    async def _act_async(self, trace_thoughts):
        self.phase_calls.append("act")
        return {"trace_outputs": self._mark_star_loop_exit(dict(trace_thoughts))}


def _log_handler(log: list[str], event_name: str):
    def _h(_event: Event):
        log.append(event_name)
        return None

    return _h


# ---------------------------------------------------------------------------
# T2.1 — handler observes see/think/act events in order (sync)
# ---------------------------------------------------------------------------


def test_t21_phase_events_fire_in_order_sync():
    agent = _MiniSTAR()
    log: list[str] = []
    for name, evt in [("see", SEE_END), ("think", THINK_END), ("act", ACT_END)]:
        agent.event_bus.subscribe(evt, _log_handler(log, name))

    agent.query(message="hi")

    assert log == ["see", "think", "act"]
    assert agent.phase_calls == ["see", "think", "act"]


# ---------------------------------------------------------------------------
# T2.2 — see_end modify is seen by think
# ---------------------------------------------------------------------------


def test_t22_modify_see_end_reaches_think():
    agent = _MiniSTAR()
    seen_by_think: list = []

    def modify_see(_e: Event):
        return {"modify": {"trace_percepts": {"hijacked": True}}}

    def capture_think(event: Event):
        seen_by_think.append(event.payload["result"])
        return None

    agent.event_bus.subscribe(SEE_END, modify_see)
    agent.event_bus.subscribe(THINK_END, capture_think)

    agent.query(message="hi")

    # _think received the hijacked percepts and wrapped them as trace_thoughts
    assert seen_by_think == [{"trace_thoughts": {"hijacked": True}}]


# ---------------------------------------------------------------------------
# T2.3 — think_end block exits BEFORE act
# ---------------------------------------------------------------------------


def test_t23_think_end_block_exits_before_act():
    agent = _MiniSTAR()
    agent.event_bus.subscribe(THINK_END, lambda _e: {"block": True, "reason": "stop"})

    result = agent.query(message="hi")

    assert "act" not in agent.phase_calls  # act never ran
    assert agent.phase_calls == ["see", "think"]
    # query returns cleanly (no crash); result is the blocked trace
    assert result is not None


# ---------------------------------------------------------------------------
# T2.4 — act_end block exits cleanly after act, loop terminates
# ---------------------------------------------------------------------------


def test_t24_act_end_block_exits_cleanly():
    agent = _MiniSTAR()
    agent.event_bus.subscribe(ACT_END, lambda _e: {"block": True, "reason": "done"})

    result = agent.query(message="hi")

    assert agent.phase_calls == ["see", "think", "act"]  # act ran, then blocked
    assert result is not None  # clean return, no crash, loop did not continue


# ---------------------------------------------------------------------------
# T2.5 — handler raise is caught (bus S1); phase continues with original payload
# ---------------------------------------------------------------------------


def test_t25_handler_raise_does_not_crash_loop():
    agent = _MiniSTAR()

    def boom(_e: Event):
        raise RuntimeError("boom")

    agent.event_bus.subscribe(SEE_END, boom)

    result = agent.query(message="hi")

    # loop completed all phases despite the raising handler
    assert agent.phase_calls == ["see", "think", "act"]
    assert result is not None


# ---------------------------------------------------------------------------
# T2.6 — async: phase events fire (parity with sync)
# ---------------------------------------------------------------------------


def test_t26_phase_events_fire_async():
    agent = _MiniSTAR()
    log: list[str] = []
    for name, evt in [("see", SEE_END), ("think", THINK_END), ("act", ACT_END)]:
        agent.event_bus.subscribe(evt, _log_handler(log, name))

    asyncio.run(agent.aquery(message="hi"))

    assert log == ["see", "think", "act"]


# ---------------------------------------------------------------------------
# T2.7 — parity: think_end block honored on both sync and async paths
# ---------------------------------------------------------------------------


def test_t27_think_block_parity_sync_async():
    def fresh():
        a = _MiniSTAR()
        a.event_bus.subscribe(THINK_END, lambda _e: {"block": True, "reason": "stop"})
        return a

    sync_agent = fresh()
    async_agent = fresh()

    sync_agent.query(message="hi")
    asyncio.run(async_agent.aquery(message="hi"))

    for agent in (sync_agent, async_agent):
        assert agent.phase_calls == ["see", "think"]  # act never ran on either path


# ---------------------------------------------------------------------------
# T2.8 — reflect_end is emitted (best-effort; reflect runs in background)
# ---------------------------------------------------------------------------


def test_t28_reflect_end_emitted():
    """reflect_end fires from the background reflect wrapper. Since reflect is
    fire-and-forget (daemon thread), we assert via a direct call to the wrapper
    path: subscribe a reflect_end handler and run query, then poll briefly."""
    agent = _MiniSTAR()
    # Disable act-exit so reflect actually triggers: override _act to NOT exit,
    # but then the loop would continue. Instead, keep exit but capture reflect
    # via the event. Reflect only runs when act did NOT mark exit — so use an
    # agent whose act does not exit (cap iterations by overriding MAX_ITERATIONS).
    agent.MAX_ITERATIONS = 1

    def _act_no_exit(self, trace_thoughts):  # type: ignore[override]
        self.phase_calls.append("act")
        return {"trace_outputs": dict(trace_thoughts)}  # no exit flag

    agent._act = _act_no_exit.__get__(agent)  # type: ignore[method-assign]

    seen: list[str] = []
    agent.event_bus.subscribe(REFLECT_END, _log_handler(seen, "reflect"))

    agent.query(message="hi")

    # reflect runs in a daemon thread; give it a moment
    for _ in range(50):
        if seen:
            break
        asyncio.run(asyncio.sleep(0.02))
    assert seen == ["reflect"]


# ---------------------------------------------------------------------------
# Adversarial fix-regression tests (Stage 3)
# ---------------------------------------------------------------------------


def test_fix_empty_modify_does_not_false_exit():
    """[Adversarial] A see_end handler that modifies the result to ``{}`` must
    NOT falsely exit the loop. The old per-phase ``_do_exit_star_loop({})`` check
    returned True for empty dicts (quirk); the precise ``EXIT_FLAG is True``
    check does not. Act must still run."""
    agent = _MiniSTAR()
    agent.event_bus.subscribe(SEE_END, lambda _e: {"modify": {}})

    agent.query(message="hi")

    assert agent.phase_calls == ["see", "think", "act"]  # no false exit before act


def test_fix_act_end_block_stops_loop_without_second_iteration():
    """[Adversarial] An act_end block must set phase_blocked so the loop does
    not iterate again (and reflect is skipped). Uses an agent whose _act does
    NOT self-exit so the block — not _act — drives the exit."""
    agent = _MiniSTAR()
    agent.MAX_ITERATIONS = 3

    def _act_no_exit(self, trace_thoughts):  # type: ignore[override]
        self.phase_calls.append("act")
        return {"trace_outputs": dict(trace_thoughts)}  # no EXIT flag

    agent._act = _act_no_exit.__get__(agent)  # type: ignore[method-assign]
    agent.event_bus.subscribe(ACT_END, lambda _e: {"block": True, "reason": "stop"})

    agent.query(message="hi")

    assert agent.phase_calls == ["see", "think", "act"]  # exactly one iteration, no repeat
