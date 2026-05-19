"""Unit tests for TaskResource factory-based sub-agent dispatch.

Covers the factory-pattern contract: every ``task()`` spawn constructs a
fresh, disjoint agent so concurrent and sequential dispatches never share
mutable state. Legacy instance registration remains supported but warns.
"""

from __future__ import annotations

import asyncio
import functools

import pytest

from dana.core.agent.base_agent import BaseAgent
from dana.core.resource.task_resource import TaskResource


class FakeSubAgent(BaseAgent):
    """Minimal agent stand-in: records the prompts it is asked to handle."""

    TASK_TOOL_DESCRIPTION = "Fake sub-agent used for TaskResource unit tests."

    def __init__(self, agent_id: str = "fake-sub", **kwargs):
        kwargs.setdefault("auto_register", False)
        super().__init__(agent_type="fake_sub", agent_id=agent_id, **kwargs)
        self.seen_prompts: list = []

    async def aquery(self, message: str | None = None, session_id: str | None = None, **kwargs) -> dict:
        # Yield control so concurrent dispatches interleave — a shared instance
        # would cross-contaminate seen_prompts here.
        await asyncio.sleep(0.01)
        self.seen_prompts.append(message)
        await asyncio.sleep(0.01)
        return {"response": f"handled: {message}", "session_id": session_id}


def _factory() -> functools.partial:
    return functools.partial(FakeSubAgent, agent_id="fake-sub")


class TestFactoryRegistration:
    """register_agent factory vs legacy-instance handling."""

    def test_task_description_from_factory_func(self):
        resource = TaskResource(resource_id="task")
        resource.register_agent("fake", _factory())
        assert FakeSubAgent.TASK_TOOL_DESCRIPTION in (resource.task.__func__.__doc__ or "")
        assert "- fake:" in (resource.task.__func__.__doc__ or "")

    def test_legacy_instance_registration_warns(self):
        resource = TaskResource(resource_id="task")
        instance = FakeSubAgent(agent_id="legacy")
        with pytest.warns(DeprecationWarning):
            resource.register_agent("legacy", instance)
        assert FakeSubAgent.TASK_TOOL_DESCRIPTION in (resource.task.__func__.__doc__ or "")

    def test_bare_lambda_factory_rejected(self):
        resource = TaskResource(resource_id="task")
        with pytest.raises(TypeError, match=r"\.func"):
            resource.register_agent("bad", lambda: FakeSubAgent())

    def test_unregister_drops_description(self):
        resource = TaskResource(resource_id="task")
        resource.register_agent("fake", _factory())
        assert resource.unregister_agent("fake") is True
        assert "fake" not in resource._descriptions
        assert resource.unregister_agent("missing") is False


class TestFactoryDispatch:
    """task() spawns fresh, isolated agents per call."""

    @pytest.mark.asyncio
    async def test_task_spawns_fresh_instance_per_call(self):
        resource = TaskResource(resource_id="task", agents={"fake": _factory()})

        out1 = await resource.task(description="d", prompt="first", subagent_type="fake")
        out2 = await resource.task(description="d", prompt="second", subagent_type="fake")

        sid1 = out1.split("session_id: ")[1].rstrip("]")
        sid2 = out2.split("session_id: ")[1].rstrip("]")
        agent1 = resource._sessions[sid1]["agent"]
        agent2 = resource._sessions[sid2]["agent"]

        assert agent1 is not agent2
        assert agent1.seen_prompts == ["first"]
        assert agent2.seen_prompts == ["second"]

    @pytest.mark.asyncio
    async def test_concurrent_same_type_tasks_isolated(self):
        resource = TaskResource(resource_id="task", agents={"fake": _factory()})

        out1, out2 = await asyncio.gather(
            resource.task(description="d", prompt="alpha", subagent_type="fake"),
            resource.task(description="d", prompt="beta", subagent_type="fake"),
        )

        sid1 = out1.split("session_id: ")[1].rstrip("]")
        sid2 = out2.split("session_id: ")[1].rstrip("]")
        agent1 = resource._sessions[sid1]["agent"]
        agent2 = resource._sessions[sid2]["agent"]

        # Each spawned agent saw exactly one prompt — no cross-contamination.
        assert agent1 is not agent2
        assert agent1.seen_prompts == ["alpha"]
        assert agent2.seen_prompts == ["beta"]

    @pytest.mark.asyncio
    async def test_resume_reuses_live_session_instance(self):
        resource = TaskResource(resource_id="task", agents={"fake": _factory()})

        out1 = await resource.task(description="d", prompt="first", subagent_type="fake")
        sid = out1.split("session_id: ")[1].rstrip("]")
        agent1 = resource._sessions[sid]["agent"]

        await resource.task(description="d", prompt="follow-up", subagent_type="fake", resume=sid)
        agent2 = resource._sessions[sid]["agent"]

        assert agent1 is agent2
        assert agent2.seen_prompts == ["first", "follow-up"]

    @pytest.mark.asyncio
    async def test_unknown_agent_type_returns_error(self):
        resource = TaskResource(resource_id="task", agents={"fake": _factory()})
        out = await resource.task(description="d", prompt="x", subagent_type="nope")
        assert "Unknown agent type" in out

    @pytest.mark.asyncio
    async def test_failed_aquery_marks_session_failed(self):
        class FailingSubAgent(FakeSubAgent):
            async def aquery(self, message=None, session_id=None, **kwargs):
                raise RuntimeError("boom")

        resource = TaskResource(
            resource_id="task",
            agents={"fail": functools.partial(FailingSubAgent, agent_id="failing")},
        )

        with pytest.raises(RuntimeError, match="boom"):
            await resource.task(description="d", prompt="x", subagent_type="fail")

        # Session must not be stuck "running" — task_output reports the failure.
        sid = next(iter(resource._sessions))
        assert resource._sessions[sid]["status"] == "failed"
        out = await resource.task_output(task_id=sid)
        assert out.startswith("Status: failed")
        assert "boom" in out
