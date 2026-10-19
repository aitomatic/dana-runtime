"""Ownership regressions use real SDK pools without making provider requests."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dana.common.llm import LLM
from dana.common.llm.providers import OpenAIProvider
from dana.core.agent.assistant_agent import AssistantAgent
from dana.core.agent.star_agent import STARAgent


_AGENT_KW: dict[str, Any] = dict(
    agent_type="lifecycle-test",
    auto_register=False,
    enable_skills=False,
    enable_web_search=False,
    enable_code_execution=False,
    enable_assistant=False,
    compress_timeline=False,
)


@pytest.fixture
def resources(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agents = []
    providers = []

    def make_agent(**overrides):
        agent = STARAgent(**(_AGENT_KW | overrides))
        agents.append(agent)
        return agent

    def make_provider():
        provider = OpenAIProvider(api_key="test-key", model="gpt-4", base_url="http://127.0.0.1:9/v1")
        providers.append(provider)
        return provider

    yield make_agent, make_provider
    for agent in reversed(agents):
        agent.close()
    for provider in providers:
        asyncio.run(provider.aclose())


def test_replacing_owned_client_closes_only_retired_pools(resources):
    make_agent, make_provider = resources
    borrowed = make_provider()
    agent = make_agent(llm_provider_instance=borrowed)
    agent.set_llm_provider(llm_provider="openai", model="gpt-4")
    first = agent._llm_client
    agent.set_llm_provider(llm_provider="openai", model="gpt-4")
    second = agent._llm_client
    assert first.provider.client.is_closed()
    assert not second.provider.client.is_closed()
    assert not borrowed.client.is_closed()

    agent.llm_client = second
    assert not second.provider.client.is_closed()
    adopted = LLM(provider=make_provider())
    agent.llm_client = adopted
    assert second.provider.client.is_closed()
    assert not adopted.provider.client.is_closed()
    agent.close()
    agent.close()
    assert adopted.provider.client.is_closed()
    assert not borrowed.client.is_closed()


@pytest.mark.asyncio
async def test_named_client_after_borrow_is_closed_without_closing_borrowed_pool(resources):
    make_agent, make_provider = resources
    borrowed = make_provider()
    agent = make_agent(llm_provider_instance=borrowed)
    agent.set_llm_provider(llm_provider="openai", model="gpt-4")
    owned = agent._llm_client.provider
    executor = agent._runtime._tool_executor
    await agent.aclose()
    await agent.aclose()
    assert owned.client.is_closed()
    assert not borrowed.client.is_closed()
    assert executor.closed


@pytest.mark.parametrize("teardown", ["sync", "async"])
def test_borrowed_child_remains_usable_after_parent_teardown(resources, teardown):
    make_agent, _ = resources
    parent = make_agent(enable_assistant=True)
    internal = parent._owned_internal_agents[0]
    internal.set_llm_provider(llm_provider="openai", model="gpt-4")
    internal_provider = internal._llm_client.provider
    child = make_agent(llm_provider="openai", model="gpt-4")
    child_provider = child.llm_client.provider
    parent.with_agents(child)
    if teardown == "async":
        asyncio.run(parent.aclose())
    else:
        parent.close()
    assert internal_provider.client.is_closed()
    assert not child_provider.client.is_closed()
    assert not child._runtime._tool_executor.closed
    child.close()
    assert child_provider.client.is_closed()


@pytest.mark.parametrize("borrowed", [False, True])
def test_memory_store_pool_ownership_survives_teardown(resources, tmp_path, borrowed):
    make_agent, make_provider = resources
    provider = make_provider() if borrowed else None
    agent = make_agent(
        ltmemory_path=str(tmp_path / "memory"),
        llm_provider="openai",
        model="gpt-4",
        llm_provider_instance=provider,
    )
    memory_provider = agent._ltmemory._rlm._llm.provider
    asyncio.run(agent.aclose())
    assert memory_provider.client.is_closed() == (not borrowed)


def test_repoint_releases_memory_store_pool_without_closing_replacement(resources, tmp_path):
    make_agent, _ = resources
    agent = make_agent(ltmemory_path=str(tmp_path / "memory"), llm_provider="openai", model="gpt-4")
    retired = agent._ltmemory._rlm._llm.provider
    agent.set_llm_provider(llm_provider="openai", model="gpt-4")
    current = agent._llm_client.provider
    assert retired.client.is_closed()
    assert not current.client.is_closed()
    agent.close()
    assert current.client.is_closed()


@pytest.mark.asyncio
async def test_executor_cleanup_failure_does_not_leak_owned_provider(resources, monkeypatch):
    make_agent, _ = resources
    agent = make_agent(llm_provider="openai", model="gpt-4")
    provider = agent.llm_client.provider
    executor = agent._runtime._tool_executor
    original_close = executor.aclose

    async def failed_close():
        await original_close()
        raise RuntimeError("executor cleanup failed")

    monkeypatch.setattr(executor, "aclose", failed_close)
    await agent.aclose()
    assert executor.closed
    assert provider.client.is_closed()


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_per_query_assistant_releases_real_resources_on_success_and_failure(resources, monkeypatch, fails, mode):
    _, make_provider = resources
    pools = []
    executors = []

    def inference(self, **kwargs):
        provider = make_provider()
        self.llm_client = LLM(provider=provider)
        pools.append(provider.client)
        executors.append(self._runtime._tool_executor)
        if fails:
            raise RuntimeError("inference failed")
        return "answer"

    async def async_inference(self, **kwargs):
        return inference(self, **kwargs)

    monkeypatch.setattr(AssistantAgent, "_do_query", inference)
    monkeypatch.setattr(AssistantAgent, "_do_aquery", async_inference)
    assistant = AssistantAgent(resources=[], **_AGENT_KW)
    try:

        def query():
            if mode == "async":
                return asyncio.run(assistant.aquery(message="hello"))
            return assistant.query(message="hello")

        if fails:
            with pytest.raises(RuntimeError, match="inference failed"):
                query()
        else:
            assert query() == "answer"
        assert pools[0].is_closed()
        assert executors[0].closed
        assert not assistant._runtime._tool_executor.closed
    finally:
        assistant.close()
