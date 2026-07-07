import pytest

from dana.common.llm import LLM
from dana.common.llm.providers import OpenAIProvider
from dana.common.resource.rlm_resource import RLMResource
from dana.core.agent.star_agent import STARAgent
from dana.core.memory import LTMemory


@pytest.fixture(autouse=True)
def _dummy_provider_env_keys(monkeypatch):
    """Tests here verify provider wiring/identity, not real API calls.
    String-path construction (llm_provider='openai'/'anthropic') reads env
    keys at provider build time; supply dummies so it succeeds offline."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def test_rlm_resource_uses_injected_llm(tmp_path):
    prov = OpenAIProvider(api_key="test-key", model="gpt-4")
    llm = LLM(provider=prov)

    rlm = RLMResource(file=str(tmp_path / "ctx.md"), llm=llm)

    assert rlm._llm is llm
    assert rlm._llm.provider is prov


def test_rlm_resource_set_llm_repoints(tmp_path):
    rlm = RLMResource(file=str(tmp_path / "ctx.md"), llm_provider="openai", llm_model="gpt-4o")
    prov2 = OpenAIProvider(api_key="k2", model="gpt-4o")
    llm2 = LLM(provider=prov2)

    rlm.set_llm(llm2)

    assert rlm._llm is llm2


def test_ltmemory_uses_injected_llm(tmp_path):
    prov = OpenAIProvider(api_key="test-key", model="gpt-4")
    llm = LLM(provider=prov)

    mem = LTMemory(path=str(tmp_path / "mem"), llm=llm)

    assert mem._rlm._llm is llm


def test_ltmemory_set_llm_repoints(tmp_path):
    mem = LTMemory(path=str(tmp_path / "mem"), llm_provider="openai", llm_model="gpt-4o")
    prov2 = OpenAIProvider(api_key="k2", model="gpt-4o")
    llm2 = LLM(provider=prov2)

    mem.set_llm(llm2)

    assert mem._rlm._llm is llm2


AGENT_KW = dict(
    agent_type="inject-test",
    auto_register=False,
    enable_web_search=False,
    enable_skills=False,
    enable_code_execution=False,
    enable_assistant=False,
)


def test_injected_provider_reaches_agent_and_call_site():
    prov = OpenAIProvider(api_key="test-key", model="gpt-4")

    agent = STARAgent(llm_provider_instance=prov, **AGENT_KW)

    # Sink 1: agent client wraps the exact provider instance
    assert agent.llm_client.provider is prov
    # Sink 2: the actual call site (LLMCaller) holds the SAME LLM — no split-brain
    assert agent._runtime._llm_caller._llm is agent.llm_client


def test_set_llm_provider_repoints_all_sinks(tmp_path):
    prov1 = OpenAIProvider(api_key="k1", model="gpt-4")
    agent = STARAgent(llm_provider_instance=prov1, ltmemory_path=str(tmp_path / "ltm"), **AGENT_KW)

    prov2 = OpenAIProvider(api_key="k2", model="gpt-4o")
    agent.set_llm_provider(llm_provider_instance=prov2)

    assert agent.llm_client.provider is prov2
    assert agent._runtime._llm_caller._llm.provider is prov2
    assert agent._ltmemory._rlm._llm.provider is prov2


def test_injected_provider_drives_ltmemory(tmp_path):
    prov = OpenAIProvider(api_key="test-key", model="gpt-4")

    agent = STARAgent(
        llm_provider_instance=prov,
        ltmemory_path=str(tmp_path / "ltm"),
        **AGENT_KW,
    )

    assert agent._ltmemory is not None
    assert agent._ltmemory._rlm._llm.provider is prov


def test_legacy_string_path_stays_lazy():
    # No instance: _llm_client must remain None until the property is touched
    # (preserves construction without API keys).
    agent = STARAgent(llm_provider="openai", model="gpt-4", **AGENT_KW)

    assert agent._llm_client is None
    assert agent._llm_config == {"provider": "openai", "model": "gpt-4"}


def test_set_llm_provider_string_path_updates_all_sinks(tmp_path):
    agent = STARAgent(
        llm_provider="anthropic",
        model="claude-sonnet-4-20250514",
        ltmemory_path=str(tmp_path / "ltm"),
        **AGENT_KW,
    )
    # Legacy path: lazy until touched
    assert agent._llm_client is None

    agent.set_llm_provider(llm_provider="openai", model="gpt-4o")

    assert agent._llm_config == {"provider": "openai", "model": "gpt-4o"}
    assert agent._llm_client is not None  # eagerly built by _apply_llm_provider
    assert agent._runtime._llm_caller._llm is agent._llm_client
    assert agent._ltmemory._rlm._llm is agent._llm_client
