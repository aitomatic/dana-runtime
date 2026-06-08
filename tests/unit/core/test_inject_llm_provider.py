from dana.common.llm import LLM
from dana.common.llm.providers import OpenAIProvider
from dana.common.resource.rlm_resource import RLMResource
from dana.core.memory import LTMemory


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
