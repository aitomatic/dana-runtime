"""Public membership/provider changes must reach both model and tool execution."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
from typing import Any
from unittest.mock import Mock

import httpx
from openai import AsyncOpenAI
import pytest
import pytest_asyncio

from dana.common.llm.providers import OpenAIProvider
from dana.common.protocols.war import named_tool, tool_use
from dana.common.utils.misc import Misc
from dana.core.agent.star_agent import STARAgent
from dana.core.ext.events import TOOL_CALL, TOOL_RESULT
from dana.core.knowledge.prompts.codecs import CSXMLCodec, NativeToolsCodec
from dana.core.resource.base_resource import BaseResource
from dana.core.resource.file_io_resource import FileIOResource
from dana.core.tool.tool_executor import ToolExecutor
from dana.core.workflow.base_workflow import BaseWorkflow


_AGENT_KW: dict[str, Any] = dict(
    auto_register=False,
    enable_skills=False,
    enable_web_search=False,
    enable_code_execution=False,
    enable_assistant=False,
    compress_timeline=False,
)


class ReadingWorkflow(BaseWorkflow):
    @named_tool(name="ReadWorkflow")
    def execute(self, file_path: str) -> str:
        """Read the requested local file."""
        return Path(file_path).read_text()


class ReadingAgent(STARAgent):
    def query(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Read the requested local file."""
        return {"response": Path(message).read_text()}

    async def aquery(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self.query(message, **kwargs)


class TextOnlyOpenAI(OpenAIProvider):
    @property
    def supports_native_tools(self) -> bool:
        return False


class TouchyResource(BaseResource):
    """Public property that constructs a client if discovery ever touches it."""

    constructions = 0

    def __init__(self, resource_id: str = "touchy", **kwargs: Any) -> None:
        super().__init__(resource_id=resource_id, auto_register=False, **kwargs)

    @property
    def llm_client(self) -> Any:
        type(self).constructions += 1
        raise RuntimeError("public property must not be evaluated during discovery")

    @tool_use
    def probe(self, value: str = "x") -> str:
        """Return the probe value.

        Args:
            value: value to echo.
        """
        return value


@pytest_asyncio.fixture
async def clients(monkeypatch):
    monkeypatch.setenv("DANA_MOCK_LLM", "false")
    bodies = []
    providers = []
    agents = []

    def respond(request):
        bodies.append(json.loads(request.content))
        if request.url.path.endswith("/responses"):
            payload = {
                "id": "resp_metadata",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-4o",
                "output": [
                    {
                        "id": "msg_metadata",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "observed",
                                "annotations": [],
                            }
                        ],
                    }
                ],
            }
        else:
            payload = {
                "id": "chat_metadata",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "observed"},
                    }
                ],
            }
        return httpx.Response(200, json=payload)

    async def make_provider(cls=OpenAIProvider):
        provider = cls(api_key="synthetic", model="gpt-4o")
        await provider.client.close()
        provider.client = AsyncOpenAI(
            api_key="synthetic",
            base_url="https://offline.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        providers.append(provider)
        return provider

    def make_agent(provider, cls=STARAgent, *, codec=None, runtime=None, agent_id=None):
        kwargs: dict[str, Any] = dict(
            agent_type="metadata-regression",
            llm_provider_instance=provider,
            **_AGENT_KW,
        )
        if codec is not None:
            kwargs["codec"] = codec
        if runtime is not None:
            kwargs["runtime"] = runtime
        if agent_id is not None:
            kwargs["agent_id"] = agent_id
        agent = cls(**kwargs)
        agents.append(agent)
        return agent

    try:
        yield make_provider, make_agent, bodies
    finally:
        for agent in reversed(agents):
            await agent.aclose()
        for provider in providers:
            await provider.aclose()


async def model_tool_names(agent, bodies):
    messages = agent._runtime.build_prompt(agent, agent._timeline)
    await agent._runtime.call_llm_async(messages)
    return {tool.get("function", tool)["name"] for tool in bodies[-1].get("tools", [])}


def system_text(body):
    for message in body.get("messages", []):
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, list):
                return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            return content or ""
    return ""


async def outbound(agent, bodies):
    """Build the prompt, send it, and return the real observed request body."""
    messages = agent._runtime.build_prompt(agent, agent._timeline)
    await agent._runtime.call_llm_async(messages)
    return bodies[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["resource", "workflow", "agent"])
async def test_public_membership_changes_reach_model_and_dispatch(clients, tmp_path, kind):
    make_provider, make_agent, bodies = clients
    provider = await make_provider()
    agent = make_agent(provider)
    source = tmp_path / "source.txt"
    source.write_text("physical metadata lifecycle read")
    if kind == "resource":
        member = FileIOResource(resource_id="metadata-files", base_path=str(tmp_path))
        name = "Read"
    elif kind == "workflow":
        member = ReadingWorkflow(workflow_id="metadata-workflow", auto_register=False)
        name = "ReadWorkflow"
    else:
        member = make_agent(provider, ReadingAgent)
        name = "metadata_regression__query"
    assert name not in await model_tool_names(agent, bodies)

    getattr(agent, f"add_{kind}")(member)
    assert name in await model_tool_names(agent, bodies)
    arguments = {"message" if kind == "agent" else "file_path": str(source)}
    call = {"function": name, "arguments": arguments, "tool_call_id": "read-member"}
    result = await agent._runtime._tool_executor.execute_tools_async(agent, [call])
    assert result[0]["success"], result
    assert "physical metadata lifecycle read" in str(result[0]["result"])

    assert getattr(agent, f"remove_{kind}")(member.object_id)
    assert name not in await model_tool_names(agent, bodies)
    result = await agent._runtime._tool_executor.execute_tools_async(agent, [call])
    assert not result[0]["success"], result


@pytest.mark.asyncio
async def test_provider_capability_switch_changes_outbound_tool_contract(clients, tmp_path):
    make_provider, make_agent, bodies = clients
    native = await make_provider()
    text_only = await make_provider(TextOnlyOpenAI)
    agent = make_agent(native)
    agent.with_resources(FileIOResource(resource_id="metadata-files", base_path=str(tmp_path)))
    assert "Read" in await model_tool_names(agent, bodies)

    agent.set_llm_provider(llm_provider_instance=text_only)
    assert "Read" not in await model_tool_names(agent, bodies)

    agent.set_llm_provider(llm_provider_instance=native)
    assert "Read" in await model_tool_names(agent, bodies)


@pytest.mark.asyncio
async def test_shared_runtime_reads_each_agents_own_resource(clients, tmp_path):
    """Two agents sharing one runtime must never dispatch the other's resource.

    A shared runtime is a supported injection mode, and the ``@named_tool``
    registry holds the *bound* resource objects of whichever agent built it.
    Same relative filename, different physical directory: each agent's read must
    return its own file.
    """
    make_provider, make_agent, bodies = clients
    provider = await make_provider()
    first = make_agent(provider, agent_id="shared-first")
    shared = first._runtime
    second = make_agent(provider, runtime=shared, agent_id="shared-second")
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "notes.txt").write_text("physical read from agent A")
    (dir_b / "notes.txt").write_text("physical read from agent B")
    first.with_resources(FileIOResource(resource_id="first-files", base_path=str(dir_a)))
    second.with_resources(FileIOResource(resource_id="second-files", base_path=str(dir_b)))

    await outbound(first, bodies)
    await outbound(second, bodies)

    call = [{"function": "Read", "arguments": {"file_path": "notes.txt"}, "tool_call_id": "shared-read"}]
    first_result = await shared.execute_tools_async(first, call)
    second_result = await shared.execute_tools_async(second, call)

    assert first_result[0]["success"], first_result
    assert "agent A" in str(first_result[0]["result"])
    assert "agent B" not in str(first_result[0]["result"])
    assert second_result[0]["success"], second_result
    assert "agent B" in str(second_result[0]["result"])
    assert "agent A" not in str(second_result[0]["result"])


@pytest.mark.asyncio
async def test_overlapping_shared_runtime_dispatch_stays_in_own_membership(clients, tmp_path):
    """Concurrent dispatches through one shared runtime must stay in their own membership.

    The registry that resolves a tool name holds the *bound* resource objects of
    the agent it was built for. Dispatch resolves it from the invoking agent
    carried by the call, never from runtime-wide "current agent" state, because
    preparation, admission and the durable hooks all await: a second invocation
    can complete in between. Here agent B is allowed to run its whole dispatch
    (both its preparation and its result handler) while agent A is suspended
    inside its own preparation, so A's registry lookup happens strictly after B
    replaced any single-agent state. Same relative filename, different physical
    directory.
    """
    make_provider, make_agent, bodies = clients
    provider = await make_provider()
    first = make_agent(provider, agent_id="overlap-first")
    shared = first._runtime
    second = make_agent(provider, runtime=shared, agent_id="overlap-second")
    dir_a = tmp_path / "overlap-a"
    dir_b = tmp_path / "overlap-b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "notes.txt").write_text("overlapping read from agent A")
    (dir_b / "notes.txt").write_text("overlapping read from agent B")
    first.with_resources(FileIOResource(resource_id="overlap-first-files", base_path=str(dir_a)))
    second.with_resources(FileIOResource(resource_id="overlap-second-files", base_path=str(dir_b)))

    second_finished = asyncio.Event()

    async def hold_first_call(_event):
        # Suspend A inside preparation until B has finished dispatch entirely.
        await asyncio.wait_for(second_finished.wait(), timeout=5)
        return None

    async def release_first_call(_event):
        second_finished.set()
        return None

    first.event_bus.subscribe(TOOL_CALL, hold_first_call)
    second.event_bus.subscribe(TOOL_RESULT, release_first_call)

    call = {"function": "Read", "arguments": {"file_path": "notes.txt"}}
    first_task = asyncio.create_task(shared.execute_tools_async(first, [{**call, "tool_call_id": "overlap-first"}]))
    await asyncio.sleep(0.05)  # let A reach its own tool_call handler and suspend
    second_result = await shared.execute_tools_async(second, [{**call, "tool_call_id": "overlap-second"}])
    first_result = await first_task

    assert second_result[0]["success"], second_result
    assert "agent B" in str(second_result[0]["result"])
    assert first_result[0]["success"], first_result
    assert "agent A" in str(first_result[0]["result"]), first_result
    assert "agent B" not in str(first_result[0]["result"]), first_result


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["zero", "defaulted", "variadic", "defaulted_variadic", "mock"])
async def test_static_registry_getters_keep_working_without_an_agent(tmp_path, shape):
    """Every supported zero-argument callable keeps its original dispatch target."""
    resource = FileIOResource(resource_id="static-files", base_path=str(tmp_path))
    (tmp_path / "notes.txt").write_text("static registry read")
    call = {"function": "Read", "arguments": {"file_path": "notes.txt"}, "tool_call_id": "static"}

    registry = {"Read": (resource, "read")}
    getters = {
        "zero": lambda: registry,
        "defaulted": lambda resource=resource: {"Read": (resource, "read")},
        "variadic": lambda *args: {} if args else registry,
        "defaulted_variadic": lambda resource=resource, *args: {"Read": (resource, "read")},
        "mock": Mock(side_effect=lambda: registry),
    }
    executor = ToolExecutor(tool_name_registry_getter=getters[shape])
    try:
        sync_result = executor.execute_tools(object(), [dict(call)])
        parallel_result = executor.execute_tools(object(), [dict(call)], parallel=True)
        async_result = await executor.execute_tools_async(object(), [dict(call)])
    finally:
        await executor.aclose()

    for result in (sync_result, parallel_result, async_result):
        assert result[0]["success"], result
        assert "static registry read" in str(result[0]["result"]), result


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["resource", "workflow", "agent"])
@pytest.mark.parametrize(
    "codec, surface",
    [(CSXMLCodec, "system"), (NativeToolsCodec, "tools")],
)
async def test_structural_reset_refreshes_rendered_codec_prompt(clients, tmp_path, codec, surface, kind):
    """Added tools must appear, removed tools disappear, in the real outbound turn.

    XML codecs render the tool list into the system prompt (which
    ``LocalPromptAPI`` caches), native codecs send it as the request ``tools``
    array. Both are read from the observed SDK payload after each public
    membership mutation, so the cache is proven to follow structural reset.
    """
    make_provider, make_agent, bodies = clients
    provider = await make_provider()
    agent = make_agent(provider, codec=codec, agent_id=f"codec-{codec.__name__}-{kind}")
    if kind == "resource":
        member = FileIOResource(resource_id="codec-files", base_path=str(tmp_path))
        name = "Read"
    elif kind == "workflow":
        member = ReadingWorkflow(workflow_id="codec-workflow", auto_register=False)
        name = "ReadWorkflow"
    else:
        # The XML codec names a sub-agent by object_id, the native codec by agent_type.
        member = make_agent(provider, ReadingAgent, agent_id="codec-sub-agent")
        name = "codec-sub-agent:query" if surface == "system" else "metadata_regression__query"

    def observed(body):
        if surface == "tools":
            return {tool.get("function", tool)["name"] for tool in body.get("tools", [])}
        return set(re.findall(r'invoke name="([^"]+)"', system_text(body)))

    assert name not in observed(await outbound(agent, bodies))

    getattr(agent, f"add_{kind}")(member)
    assert name in observed(await outbound(agent, bodies))

    assert getattr(agent, f"remove_{kind}")(member.object_id)
    assert name not in observed(await outbound(agent, bodies))


@pytest.mark.asyncio
async def test_structural_reset_keeps_template_override(clients, tmp_path):
    """Invalidating the rendered prompt must not drop the template override."""
    make_provider, make_agent, bodies = clients
    provider = await make_provider()
    agent = make_agent(provider, codec=CSXMLCodec, agent_id="codec-override")
    agent.override_system_prompt_template("override {{available_tools_prompt}}")
    assert "override" in system_text(await outbound(agent, bodies))

    agent.with_resources(FileIOResource(resource_id="codec-files", base_path=str(tmp_path)))
    refreshed = system_text(await outbound(agent, bodies))
    assert "override" in refreshed
    assert 'invoke name="Read"' in refreshed


def test_tool_discovery_does_not_evaluate_public_properties():
    """Metadata discovery must not construct clients through public properties."""
    TouchyResource.constructions = 0
    resource = TouchyResource()
    names = [name for name, _ in Misc.extract_tool_use_methods(resource)]
    assert "probe" in names
    assert TouchyResource.constructions == 0


@pytest.mark.asyncio
async def test_instance_installed_tool_retains_required_argument(clients):
    """A dynamic function must not lose its first argument through rebinding."""
    make_provider, make_agent, bodies = clients
    provider = await make_provider()
    agent = make_agent(provider)
    resource = BaseResource(resource_id="dynamic", auto_register=False)

    @named_tool(name="DynamicEcho")
    def echo(value: str) -> str:
        """Return the requested value."""
        return f"echo:{value}"

    resource.echo = echo
    agent.with_resources(resource)
    body = await outbound(agent, bodies)
    tool = next(tool.get("function", tool) for tool in body["tools"] if tool.get("function", tool)["name"] == "DynamicEcho")
    assert tool["parameters"]["properties"]["value"]["type"] == "string"
    assert tool["parameters"]["required"] == ["value"]
    results = await agent._runtime.execute_tools_async(
        agent, [{"function": "DynamicEcho", "arguments": {"value": "kept"}, "tool_call_id": "dynamic"}]
    )
    assert results[0]["success"] and results[0]["result"] == "echo:kept"
