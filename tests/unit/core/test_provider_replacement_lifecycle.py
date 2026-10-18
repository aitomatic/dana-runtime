"""Provider-replacement ownership across the agent, the call site and long-term memory.

Consumer-visible contract under test: whichever pool an installed LLM facade points
at is released exactly once, by the agent that owns it, and never by somebody else.

* Reinjecting the provider this agent currently owns through the public
  ``set_llm_provider(llm_provider_instance=...)`` API must not hand the replacement
  facade a pool that the retirement of the replaced facade already closed: two
  wrappers over one provider are not independent retirement targets.
* The memory store's sub-LLM is a second holder of the installed client, so it
  must follow the replacement and reach teardown with a live pool.
* A provider instance injected by a caller stays the caller's: plain teardown
  leaves it open and usable, and a same-wrapper ``switch_provider`` must not end a
  pool whose lifetime is not this agent's.
* The replacement a ``switch_provider`` creates inside an installed wrapper *is*
  this agent's, so it is released at teardown (it used to leak).
* Ordinary different-provider replacement still retires the owned pool it
  displaces, ``close``/``aclose`` stay idempotent, and re-installing the same
  wrapper is still a no-op.

Real ``AsyncOpenAI`` clients over ``httpx.MockTransport``: no network, no live
provider, and every pool is exercised through a real request rather than by
asserting on source.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any
from unittest.mock import patch


# Set before the Dana imports so string-path construction can read an API key.
os.environ.setdefault("OPENAI_API_KEY", "synthetic-provider-replacement")
os.environ.setdefault("ANTHROPIC_API_KEY", "synthetic-provider-replacement")

import httpx
from openai import AsyncOpenAI
import pytest

from dana.common.llm.llm import LLM
from dana.common.llm.providers import OpenAIProvider
from dana.common.llm.types import LLMMessage, LLMProvider
from dana.core.agent.star_agent import STARAgent, _client_is_borrowed


_AGENT_KW: dict[str, Any] = dict(
    auto_register=False,
    enable_skills=False,
    enable_web_search=False,
    enable_code_execution=False,
    enable_assistant=False,
    compress_timeline=False,
)

_MODEL = "gpt-4o"


def _respond(request: httpx.Request) -> httpx.Response:
    """Serve both wire shapes so the offline pool answers whichever path is taken."""
    if request.url.path.endswith("/responses"):
        payload = {
            "id": "resp_provider_replacement",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": _MODEL,
            "output": [
                {
                    "id": "msg_provider_replacement",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "pong", "annotations": []}],
                }
            ],
        }
    else:
        payload = {
            "id": "chatcmpl_provider_replacement",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "pong"},
                }
            ],
        }
    return httpx.Response(200, content=json.dumps(payload).encode())


async def _install_mock_pool(provider: OpenAIProvider) -> None:
    """Replace the provider's real SDK client with an offline one.

    The constructor's own client is closed first so a replaced pool cannot linger;
    the replacement is a real ``AsyncOpenAI`` speaking to a ``MockTransport``.
    """
    previous = getattr(provider, "client", None)
    if previous is not None:
        closer = getattr(previous, "close", None) or getattr(previous, "aclose", None)
        if closer is not None:
            await closer()
    provider.client = AsyncOpenAI(
        api_key="synthetic",
        base_url="https://offline.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_respond)),
    )


async def _usable(facade: LLM) -> str | None:
    """Prove the facade's pool answers by driving one real completion through it."""
    try:
        return await facade.chat([LLMMessage(role="user", content="ping")])
    except Exception:
        return None


async def _provider_usable(provider: LLMProvider) -> str | None:
    """Same, for a provider the test holds directly (caller-owned pools)."""
    try:
        response = await provider.chat([LLMMessage(role="user", content="ping")])
    except Exception:
        return None
    return getattr(response, "content", None)


async def _drain(*facades: LLM | None) -> None:
    """Let any close the facade scheduled on the running loop finish."""
    tasks: list[asyncio.Task[Any]] = []
    for facade in facades:
        if facade is None:
            continue
        tasks.extend(getattr(facade, "_close_tasks", ()) or ())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _make_provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="synthetic", model=_MODEL, base_url="https://offline.invalid/v1")


def _make_agent(**overrides: Any) -> STARAgent:
    return STARAgent(agent_type="provider-replacement", **(_AGENT_KW | overrides))


async def _scenario_reinject_owned_provider(tmp_path: Any) -> None:
    """M3: the pool the agent built survives its own re-installation."""
    agent = _make_agent(llm_provider="openai", model=_MODEL)
    original = agent.llm_client
    provider = original.provider
    await _install_mock_pool(provider)

    agent.set_llm_provider(llm_provider_instance=provider)
    installed = agent.llm_client
    await _drain(original, installed)

    assert installed is not original, "reinstalling a provider must install a facade for it"
    assert not provider.client.is_closed(), "the installed facade was handed a freshly closed pool"
    assert await _usable(installed) == "pong", "the reinstalled pool could not serve a request"

    await agent.aclose()
    assert provider.client.is_closed(), "the agent built this pool, so teardown must release it"


async def _scenario_memory_alias_survives_reinjection(tmp_path: Any) -> None:
    """M3 memory alias: the store's holder follows the replacement and keeps a live pool."""
    agent = _make_agent(llm_provider="openai", model=_MODEL, ltmemory_path=str(tmp_path / "memory"))
    original = agent.llm_client
    provider = original.provider
    await _install_mock_pool(provider)

    agent.set_llm_provider(llm_provider_instance=provider)
    installed = agent.llm_client
    await _drain(original, installed)

    memory_facade = agent._ltmemory._rlm._llm
    assert memory_facade is installed, "the memory store kept a facade the agent had replaced"
    assert not provider.client.is_closed(), "the store's pool was closed by the retirement it survived"
    assert await _usable(memory_facade) == "pong", "the store's pool could not serve a request"

    await agent.aclose()
    assert provider.client.is_closed(), "the reinstalled pool must still be released at teardown"


async def _scenario_reinject_memory_held_pool(tmp_path: Any) -> None:
    """M3 memory branch: the store's own pool must not be closed by its own re-installation.

    The store built this pool, so this agent owns it; the retirement of the agent's
    client must not end it, and teardown must still release it.
    """
    agent = _make_agent(llm_provider="openai", model=_MODEL, ltmemory_path=str(tmp_path / "memory"))
    original = agent.llm_client
    displaced = original.provider
    await _install_mock_pool(displaced)

    store_built_provider = agent._ltmemory._rlm._llm.provider
    await _install_mock_pool(store_built_provider)

    agent.set_llm_provider(llm_provider_instance=store_built_provider)
    installed = agent.llm_client
    await _drain(original, agent._ltmemory._rlm._llm, installed)

    assert displaced.client.is_closed(), "the displaced owned pool leaked"
    assert installed.provider is store_built_provider
    assert not store_built_provider.client.is_closed(), "the store retired the pool it was reinstalled with"
    assert await _usable(installed) == "pong", "the reinstalled store pool could not serve a request"

    await agent.aclose()
    assert store_built_provider.client.is_closed(), "a pool this agent built must be released at teardown"


async def _scenario_switch_from_borrowed_instance(tmp_path: Any) -> None:
    """M4: a same-wrapper switch hands ownership of the new pool to this agent."""
    caller_provider = _make_provider()
    await _install_mock_pool(caller_provider)
    agent = _make_agent(
        llm_provider_instance=caller_provider,
        ltmemory_path=str(tmp_path / "memory"),
    )
    facade = agent.llm_client
    created: list[OpenAIProvider] = []

    def make(name: str, model: str | None = None) -> LLMProvider:
        provider = _make_provider()
        created.append(provider)
        return provider

    with patch("dana.common.llm.llm.create_provider", side_effect=make):
        facade.switch_provider("openai", model=_MODEL)
    replacement = facade.provider
    assert replacement is created[0]
    await _install_mock_pool(replacement)
    await _drain(facade)

    assert not replacement.client.is_closed(), "the switch installed a pool that was already closed"
    assert await _usable(facade) == "pong", "the switched-to pool could not serve a request"

    await agent.aclose()
    assert replacement.client.is_closed(), "the pool this switch created leaked past teardown"
    assert agent._ltmemory._rlm._llm is facade, "the memory store kept a stale facade after the switch"
    assert not caller_provider.client.is_closed(), "teardown closed a pool the caller still owns"
    assert await _provider_usable(caller_provider) == "pong", "the caller's borrowed pool was ended"


async def _scenario_injected_provider_stays_caller_owned(tmp_path: Any) -> None:
    """Control: an injected instance is caller-owned, with or without teardown."""
    provider = _make_provider()
    await _install_mock_pool(provider)
    agent = _make_agent(llm_provider_instance=provider)

    await agent.aclose()

    assert not provider.client.is_closed(), "teardown closed a pool it merely borrowed"
    assert await _provider_usable(provider) == "pong", "the borrowed pool was ended by teardown"


async def _scenario_different_provider_retires_owned_pool(tmp_path: Any) -> None:
    """Control: replacing an owned pool with another provider still retires it."""
    agent = _make_agent(llm_provider="openai", model=_MODEL)
    first = agent.llm_client
    retired = first.provider
    await _install_mock_pool(retired)

    other = _make_provider()
    await _install_mock_pool(other)
    agent.set_llm_provider(llm_provider_instance=other)
    await _drain(first, agent.llm_client)

    assert retired.client.is_closed(), "the displaced owned pool leaked"
    assert not other.client.is_closed(), "the borrowed replacement was closed on installation"

    await agent.aclose()
    assert not other.client.is_closed(), "teardown closed the caller's pool"


async def _scenario_direct_switch_leaves_borrowed_pool(tmp_path: Any) -> None:
    """The direct LLM API: a borrowed pool is not the facade's to retire."""
    borrowed = _make_provider()
    await _install_mock_pool(borrowed)
    facade = LLM(provider=borrowed)
    # A host that injects an instance records the borrow on the facade it hands out.
    facade.provider_borrowed = True
    created: list[OpenAIProvider] = []

    def make(name: str, model: str | None = None) -> LLMProvider:
        provider = _make_provider()
        created.append(provider)
        return provider

    with patch("dana.common.llm.llm.create_provider", side_effect=make):
        facade.switch_provider("openai", model=_MODEL)
    replacement = facade.provider
    await _install_mock_pool(replacement)
    await _drain(facade)

    assert not borrowed.client.is_closed(), "the switch ended a pool its owner still holds"
    assert await _provider_usable(borrowed) == "pong"
    assert not replacement.client.is_closed()
    assert facade.provider_borrowed is False, "the replacement this facade created is its own"

    await facade.aclose()
    assert replacement.client.is_closed(), "the replacement pool was not released"
    assert not borrowed.client.is_closed()


async def _scenario_teardown_idempotent_and_reinstall_is_noop(tmp_path: Any) -> None:
    """Control: the same-object guard and double teardown still hold."""
    agent = _make_agent(llm_provider="openai", model=_MODEL)
    facade = agent.llm_client
    provider = facade.provider
    await _install_mock_pool(provider)

    agent.llm_client = facade
    await _drain(facade)
    assert not provider.client.is_closed(), "re-installing the same facade closed a live pool"

    await agent.aclose()
    await agent.aclose()
    agent.close()
    assert provider.client.is_closed(), "the owned pool was never released"


class _GatedProvider(OpenAIProvider):
    """Real offline provider whose ``aclose`` blocks until the test releases it.

    Used to observe the difference between "cleanup scheduled" and "cleanup done":
    without a gate, a leaked-or-orphaned task still finishes before the assertion.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()

    async def aclose(self) -> None:
        self.close_started.set()
        await self.close_release.wait()
        await super().aclose()


async def _make_gated_provider() -> _GatedProvider:
    """Real offline provider whose close is gated; one pool allocated, no spare."""
    provider = _GatedProvider(api_key="synthetic", model=_MODEL, base_url="https://offline.invalid/v1")
    await _install_mock_pool(provider)
    return provider


class _SettleProbe:
    """Deterministic view of an awaited teardown, without loop-yield guessing.

    Wraps the coroutine so the test can (a) await a signal that teardown has really
    entered, instead of sleeping a fixed number of loop turns, and (b) ask whether the
    teardown *completed*, which is the only thing a settled lifecycle may claim.
    """

    def __init__(self, coro: Any, entered: asyncio.Event) -> None:
        self._entered = entered
        self.task = asyncio.ensure_future(self._run(coro))

    async def _run(self, coro: Any) -> None:
        self._entered.set()
        await coro

    async def entered_and_check_pending(self) -> bool:
        """Return True when teardown has entered but has not finished yet."""
        await asyncio.wait_for(self._entered.wait(), 5)
        return not self.task.done()

    async def finish(self) -> None:
        await asyncio.wait_for(self.task, 5)


def _gated_providers(agent: STARAgent) -> list[_GatedProvider]:
    """Every gated provider this agent currently points at, so cleanup can drain."""
    memory_facade = getattr(getattr(getattr(agent, "_ltmemory", None), "_rlm", None), "_llm", None)
    providers = []
    for facade in (agent._llm_client, memory_facade):
        provider = getattr(facade, "provider", None)
        if isinstance(provider, _GatedProvider):
            providers.append(provider)
    return providers


async def _release_gated(*providers: _GatedProvider) -> None:
    """Unblock every gated provider a scenario may have created, then let it finish."""
    released = {id(p) for p in providers if isinstance(p, _GatedProvider)}
    for provider in providers:
        if isinstance(provider, _GatedProvider):
            provider.close_release.set()
    for _ in range(len(released)):  # each pending close gets a turn to observe release
        await asyncio.sleep(0)


async def _scenario_teardown_awaits_retired_facade_cleanup(tmp_path: Any) -> None:
    """Teardown must finish cleanup that a *retired* facade only scheduled.

    The agent replaces its client, which schedules the retired pool's close on the
    retired facade. That facade is no longer reachable from the agent, so unless the
    replacement takes the task over, an awaited teardown returns while the retired
    pool is still open and loop shutdown cancels the orphaned close.
    """
    agent = _make_agent(llm_provider="openai", model=_MODEL)
    retirable = await _make_gated_provider()
    replacement = await _make_gated_provider()
    try:
        agent.llm_client = LLM(provider=retirable)
        replacement.close_release.set()
        agent.llm_client = LLM(provider=replacement)

        await asyncio.wait_for(retirable.close_started.wait(), 5)
        probe = _SettleProbe(agent.aclose(), asyncio.Event())
        assert await probe.entered_and_check_pending(), "teardown returned while the retired pool's close was still pending"

        retirable.close_release.set()
        await probe.finish()
        assert retirable.client.is_closed(), "the retired pool was still open when teardown finished"
        assert replacement.client.is_closed(), "the installed pool was not released"
    finally:
        await _release_gated(*_gated_providers(agent), retirable, replacement)
        await agent.aclose()


async def _scenario_aclose_is_barrier_after_sync_close(tmp_path: Any) -> None:
    """A sync ``close()`` schedules cleanup; a later awaited ``aclose()`` must finish it.

    ``close`` has an explicit non-blocking contract, so it cannot wait. That makes
    the awaited teardown the only place a session host can observe the owned pool
    actually released, which is why it must re-enter instead of returning early.
    """
    agent = _make_agent(llm_provider="openai", model=_MODEL)
    provider = await _make_gated_provider()
    agent.llm_client = LLM(provider=provider)
    try:
        agent.close()
        await asyncio.wait_for(provider.close_started.wait(), 5)

        probe = _SettleProbe(agent.aclose(), asyncio.Event())
        assert await probe.entered_and_check_pending(), "aclose returned although the cleanup close() scheduled was still pending"

        provider.close_release.set()
        await probe.finish()
        assert provider.client.is_closed(), "the owned pool was still open when the awaited teardown finished"

        await agent.aclose()
        assert provider.client.is_closed()
    finally:
        await _release_gated(*_gated_providers(agent), provider)
        await agent.aclose()


async def _scenario_teardown_awaits_memory_cleanup(tmp_path: Any) -> None:
    """The store's own retired facade is covered by the same barrier.

    Displacing the store's facade schedules that pool's close on the *store's*
    facade. The agent then holds only the replacement, so unless the pending task
    moves with the fan-out, the awaited teardown returns while the displaced pool is
    still open — and loop shutdown cancels the orphaned close.
    """
    agent = _make_agent(llm_provider="openai", model=_MODEL, ltmemory_path=str(tmp_path / "memory"))
    gated = await _make_gated_provider()
    try:
        agent._ltmemory.set_llm(LLM(provider=gated))
        agent.set_llm_provider(llm_provider="openai", model=_MODEL)
        installed = agent._llm_client
        assert installed.provider is not gated

        await asyncio.wait_for(gated.close_started.wait(), 5)
        probe = _SettleProbe(agent.aclose(), asyncio.Event())
        assert await probe.entered_and_check_pending(), "teardown returned while the store cleanup was still pending"

        gated.close_release.set()
        await probe.finish()
        assert gated.client.is_closed(), "the displaced store pool was still open when teardown finished"
    finally:
        await _release_gated(*_gated_providers(agent), gated)
        await agent.aclose()


async def _scenario_borrowed_replacement_carries_displaced_cleanup(tmp_path: Any) -> None:
    """An owned pool displaced by a *borrowed* replacement is still drained.

    The replacement is caller-owned, so teardown must not close its pool — but the
    drain of cleanup the agent itself scheduled when retiring its own pool must not
    depend on the replacement being owned either. Awaiting pending cleanup never
    closes the pool a borrowed facade points at.
    """
    agent = _make_agent(llm_provider="openai", model=_MODEL)
    owned = await _make_gated_provider()
    borrowed = await _make_gated_provider()
    borrowed.close_release.set()
    try:
        agent.llm_client = LLM(provider=owned)
        agent.set_llm_provider(llm_provider_instance=borrowed)
        assert _client_is_borrowed(agent._llm_client)

        await asyncio.wait_for(owned.close_started.wait(), 5)
        probe = _SettleProbe(agent.aclose(), asyncio.Event())
        assert await probe.entered_and_check_pending(), "teardown returned while the owned pool's close was still pending"
        assert not owned.client.is_closed()

        owned.close_release.set()
        await probe.finish()
        assert owned.client.is_closed(), "the retired owned pool was not released by teardown"
        assert not borrowed.client.is_closed(), "teardown closed the caller-owned replacement pool"
    finally:
        await _release_gated(*_gated_providers(agent), owned, borrowed)
        await agent.aclose()


async def _scenario_concurrent_teardowns_join_and_retry(tmp_path: Any) -> None:
    """Teardown completion is joinable, and an interrupted pass does not claim settled.

    A second ``aclose`` must wait for the pass already in flight instead of returning
    from a flag that was set before the work finished; and a cancelled pass must leave
    teardown re-runnable, so a later ``aclose`` still finishes the owned cleanup.
    """
    agent = _make_agent(llm_provider="openai", model=_MODEL)
    provider = await _make_gated_provider()
    agent.llm_client = LLM(provider=provider)
    try:
        first = _SettleProbe(agent.aclose(), asyncio.Event())
        await asyncio.wait_for(provider.close_started.wait(), 5)
        second = asyncio.ensure_future(agent.aclose())
        with pytest.raises(asyncio.TimeoutError):
            # A joined teardown must not complete while the gated close is unreleased;
            # the bound proves non-completion rather than waiting for it to finish.
            await asyncio.wait_for(asyncio.shield(second), 0.2)

        provider.close_release.set()
        await first.finish()
        await asyncio.wait_for(second, 5)
        assert provider.client.is_closed()

        await agent.aclose()
        assert provider.client.is_closed(), "a settled agent did not keep its pool released"
    finally:
        await _release_gated(*_gated_providers(agent), provider)
        await agent.aclose()

    # Interrupted pass: cancelling the first teardown must not look settled.
    agent2 = _make_agent(llm_provider="openai", model=_MODEL)
    provider2 = await _make_gated_provider()
    agent2.llm_client = LLM(provider=provider2)
    try:
        interrupted = asyncio.ensure_future(agent2.aclose())
        await asyncio.wait_for(provider2.close_started.wait(), 5)
        interrupted.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await interrupted

        # The cancelled pass released nothing, so the pool is still gated: release it
        # and require a later teardown to carry the owned cleanup to completion.
        provider2.close_release.set()
        await asyncio.wait_for(agent2.aclose(), 5)
        assert provider2.client.is_closed(), "teardown after a cancelled pass did not finish the owned cleanup"
    finally:
        await _release_gated(*_gated_providers(agent2), provider2)
        await agent2.aclose()


async def _scenario_owned_child_cleanup_drained_after_sync_close(tmp_path: Any) -> None:
    """An owned internal child's scheduled cleanup is finished by the parent's aclose.

    ``parent.close()`` starts each owned child's teardown, which only *schedules* the
    child's SDK close. The parent's awaited teardown must therefore still visit those
    children, or the child pool is left open and loop shutdown cancels its close. A
    child added with ``with_agents`` is caller-owned and is never visited.
    """
    parent = _make_agent(enable_assistant=True)
    child = parent._owned_internal_agents[0]
    child_provider = await _make_gated_provider()
    child.llm_client = LLM(provider=child_provider)
    borrowed_child = _make_agent()
    borrowed_provider = await _make_gated_provider()
    borrowed_child.llm_client = LLM(provider=borrowed_provider)
    parent.with_agents(borrowed_child)
    try:
        parent.close()
        await asyncio.wait_for(child_provider.close_started.wait(), 5)

        probe = _SettleProbe(parent.aclose(), asyncio.Event())
        assert await probe.entered_and_check_pending(), "parent teardown returned while the owned child's cleanup was still pending"

        child_provider.close_release.set()
        await probe.finish()
        assert child_provider.client.is_closed(), "the owned child's pool was not released"
        assert not borrowed_child._lifecycle_closed, "a borrowed child was torn down by its parent"
    finally:
        await _release_gated(child_provider, borrowed_provider)
        for agent in (borrowed_child, parent):
            await agent.aclose()
        await asyncio.wait_for(borrowed_child.aclose(), 5)


async def _scenario_concurrent_caller_lifetime_is_not_joined(tmp_path: Any) -> None:
    """A joining ``aclose`` waits for the teardown, not for the first caller's lifetime.

    Consumers commonly hold the teardown future and keep working after it: calling
    ``aclose`` from a coroutine that then awaits unrelated work must not make a second
    ``aclose`` wait for that work. If the join tracked the first *caller*, two
    overlapping callers form a cycle and both hang even though the pool is released.
    """
    agent = _make_agent(llm_provider="openai", model=_MODEL)
    provider = await _make_gated_provider()
    agent.llm_client = LLM(provider=provider)
    first_returned = asyncio.Event()
    second_entered = asyncio.Event()
    second: asyncio.Task[None] | None = None
    try:
        assert await _usable(agent.llm_client) == "pong"

        async def first_caller() -> None:
            await agent.aclose()
            first_returned.set()
            assert second is not None
            await second  # unrelated remaining caller work after teardown

        async def second_caller() -> None:
            second_entered.set()
            await agent.aclose()

        first = asyncio.ensure_future(first_caller())
        await asyncio.wait_for(provider.close_started.wait(), 5)
        second = asyncio.ensure_future(second_caller())
        await asyncio.wait_for(second_entered.wait(), 5)

        provider.close_release.set()
        await asyncio.wait_for(first_returned.wait(), 5)
        try:
            await asyncio.wait_for(asyncio.shield(first), 2)
        except TimeoutError:  # pragma: no cover - only on the reverted join
            pytest.fail("joining aclose waited on the first caller's remaining lifetime instead of the teardown")
        await asyncio.wait_for(second, 5)
        assert provider.client.is_closed()
    finally:
        provider.close_release.set()
        for task in (locals().get("first"), second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (locals().get("first"), second) if t is not None), return_exceptions=True)
        await _release_gated(*_gated_providers(agent), provider)
        await agent.aclose()


@pytest.mark.asyncio
async def test_concurrent_aclose_joins_teardown_not_the_first_caller(tmp_path):
    await _scenario_concurrent_caller_lifetime_is_not_joined(tmp_path)


@pytest.mark.asyncio
async def test_owned_child_cleanup_is_drained_by_parent_teardown(tmp_path):
    await _scenario_owned_child_cleanup_drained_after_sync_close(tmp_path)


@pytest.mark.asyncio
async def test_borrowed_replacement_still_drains_displaced_owned_cleanup(tmp_path):
    await _scenario_borrowed_replacement_carries_displaced_cleanup(tmp_path)


@pytest.mark.asyncio
async def test_concurrent_teardowns_join_and_a_cancelled_pass_is_retryable(tmp_path):
    await _scenario_concurrent_teardowns_join_and_retry(tmp_path)


@pytest.mark.asyncio
async def test_teardown_awaits_retired_facade_cleanup(tmp_path):
    await _scenario_teardown_awaits_retired_facade_cleanup(tmp_path)


@pytest.mark.asyncio
async def test_aclose_is_a_barrier_for_cleanup_scheduled_by_sync_close(tmp_path):
    await _scenario_aclose_is_barrier_after_sync_close(tmp_path)


@pytest.mark.asyncio
async def test_teardown_awaits_memory_facade_cleanup(tmp_path):
    await _scenario_teardown_awaits_memory_cleanup(tmp_path)


@pytest.mark.asyncio
async def test_reinjected_current_provider_pool_stays_open_and_usable(tmp_path):
    await _scenario_reinject_owned_provider(tmp_path)


@pytest.mark.asyncio
async def test_memory_alias_shares_reinjected_pool_and_survives_retirement(tmp_path):
    await _scenario_memory_alias_survives_reinjection(tmp_path)


@pytest.mark.asyncio
async def test_reinstalling_the_store_held_pool_keeps_it_open(tmp_path):
    await _scenario_reinject_memory_held_pool(tmp_path)


@pytest.mark.asyncio
async def test_switch_from_borrowed_instance_closes_replacement_on_teardown(tmp_path):
    await _scenario_switch_from_borrowed_instance(tmp_path)


@pytest.mark.asyncio
async def test_injected_provider_remains_caller_owned_through_teardown():
    await _scenario_injected_provider_stays_caller_owned(None)


@pytest.mark.asyncio
async def test_different_provider_replacement_still_retires_owned_pool():
    await _scenario_different_provider_retires_owned_pool(None)


@pytest.mark.asyncio
async def test_direct_switch_provider_leaves_borrowed_pool_to_its_owner():
    await _scenario_direct_switch_leaves_borrowed_pool(None)


@pytest.mark.asyncio
async def test_teardown_idempotent_and_reinstall_same_wrapper_is_a_noop():
    await _scenario_teardown_idempotent_and_reinstall_is_noop(None)
