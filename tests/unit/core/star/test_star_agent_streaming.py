"""
Unit tests for STARAgent streaming: call_llm_stream(), aquery_stream().
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dana.common.llm.types import LLMMessage, LLMResponse, LLMStreamChunk
from dana.core.llm.llm_caller import LLMCaller
from dana.core.runtime.protocols import StreamEvent, StreamEventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(content: str = "Hello, world!") -> LLMResponse:
    return LLMResponse(content=content, model="test-model")


def _make_agent():
    """Create a minimal STARAgent without registry or LLM side-effects."""
    from dana.core.agent.star_agent import STARAgent

    return STARAgent(
        agent_type="test-agent",
        auto_register=False,
        enable_skills=False,
        enable_web_search=False,
        enable_code_execution=False,
        enable_assistant=False,
        compress_timeline=False,
    )


async def _collect_stream(agen) -> list[StreamEvent]:
    """Collect all events from an async generator into a list."""
    events: list[StreamEvent] = []
    async for event in agen:
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# LLMCaller.call_llm_stream()
# ---------------------------------------------------------------------------


class TestCallLlmStream:
    """Tests for LLMCaller.call_llm_stream()."""

    @pytest.mark.asyncio
    async def test_yields_chunks_from_llm_stream(self):
        """Should yield each LLMStreamChunk from the underlying LLM stream."""

        async def _fake_stream(messages, **kwargs):
            for text in ["Hello", ", ", "world", "!"]:
                yield LLMStreamChunk(type="text_delta", content=text)

        mock_llm = MagicMock()
        mock_llm.stream = _fake_stream
        caller = LLMCaller(llm=mock_llm)

        chunks = []
        async for chunk in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
            chunks.append(chunk)

        assert [c.content for c in chunks] == ["Hello", ", ", "world", "!"]
        assert all(c.type == "text_delta" for c in chunks)

    @pytest.mark.asyncio
    async def test_passes_agent_info_to_llm_stream(self):
        """Should forward agent_id and agent_type to llm.stream()."""
        received_kwargs: dict = {}

        async def _fake_stream(messages, **kwargs):
            received_kwargs.update(kwargs)
            yield LLMStreamChunk(type="text_delta", content="ok")

        mock_llm = MagicMock()
        mock_llm.stream = _fake_stream

        mock_agent = MagicMock()
        mock_agent.object_id = "agent-123"
        mock_agent.agent_type = "test"

        caller = LLMCaller(llm=mock_llm, agent_getter=lambda: mock_agent)

        async for _ in caller.call_llm_stream([LLMMessage(role="user", content="hi")]):
            pass

        assert received_kwargs.get("agent_id") == "agent-123"
        assert received_kwargs.get("agent_type") == "test"

    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self):
        """If the LLM stream yields nothing, call_llm_stream should be empty."""

        async def _empty_stream(messages, **kwargs):
            return
            yield  # makes it an async generator

        mock_llm = MagicMock()
        mock_llm.stream = _empty_stream
        caller = LLMCaller(llm=mock_llm)

        chunks = []
        async for chunk in caller.call_llm_stream([]):
            chunks.append(chunk)

        assert chunks == []


# ---------------------------------------------------------------------------
# STARAgent.aquery_stream() — basic TEXT_DELTA + DONE flow
# ---------------------------------------------------------------------------


class TestAqueryStreamBasic:
    """aquery_stream() yields TEXT_DELTA events then a DONE event."""

    @pytest.mark.asyncio
    async def test_yields_text_delta_then_done(self):
        """A simple text-only response should produce TEXT_DELTA(s) + DONE."""
        agent = _make_agent()

        # Mock the think_stream to yield one TEXT_DELTA and set result_holder
        async def _mock_think_stream(trace_percepts, result_holder):
            yield StreamEvent(
                event_type=StreamEventType.TEXT_DELTA,
                data="Hello, world!",
                iteration=1,
            )
            # Populate the result holder so _run_aquery_stream can continue
            from dana.core.agent.base_star_agent import EXIT_STAR_LOOP_FLAG

            result_holder["trace_thoughts"] = {
                "trace_thoughts": {
                    "response": "Hello, world!",
                    "tool_calls": [],
                    "done": True,
                    EXIT_STAR_LOOP_FLAG: True,
                }
            }

        # Mock _act_async to return exit signal immediately
        async def _mock_act_async(trace_thoughts):
            from dana.core.agent.base_star_agent import EXIT_STAR_LOOP_FLAG

            return {"trace_outputs": {EXIT_STAR_LOOP_FLAG: True}}

        with patch.object(agent, "_think_stream", _mock_think_stream), patch.object(agent, "_act_async", _mock_act_async):
            events = await _collect_stream(agent.aquery_stream(message="hello"))

        event_types = [e.event_type for e in events]
        assert StreamEventType.TEXT_DELTA in event_types
        assert events[-1].event_type == StreamEventType.DONE

    @pytest.mark.asyncio
    async def test_yields_tool_call_start_and_tool_result(self):
        """Should yield TOOL_CALL_START before tool execution and TOOL_RESULT after."""
        agent = _make_agent()

        tool_calls = [{"function": "search:web", "arguments": {"query": "python"}}]
        tool_results = [{"type": "resource", "result": "some results"}]

        # First iteration: think yields tool calls
        call_count = {"n": 0}

        async def _mock_think_stream(trace_percepts, result_holder):
            from dana.core.agent.base_star_agent import EXIT_STAR_LOOP_FLAG

            call_count["n"] += 1
            yield StreamEvent(
                event_type=StreamEventType.TEXT_DELTA,
                data="Let me search...",
                iteration=1,
            )
            if call_count["n"] == 1:
                # First call: signal tool calls (TOOL_CALL_START now emitted by _think_stream)
                yield StreamEvent(
                    event_type=StreamEventType.TOOL_CALL_START,
                    data=tool_calls,
                    iteration=1,
                )
                result_holder["trace_thoughts"] = {
                    "trace_thoughts": {
                        "response": "",
                        "tool_calls": tool_calls,
                        "done": False,
                    }
                }
            else:
                # Second call: signal done
                result_holder["trace_thoughts"] = {
                    "trace_thoughts": {
                        "response": "Found results",
                        "tool_calls": [],
                        "done": True,
                        EXIT_STAR_LOOP_FLAG: True,
                    }
                }

        async def _mock_act_async(trace_thoughts):
            from dana.core.agent.base_star_agent import EXIT_STAR_LOOP_FLAG

            tcs = trace_thoughts.get("tool_calls", [])
            if tcs:
                return {
                    "trace_outputs": {
                        "tool_results": tool_results,
                        "tool_calls": tcs,
                        "response": "",
                    }
                }
            return {"trace_outputs": {EXIT_STAR_LOOP_FLAG: True}}

        with patch.object(agent, "_think_stream", _mock_think_stream), patch.object(agent, "_act_async", _mock_act_async):
            events = await _collect_stream(agent.aquery_stream(message="search python"))

        event_types = [e.event_type for e in events]
        assert StreamEventType.TOOL_CALL_START in event_types
        assert StreamEventType.TOOL_RESULT in event_types
        assert events[-1].event_type == StreamEventType.DONE

    @pytest.mark.asyncio
    async def test_yields_error_event_on_exception(self):
        """When _think_stream raises, aquery_stream should yield ERROR."""
        agent = _make_agent()

        async def _mock_think_stream_error(trace_percepts, result_holder):
            raise RuntimeError("LLM failed")
            yield  # make it an async generator

        async def _mock_act_async(trace_thoughts):
            from dana.core.agent.base_star_agent import EXIT_STAR_LOOP_FLAG

            return {"trace_outputs": {EXIT_STAR_LOOP_FLAG: True}}

        with patch.object(agent, "_think_stream", _mock_think_stream_error), patch.object(agent, "_act_async", _mock_act_async):
            events = await _collect_stream(agent.aquery_stream(message="hello"))

        assert any(e.event_type == StreamEventType.ERROR for e in events)
        # Should NOT have DONE after error (loop exits)
        error_idx = next(i for i, e in enumerate(events) if e.event_type == StreamEventType.ERROR)
        assert error_idx == len(events) - 1  # ERROR is the last event


# ---------------------------------------------------------------------------
# _think_stream() integration-style test (no real LLM calls)
# ---------------------------------------------------------------------------


class TestThinkStream:
    """Tests for STARAgent._think_stream() with a mocked LLM stream."""

    @pytest.mark.asyncio
    async def test_provisional_text_precedes_provider_completion_and_close_releases_provider(self):
        agent = _make_agent()
        provider_completed = asyncio.Event()
        provider_closed = asyncio.Event()

        async def provider_stream(messages):
            try:
                yield LLMStreamChunk(type="thinking", content="private reasoning")
                yield LLMStreamChunk(type="text_delta", content="Visible answer")
                await asyncio.Event().wait()
                provider_completed.set()
            finally:
                provider_closed.set()

        runtime = MagicMock()
        runtime.build_prompt.return_value = []
        runtime._llm_caller.call_llm_stream = provider_stream
        with patch.object(agent, "_runtime", runtime), patch.object(agent, "_maybe_compress_timeline_async", new=AsyncMock()):
            stream = agent._think_stream({"timeline": agent._timeline}, {})
            thought = await asyncio.wait_for(anext(stream), 1)
            text = await asyncio.wait_for(anext(stream), 1)
            assert thought.event_type == StreamEventType.THINKING
            assert text.event_type == StreamEventType.TEXT_DELTA
            assert not provider_completed.is_set()
            await stream.aclose()
            assert provider_closed.is_set()

    @pytest.mark.asyncio
    async def test_exits_early_on_exit_star_loop_flag(self):
        """If trace_percepts has EXIT flag set, _think_stream yields nothing."""
        agent = _make_agent()
        from dana.core.agent.base_star_agent import EXIT_STAR_LOOP_FLAG

        trace_percepts = {EXIT_STAR_LOOP_FLAG: True}
        result_holder: dict = {}

        events = []
        async for event in agent._think_stream(trace_percepts, result_holder):
            events.append(event)

        assert events == []
        assert "trace_thoughts" in result_holder
