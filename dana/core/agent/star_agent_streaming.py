"""
STARAgentStreamingMixin — streaming extensions for STARAgent.

Provides aquery_stream(), _run_aquery_stream(), _think_stream(), and
aquery_text_stream(). Mixed into STARAgent so all methods retain self access.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

import structlog

from dana.common.protocols import DictParams
from dana.common.protocols.types import LearningPhase

from ..runtime.protocols import StreamEvent, StreamEventType
from ..timeline.timeline import Timeline


logger = structlog.get_logger()


class STARAgentStreamingMixin:
    """Mixin that adds streaming query support to STARAgent.

    Depends on the following attributes / methods being available on self:
        - _star_loop_count, _session_id, _event_log, _timeline
        - _runtime, llm_client, object_id, agent_type, agent_id
        - set_session_id(), _do_exit_star_loop(), _mark_star_loop_exit()
        - _see(), _act_async(), _reflect()
        - _maybe_compress_timeline_async(), _record_think_results()
        - MAX_ITERATIONS
    """

    # AgentSession retains this mixin's coherent working timeline on success.
    _stream_updates_timeline = True

    async def _think_stream(
        self,
        trace_percepts: DictParams,
        result_holder: dict,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming think phase. Yields StreamEvents while the LLM responds.

        Handles typed LLMStreamChunk objects: text_delta, tool_use, thinking,
        and the optional terminal "response" carrier.

        Builds a synthetic LLMResponse with both text and tool_calls so
        parse_response works for native tool calling agents. When the provider
        emits a terminal "response" carrier, its non-delta fields (reasoning
        items, response id, usage, finish reason) are preserved through
        parse_response into _record_think_results; providers without one keep
        their existing delta-derived behavior.

        Args:
            trace_percepts: Percepts from the SEE phase (timeline must be present).
            result_holder:  Mutable dict; populated with {"trace_thoughts": ...}
                            after streaming completes.

        Yields:
            StreamEvent: TEXT_DELTA, TOOL_CALL_START, THINKING events.
        """
        import json

        from dana.common.llm.types import LLMResponse, ProviderError

        trace_percepts = trace_percepts or {}
        if self._do_exit_star_loop(trace_percepts) or not trace_percepts:
            result_holder["trace_thoughts"] = {"trace_thoughts": self._mark_star_loop_exit(trace_percepts)}
            return

        iteration = self._star_loop_count
        timeline: Timeline = trace_percepts.get("timeline", self._timeline)
        trace_percepts.pop("timeline", None)

        await self._maybe_compress_timeline_async(timeline)
        llm_messages = self._runtime.build_prompt(self, timeline)

        # Resolve stream source
        if hasattr(self._runtime, "_llm_caller"):
            stream_src = self._runtime._llm_caller.call_llm_stream(llm_messages)
        else:
            stream_src = self.llm_client.stream(
                llm_messages,
                agent_id=self.object_id,
                agent_type=self.agent_type,
            )

        # Assistant narration is provisional until the turn commits. Only
        # provider reasoning is a THINKING event; tools may follow visible text.
        text_chunks: list[str] = []
        tool_calls_from_stream: list[dict] = []
        thinking_chunks: list[str] = []
        terminal_response: LLMResponse | None = None

        async with aclosing(stream_src):
            async for chunk in stream_src:
                if chunk.type == "text_delta":
                    text_chunks.append(chunk.content)
                    yield StreamEvent(event_type=StreamEventType.TEXT_DELTA, data=chunk.content, iteration=iteration)
                elif chunk.type == "tool_use":
                    tool_calls_from_stream.append(chunk.tool_call)
                    yield StreamEvent(event_type=StreamEventType.TOOL_CALL_START, data=[chunk.tool_call], iteration=iteration)
                elif chunk.type == "thinking":
                    thinking_chunks.append(chunk.content)
                    yield StreamEvent(event_type=StreamEventType.THINKING, data=chunk.content, iteration=iteration)
                elif chunk.type == "response" and chunk.response is not None:
                    terminal_response = chunk.response
                    yield StreamEvent(
                        event_type=StreamEventType.PROVIDER_RESPONSE,
                        data={
                            "iteration": iteration,
                            "model": terminal_response.model,
                            "finish_reason": terminal_response.finish_reason,
                            "usage": terminal_response.usage,
                            "reasoning_tokens": terminal_response.reasoning_tokens,
                            "response_id": terminal_response.response_id,
                        },
                        iteration=iteration,
                    )

        buffered_text = "".join(text_chunks)

        # Some providers supply content only in their completion carrier.
        if not buffered_text and terminal_response is not None and terminal_response.content:
            yield StreamEvent(event_type=StreamEventType.TEXT_DELTA, data=terminal_response.content, iteration=iteration)

        if terminal_response is not None and (terminal_response.finish_reason or "").lower() in {
            "length",
            "max_tokens",
            "incomplete",
            "content_filter",
            "safety",
            "recitation",
            "prohibited_content",
        }:
            # Provisional text remains visible, but a truncated/refused answer
            # must not become a committed final answer or execute partial tools.
            # Deterministic provider outcomes are permanent, never retried.
            raise ProviderError(
                f"Provider did not complete the answer: {terminal_response.finish_reason}",
                terminal_response=terminal_response,
            )

        # Build synthetic LLMResponse with BOTH text and tool calls.
        # Stream deltas stay authoritative for content and tool calls (they are
        # what the host already saw); the terminal carrier only fills fields the
        # deltas cannot express. Legacy providers that emit no carrier leave every
        # value at its pre-existing delta-derived default.
        streamed_tool_calls = (
            [
                type(
                    "ToolCall",
                    (),
                    {
                        "id": tc["id"],
                        "function": type(
                            "Function",
                            (),
                            {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["input"]) if isinstance(tc["input"], dict) else tc["input"],
                            },
                        )(),
                    },
                )()
                for tc in tool_calls_from_stream
            ]
            if tool_calls_from_stream
            else None
        )
        resolved_tool_calls = streamed_tool_calls or (terminal_response.tool_calls if terminal_response else None)

        synthetic_response = LLMResponse(
            content=buffered_text or (terminal_response.content if terminal_response else ""),
            model=(terminal_response.model if terminal_response else None) or "streamed",
            usage=terminal_response.usage if terminal_response else None,
            finish_reason=terminal_response.finish_reason if terminal_response else None,
            tool_calls=resolved_tool_calls,
            reasoning_content="".join(thinking_chunks) or (terminal_response.reasoning_content if terminal_response else None),
            reasoning_tokens=terminal_response.reasoning_tokens if terminal_response else None,
            reasoning_items=terminal_response.reasoning_items if terminal_response else None,
            response_id=terminal_response.response_id if terminal_response else None,
        )

        parsed = self._runtime.parse_response(synthetic_response)
        response, reasoning, tool_calls, done, todo_list = (
            parsed.response,
            parsed.reasoning,
            parsed.tool_calls,
            parsed.done,
            parsed.todo_list,
        )

        has_tool_calls = bool(tool_calls)
        has_response = bool(response and response.strip())
        output_state = self._runtime.validate_done_output(done, has_tool_calls, has_response)

        # No retry loop for streaming (YAGNI — keep simple)
        if output_state == "retry":
            output_state = "exit"

        trace_result = self._record_think_results(
            timeline,
            trace_percepts,
            response,
            reasoning,
            tool_calls,
            done,
            todo_list,
            output_state,
            reasoning_items=parsed.reasoning_items,
            response_id=parsed.response_id,
        )
        result_holder["trace_thoughts"] = trace_result

    async def aquery_text_stream(
        self,
        *,
        message: str,
        cancel_event: asyncio.Event,
        result_holder: dict | None = None,
    ) -> AsyncIterator[str]:
        """Stream text-only response chunks for a D1 text turn.

        Unlike ``_think_stream``, this does NOT buffer text or emit THINKING
        events. Text deltas are yielded immediately as they arrive. Tool calls
        are not handled (D1 is text-only). No reflection, timeline compression,
        or other post-processing runs — this is a pure text stream.

        The caller (e.g. :class:`~dana.core.session.agent_session.AgentSession`)
        is responsible for adding the user message to the timeline BEFORE calling
        this method, so ``build_prompt`` includes it.

        After the generator is exhausted, ``result_holder`` (if provided) is
        populated with::

            {"full_text": str, "protected_payload": bytes | None, "finish_reason": str | None}

        Args:
            message: The user message text (already added to the timeline).
            cancel_event: Set by the caller to request cancellation; checked
                between chunks and raises :class:`asyncio.CancelledError`.
            result_holder: Optional mutable dict populated with the final result.

        Yields:
            str: Text response chunks (immediate, not buffered).
        """
        llm_messages = self._runtime.build_prompt(self, self._timeline)

        if hasattr(self._runtime, "_llm_caller"):
            stream_src = self._runtime._llm_caller.call_llm_stream(llm_messages)
        else:
            stream_src = self.llm_client.stream(
                llm_messages,
                agent_id=self.object_id,
                agent_type=self.agent_type,
            )

        full_text_parts: list[str] = []
        async with aclosing(stream_src) as chunks:
            async for chunk in chunks:
                if cancel_event.is_set():
                    raise asyncio.CancelledError
                if chunk.type == "text_delta" and chunk.content:
                    full_text_parts.append(chunk.content)
                    yield chunk.content
                elif chunk.type == "response" and chunk.response is not None and result_holder is not None:
                    response = chunk.response
                    result_holder["finish_reason"] = response.finish_reason
                    result_holder["usage"] = response.usage
                    result_holder["response_id"] = response.response_id
                    result_holder["reasoning_tokens"] = response.reasoning_tokens

        if result_holder is not None:
            result_holder["full_text"] = "".join(full_text_parts)
            result_holder["protected_payload"] = None
            result_holder.setdefault("finish_reason", None)

    async def aquery_stream(self, **kwargs) -> AsyncIterator[StreamEvent]:
        """Streaming version of aquery with session management.

        Yields StreamEvent objects: TEXT_DELTA during LLM generation,
        TOOL_CALL_START and TOOL_RESULT during tool execution, then DONE.

        Args:
            **kwargs: Same kwargs as aquery() (message, caller_message, etc.)

        Yields:
            StreamEvent: Stream events throughout the STAR loop.
        """
        # Session management (mirrors aquery())
        # session_id relabels the in-memory session / write target (see
        # set_session_id). Call resume(session_id) beforehand to reload a
        # persisted session from disk — relabel never reloads.
        new_session_id = kwargs.get("session_id")
        if new_session_id is not None:
            self.set_session_id(new_session_id)
        session_id = self._session_id

        self._star_loop_count = 0

        if hasattr(self, "_event_log") and self._event_log is not None:
            self._event_log._current_session_id = session_id

        try:
            async with aclosing(self._run_aquery_stream(kwargs)) as events:
                async for event in events:
                    yield event
        finally:
            # Observation events have no journal counterpart, so the event log is
            # written even for a journal-managed session (when a host wired an
            # observer at all — the session host wires none).
            if hasattr(self, "_event_log") and self._event_log is not None:
                self._event_log.save(session_id)
            # The derived legacy timeline snapshot is only written when this agent
            # owns durable history (see STARAgent._legacy_timeline_persistence_enabled).
            if self._legacy_timeline_persistence_enabled() and hasattr(self, "_timeline") and self._timeline is not None:
                self._timeline.save(session_id)

    async def _run_aquery_stream(self, kwargs: DictParams) -> AsyncIterator[StreamEvent]:
        """Inner streaming STAR loop (no session management).

        Yields StreamEvent objects from the STAR loop iterations.
        """
        from contextlib import aclosing

        trace_inputs: DictParams = {"trace_inputs": kwargs}
        trace_outputs: DictParams = {}

        for _ in range(self.MAX_ITERATIONS):
            try:
                # SEE phase (sync — no streaming needed)
                trace_percepts = self._see(trace_inputs.get("trace_inputs", {}))

                # THINK phase — stream text deltas
                result_holder: dict = {}
                async with aclosing(self._think_stream(trace_percepts.get("trace_percepts", {}), result_holder)) as events:
                    async for event in events:
                        yield event

                # result_holder now contains {"trace_thoughts": ...}
                trace_thoughts_wrapper = result_holder.get("trace_thoughts", {"trace_thoughts": {}})
                trace_thoughts_inner = trace_thoughts_wrapper.get("trace_thoughts", {})
                iteration = self._star_loop_count

                # TOOL_CALL_START already emitted by _think_stream() — no duplicate here

                # ACT phase (async — uses existing _act_async)
                trace_outputs = await self._act_async(trace_thoughts_inner)
                tool_results = trace_outputs.get("trace_outputs", {}).get("tool_results", [])

                # Yield TOOL_RESULT events after execution
                if tool_results:
                    yield StreamEvent(
                        event_type=StreamEventType.TOOL_RESULT,
                        data=tool_results,
                        iteration=iteration,
                    )

                # Reflection mutates agent state. Finish it in this turn rather
                # than leaking work into the next turn or beyond cancellation.
                if not self._do_exit_star_loop(trace_outputs.get("trace_outputs", {})):
                    acquisitive_input = trace_outputs.get("trace_outputs", {}).copy()
                    acquisitive_input["phase"] = LearningPhase.ACQUISITIVE
                    self._reflect(acquisitive_input)

                if self._do_exit_star_loop(trace_outputs.get("trace_outputs", {})):
                    break

                # Prepare next iteration inputs from tool results
                trace_inputs = {"trace_inputs": trace_outputs.get("trace_outputs", {})}

            except Exception as exc:
                from dana.core.llm.llm_caller import is_transient_llm_error

                logger.error(
                    "Provider stream failed",
                    error_type=type(exc).__name__,
                    transient=is_transient_llm_error(exc),
                )
                yield StreamEvent(
                    event_type=StreamEventType.ERROR,
                    data=exc,
                    iteration=self._star_loop_count,
                )
                return

        yield StreamEvent(
            event_type=StreamEventType.DONE,
            data=None,
            iteration=self._star_loop_count,
        )
