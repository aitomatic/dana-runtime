"""OpenAI-compatible provider base class for OpenAI and Azure."""

import json
import os
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError
import structlog

from ..types import (
    EmbeddingNotSupportedError,
    EmbeddingResponse,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    LLMTimeoutError,
    is_multimodal_content,
    read_media_as_base64,
    unsupported_placeholder,
)


logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# OpenAI multimodal format helpers
# ---------------------------------------------------------------------------


def _extract_audio_format(media_type: str) -> str:
    """Extract audio format from media_type (e.g. 'audio/wav' -> 'wav')."""
    if "/" in media_type:
        fmt = media_type.split("/", 1)[1]
        if fmt in ("mpeg", "mp3"):
            return "mp3"
        return fmt
    return "wav"


# Model families with known parameter restrictions.
# Key is a model prefix matched against the model name.
# "rename" remaps the parameter key (e.g. max_tokens → max_completion_tokens).
MODEL_RESTRICTIONS: dict[str, dict] = {
    "gpt-5": {
        "temperature": {"allowed_values": [1], "default": 1},
        "max_tokens": {"rename": "max_completion_tokens"},
    },
}

# Model prefixes that default to Responses API
RESPONSES_API_PREFIXES = ("gpt-5", "o3-", "o4-", "o3", "o4")

# Valid reasoning effort levels accepted by the OpenAI Responses API.
# "minimal" is gpt-5-only; the SDK rejects unknown values, so validate at the wrapper.
VALID_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})

# Generic fallback env var when no provider-specific override is set.
GENERIC_REASONING_EFFORT_ENV = "LLM_REASONING_EFFORT"


DEFAULT_REASONING_EFFORT = "low"


def _resolve_reasoning_effort(provider_env_var: str | None) -> str:
    """Resolve default reasoning effort from env, with validation.

    Precedence:
      1. provider-specific env var (e.g. ``AZURE_THINKING_EFFORT``)
      2. generic ``LLM_REASONING_EFFORT``
      3. hardcoded ``DEFAULT_REASONING_EFFORT`` ("low" — favors latency/cost;
         operators can opt in to deeper reasoning per provider via env)

    Invalid values are logged and ignored so a typo in env doesn't break calls.
    """
    for env_name in (provider_env_var, GENERIC_REASONING_EFFORT_ENV):
        if not env_name:
            continue
        raw = os.getenv(env_name)
        if not raw:
            continue
        normalized = raw.strip().lower()
        if normalized in VALID_REASONING_EFFORTS:
            return normalized
        logger.warning(
            "ignoring invalid reasoning effort env var",
            env_var=env_name,
            value=raw,
            valid=sorted(VALID_REASONING_EFFORTS),
        )
    return DEFAULT_REASONING_EFFORT


def make_logging_http_client(timeout_seconds: int) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` with request/response hooks.

    The OpenAI SDK retries failed requests internally (default ``max_retries=2``).
    Without these hooks those retry attempts are invisible — a single user-facing
    "Connection error." can hide ~7+ minutes of silent retrying. Logging each
    HTTP request surfaces the retry sequence in normal logs.
    """

    async def _on_request(request: httpx.Request) -> None:
        logger.info("LLM HTTP request", method=request.method, url=str(request.url))

    async def _on_response(response: httpx.Response) -> None:
        if response.status_code >= 400:
            logger.warning(
                "LLM HTTP non-2xx response",
                status=response.status_code,
                url=str(response.request.url),
            )

    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds),
        event_hooks={"request": [_on_request], "response": [_on_response]},
    )


class OpenAICompatibleProvider(LLMProvider):
    """Base for providers using OpenAI-compatible API (OpenAI, Azure)."""

    client: Any  # AsyncOpenAI or AsyncAzureOpenAI
    model: str
    _use_responses_api: bool | None = None
    # Subclasses set this to expose a provider-specific knob, e.g.
    # ``AZURE_THINKING_EFFORT`` or ``OPENAI_THINKING_EFFORT``. Resolved at call
    # time so env changes take effect without process restart in tests.
    _REASONING_EFFORT_ENV_VAR: str | None = None

    def _default_reasoning_effort(self) -> str:
        return _resolve_reasoning_effort(self._REASONING_EFFORT_ENV_VAR)

    @property
    def supports_native_tools(self) -> bool:
        return True

    @staticmethod
    def _get_model_family(model: str) -> str | None:
        """Extract model family from model name for restriction lookup."""
        for family_prefix in MODEL_RESTRICTIONS:
            if model.startswith(family_prefix) and (len(model) == len(family_prefix) or model[len(family_prefix)] in "-_."):
                return family_prefix
        return None

    @staticmethod
    def _filter_params_for_model(model: str, params: dict) -> dict:
        """Filter/adjust parameters based on model-specific restrictions."""
        family = OpenAICompatibleProvider._get_model_family(model)
        if not family:
            return params

        restrictions = MODEL_RESTRICTIONS.get(family, {})
        if not restrictions:
            return params

        filtered = params.copy()
        adjustments_made = []

        for param_name, restriction in restrictions.items():
            if param_name not in filtered:
                continue
            current_value = filtered[param_name]
            # Rename parameter (e.g. max_tokens → max_completion_tokens)
            rename_to = restriction.get("rename")
            if rename_to:
                del filtered[param_name]
                filtered[rename_to] = current_value
                adjustments_made.append(f"{param_name} renamed to {rename_to}")
                continue
            # Remove disallowed values
            allowed_values = restriction.get("allowed_values")
            if allowed_values is not None and current_value not in allowed_values:
                del filtered[param_name]
                adjustments_made.append(f"{param_name}={current_value} removed (model only supports {allowed_values})")

        if adjustments_made:
            logger.debug(
                "Adjusted parameters for model compatibility",
                model=model,
                model_family=family,
                adjustments=adjustments_made,
            )

        return filtered

    def convert_multimodal_content(self, blocks: list[dict]) -> list[dict]:
        """Convert canonical path-based blocks to OpenAI Chat Completions wire format."""
        result: list[dict] = []
        for block in blocks:
            btype = block.get("type", "text")
            if btype == "text":
                result.append(block)
            elif btype == "image":
                if getattr(self, "supports_vision", True):
                    b64 = read_media_as_base64(block)
                    media_type = block["media_type"]
                    url = f"data:{media_type};base64,{b64}"
                    result.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
                else:
                    result.append(unsupported_placeholder(block))
            elif btype == "audio":
                if getattr(self, "supports_audio", False):
                    b64 = read_media_as_base64(block)
                    fmt = _extract_audio_format(block["media_type"])
                    result.append({"type": "input_audio", "input_audio": {"data": b64, "format": fmt}})
                else:
                    result.append(unsupported_placeholder(block))
            elif btype == "video":
                # Standard OpenAI doesn't support video input
                result.append(unsupported_placeholder(block))
            elif btype == "document":
                if getattr(self, "supports_vision", True):
                    b64 = read_media_as_base64(block)
                    media_type = block["media_type"]
                    result.append({"type": "file", "file": {"file_data": f"data:{media_type};base64,{b64}"}})
                else:
                    result.append(unsupported_placeholder(block))
            else:
                result.append(block)
        return result

    def prepare_messages(self, messages: list[LLMMessage]) -> tuple[str | None, list[dict]]:
        """Convert LLMMessage[] to OpenAI API wire format."""
        system = None
        openai_messages = []
        for msg in messages:
            # Guard against None content — Azure/OpenAI APIs reject null content
            safe_content = msg.content if msg.content is not None else ""
            if msg.role == "system":
                system = safe_content
                openai_messages.append({"role": "system", "content": safe_content})
            elif msg.role == "user":
                if isinstance(msg.content, list) and is_multimodal_content(msg.content):
                    converted = self.convert_multimodal_content(msg.content)
                    openai_messages.append({"role": "user", "content": converted})
                else:
                    openai_messages.append({"role": "user", "content": safe_content})
            elif msg.role == "tool":
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": safe_content,
                    }
                )
            elif msg.role == "assistant":
                if msg.tool_calls:
                    formatted_tool_calls = []
                    for tc in msg.tool_calls:
                        tc_id = tc.get("tool_call_id") or tc.get("id", "")
                        tc_name = tc.get("function") or tc.get("name", "")
                        formatted_tool_calls.append(
                            {
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": tc_name,
                                    "arguments": json.dumps(tc.get("arguments", {}))
                                    if isinstance(tc.get("arguments"), dict)
                                    else str(tc.get("arguments", "{}")),
                                },
                            }
                        )
                    openai_messages.append(
                        {
                            "role": "assistant",
                            "content": safe_content,
                            "tool_calls": formatted_tool_calls,
                        }
                    )
                else:
                    openai_messages.append({"role": "assistant", "content": safe_content})
        return system, openai_messages

    def prepare_tools(self, tools) -> list[dict]:
        """Convert tool definitions to OpenAI tool schema format."""
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                result.append(tool)
                continue
            params = {"type": "object", "properties": {}, "required": []}
            for p in tool.parameters:
                prop = {"type": p.type if hasattr(p, "type") else "string"}
                if p.description:
                    prop["description"] = p.description
                params["properties"][p.name] = prop
                if not p.has_default:
                    params["required"].append(p.name)
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": params,
                    },
                }
            )
        return result

    async def chat(self, messages: list[LLMMessage], tools: list[dict] | None = None, **kwargs) -> LLMResponse:
        """Send messages and get a response.

        Routes to the Responses API for reasoning models (gpt-5/o3/o4) when the
        endpoint supports it, so callers get reasoning text in
        ``LLMResponse.reasoning_content``. Falls back to Chat Completions
        otherwise. Mirrors ``stream()`` routing.
        """
        try:
            if self._should_use_responses_api():
                return await self._chat_via_responses(messages, tools, **kwargs)
            return await self._chat_via_chat_completions(messages, tools, **kwargs)
        except (APITimeoutError, httpx.TimeoutException) as e:
            raise LLMTimeoutError(f"OpenAI-compatible API timeout: {e}") from e
        except APIConnectionError as e:
            # Network failure — surface explicitly so it's not buried under a generic
            # "OpenAI-compatible API error" line. Classified transient by LLMCaller.
            logger.error(
                "OpenAI-compatible API connection error",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise
        except Exception as e:
            # Map OpenAI-compat context_length_exceeded to typed PromptTooLongError.
            # Covers OpenAI, Azure, Moonshot uniformly.
            try:
                import openai as _openai

                if isinstance(e, _openai.APIStatusError):
                    from dana.common.llm.types import PromptTooLongError

                    err_body: dict = {}
                    try:
                        resp = getattr(e, "response", None)
                        if resp is not None and hasattr(resp, "json"):
                            err_body = (resp.json() or {}).get("error", {}) or {}
                    except Exception:
                        err_body = {}
                    err_code = err_body.get("code") or getattr(e, "code", "") or ""
                    err_msg = err_body.get("message") or str(e)
                    if err_code == "context_length_exceeded":
                        raise PromptTooLongError(f"OpenAI-compat: {err_msg}") from e
            except ImportError:
                pass
            logger.error("OpenAI-compatible API error", error=str(e), error_type=type(e).__name__, exc_info=True)
            raise

    async def _chat_via_chat_completions(self, messages: list[LLMMessage], tools: list[dict] | None = None, **kwargs) -> LLMResponse:
        """Non-streaming chat via the legacy Chat Completions endpoint.

        Used for non-reasoning models or when the Responses API isn't available
        (e.g. Azure with api-version < 2025-03-01-preview). Reasoning text is not
        surfaced on this path; only ``reasoning_tokens`` count if the API returns it.
        """
        _, openai_messages = self.prepare_messages(messages)

        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["json_mode"] and v is not None}
        filtered_kwargs = self._filter_params_for_model(self.model, filtered_kwargs)

        request_kwargs = {"model": self.model, "messages": openai_messages, **filtered_kwargs}

        if tools:
            request_kwargs["tools"] = self.prepare_tools(tools)
            request_kwargs["tool_choice"] = "auto"

        if kwargs.get("json_mode", False):
            request_kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(
            **request_kwargs,
            timeout=httpx.Timeout(self.DEFAULT_TIMEOUT_SECONDS),
        )

        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        tool_calls = message.tool_calls if (hasattr(message, "tool_calls") and message.tool_calls) else None

        usage = None
        reasoning_tokens = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
                cached = getattr(response.usage.prompt_tokens_details, "cached_tokens", None)
                if cached is not None:
                    usage["cached_tokens"] = cached
            if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                reasoning_tokens = getattr(response.usage.completion_tokens_details, "reasoning_tokens", None) or None

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
            reasoning_tokens=reasoning_tokens,
        )

    async def _chat_via_responses(self, messages: list[LLMMessage], tools: list[dict] | None = None, **kwargs) -> LLMResponse:
        """Non-streaming chat via the Responses API.

        Surfaces reasoning summary text in ``LLMResponse.reasoning_content`` for
        gpt-5/o3/o4 models. Tool calls are returned in the same Pydantic shape
        Chat Completions emits (``ChatCompletionMessageToolCall``) so downstream
        parsers (e.g. ``response_parser._to_tool_call_dicts``) see no difference.
        """
        from openai.types.chat import ChatCompletionMessageToolCall
        from openai.types.chat.chat_completion_message_tool_call import Function

        _, openai_messages = self.prepare_messages(messages)
        responses_input = self._convert_to_responses_input(openai_messages)

        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["json_mode"] and v is not None}
        filtered_kwargs = self._filter_params_for_model(self.model, filtered_kwargs)

        request_kwargs = {"model": self.model, "input": responses_input, **filtered_kwargs}

        # Default reasoning config:
        #   effort  — without explicit effort, gpt-5* sometimes skips reasoning entirely,
        #             making reasoning_content nondeterministic. Resolved from the provider's
        #             env var (e.g. AZURE_THINKING_EFFORT) → LLM_REASONING_EFFORT → "low".
        #             Caller-supplied reasoning.effort always wins.
        #   summary="auto"  — required for reasoning summary text to be returned at all.
        reasoning_cfg = dict(request_kwargs.get("reasoning") or {})
        reasoning_cfg.setdefault("effort", self._default_reasoning_effort())
        reasoning_cfg.setdefault("summary", "auto")
        request_kwargs["reasoning"] = reasoning_cfg

        if tools:
            request_kwargs["tools"] = self._prepare_tools_for_responses(tools)

        if kwargs.get("json_mode", False):
            # Responses API uses text.format instead of response_format.
            request_kwargs["text"] = {"format": {"type": "json_object"}}

        response = await self.client.responses.create(
            **request_kwargs,
            timeout=httpx.Timeout(self.DEFAULT_TIMEOUT_SECONDS),
        )

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_list: list = []

        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "reasoning":
                for s in item.summary or []:
                    text = getattr(s, "text", None)
                    if text:
                        reasoning_parts.append(text)
            elif item_type == "message":
                for c in item.content or []:
                    if getattr(c, "type", None) == "output_text":
                        text = getattr(c, "text", "") or ""
                        if text:
                            content_parts.append(text)
            elif item_type == "function_call":
                tool_calls_list.append(
                    ChatCompletionMessageToolCall(
                        id=item.call_id,
                        type="function",
                        function=Function(name=item.name, arguments=item.arguments or ""),
                    )
                )

        # Map Responses API status to a Chat-Completions-style finish_reason so
        # callers don't need to know which path was used.
        if tool_calls_list:
            finish_reason = "tool_calls"
        elif response.status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) if details else None
            finish_reason = "length" if reason == "max_output_tokens" else "incomplete"
        else:
            finish_reason = "stop"

        usage = None
        reasoning_tokens = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            input_details = getattr(response.usage, "input_tokens_details", None)
            if input_details:
                cached = getattr(input_details, "cached_tokens", None)
                if cached is not None:
                    usage["cached_tokens"] = cached
            output_details = getattr(response.usage, "output_tokens_details", None)
            if output_details:
                reasoning_tokens = getattr(output_details, "reasoning_tokens", None) or None

        return LLMResponse(
            content="".join(content_parts),
            model=response.model,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls_list or None,
            reasoning_tokens=reasoning_tokens,
            reasoning_content="".join(reasoning_parts) or None,
        )

    # --- Embedding methods ---

    # Subclasses that support embeddings set this to the default model name.
    embedding_model: str | None = None

    @property
    def supports_embeddings(self) -> bool:
        return self.embedding_model is not None

    async def embed(self, text: str, model: str | None = None, **kwargs) -> EmbeddingResponse:
        """Generate embedding for a single text."""
        return await self.embed_batch([text], model=model, **kwargs)

    async def embed_batch(self, texts: list[str], model: str | None = None, **kwargs) -> EmbeddingResponse:
        """Generate embeddings for multiple texts using OpenAI embeddings API."""
        if not self.supports_embeddings:
            raise EmbeddingNotSupportedError(f"{self.__class__.__name__} does not support embeddings.")

        embed_model = model or self.embedding_model or "text-embedding-3-small"
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=embed_model,
                **kwargs,
            )
            embeddings = [item.embedding for item in response.data]
            dimensions = len(embeddings[0]) if embeddings else 0
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return EmbeddingResponse(
                embeddings=embeddings,
                model=response.model,
                usage=usage,
                dimensions=dimensions,
            )
        except Exception as e:
            logger.error("Embedding API error", error=str(e), model=embed_model)
            raise

    # --- Streaming methods (Phases 2-4) ---

    async def _stream_chat_completions(self, messages: list[LLMMessage], tools: list | None = None, **kwargs):
        """Stream via client.chat.completions.create(stream=True).

        Yields LLMStreamChunk. Accumulates incremental tool call deltas.
        """
        _, openai_messages = self.prepare_messages(messages)

        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["json_mode"] and v is not None}
        filtered_kwargs = self._filter_params_for_model(self.model, filtered_kwargs)

        request_kwargs = {"model": self.model, "messages": openai_messages, "stream": True, **filtered_kwargs}

        if tools:
            request_kwargs["tools"] = self.prepare_tools(tools)
            request_kwargs["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(
            **request_kwargs,
            timeout=httpx.Timeout(self.DEFAULT_TIMEOUT_SECONDS),
        )

        tool_calls: dict[int, dict] = {}

        async for chunk in response:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta and delta.content:
                yield LLMStreamChunk(type="text_delta", content=delta.content)

            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls[idx]["arguments"] += tc_delta.function.arguments

            if choice.finish_reason == "tool_calls":
                for tc in tool_calls.values():
                    try:
                        parsed_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except json.JSONDecodeError:
                        parsed_args = {}
                    yield LLMStreamChunk(
                        type="tool_use",
                        tool_call={"id": tc["id"], "name": tc["name"], "input": parsed_args},
                    )
                tool_calls.clear()

        # Post-loop flush: yield any remaining tool calls not yet emitted
        if tool_calls:
            for tc in tool_calls.values():
                try:
                    parsed_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    parsed_args = {}
                yield LLMStreamChunk(
                    type="tool_use",
                    tool_call={"id": tc["id"], "name": tc["name"], "input": parsed_args},
                )

    def _prepare_tools_for_responses(self, tools: list) -> list[dict]:
        """Convert tools to Responses API format (flat, no function wrapper)."""
        chat_tools = self.prepare_tools(tools)
        return [
            {
                "type": "function",
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "parameters": t["function"].get("parameters", {}),
            }
            for t in chat_tools
        ]

    def _convert_to_responses_input(self, openai_messages: list[dict]) -> list[dict]:
        """Convert Chat Completions wire messages to Responses API input format.

        Chat Completions uses:
          {"role": "assistant", "tool_calls": [...]}
          {"role": "tool", "tool_call_id": "...", "content": "..."}

        Responses API uses:
          {"type": "function_call", "id": "...", "call_id": "...", "name": "...", "arguments": "...", "status": "completed"}
          {"type": "function_call_output", "call_id": "...", "output": "..."}
        """
        result = []
        for msg in openai_messages:
            role = msg.get("role")
            # Convert multimodal user messages to Responses API format
            if role == "user" and isinstance(msg.get("content"), list):
                content = msg["content"]
                # Check if content has OpenAI Chat format blocks (image_url, input_audio)
                # and convert to Responses API format (input_image, input_text)
                has_openai_blocks = any(b.get("type") in ("image_url", "input_audio", "file") for b in content if isinstance(b, dict))
                if has_openai_blocks:
                    responses_content = []
                    for block in content:
                        btype = block.get("type", "")
                        if btype == "text":
                            responses_content.append({"type": "input_text", "text": block.get("text", "")})
                        elif btype == "image_url":
                            url = block.get("image_url", {}).get("url", "")
                            responses_content.append({"type": "input_image", "image_url": url})
                        else:
                            responses_content.append(block)
                    result.append({"role": "user", "content": responses_content})
                else:
                    result.append(msg)
                continue
            if role == "assistant" and msg.get("tool_calls"):
                # Emit any text content as a regular assistant message
                if msg.get("content"):
                    result.append({"role": "assistant", "content": msg["content"]})
                # Convert each tool call to a function_call item
                for tc in msg["tool_calls"]:
                    call_id = tc.get("id", "")
                    func = tc.get("function", {})
                    # Responses API requires 'id' to start with 'fc_'
                    fc_id = call_id if call_id.startswith("fc_") else f"fc_{call_id}"
                    result.append(
                        {
                            "type": "function_call",
                            "id": fc_id,
                            "call_id": call_id,
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", "{}"),
                            "status": "completed",
                        }
                    )
            elif role == "tool":
                result.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", ""),
                        "output": msg.get("content", ""),
                    }
                )
            else:
                result.append(msg)
        return result

    async def _stream_responses(self, messages: list[LLMMessage], tools: list | None = None, **kwargs):
        """Stream via client.responses.create(stream=True).

        For gpt-5+, o3+, o4+ models. Tool calls arrive complete.
        """
        _, openai_messages = self.prepare_messages(messages)
        responses_input = self._convert_to_responses_input(openai_messages)

        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["json_mode"] and v is not None}
        filtered_kwargs = self._filter_params_for_model(self.model, filtered_kwargs)

        request_kwargs = {"model": self.model, "input": responses_input, "stream": True, **filtered_kwargs}

        # Default reasoning config (mirrors _chat_via_responses):
        #   effort  — env-driven default (provider env → LLM_REASONING_EFFORT → "low")
        #             so gpt-5* deterministically reasons. Caller can override per-call.
        #   summary="auto"  — required for reasoning summary delta events to fire.
        reasoning_cfg = dict(request_kwargs.get("reasoning") or {})
        reasoning_cfg.setdefault("effort", self._default_reasoning_effort())
        reasoning_cfg.setdefault("summary", "auto")
        request_kwargs["reasoning"] = reasoning_cfg

        if tools:
            request_kwargs["tools"] = self._prepare_tools_for_responses(tools)

        stream = await self.client.responses.create(
            **request_kwargs,
            timeout=httpx.Timeout(self.DEFAULT_TIMEOUT_SECONDS),
        )

        async for event in stream:
            if event.type == "response.output_text.delta":
                yield LLMStreamChunk(type="text_delta", content=event.delta)

            elif event.type == "response.output_item.done":
                item = event.item
                if getattr(item, "type", None) == "function_call":
                    try:
                        parsed_args = json.loads(item.arguments) if item.arguments else {}
                    except json.JSONDecodeError:
                        parsed_args = {}
                    yield LLMStreamChunk(
                        type="tool_use",
                        tool_call={
                            "id": item.call_id,
                            "name": item.name,
                            "input": parsed_args,
                        },
                    )

            elif event.type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                # Both summary deltas (when reasoning.summary="auto") and raw reasoning
                # text deltas (trusted access) carry .delta strings; surface as "thinking".
                yield LLMStreamChunk(type="thinking", content=event.delta)

    def _responses_api_supported(self) -> bool:
        """Whether the underlying endpoint exposes the Responses API at all.

        OpenAI-compatible endpoints support it unconditionally. Azure subclasses
        override this to gate on api-version (Responses API requires
        api-version >= 2025-03-01-preview).
        """
        return True

    def _should_use_responses_api(self) -> bool:
        """Determine whether to use Responses API or Chat Completions.

        Priority: explicit config flag > endpoint capability + model prefix.
        """
        if self._use_responses_api is not None:
            return self._use_responses_api

        if not self._responses_api_supported():
            return False

        model_lower = self.model.lower()
        for prefix in RESPONSES_API_PREFIXES:
            if model_lower.startswith(prefix):
                return True

        return False

    async def stream(self, messages: list[LLMMessage], tools: list | None = None, **kwargs):
        """Stream LLMStreamChunk, auto-selecting API based on model/config.

        Routes to Responses API for gpt-5+/o3+/o4+ models (or when configured),
        falls back to Chat Completions API otherwise.
        """
        try:
            if self._should_use_responses_api():
                logger.debug("Using Responses API for streaming", model=self.model)
                async for chunk in self._stream_responses(messages, tools=tools, **kwargs):
                    yield chunk
            else:
                logger.debug("Using Chat Completions API for streaming", model=self.model)
                async for chunk in self._stream_chat_completions(messages, tools=tools, **kwargs):
                    yield chunk
        except (APITimeoutError, httpx.TimeoutException) as e:
            raise LLMTimeoutError(f"OpenAI-compatible stream timeout: {e}") from e
        except APIConnectionError as e:
            logger.error(
                "OpenAI-compatible stream connection error",
                model=self.model,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error("Stream error", model=self.model, error=str(e), error_type=type(e).__name__, exc_info=True)
            raise
