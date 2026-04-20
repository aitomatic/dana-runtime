"""
LLM Types and Base Classes

Core types, Protocol definition, and base class for LLM providers.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, runtime_checkable


if TYPE_CHECKING:
    from dana.common.schemas.tool_call import MethodSignature


# ---------------------------------------------------------------------------
# Canonical content block types (path-based, no inline encoding)
# ---------------------------------------------------------------------------

# Canonical internal format:
#   Text:  {"type": "text", "text": "..."}
#   Media: {"type": "image"|"audio"|"video"|"document", "media_type": "image/png", "path": "/abs/path"}
#
# Providers read files and encode as needed in convert_multimodal_content().


class TextBlock(TypedDict):
    """Text content block."""

    type: Literal["text"]
    text: str


class MediaBlock(TypedDict):
    """Media content block — carries file path, not encoding."""

    type: Literal["image", "audio", "video", "document"]
    media_type: str  # MIME type: "image/png", "audio/wav", etc.
    path: str  # absolute path to the file


ContentBlock = TextBlock | MediaBlock

# Block types that carry binary media
MULTIMODAL_TYPES = {"image", "audio", "video", "document"}


def is_multimodal_content(content: str | list[ContentBlock]) -> bool:
    """Check if content contains multimodal (non-text) blocks."""
    if isinstance(content, str):
        return False
    return any(block.get("type") in MULTIMODAL_TYPES for block in content)


def unsupported_placeholder(block: ContentBlock) -> dict:
    """Create a text placeholder for an unsupported media block."""
    block_type = block.get("type", "unknown")
    media_type = block.get("media_type", "")
    return {"type": "text", "text": f"[{block_type} content ({media_type}) not supported by this model]"}


def read_media_as_base64(block: MediaBlock) -> str:
    """Read a media block's file and return base64-encoded string.

    Raises FileNotFoundError if the file no longer exists at conversion time.
    """
    file_path = Path(block["path"])
    if not file_path.is_file():
        raise FileNotFoundError(f"Media file not found: {block['path']}")
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


class LLMError(Exception):
    """Base exception for LLM operations."""

    pass


class ProviderError(LLMError):
    """Exception raised when provider operations fail."""

    pass


class PromptTooLongError(ProviderError):
    """Raised when a provider signals the prompt exceeds its context window.

    Providers map their native token-limit error to this type (Anthropic
    `BadRequestError` with "prompt is too long"; OpenAI-compat
    `APIStatusError` with `error.code='context_length_exceeded'`). Stays
    OUT of LLMCaller `_TRANSIENT_KEYWORDS` so failover never retries it —
    the caller-layer catches, calls `timeline.reactive_compact`, and retries.
    """

    pass


class CompactCircuitOpenError(LLMError):
    """Raised when `reactive_compact` exhausts its retry budget.

    Marks a session's compaction subsystem as temporarily disabled;
    recovery via time-based cooldown + half-open probe.
    """

    pass


class LLMTimeoutError(ProviderError):
    """Exception raised when an LLM API call times out.

    Providers catch SDK-specific timeout exceptions and re-raise as this
    unified type so the retry/failover layer can detect timeouts without
    knowing about individual SDK exception hierarchies.
    """

    pass


class ConfigurationError(LLMError):
    """Exception raised for configuration issues."""

    pass


class EmbeddingNotSupportedError(ProviderError):
    """Raised when a provider does not support embeddings (e.g. Anthropic, Moonshot)."""

    pass


# ---------------------------------------------------------------------------
# Embedding types
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingResponse:
    """Response from an embedding call."""

    embeddings: list[list[float]]
    model: str
    usage: dict[str, int] | None = None
    dimensions: int = 0


@dataclass
class LLMMessage:
    """A single message in a conversation."""

    content: str | list[ContentBlock]
    role: str  # "system", "user", "assistant", "tool"
    cache_control: dict | None = None  # For Anthropic prompt caching
    tool_calls: list | None = None  # For assistant messages with native tool calls
    tool_call_id: str | None = None  # For tool result messages (role="tool")


@dataclass
class SystemLLMMessage(LLMMessage):
    """A system message in a conversation."""

    content: str
    role: str = "system"  # Hard-coded role
    cache_control: dict | None = None  # For Anthropic prompt caching


@dataclass
class UserLLMMessage(LLMMessage):
    """A user message in a conversation."""

    content: str | list[ContentBlock]
    role: str = "user"  # Hard-coded role


@dataclass
class AssistantLLMMessage(LLMMessage):
    """An assistant message in a conversation."""

    content: str
    role: str = "assistant"  # Hard-coded role


@dataclass
class ToolLLMMessage(LLMMessage):
    """A tool result message for native OpenAI tool calling."""

    content: str
    tool_call_id: str
    role: str = "tool"  # Hard-coded role


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    tool_calls: list | None = None  # For function calling support
    reasoning_content: str | None = None  # From providers that expose thinking (DeepSeek, future Claude extended)
    reasoning_tokens: int | None = None  # Token count from OpenAI thinking models


@dataclass
class LLMStreamChunk:
    """A single chunk from a streaming LLM response."""

    type: str  # "text_delta", "tool_use", "thinking"
    content: str = ""
    tool_call: dict | None = None  # {"id": str, "name": str, "input": dict}


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """Structural typing protocol for LLM providers.

    Third-party providers can satisfy this protocol without inheriting from LLMProvider.
    Use isinstance(x, LLMProviderProtocol) for runtime checks.
    """

    @property
    def supports_native_tools(self) -> bool: ...

    def convert_multimodal_content(self, blocks: list[dict]) -> list: ...

    def prepare_messages(self, messages: list[LLMMessage]) -> tuple[Any, list[dict]]:
        """Convert LLMMessage[] to provider wire format.

        Returns: (system_param, messages_list) — system_param is provider-specific.
        """
        ...

    def prepare_tools(self, tools: list[MethodSignature]) -> list[dict]:
        """Convert MethodSignature[] to provider-specific tool schema."""
        ...

    async def chat(self, messages: list[LLMMessage], tools: list | None = None, **kwargs) -> LLMResponse: ...

    async def stream(self, messages: list[LLMMessage], tools: list | None = None, **kwargs): ...


class LLMProvider:
    """Base class for LLM providers. Satisfies LLMProviderProtocol.

    Provides default implementations for prepare_messages() and prepare_tools().
    Subclasses override chat() and optionally the prepare methods.
    """

    # Default timeout in seconds for LLM API calls (2 minutes).
    # Prevents long-running calls (e.g. timeline compression) from blocking the agent loop.
    DEFAULT_TIMEOUT_SECONDS = 360

    @property
    def supports_native_tools(self) -> bool:
        """Whether this provider supports native function/tool calling."""
        return False

    # supports_vision, supports_video, supports_audio are set as instance
    # attributes by LLMResource at runtime (based on env/config).
    # Not @property — must be assignable per-instance.
    # Use getattr(provider, "supports_video", False) for safe access.

    def convert_multimodal_content(self, blocks: list[dict]) -> list:
        """Convert canonical multimodal blocks to provider wire format.

        Base implementation: text passthrough, all media types → unsupported placeholder.
        Subclasses override for provider-specific formats.
        """
        result: list[dict] = []
        for block in blocks:
            btype = block.get("type", "text")
            if btype == "text":
                result.append(block)
            else:
                result.append(unsupported_placeholder(block))
        return result

    def prepare_messages(self, messages: list[LLMMessage]) -> tuple[Any, list[dict]]:
        """Default: extract system message, convert rest to basic dicts."""
        system = None
        converted = []
        for msg in messages:
            # Guard against None content — APIs reject null content
            safe_content = msg.content if msg.content is not None else ""
            # Base class doesn't support multimodal — extract text from list[dict] blocks
            if isinstance(safe_content, list):
                safe_content = " ".join(block.get("text", f"[{block.get('type', 'unknown')} content]") for block in safe_content)
            if msg.role == "system":
                system = safe_content
            elif msg.role == "tool":
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": safe_content,
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
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
                                "arguments": str(tc.get("arguments", {})),
                            },
                        }
                    )
                converted.append(
                    {
                        "role": "assistant",
                        "content": safe_content,
                        "tool_calls": formatted_tool_calls,
                    }
                )
            else:
                converted.append({"role": msg.role, "content": safe_content})
        return system, converted

    def prepare_tools(self, tools: list[MethodSignature]) -> list[dict]:
        """Default: convert MethodSignature[] to OpenAI-compatible tool schema format."""
        result = []
        for sig in tools:
            params: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            for p in sig.parameters:
                prop: dict[str, Any] = {"type": p.type if hasattr(p, "type") else "string"}
                if p.description:
                    prop["description"] = p.description
                params["properties"][p.name] = prop
                if not p.has_default:
                    params["required"].append(p.name)
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": sig.name,
                        "description": sig.description,
                        "parameters": params,
                    },
                }
            )
        return result

    @property
    def supports_embeddings(self) -> bool:
        """Whether this provider supports text embeddings."""
        return False

    async def chat(self, messages: list[LLMMessage], tools: list | None = None, **kwargs) -> LLMResponse:
        """Send messages to the LLM and get a response."""
        raise NotImplementedError

    async def stream(self, messages: list[LLMMessage], tools: list | None = None, **kwargs):
        """Stream LLMStreamChunk from the LLM."""
        raise NotImplementedError

    async def embed(self, text: str, model: str | None = None, **kwargs) -> EmbeddingResponse:
        """Generate embedding for a single text. Raises EmbeddingNotSupportedError if not supported."""
        raise EmbeddingNotSupportedError(f"{self.__class__.__name__} does not support embeddings.")

    async def embed_batch(self, texts: list[str], model: str | None = None, **kwargs) -> EmbeddingResponse:
        """Generate embeddings for multiple texts. Raises EmbeddingNotSupportedError if not supported."""
        raise EmbeddingNotSupportedError(f"{self.__class__.__name__} does not support embeddings.")
