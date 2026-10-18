#!/usr/bin/env python3
"""
Adana LLM Library - Minimal API for LLM Integration

A simple, clean interface for interacting with any LLM provider.
Follows KISS principle with just the essential methods most clients need.
"""

try:
    import structlog
except ModuleNotFoundError:
    import logging

    class _StructLogShim:
        @staticmethod
        def get_logger() -> logging.Logger:
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger("dana")

    structlog = _StructLogShim()

from ..config import config_manager
from .providers.factory import create_provider
from .types import LLMError, LLMMessage, LLMProvider, LLMResponse, LLMTimeoutError, ProviderError


logger = structlog.get_logger()

import asyncio
import atexit
import os
import time

from dana.common.observable import observable

from .debug_logger import get_debug_logger


debug_logger = get_debug_logger()

# Sync facade ceiling. The provider SDK itself applies DEFAULT_TIMEOUT_SECONDS
# (360s) per HTTP attempt. The persistent sync loop has no other upper bound, so
# a wedged connection would otherwise block the calling thread forever. Env var
# ``DANA_LLM_SYNC_TIMEOUT_SECONDS`` overrides; 0 disables the ceiling.
DEFAULT_SYNC_CALL_TIMEOUT_SECONDS = 600

# Module-level event loop for sync operations
# This avoids "Event loop is closed" errors when async HTTP clients (httpx/anthropic/openai)
# try to close connections after asyncio.run() closes the loop
_sync_event_loop: asyncio.AbstractEventLoop | None = None


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for sync operations.

    This reuses the same event loop across multiple sync calls to avoid
    the "Event loop is closed" error that occurs when async HTTP clients
    try to clean up after asyncio.run() closes the loop.
    """
    global _sync_event_loop
    if _sync_event_loop is None or _sync_event_loop.is_closed():
        _sync_event_loop = asyncio.new_event_loop()
        # Don't set as the default loop to avoid interfering with other async code
    return _sync_event_loop


def _sync_call_timeout_seconds() -> float:
    """Resolve the sync-facade ceiling from env; invalid values fall back."""
    raw = os.getenv("DANA_LLM_SYNC_TIMEOUT_SECONDS")
    if raw is None or not raw.strip():
        return float(DEFAULT_SYNC_CALL_TIMEOUT_SECONDS)
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "ignoring invalid DANA_LLM_SYNC_TIMEOUT_SECONDS",
            value=raw,
            fallback=DEFAULT_SYNC_CALL_TIMEOUT_SECONDS,
        )
        return float(DEFAULT_SYNC_CALL_TIMEOUT_SECONDS)


def _cleanup_event_loop():
    """Clean up the persistent event loop on exit."""
    global _sync_event_loop
    if _sync_event_loop is not None and not _sync_event_loop.is_closed():
        try:
            # Cancel all pending tasks
            pending = asyncio.all_tasks(_sync_event_loop)
            for task in pending:
                task.cancel()
            # Run until all tasks are cancelled
            if pending:
                _sync_event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            _sync_event_loop.close()
        except Exception:
            pass  # Ignore errors during cleanup


def _log_close_failure(task: asyncio.Task) -> None:
    """Retrieve a scheduled provider-close result so failures are visible, not warnings."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("llm_provider_close_failed", error_type=type(exc).__name__)


# Register cleanup handler
atexit.register(_cleanup_event_loop)


class LLM:
    """
    Stateless LLM interface - KISS principle applied.

    This LLM class is stateless - it does not maintain conversation history.
    The caller is responsible for managing conversation context.

    Essential methods:
    - chat(messages) - for conversations with full context
    - ask(question, system_prompt) - for single questions
    - stream(messages) - for streaming responses
    - switch_provider() - to change LLM provider

    Usage Examples:

    1. Single Question:
       llm = LLM(provider="openai", model="gpt-4")
       answer = await llm.ask("What is 2+2?")

    2. Conversation with Context:
       llm = LLM(provider="anthropic", model="claude-3-sonnet")
       messages = [
           LLMMessage(role="system", content="You are a helpful assistant"),
           LLMMessage(role="user", content="Hello"),
           LLMMessage(role="assistant", content="Hi there!"),
           LLMMessage(role="user", content="How are you?")
       ]
       response = await llm.chat(messages)

    3. Quick One-off Questions:
       answer = await LLM.ask("Hello", provider="openai", model="gpt-4-turbo")

    4. Check Available Models:
       models = LLM.get_provider_models("openai")
       print(models)  # {'gpt-3.5-turbo': 'gpt-3.5-turbo', 'gpt-4': 'gpt-4', ...}

    Available Models (from config):
    - OpenAI: gpt-3.5-turbo, gpt-4, gpt-4-turbo
    - Anthropic: claude-3-haiku, claude-3-sonnet, claude-3-opus
    - Groq: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768
    - Ollama: llama2, codellama, mistral (and any locally installed models)
    - Moonshot: moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k
    - Hugging Face: microsoft/DialoGPT-medium, facebook/blenderbot-400M-distill, google/flan-t5-large
    - Qwen: qwen-turbo, qwen-plus, qwen-max, qwen-long
    - DeepSeek: deepseek-chat, deepseek-coder, deepseek-coder-6.7b, deepseek-coder-33b
    - OpenRouter: Multiple providers (OpenAI, Anthropic, Meta, Mistral, Google, Qwen, DeepSeek, etc.)
    """

    def __init__(self, provider: str | LLMProvider | None = None, model: str | None = None):
        """
        Initialize LLM with a provider.

        Args:
            provider: Provider name ('openai', 'anthropic', 'ollama') or provider instance
            model: Model name to use (defaults to provider's default)
        """
        self._close_tasks: set[asyncio.Task[None]] = set()
        #: Whether the pool behind :attr:`provider` belongs to somebody else, in
        #: which case a host that holds this facade must not close it.
        #:
        #: ``False`` means this facade is responsible for the pool it currently
        #: points at, so ``close``/``aclose`` release it. The marker always
        #: describes the pool the facade points at *now*, and it is re-written at
        #: both points where that pool changes: :meth:`switch_provider` (the
        #: replacement pool was created here, so it is this facade's) and a host
        #: that installs the facade into its own lifecycle with a different
        #: ownership (``STARAgent._install_llm_client``: an injected instance is
        #: borrowed, an adopted or name-built client is the host's).
        self.provider_borrowed = False
        if isinstance(provider, str):
            # Create provider by name
            self.provider = create_provider(provider, model=model)
            self.provider_name = provider
        elif isinstance(provider, LLMProvider):
            # Use provided instance
            self.provider = provider
            self.provider_name = "custom"
        else:
            # Auto-select first available provider by priority
            first_provider = config_manager.get_first_available_provider()
            if first_provider:
                self.provider = create_provider(first_provider, model=model)
                self.provider_name = first_provider
            else:
                # Fallback to OpenAI if no providers available
                self.provider = create_provider("openai", model=model)
                self.provider_name = "openai"

        self.model = getattr(self.provider, "model", "unknown")
        self.provider.supports_vision = self._resolve_vision_support()
        self.provider.supports_video = self._resolve_capability("video")
        self.provider.supports_audio = self._resolve_capability("audio")

    def _resolve_vision_support(self) -> bool:
        """Resolve vision support: env var > config vision_models > False."""
        return self._resolve_capability("vision")

    def _resolve_capability(self, capability: str) -> bool:
        """Resolve a multimodal capability: env var > config {capability}_models > False.

        Args:
            capability: One of 'vision', 'video', 'audio'.
        """
        env_key = f"{self.provider_name.upper()}_SUPPORTS_{capability.upper()}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes")

        config = config_manager.get_provider_config(self.provider_name)
        if config:
            model_list = config.get(f"{capability}_models", [])
            if model_list:
                # Match against both model keys (aliases) and values (full IDs)
                models_map = config.get("models", {})
                ids = set()
                for m in model_list:
                    ids.add(models_map[m] if m in models_map else m)
                return self.model in ids

        return False

    @property
    def supports_vision(self) -> bool:
        return getattr(self.provider, "supports_vision", False)

    @property
    def supports_video(self) -> bool:
        return getattr(self.provider, "supports_video", False)

    @property
    def supports_audio(self) -> bool:
        return getattr(self.provider, "supports_audio", False)

    async def chat(self, messages: list[LLMMessage], **kwargs) -> str:
        """
        Send messages and get a response.

        Args:
            messages: List of LLMMessage objects representing the full conversation context
            **kwargs: Additional parameters for the LLM call

        Returns:
            The response content as a string

        Raises:
            ValueError: If messages list is empty
            ProviderError: If the provider operation fails
        """
        response = await self.chat_response(messages, **kwargs)
        return response.content

    async def chat_response(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        """
        Send messages and get a full LLM response with metadata.

        Args:
            messages: List of LLMMessage objects representing the full conversation context
            **kwargs: Additional parameters for the LLM call

        Returns:
            Full LLMResponse object with content, tool_calls, usage, etc.

        Raises:
            ValueError: If messages list is empty
            ProviderError: If the provider operation fails
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        # Debug logging - log the request
        request_id = debug_logger.log_request(provider=self.provider_name, model=self.model, messages=messages, **kwargs)
        start_time = time.time()

        try:
            # Create agent label for logging if agent info is available
            agent_label = ""
            if "agent_id" in kwargs and "agent_type" in kwargs:
                short_id = kwargs["agent_id"][:8] + "..." if len(kwargs["agent_id"]) > 8 else kwargs["agent_id"]
                agent_label = f" [{kwargs['agent_type']}({short_id})]"

            # Filter out agent parameters before passing to provider
            provider_kwargs = {k: v for k, v in kwargs.items() if k not in ["agent_id", "agent_type"]}

            logger.info("Starting chat", provider=self.provider_name, message_count=len(messages), agent=agent_label)
            response = await self.provider.chat(messages, **provider_kwargs)
            logger.info("Chat completed", provider=self.provider_name, response_length=len(response.content), agent=agent_label)

            # Debug logging - log the response
            duration_ms = (time.time() - start_time) * 1000
            debug_logger.log_response(
                request_id=request_id, response=response, provider=self.provider_name, model=self.model, duration_ms=duration_ms
            )
            # Log full response content for detailed debugging
            if response.content:
                debug_logger.log_full_response_content(
                    request_id=request_id, content=response.content, provider=self.provider_name, model=self.model
                )

            return response
        except Exception as e:
            # Create agent label for error logging if agent info is available
            agent_label = ""
            if "agent_id" in kwargs and "agent_type" in kwargs:
                short_id = kwargs["agent_id"][:8] + "..." if len(kwargs["agent_id"]) > 8 else kwargs["agent_id"]
                agent_label = f" [{kwargs['agent_type']}({short_id})]"

            logger.error("Chat failed", provider=self.provider_name, error=str(e), agent=agent_label)

            # Debug logging - log the error
            debug_logger.log_error(error=e, context="chat_response", request_id=request_id, provider=self.provider_name, model=self.model)

            # Typed pass-through: LLMError subclasses (ProviderError, PromptTooLongError,
            # LLMTimeoutError, CompactCircuitOpenError, ConfigurationError, ...) carry
            # control-flow meaning for the caller layer — retry classification, reactive
            # compaction, failover. Re-wrapping them as a generic ProviderError destroys
            # the type and silently disables all three. Unknown exceptions still get the
            # documented ProviderError contract.
            if isinstance(e, LLMError):
                raise
            raise ProviderError(f"Chat failed with {self.provider_name}: {e}") from e

    @observable
    def chat_response_sync(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        """
        Synchronous version of chat_response - runs the async version in an event loop.

        Args:
            messages: List of LLMMessage objects representing the full conversation context
            **kwargs: Additional parameters for the LLM call

        Returns:
            Full LLMResponse object with content, tool_calls, usage, etc.

        Raises:
            ValueError: If messages list is empty
            ProviderError: If the provider operation fails
        """
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, but this is a sync method
            # This should not normally happen, but if it does, raise an error
            raise RuntimeError("chat_response_sync() cannot be called from within an async context. Use chat_response() instead.")
        except RuntimeError:
            # No running loop, which is expected for sync method
            pass

        # Reuse a persistent event loop to avoid "Event loop is closed" errors
        # when async HTTP clients (httpx) try to close connections after asyncio.run() closes the loop
        loop = _get_or_create_event_loop()
        # Caller-supplied per-attempt budget (LLMCaller) wins; otherwise the facade's
        # own ceiling applies. Either way the sync thread cannot be blocked forever.
        ceiling = kwargs.get("timeout") or _sync_call_timeout_seconds()
        if ceiling <= 0:
            return loop.run_until_complete(self.chat_response(messages, **kwargs))
        try:
            return loop.run_until_complete(asyncio.wait_for(self.chat_response(messages, **kwargs), timeout=ceiling))
        except TimeoutError as e:
            # Thread-safe ceiling: asyncio.wait_for cancels the pending task on the
            # owning loop, which is what propagates cancellation into the SDK request.
            raise LLMTimeoutError(f"{self.provider_name}: sync chat exceeded {ceiling}s ceiling") from e

    async def ask(self, question: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Ask a single question and get an answer.

        Args:
            question: The question to ask
            system_prompt: Optional system prompt to set context
            **kwargs: Additional parameters for the LLM call

        Returns:
            The answer as a string

        Raises:
            ValueError: If question is empty
            ProviderError: If the provider operation fails
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        messages = []

        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))

        messages.append(LLMMessage(role="user", content=question))

        return await self.chat(messages, **kwargs)

    async def stream(self, messages: list[LLMMessage], tools: list | None = None, **kwargs):
        """Stream LLMStreamChunk from the LLM provider.

        Args:
            messages: Conversation context.
            tools: Optional tool definitions for native tool calling.
            **kwargs: Additional parameters for the LLM call.

        Yields:
            LLMStreamChunk with type, content, and optional tool_call.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        try:
            agent_label = ""
            if "agent_id" in kwargs and "agent_type" in kwargs:
                short_id = kwargs["agent_id"][:8] + "..." if len(kwargs["agent_id"]) > 8 else kwargs["agent_id"]
                agent_label = f" [{kwargs['agent_type']}({short_id})]"

            provider_kwargs = {k: v for k, v in kwargs.items() if k not in ["agent_id", "agent_type"]}

            logger.info("Starting stream", provider=self.provider_name, message_count=len(messages), agent=agent_label)
            async for chunk in self.provider.stream(messages, tools=tools, **provider_kwargs):
                yield chunk  # LLMStreamChunk passthrough
            logger.info("Stream completed", provider=self.provider_name, agent=agent_label)
        except Exception as e:
            agent_label = ""
            if "agent_id" in kwargs and "agent_type" in kwargs:
                short_id = kwargs["agent_id"][:8] + "..." if len(kwargs["agent_id"]) > 8 else kwargs["agent_id"]
                agent_label = f" [{kwargs['agent_type']}({short_id})]"

            logger.error("Stream failed", provider=self.provider_name, error=str(e), agent=agent_label)
            # Typed pass-through — see chat_response. A stream that fails after the
            # first yielded chunk must surface the provider's own type so the caller
            # can distinguish "no output yet, safe to retry" from "output already
            # delivered, must not re-request".
            if isinstance(e, LLMError):
                raise
            raise ProviderError(f"Stream failed with {self.provider_name}: {e}") from e

    def switch_provider(self, provider: str, model: str | None = None):
        """
        Switch to a different LLM provider.

        The retired provider owns an SDK/HTTP client that must not outlive its
        selection — otherwise every switch leaks a connection pool (and the
        abandoned client may still be mid-request on another task's loop). The
        old client is closed best-effort on the loop that can drive it; the new
        provider is installed regardless, so a failing close cannot block the
        switch.

        A *borrowed* pool is the one exception: a host that installed this facade
        with ``provider_borrowed = True`` told it that somebody else — the caller
        who injected the instance — owns that pool's lifetime, so retiring it
        would end a pool still in use elsewhere. Ownership of the replacement is
        unaffected: this facade created it, so it is this facade's to close.

        Args:
            provider: Provider name ('openai', 'anthropic', 'ollama', etc.)
            model: Model to use (defaults to provider's default)
        """
        previous = self.provider
        previous_borrowed = self.provider_borrowed
        self.provider = create_provider(provider, model=model)
        self.provider_name = provider
        # The replacement pool is created by *this* facade, so ownership moves to
        # the facade even when the wrapper was originally built around a borrowed
        # instance. Without this, a wrapper installed as borrowed by a host would
        # keep reporting the old, caller-owned pool's status and the new pool would
        # never be released at teardown.
        self.provider_borrowed = False
        self.model = getattr(self.provider, "model", "unknown")
        self.provider.supports_vision = self._resolve_capability("vision")
        self.provider.supports_video = self._resolve_capability("video")
        self.provider.supports_audio = self._resolve_capability("audio")
        if not previous_borrowed:
            self._close_provider(previous)
        logger.info("Switched LLM provider", provider=provider, model=self.model)

    async def aclose(self) -> None:
        """Close this LLM's provider client on the current (owning) event loop.

        Call from session teardown for the loop that constructed the client —
        clients are loop-affine, so closing a client from a different loop can
        raise. Prefer this over :meth:`close` inside async code.
        """
        try:
            await self.provider.aclose()
        finally:
            await self.await_pending_cleanup()

    async def await_pending_cleanup(self) -> None:
        """Wait for cleanup this facade already scheduled on the running loop.

        This is the barrier :meth:`close` cannot be: with a loop already running,
        ``close`` only *schedules* the close, so a later async teardown is the only
        place that can turn "cleanup started" into "cleanup finished".

        Awaiting is not closing: it never touches :attr:`provider`, so a host may
        call it on a facade whose pool it does not own (a borrowed client) to make
        sure cleanup that facade merely carries — for example the close scheduled for
        a pool the host itself retired — has actually finished before teardown
        returns.
        """
        pending = tuple(self._close_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def take_over_pending_cleanup(self, *retired) -> None:
        """Adopt cleanup that ``retired`` facades already scheduled on this loop.

        Each facade schedules an in-flight close on its *own* task set, and only
        :meth:`aclose` awaits it. A host that retires facade A in favour of facade B
        holds only B at shutdown, so A's pending cleanup would be orphaned and
        cancelled by loop shutdown. Moving the tasks onto the facade the host still
        holds keeps teardown a completion barrier for every pool it retired, without
        closing anything that is still live.
        """
        for facade in retired:
            if facade is None or facade is self:
                continue
            incoming = getattr(facade, "_close_tasks", None)
            if not incoming:
                continue
            for task in tuple(incoming):
                incoming.discard(task)
                task.add_done_callback(self._close_tasks.discard)
                self._close_tasks.add(task)

    def close(self) -> None:
        """Synchronously close this LLM's provider client.

        For sync callers: uses the persistent sync loop when no loop is running,
        otherwise schedules the close on the running loop. Best-effort — a
        teardown failure is logged, never raised, so it cannot mask a turn result.
        """
        self._close_provider(self.provider)

    def _close_provider(self, provider: LLMProvider | None) -> None:
        """Best-effort close of a provider client, loop-affine aware."""
        if provider is None:
            return
        try:
            running: asyncio.AbstractEventLoop | None
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None

            if running is not None:
                # Inside async code: schedule and attach a done-callback so the
                # exception is retrieved instead of surfacing as "never awaited".
                task = running.create_task(provider.aclose())
                self._close_tasks.add(task)
                task.add_done_callback(self._close_tasks.discard)
                task.add_done_callback(_log_close_failure)
                return

            loop = _get_or_create_event_loop()
            loop.run_until_complete(provider.aclose())
        except Exception as e:
            logger.warning(
                "llm_provider_close_failed",
                provider=getattr(provider, "name", provider.__class__.__name__),
                error_type=type(e).__name__,
            )

    @staticmethod
    async def ask_question(question: str, provider: str | None = None, model: str | None = None, **kwargs) -> str:
        """
        Quick static method to ask a single question.

        Args:
            question: The question to ask
            provider: Provider to use (defaults to first available by priority)
            model: Model to use (defaults to provider's default)
            **kwargs: Additional parameters

        Returns:
            The answer as a string
        """
        llm = LLM(provider=provider, model=model)
        return await llm.ask(question, **kwargs)

    @staticmethod
    def get_available_providers() -> list[str]:
        """Get list of available providers."""
        return config_manager.get_available_providers()

    @staticmethod
    def is_provider_available(provider: str) -> bool:
        """Check if a provider is available."""
        return config_manager.is_provider_available(provider)

    @staticmethod
    def get_provider_models(provider: str) -> dict[str, str]:
        """Get available models for a provider."""
        return config_manager.get_provider_models(provider)

    @staticmethod
    def show_config_documentation():
        """Display configuration information from adana/config.json."""
        import json
        from pathlib import Path

        # Load the config file
        config_path = Path(__file__).parent.parent.parent / "config.json"

        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return

        print("📚 Adana LLM Configuration")
        print("=" * 50)

        # Required environment variables (from config)
        print("\n🔑 Required Environment Variables:")
        print("-" * 30)
        providers = config.get("llm", {}).get("providers", {})
        for provider, info in providers.items():
            api_key_env = info.get("api_key_env")
            if api_key_env:
                name = info.get("name", provider)
                print(f"  {api_key_env:<25} - {name} API key")

        # Optional environment variables (from config)
        print("\n⚙️  Optional Environment Variables:")
        print("-" * 30)
        for provider, info in providers.items():
            base_url_env = info.get("base_url_env")
            if base_url_env:
                name = info.get("name", provider)
                print(f"  {base_url_env:<25} - Override {name} endpoint")

            api_version_env = info.get("api_version_env")
            if api_version_env:
                name = info.get("name", provider)
                print(f"  {api_version_env:<25} - Override {name} API version")

        # Show available providers
        print("\n🤖 Available Providers:")
        print("-" * 30)
        for provider, info in providers.items():
            name = info.get("name", provider)
            priority = info.get("priority", 0)
            api_key_env = info.get("api_key_env")
            if api_key_env:
                print(f"  {provider:<12} - {name} (priority: {priority}, requires {api_key_env})")
            else:
                print(f"  {provider:<12} - {name} (priority: {priority}, no API key required)")

        print("\n💡 Example .env file:")
        print("-" * 30)
        print("# Required API Keys")
        for _provider, info in providers.items():
            api_key_env = info.get("api_key_env")
            if api_key_env:
                print(f"# {api_key_env}=your-{api_key_env.lower().replace('_', '-')}-here")

        print("\n# Optional: Custom endpoints and settings")
        for _provider, info in providers.items():
            base_url_env = info.get("base_url_env")
            if base_url_env:
                print(f"# {base_url_env}=your-custom-endpoint")

            api_version_env = info.get("api_version_env")
            if api_version_env:
                print(f"# {api_version_env}=your-api-version")

        print("\n🎯 Quick Start:")
        print("-" * 30)
        print("1. Set at least one API key from the required list above")
        print("2. For Azure: Update the base_url in adana/config.json")
        print("3. For Ollama: Set OLLAMA_API_KEY='ollama' to enable local Ollama")
        print("4. Run: python examples/llm_example.py")
