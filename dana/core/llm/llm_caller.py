"""
LLMCaller — wraps the LLM invocation logic extracted from AgentRuntime.

Responsible for: lazy LLM resolution, sync and async chat calls,
retry with exponential backoff, and fallback provider support (Phase 7).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Any

import structlog

from dana.common.llm.llm import LLM
from dana.common.llm.types import (
    CompactCircuitOpenError,
    ConfigurationError,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    LLMTimeoutError,
    PromptTooLongError,
    ProviderError,
)
from dana.common.observable import observable


if TYPE_CHECKING:
    pass

logger = structlog.get_logger()

# Wall-clock ceiling for one logical request (all attempts and failovers).
# ``DANA_LLM_REQUEST_DEADLINE_SECONDS`` overrides; 0 disables the ceiling.
REQUEST_DEADLINE_ENV = "DANA_LLM_REQUEST_DEADLINE_SECONDS"

# Default wall-clock ceiling for one logical request (all attempts + failover).
# Bounds the structural worst case of (max_retries + 1) outer attempts per
# provider, which no other layer bounds. Must exceed one slow-but-legitimate
# attempt: the SDK per-attempt timeout is 360s, so a ceiling below that would
# silently turn every slow attempt into a deadline failure instead of letting it
# finish, and would also suppress the configured retries. 900s leaves room for
# the full retry sequence while still capping a wedged connection.
DEFAULT_REQUEST_DEADLINE_SECONDS = 900.0

# Idle bound between streamed frames. The logical-request budget only gates the
# wait for an attempt's *first* frame: a stream that keeps delivering may
# legitimately run past the deadline, while a connection that stops delivering is
# dead. Each later frame wait is therefore bounded by this constant instead, i.e.
# the provider SDK's own per-attempt timeout (360s).
STREAM_IDLE_TIMEOUT_SECONDS = 360.0

# Sampling temperature the runtimes configure by default (``AgentRuntime``'s
# ``temperature=0``). It is the *default*, not a requested policy: streaming omits
# it so the provider's own default applies, while the non-streaming wire shape is
# unchanged; see :meth:`LLMCaller._build_request_kwargs`.
DEFAULT_TEMPERATURE = 0.0


class _Unset:
    """Sentinel for a request control the caller never supplied.

    The distinction that matters is *intent*, not value: the runtimes construct
    their caller with ``temperature=0`` whether or not sampling was ever asked
    for, so the value ``0`` cannot tell "the runtime default" from "an explicit
    0". Reading a value as intent silently drops the explicit choice (I11). This
    sentinel is what a constructor default binds to, so ``UNSET`` means "no
    policy" while any concrete value, including ``0`` and ``None``, is a caller
    decision that reaches the wire.

    Mirrors the ``fields_set`` distinction Pydantic exposes for models; the
    request controls here are plain constructor parameters, so the equivalent is
    a sentinel default.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        # Falsy so ``UNSET`` reads as "not set" in boolean contexts.
        return False


#: A control that was never supplied by any caller.
UNSET = _Unset()

# Keywords that indicate a transient (retriable) provider error.
# "connection" matches openai.APIConnectionError → wrapped as ProviderError("...: Connection error.").
_TRANSIENT_KEYWORDS = (
    "rate limit",
    "timeout",
    "5xx",
    "503",
    "502",
    "429",
    "overloaded",
    "down",
    "unavailable",
    "unreachable",
    "connection",
)

# OpenAI SDK exception class names that indicate transient network failures.
# Checked via class-name match to avoid importing openai here (keeps llm_caller provider-agnostic).
_TRANSIENT_OPENAI_EXC_NAMES = ("APIConnectionError", "APIConnectionTimeoutError")


def is_transient_llm_error(exc: BaseException) -> bool:
    """Module-level helper so other layers (e.g. STAR loop) can classify errors
    using the same rules as :class:`LLMCaller`."""
    return LLMCaller._is_transient_error(exc)


def _resolve_timeline(agent: Any | None):
    """Return the agent's timeline if it exposes `reactive_compact`, else None."""
    if agent is None:
        return None
    tl = getattr(agent, "_timeline", None)
    if tl is None or not hasattr(tl, "reactive_compact"):
        return None
    return tl


def _reactive_enabled(timeline: Any) -> bool:
    """Kill switch: env `DANA_DISABLE_REACTIVE_COMPACT=1` OR config flag False."""
    import os

    if os.getenv("DANA_DISABLE_REACTIVE_COMPACT") == "1":
        return False
    cfg = getattr(timeline, "_compressed_config", None)
    if cfg is None:
        return False
    return bool(getattr(cfg, "enable_reactive_compact", True))


@dataclass
class ProviderConfig:
    """Configuration for a fallback LLM provider."""

    provider: str
    model: str | None = None
    api_key: str | None = None  # optional override
    priority: int = 0  # lower = higher priority (used for sorting fallbacks)


class LLMCaller:
    """
    Encapsulates LLM invocation for AgentRuntime.

    Parameters
    ----------
    llm:
        Pre-built LLM instance. When None, one is lazily created from
        ``provider`` / ``model`` on first call.
    model:
        Model name passed to :class:`LLM` when lazy-creating.
    provider:
        Provider name (e.g. ``"anthropic"``, ``"openai"``).
    temperature:
        Sampling temperature forwarded to the LLM call. Defaults to
        :data:`UNSET`, meaning **no caller ever expressed a sampling policy**
        (the runtime's own ``temperature=0`` default). ``UNSET`` keeps the
        streaming path's historical omission so the provider applies its own
        default, while a caller-supplied value always reaches the wire on both
        paths, including an explicit ``0``, which is a decision, not "unset".
        ``None`` explicitly asks for no temperature on either path.
    max_tokens:
        Max tokens forwarded to the LLM call. ``None`` (the default) means
        unset and is omitted rather than sent as null.
    native_tools_getter:
        Zero-argument callable that returns the current native tools list
        (or ``None``).  Called on every invocation so changes are reflected.
    agent_getter:
        Zero-argument callable that returns the current agent instance
        (or ``None``).  Used to read ``object_id``, ``agent_type``, and
        ``llm_client``.
    json_mode:
        Whether to request JSON output from the LLM.  Defaults to ``True``
        (used by DefaultRuntime / AgentRuntime).
    fallback_providers:
        Ordered list of fallback :class:`ProviderConfig` to try when the
        primary provider fails transiently.  ``None`` disables failover.
    max_retries:
        Number of retries per provider before switching to the next one.
    base_delay:
        Base delay in seconds for exponential backoff between retries.
        ``retry_max_delay`` caps the exponential growth.
    deadline_seconds:
        Wall-clock ceiling for one logical request, spanning every attempt and
        provider failover. ``None`` resolves from
        ``DANA_LLM_REQUEST_DEADLINE_SECONDS`` (default
        :data:`DEFAULT_REQUEST_DEADLINE_SECONDS`); ``0`` disables the ceiling.
        Exceeding it raises :class:`LLMTimeoutError`.
    stream_max_retries:
        Attempts for a streaming request **only while no chunk has been
        emitted**. Once any chunk (text/thinking/tool) reaches the consumer,
        retrying would duplicate already-visible output, so the failure is
        raised immediately. Default 1 (one re-issue of a stream that failed
        before producing output).
    cancel_event_getter:
        Optional zero-argument callable returning the owning session's
        :class:`asyncio.Event`, or ``None``. When set, a cancelled session is
        detected during backoff sleeps and raises :class:`asyncio.CancelledError`
        instead of waiting out the delay.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: str | None = None,
        provider: str = "anthropic",
        temperature: float | None | _Unset = UNSET,
        max_tokens: int | None = None,
        native_tools_getter: Callable[[], list | None] | None = None,
        agent_getter: Callable[[], Any] | None = None,
        json_mode: bool = True,
        fallback_providers: list[ProviderConfig] | None = None,
        max_retries: int = 2,
        base_delay: float = 1.0,
        retry_max_delay: float = 8.0,
        deadline_seconds: float | None = None,
        stream_max_retries: int = 1,
        cancel_event_getter: Callable[[], Any] | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        self._provider = provider
        #: Raw request control, intentionally *not* resolved here: the
        #: distinction between "no policy" (:data:`UNSET`) and "the value 0" is
        #: the whole point (I11), and resolving it at construction would lose it.
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._native_tools_getter = native_tools_getter or (lambda: None)
        self._agent_getter = agent_getter or (lambda: None)
        self._json_mode = json_mode
        self._fallback_providers = sorted(fallback_providers, key=lambda p: p.priority) if fallback_providers else None
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._retry_max_delay = retry_max_delay
        self._deadline_seconds = deadline_seconds
        self._stream_max_retries = max(0, stream_max_retries)
        self._cancel_event_getter = cancel_event_getter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_llm(self, llm: LLM) -> None:
        """Replace the current LLM instance.

        Does not close the replaced instance: the same ``LLM`` object is usually
        shared with the agent, LTMemory and codec runtimes, so closing here would
        pull the client out from under its other users. Retire a client only when
        the LLM object itself is being discarded — call ``LLM.aclose()`` from
        session teardown.
        """
        self._llm = llm

    @observable
    def call_llm(
        self,
        messages: list[LLMMessage],
        messages_fn: Callable[[], list[LLMMessage]] | None = None,
    ) -> LLMResponse:
        """Synchronous LLM call with retry + exponential backoff.

        Retry always runs for transient errors on the primary provider.
        Failover only runs when ``fallback_providers`` is configured.

        When ``messages_fn`` is provided, it is invoked to rebuild the message
        list after each successful ``reactive_compact`` in the PTL retry loop,
        so the retry observes the compacted timeline rather than a stale
        snapshot. Non-PTL retries continue to use the original ``messages``.
        """
        return self._call_with_failover(messages, messages_fn)

    @observable
    async def call_llm_async(
        self,
        messages: list[LLMMessage],
        messages_fn: Callable[[], list[LLMMessage]] | None = None,
    ) -> LLMResponse:
        """Asynchronous LLM call with retry + exponential backoff.

        Retry always runs for transient errors on the primary provider.
        Failover only runs when ``fallback_providers`` is configured.

        See :meth:`call_llm` for ``messages_fn`` semantics.
        """
        return await self._call_with_failover_async(messages, messages_fn)

    async def call_llm_stream(self, messages: list[LLMMessage]) -> AsyncIterator[LLMStreamChunk]:
        """Stream LLM response, yielding typed LLMStreamChunk objects.

        Yields text_delta, tool_use, thinking, and the terminal ``response``
        carrier as they arrive. Request controls (temperature, max_tokens,
        json_mode) are assembled by the same :meth:`_build_request_kwargs` used by
        the non-streaming path, so streaming and non-streaming send the same
        provider-neutral policy and the adapters translate it per capability.
        Temperature is resolved from caller *intent*: an unset control keeps the
        historical omission (the provider's own sampling default applies), while
        any explicitly supplied value, including ``0``, reaches the wire (see
        :meth:`_resolve_temperature`).

        Retry policy: bounded re-issue **only while nothing has been emitted**.
        Once a chunk reaches the consumer, a failure is raised as-is — replaying
        would duplicate output the host has already rendered. No provider
        failover: a second provider would restart a partially-delivered turn.

        Budget: the logical-request deadline bounds the wait for the *first*
        frame, where a stalled connection is otherwise invisible. Once frames are
        flowing, each subsequent wait is bounded by
        :data:`STREAM_IDLE_TIMEOUT_SECONDS` (or the provider's own stricter
        timeout) instead, so a healthy long stream is never cut off mid-turn while
        a stream that stops delivering still fails as a typed
        :class:`LLMTimeoutError`. The provider stream is closed on every exit path
        and nothing already emitted is replayed.
        """
        agent = self._agent_getter()
        agent_kwargs = self._agent_kwargs(agent)
        tools = self._native_tools_getter() or None
        request_kwargs = self._build_request_kwargs(streaming=True)
        deadline = self._new_deadline()

        last_exc: Exception | None = None
        for attempt in range(self._stream_max_retries + 1):
            self._raise_if_deadline_exhausted(deadline, self._provider)
            emitted = False
            stream: AsyncIterator[LLMStreamChunk] | None = None
            try:
                llm = self._resolve_llm()
                stream = llm.stream(
                    messages,
                    tools=tools,
                    timeout=self._attempt_timeout(deadline, llm),
                    **request_kwargs,
                    **agent_kwargs,
                )
                first_frame = True
                while True:
                    if first_frame:
                        # The first frame is where a stalled connection is invisible to
                        # the transport timeout, so it is bounded by the same
                        # per-attempt ceiling the provider was handed: the smaller of
                        # the remaining budget and the provider's own timeout. Capping
                        # at the latter keeps one hung connection from consuming the
                        # whole budget before the configured retry can run, and also
                        # covers the budget-disabled case, where it repeats the bound
                        # the SDK applies on the wire.
                        frame_timeout = self._attempt_timeout(deadline, llm) or self._stream_idle_timeout(llm)
                    else:
                        # Frames are flowing: the turn is alive, so only idleness counts
                        # and the request deadline no longer applies.
                        frame_timeout = self._stream_idle_timeout(llm)
                    chunk = await self._next_chunk(stream, frame_timeout, self._provider)
                    first_frame = False
                    emitted = True
                    yield chunk
            except (GeneratorExit, asyncio.CancelledError):
                # This generator was closed while a provider stream was open:
                # release its response before unwinding, then re-raise so the
                # cancellation contract is preserved.
                if stream is not None:
                    await self._aclose_stream(stream)  # Must complete before teardown.
                raise
            except StopAsyncIteration:
                return
            except Exception as exc:
                if emitted or not self._is_transient_error(exc):
                    raise
                last_exc = exc
                if attempt < self._stream_max_retries:
                    # Same interruptible backoff the non-streaming path uses, so a
                    # cancelled session stops here instead of after the delay.
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "llm_stream_retry",
                        attempt=attempt + 1,
                        max_attempts=self._stream_max_retries + 1,
                        delay=delay,
                        deadline_remaining=self._budget_left(deadline),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    await self._async_backoff(delay, deadline, self._provider)
                    continue
                if not isinstance(exc, LLMError):
                    logger.error(
                        "llm_stream_retries_exhausted",
                        attempts=self._stream_max_retries + 1,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
        if last_exc is not None:
            raise last_exc

    async def _next_chunk(self, stream: Any, timeout: float | None, provider_label: str) -> LLMStreamChunk:
        """Await the next stream frame, bounded by ``timeout`` seconds.

        The caller passes the remaining logical-request budget for the *first*
        frame — a provider that accepts the request and then stalls before sending
        anything would otherwise hang the turn indefinitely, because the transport
        timeout only starts once the response headers arrive. Later frames are
        awaited with the fixed idle bound instead (``timeout=None``), so a healthy
        but slow stream is not cut off between chunks while a stream that stops
        delivering is still detected. Either way the failure is a typed
        :class:`LLMTimeoutError` and the underlying response is closed before
        propagating, so nothing already emitted is replayed.
        """
        if timeout is None:
            return await stream.__anext__()
        try:
            # ``asyncio.timeout`` (3.11+) cancels the pending ``__anext__`` directly,
            # unlike wrapping it in a Task, and converts expiry into TimeoutError.
            async with asyncio.timeout(timeout):
                return await stream.__anext__()
        except TimeoutError as e:
            await self._aclose_stream(stream)
            raise LLMTimeoutError(f"{provider_label}: stream idle timeout exceeded waiting for the next chunk") from e
        except asyncio.CancelledError:
            # Owner cancellation: release the connection before unwinding.
            await self._aclose_stream(stream)
            raise
        except StopAsyncIteration:
            raise
        except Exception:
            await self._aclose_stream(stream)
            raise

    @staticmethod
    async def _aclose_stream(stream: Any) -> None:
        """Best-effort close of a provider stream so its response is released."""
        closer = getattr(stream, "aclose", None)
        if closer is None:
            return
        try:
            await closer()
        except Exception as e:  # pragma: no cover - teardown must not raise
            logger.warning("llm_stream_close_failed", error_type=type(e).__name__, error=str(e))

    async def _await_within_budget(self, factory: Any, deadline: float, provider_label: str) -> LLMResponse:
        """Await one provider operation, hard-bounded by the remaining budget.

        The provider is *also* handed the same remaining budget as its SDK timeout
        (:meth:`_attempt_timeout`), so a cooperative provider fails fast on the wire.
        This wrapper is the second line of defence for the async path: it guarantees
        the caller stops waiting even when the provider ignores the timeout or stalls
        below it, which is what makes the deadline a real ceiling rather than a
        request to the SDK. Owner cancellation is never converted — ``asyncio.timeout``
        only raises :class:`TimeoutError` for its own expiry.
        """
        remaining = self._budget_left(deadline)
        if remaining == float("inf"):
            return await factory()
        try:
            async with asyncio.timeout(remaining):
                return await factory()
        except TimeoutError as e:
            raise LLMTimeoutError(f"{provider_label}: request deadline exceeded after {remaining:.1f}s") from e

    # ------------------------------------------------------------------
    # Request policy
    # ------------------------------------------------------------------

    def _agent_kwargs(self, agent: Any | None) -> dict[str, Any]:
        """Provider-passed agent identity. Stripped by ``LLM`` before the wire."""
        if agent is None:
            return {}
        return {"agent_id": agent.object_id, "agent_type": agent.agent_type}

    def _build_request_kwargs(self, *, streaming: bool = False) -> dict[str, Any]:
        """Single assembly point for provider-neutral request controls.

        Both :meth:`call_llm_stream` and the non-streaming invocations read the
        same values, so a control can no longer reach the wire on one path and
        silently vanish on the other. Adapters translate this policy per
        capability (e.g. Anthropic's ``max_tokens`` requirement, Gemini's
        ``max_output_tokens``, gpt-5's ``max_completion_tokens`` rename and
        temperature restriction) and keep their existing unsupported-parameter
        degradation.

        ``max_tokens`` is omitted when ``None`` rather than sent as null: providers
        treat an explicit null differently from an absent key, and the adapters
        rely on presence checks.

        Temperature is resolved from *intent*, not from the value
        (:meth:`_resolve_temperature`): the runtimes construct their caller with
        the same default whether or not sampling was ever requested, so a value
        cannot express that distinction. The streaming path historically omitted
        that runtime default so each provider applied its own sampling default;
        sending the default instead is not a policy any caller expressed, and it
        is a hard 400 on families and deployments that reject sampling (o3/o4,
        Azure gpt-5 deployments not named ``gpt-5*``) and a silent behaviour
        change everywhere else. An explicit value, including an explicit ``0``,
        always reaches the wire on both paths, subject to the same per-model
        restriction the non-streaming path uses. The non-streaming wire shape is
        unchanged by this PR.
        """
        kwargs: dict[str, Any] = {"json_mode": self._json_mode}
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        temperature = self._resolve_temperature(streaming=streaming)
        if temperature is not None:
            kwargs["temperature"] = temperature
        return kwargs

    def _resolve_temperature(self, *, streaming: bool) -> float | None:
        """The temperature to put on the wire for this path, or ``None`` to omit.

        The value alone cannot tell "the runtime's own default" from "the caller
        asked for 0", so the constructor keeps the distinction and resolves it
        here (I11):

        * :data:`UNSET`: no sampling policy was ever expressed. Streaming keeps
          its historical omission (the provider default applies); non-streaming
          keeps its historical wire shape, which carried
          :data:`DEFAULT_TEMPERATURE`.
        * Any concrete value, including ``0`` and a non-zero float, is caller
          intent and is forwarded unchanged on **both** paths.
        * ``None`` explicitly asks for no sampling parameter, so it is omitted on
          both paths.
        """
        temperature = self._temperature
        if isinstance(temperature, _Unset):
            return None if streaming else DEFAULT_TEMPERATURE
        return temperature

    # ------------------------------------------------------------------
    # Deadline / cancellation
    # ------------------------------------------------------------------

    def _resolve_deadline_seconds(self) -> float:
        """Resolve the logical-request ceiling; <= 0 disables it."""
        if self._deadline_seconds is not None:
            return float(self._deadline_seconds)
        import os

        raw = os.getenv(REQUEST_DEADLINE_ENV)
        if raw is None or not raw.strip():
            return float(DEFAULT_REQUEST_DEADLINE_SECONDS)
        try:
            return float(raw)
        except ValueError:
            logger.warning(
                "ignoring invalid request deadline env var",
                env_var=REQUEST_DEADLINE_ENV,
                value=raw,
                fallback=DEFAULT_REQUEST_DEADLINE_SECONDS,
            )
            return float(DEFAULT_REQUEST_DEADLINE_SECONDS)

    def _budget_left(self, deadline: float) -> float:
        """Seconds remaining before the logical request deadline (inf when disabled)."""
        if deadline <= 0:
            return float("inf")
        return deadline - time.monotonic()

    def _provider_timeout(self, llm: LLM | None) -> float:
        """The provider's own configured per-attempt ceiling.

        Each provider class advertises :attr:`LLMProvider.DEFAULT_TIMEOUT_SECONDS`
        and hands it to its SDK/HTTP client. Reading it here keeps one attempt from
        being handed a budget so large that a single hung connection consumes the
        whole request before any configured retry can run.
        """
        provider = getattr(llm, "provider", None) if llm is not None else None
        configured = getattr(provider, "DEFAULT_TIMEOUT_SECONDS", None)
        if not isinstance(configured, (int, float)) or configured <= 0:
            return float(LLMProvider.DEFAULT_TIMEOUT_SECONDS)
        return float(configured)

    def _attempt_timeout(self, deadline: float, llm: LLM | None = None) -> float | None:
        """Per-attempt ceiling: the smaller of the budget left and the provider's own.

        This is the value handed to the awaited provider operation, so the budget
        bounds the actual call rather than only the backoff between calls. Returns
        ``None`` when the budget is disabled (``deadline <= 0``) so the provider
        keeps its own timeout.

        Capping at the provider's configured timeout is what keeps a stalled
        attempt retriable: handed the whole remaining budget (900s by default), one
        hung connection would outlive the request deadline and no configured retry
        could ever run. An exhausted remainder is returned as-is — the caller
        refuses to start such an attempt in the first place
        (:meth:`_raise_if_deadline_exhausted`), so it is never inflated into a
        timeout the budget cannot pay for.
        """
        if deadline <= 0:
            return None
        return min(self._budget_left(deadline), self._provider_timeout(llm))

    def _stream_idle_timeout(self, llm: LLM | None = None) -> float:
        """Bound on the wait for one *later* stream frame.

        The logical-request budget gates only the first frame (a stalled connection
        is otherwise invisible). After that, a stream that keeps delivering may run
        as long as it likes — the caller is making progress — while one that stops
        delivering is dead. The provider's configured timeout wins when it is
        stricter than :data:`STREAM_IDLE_TIMEOUT_SECONDS`.
        """
        return min(STREAM_IDLE_TIMEOUT_SECONDS, self._provider_timeout(llm))

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff capped by ``retry_max_delay`` (never negative)."""
        return max(0.0, min(self._base_delay * (2**attempt), self._retry_max_delay))

    def _raise_if_cancelled(self) -> None:
        """Surface owner cancellation during backoff instead of sleeping it out.

        ``asyncio.CancelledError`` derives from ``BaseException``, so no
        ``except Exception`` in this module or the caller layers can swallow it.
        """
        if self._cancel_event_getter is None:
            return
        event = self._cancel_event_getter()
        if event is not None and event.is_set():
            raise asyncio.CancelledError("LLM request cancelled during backoff")

    # ------------------------------------------------------------------
    # Failover logic
    # ------------------------------------------------------------------

    def _call_with_failover(
        self,
        messages: list[LLMMessage],
        messages_fn: Callable[[], list[LLMMessage]] | None = None,
    ) -> LLMResponse:
        """Sync call with retry + exponential backoff + optional provider failover.

        Every attempt and failover shares one wall-clock budget
        (:meth:`_new_deadline`). The budget bounds the actual call, not just the
        sequence between calls: each attempt is handed the *remaining* budget as
        the per-attempt SDK/HTTP timeout (:meth:`_attempt_timeout`), so a stalled
        provider cannot hold the caller past the deadline. The only slack is the
        provider honoring its own timeout — the same cooperative limit
        :class:`LLMTimeoutError` has always depended on.
        """
        deadline = self._new_deadline()
        providers: list[ProviderConfig | None] = [None, *(self._fallback_providers or [])]
        last_exc: Exception | None = None

        for fallback in providers:
            llm = self._resolve_llm() if fallback is None else LLM(provider=fallback.provider, model=fallback.model)
            provider_label = self._provider if fallback is None else fallback.provider

            for attempt in range(self._max_retries + 1):
                self._raise_if_deadline_exhausted(deadline, provider_label)
                try:
                    return self._invoke_llm_sync(llm, messages, messages_fn, timeout=self._attempt_timeout(deadline, llm))
                except Exception as exc:
                    if not self._is_transient_error(exc):
                        raise
                    last_exc = exc
                    if attempt < self._max_retries:
                        delay = self._backoff_delay(attempt)
                        logger.warning(
                            "llm_retry",
                            provider=provider_label,
                            attempt=attempt + 1,
                            max_attempts=self._max_retries + 1,
                            delay=delay,
                            deadline_remaining=self._budget_left(deadline),
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                        self._sleep_within_deadline(delay, deadline, provider_label)
                    elif self._fallback_providers:
                        logger.warning("llm_failover", from_provider=provider_label, error_type=type(exc).__name__, error=str(exc))
                    else:
                        logger.error(
                            "llm_retries_exhausted",
                            provider=provider_label,
                            attempts=self._max_retries + 1,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )

        raise last_exc  # type: ignore[misc]

    async def _call_with_failover_async(
        self,
        messages: list[LLMMessage],
        messages_fn: Callable[[], list[LLMMessage]] | None = None,
    ) -> LLMResponse:
        """Async call with retry + exponential backoff + optional provider failover.

        Same shared wall-clock budget as the sync path; each attempt is handed the
        remaining budget as its per-attempt provider/SDK timeout, and backoff sleeps
        are interruptible: a cancelled session stops the loop immediately instead of
        waiting out the delay.
        """
        deadline = self._new_deadline()
        providers: list[ProviderConfig | None] = [None, *(self._fallback_providers or [])]
        last_exc: Exception | None = None

        for fallback in providers:
            llm = self._resolve_llm() if fallback is None else LLM(provider=fallback.provider, model=fallback.model)
            provider_label = self._provider if fallback is None else fallback.provider

            for attempt in range(self._max_retries + 1):
                self._raise_if_deadline_exhausted(deadline, provider_label)
                try:
                    return await self._await_within_budget(
                        lambda llm=llm: self._invoke_llm_async(llm, messages, messages_fn, timeout=self._attempt_timeout(deadline, llm)),
                        deadline,
                        provider_label,
                    )
                except asyncio.CancelledError:
                    # Owner cancellation — never reclassify as a provider fault.
                    raise
                except Exception as exc:
                    if not self._is_transient_error(exc):
                        raise
                    last_exc = exc
                    if attempt < self._max_retries:
                        delay = self._backoff_delay(attempt)
                        logger.warning(
                            "llm_retry",
                            provider=provider_label,
                            attempt=attempt + 1,
                            max_attempts=self._max_retries + 1,
                            delay=delay,
                            deadline_remaining=self._budget_left(deadline),
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                        await self._async_backoff(delay, deadline, provider_label)
                    elif self._fallback_providers:
                        logger.warning("llm_failover", from_provider=provider_label, error_type=type(exc).__name__, error=str(exc))
                    else:
                        logger.error(
                            "llm_retries_exhausted",
                            provider=provider_label,
                            attempts=self._max_retries + 1,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Deadline helpers
    # ------------------------------------------------------------------

    def _new_deadline(self) -> float:
        """Absolute monotonic deadline for one logical request (<= 0 → disabled)."""
        seconds = self._resolve_deadline_seconds()
        return (time.monotonic() + seconds) if seconds > 0 else 0.0

    def _raise_if_deadline_exhausted(self, deadline: float, provider_label: str) -> None:
        """Refuse to start an attempt the budget can no longer cover."""
        if deadline <= 0:
            return
        remaining = self._budget_left(deadline)
        if remaining <= 0:
            raise LLMTimeoutError(f"{provider_label}: request deadline exceeded before attempt")

    def _sleep_within_deadline(self, delay: float, deadline: float, provider_label: str) -> None:
        """Backoff sleep that cannot overshoot the logical-request deadline."""
        remaining = self._budget_left(deadline)
        if delay > remaining:
            raise LLMTimeoutError(f"{provider_label}: request deadline leaves {remaining:.1f}s, backoff needs {delay:.1f}s")
        time.sleep(delay)

    async def _async_backoff(self, delay: float, deadline: float, provider_label: str) -> None:
        """Backoff sleep that honours both the deadline and owner cancellation.

        Same affordability rule as :meth:`_sleep_within_deadline` — a retry is
        only attempted when the remaining budget can pay for the wait, so the two
        paths cannot drift into different notions of when the request is over.

        The sleep is interruptible rather than merely short: waiting out a 30s
        delay while the session is already cancelled delays teardown by the whole
        delay. When the owner exposes a cancel event the wait races against it and
        a set event raises :class:`asyncio.CancelledError` immediately.
        """
        self._raise_if_cancelled()
        remaining = self._budget_left(deadline)
        if delay > remaining:
            raise LLMTimeoutError(f"{provider_label}: request deadline leaves {remaining:.1f}s, backoff needs {delay:.1f}s")

        event = self._cancel_event_getter() if self._cancel_event_getter is not None else None
        if event is None:
            await asyncio.sleep(delay)
        else:
            try:
                await asyncio.wait_for(event.wait(), timeout=delay)
            except TimeoutError:
                pass
        self._raise_if_cancelled()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke_llm_sync(
        self,
        llm: LLM,
        messages: list[LLMMessage],
        messages_fn: Callable[[], list[LLMMessage]] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Execute a single synchronous LLM chat call with PTL reactive retry.

        On `PromptTooLongError`, calls `timeline.reactive_compact(attempt)` and
        retries up to 3 times with exponential backoff (1s, 3s). After each
        successful `reactive_compact`, if ``messages_fn`` was provided, the
        message list is rebuilt from it so the retry observes the compacted
        timeline (CRITICAL-1 fix — without this, retries send the same
        oversized payload and the circuit opens on recoverable sessions).
        After 3 failures, `CompactCircuitOpenError` is raised. Kill switch via
        `DANA_DISABLE_REACTIVE_COMPACT=1` or `timeline.config.enable_reactive_compact=False`.

        ``timeout`` is the caller's remaining logical-request budget, forwarded as
        the per-attempt SDK timeout.
        """
        agent = self._agent_getter()
        tools = self._native_tools_getter() or None
        request_kwargs = self._build_request_kwargs()

        def _do_call(msgs: list[LLMMessage]) -> LLMResponse:
            return llm.chat_response_sync(
                msgs,
                **self._agent_kwargs(agent),
                tools=tools,
                timeout=timeout,
                **request_kwargs,
            )

        timeline = _resolve_timeline(agent)
        if timeline is None or not _reactive_enabled(timeline):
            return _do_call(messages)

        backoff = {1: 1.0, 2: 3.0}
        last_exc: PromptTooLongError | None = None
        current_messages = messages
        for attempt in range(1, 4):
            try:
                return _do_call(current_messages)
            except PromptTooLongError as e:
                last_exc = e
                logger.warning(
                    "prompt_too_long_reactive_compact",
                    attempt=attempt,
                    error=str(e),
                )
                try:
                    timeline.reactive_compact(attempt)
                except CompactCircuitOpenError:
                    raise
                current_messages = self._rebuild_messages_after_compact(current_messages, messages_fn, attempt)
                if attempt < 3:
                    time.sleep(backoff.get(attempt, 0))
        # All 3 attempts exhausted — open circuit and surface.
        timeline._consecutive_compact_failures = max(timeline._consecutive_compact_failures, 3)
        timeline._compaction_disabled = True
        from datetime import datetime as _dt

        timeline._circuit_opened_at = _dt.now()
        raise CompactCircuitOpenError(f"PTL retry exhausted after 3 attempts; last: {last_exc}") from last_exc

    async def _invoke_llm_async(
        self,
        llm: LLM,
        messages: list[LLMMessage],
        messages_fn: Callable[[], list[LLMMessage]] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Async version of `_invoke_llm_sync` with PTL reactive retry.

        See :meth:`_invoke_llm_sync` for the ``messages_fn`` rebuild semantics
        (CRITICAL-1 fix). ``timeout`` is the caller's remaining logical-request
        budget, forwarded as the provider/SDK deadline for this attempt.
        """
        import asyncio

        agent = self._agent_getter()
        tools = self._native_tools_getter() or None
        request_kwargs = self._build_request_kwargs()

        async def _do_call(msgs: list[LLMMessage]) -> LLMResponse:
            return await llm.chat_response(
                msgs,
                **self._agent_kwargs(agent),
                tools=tools,
                timeout=timeout,
                **request_kwargs,
            )

        timeline = _resolve_timeline(agent)
        if timeline is None or not _reactive_enabled(timeline):
            return await _do_call(messages)

        backoff = {1: 1.0, 2: 3.0}
        last_exc: PromptTooLongError | None = None
        current_messages = messages
        for attempt in range(1, 4):
            try:
                return await _do_call(current_messages)
            except PromptTooLongError as e:
                last_exc = e
                logger.warning(
                    "prompt_too_long_reactive_compact",
                    attempt=attempt,
                    error=str(e),
                )
                try:
                    # Await the timeline's own cancellation-capable compaction when it
                    # has one. ``asyncio.to_thread`` cannot be stopped: when the
                    # logical-request deadline or an owner cancellation ends the
                    # await, the worker keeps running and later installs a whole new
                    # projection (and stamps the compaction marker) on a timeline the
                    # caller has already released. ``reactive_compact_async`` keeps
                    # every mutation on this awaiting thread and joins any sync-only
                    # summarizer before returning, so an abandoned await leaves no
                    # late writer behind.
                    compact_async = getattr(timeline, "reactive_compact_async", None)
                    if compact_async is not None:
                        await compact_async(attempt)
                    else:
                        # Foreign timeline without the async seam: its compaction is
                        # synchronous and cannot be stopped, so the worker is joined
                        # before a cancellation is re-raised instead of being
                        # orphaned. The deadline may overrun by that callback's
                        # remaining physical duration.
                        task = asyncio.ensure_future(asyncio.to_thread(timeline.reactive_compact, attempt))
                        try:
                            await asyncio.shield(task)
                        except asyncio.CancelledError:
                            # The owner cancelled. ``shield`` keeps the worker
                            # running, so join it here: it owns a mutation on a
                            # timeline the caller has already released, and
                            # abandoning it would let that write land late.
                            # Its outcome is *not* this call's result, though:
                            # the owner's cancellation is, so a worker failure
                            # observed while joining is retrieved and discarded
                            # rather than allowed to replace the cancellation
                            # (``shield`` re-raises the worker's own exception
                            # the moment the worker completes).
                            while not task.done():
                                try:
                                    await asyncio.shield(task)
                                except asyncio.CancelledError:
                                    continue
                                except BaseException:
                                    # The worker failed while being joined. That is
                                    # not the owner's outcome; the owner's
                                    # cancellation must be what propagates, so the
                                    # failure is drained and retrieved below rather
                                    # than allowed to escape (``shield`` re-raises
                                    # the worker's own exception the moment the
                                    # worker completes).
                                    break
                            if not task.cancelled():
                                try:
                                    task.exception()  # retrieved, deliberately not raised
                                except BaseException:
                                    pass
                            raise  # the owner's CancelledError, unchanged
                except CompactCircuitOpenError:
                    raise
                current_messages = self._rebuild_messages_after_compact(current_messages, messages_fn, attempt)
                if attempt < 3:
                    await asyncio.sleep(backoff.get(attempt, 0))
        timeline._consecutive_compact_failures = max(timeline._consecutive_compact_failures, 3)
        timeline._compaction_disabled = True
        from datetime import datetime as _dt

        timeline._circuit_opened_at = _dt.now()
        raise CompactCircuitOpenError(f"PTL retry exhausted after 3 attempts; last: {last_exc}") from last_exc

    @staticmethod
    def _rebuild_messages_after_compact(
        current: list[LLMMessage],
        messages_fn: Callable[[], list[LLMMessage]] | None,
        attempt: int,
    ) -> list[LLMMessage]:
        """Re-invoke ``messages_fn`` after a successful ``reactive_compact`` so
        the next retry sends the compacted payload. Falls back to ``current``
        on failure or when no factory was provided.
        """
        if messages_fn is None:
            return current
        try:
            rebuilt = messages_fn()
            logger.info(
                "ptl_messages_rebuilt",
                attempt=attempt,
                old_count=len(current),
                new_count=len(rebuilt),
            )
            return rebuilt
        except Exception as exc:
            logger.warning(
                "ptl_messages_fn_raised_keeping_stale",
                attempt=attempt,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return current

    @staticmethod
    def _is_transient_error(exc: BaseException) -> bool:
        """Return True if the error is transient and should trigger a retry."""
        if isinstance(exc, ConfigurationError):
            return False
        # Prompt-too-long is deterministic for a fixed payload: it is handled by
        # the reactive-compaction path inside _invoke_llm_*, never by re-sending
        # the identical oversized request. Explicit because PromptTooLongError is
        # a ProviderError subclass and would otherwise reach the keyword list.
        if isinstance(exc, PromptTooLongError):
            return False
        # Unified timeout error raised by all providers
        if isinstance(exc, LLMTimeoutError):
            return True
        if isinstance(exc, TimeoutError | ConnectionError):
            return True
        # Walk the __cause__ chain to detect openai.APIConnectionError without
        # importing openai here (the SDK exception is preserved via `raise ... from e`).
        cur: BaseException | None = exc
        for _ in range(5):
            if cur is None:
                break
            if type(cur).__name__ in _TRANSIENT_OPENAI_EXC_NAMES:
                return True
            cur = cur.__cause__
        if isinstance(exc, ProviderError):
            msg = str(exc).lower()
            return any(kw in msg for kw in _TRANSIENT_KEYWORDS)
        # Generic LLMError without transient keywords → permanent
        if isinstance(exc, LLMError):
            return False
        return False

    def _resolve_llm(self) -> LLM:
        """Lazy LLM resolution.

        Priority:
        1. Cached ``self._llm``
        2. Agent's ``llm_client`` attribute (set externally)
        3. Create a new :class:`LLM` from provider/model and cache it on the
           agent if one is available.
        """
        if self._llm is not None:
            return self._llm

        agent = self._agent_getter()
        existing = getattr(agent, "_llm_client", None) if agent is not None else None
        if existing is not None:
            # Reuse the exact instance the agent already owns. Reading the
            # ``llm_client`` property instead would lazily construct a second LLM
            # for agents that build one on access, and assigning over a non-empty
            # slot would orphan the existing client's connection pool.
            self._llm = existing
            return self._llm

        self._llm = LLM(provider=self._provider, model=self._model)
        if agent is not None:
            agent.llm_client = self._llm
        return self._llm
