"""
Observable decorator for tracking function calls with a tracing backend.

Dispatches to ONE tracing backend per process, selected at decoration time:
  1. LangSmith  — when `langsmith` is importable AND
                  (`LANGSMITH_TRACING=true` OR `DANA_LANGSMITH_ENABLED` truthy)
  2. Langfuse   — when `langfuse` is importable AND `LANGFUSE_ENABLED` truthy
  3. no-op      — default (decorator returns the function unchanged)

Backends are EXCLUSIVE: LangSmith takes precedence over Langfuse. Toggle the
active backend via environment variables, not by editing call sites.

Neither backend flushes per call. Both SDKs batch in background threads, and a
synchronous flush after every nested span is a network round trip on the
critical path. Call `flush_observability()` at explicit boundaries where
traces must be delivered (a session ending, a CLI exiting) and
`shutdown_observability()` once when the process stops tracing.

A backend is imported only when its environment flag asks for it, so a process
with tracing disabled pays nothing for the SDKs at startup.
"""

from collections.abc import Callable
import logging
import os
import threading
from typing import cast

from dana.common.trace_sanitizer import safe_trace_inputs, safe_trace_outputs


logger = logging.getLogger(__name__)

_TRUTHY = ("true", "1", "yes")
# LangSmith run_type vocabulary. langfuse `as_type` has no 1:1 mapping.
_VALID_RUN_TYPES = {"chain", "llm", "tool", "prompt", "retriever"}

# Bounded budget for a lifecycle flush. A tracing backend that is slow, offline
# or wedged must not be able to hold a process open at exit.
_FLUSH_TIMEOUT_S = 5.0

# Set once by `shutdown_observability()`. Later flushes become no-ops so
# shutdown is idempotent and cannot resurrect a released client.
_shutdown_done = threading.Event()

# Enablement is read ONCE at module load. Toggling requires a process restart
# (or `importlib.reload(dana.common.observable)` in tests).
LANGFUSE_ENABLED = False
OBSERVER = None
if os.getenv("LANGFUSE_ENABLED", "false").lower() in _TRUTHY:
    try:
        from langfuse import Langfuse
        from langfuse import observe as langfuse_observe
    except ImportError:
        # A requested but unavailable backend stays disabled.
        pass
    else:
        LANGFUSE_ENABLED = True
        OBSERVER = Langfuse()

LANGSMITH_ENABLED = False
langsmith_traceable = None
if os.getenv("LANGSMITH_TRACING", "false").lower() == "true" or os.getenv("DANA_LANGSMITH_ENABLED", "false").lower() in _TRUTHY:
    try:
        from langsmith import traceable as langsmith_traceable
    except ImportError:
        # LangSmith ships in the optional `dana-agent[langsmith]` extra.
        pass
    else:
        LANGSMITH_ENABLED = True


def _langsmith_kwargs(kwargs: dict) -> dict:
    """Translate langfuse-style @observe kwargs to langsmith @traceable kwargs.

    name                -> name
    as_type             -> run_type (default "chain"; "generation" and unknowns -> "chain")
    tags                -> tags
    session_id, user_id -> folded into metadata (langsmith has no direct equivalent)
    metadata            -> metadata (merged)
    """
    ls: dict = {}
    if "name" in kwargs:
        ls["name"] = kwargs["name"]
    run_type = kwargs.get("as_type", "chain")
    if run_type == "generation" or run_type not in _VALID_RUN_TYPES:
        run_type = "chain"
    ls["run_type"] = run_type
    if "tags" in kwargs:
        ls["tags"] = kwargs["tags"]
    meta = dict(kwargs.get("metadata") or {})
    if "session_id" in kwargs:
        meta["session_id"] = kwargs["session_id"]
    if "user_id" in kwargs:
        meta["user_id"] = kwargs["user_id"]
    if meta:
        ls["metadata"] = meta
    ls["process_inputs"] = kwargs.get("process_inputs", safe_trace_inputs)
    ls["process_outputs"] = kwargs.get("process_outputs", safe_trace_outputs)
    return ls


def observable(*args, **kwargs) -> Callable:
    """Decorator that tracks function calls with the active tracing backend.

    Backend selection (exclusive, evaluated at decoration time):
    LangSmith > Langfuse > no-op. See module docstring for env-var triggers.

    Supports both `@observable` and `@observable(...)` syntax, and sync + async.

    Args:
        *args: Positional args forwarded to the backend decorator.
        **kwargs: Keyword args forwarded to the backend decorator
            (langfuse-style: name, as_type, tags, session_id, user_id, metadata).

    Returns:
        Decorated callable that tracks inputs/outputs via the active backend,
        or the original callable unchanged when no backend is enabled.
    """

    def _apply(func: Callable, dec_args: tuple, dec_kwargs: dict) -> Callable:
        # Branch 1: LangSmith (precedence). traceable is sync+async native.
        # Positional decorator params are dropped (none used by any call site);
        # only translated kwargs are forwarded. LANGSMITH_ENABLED guarantees the import succeeded.
        if LANGSMITH_ENABLED:
            traceable = cast("Callable[..., Callable]", langsmith_traceable)
            return traceable(**_langsmith_kwargs(dec_kwargs))(func)

        # Branch 2: Langfuse. The SDK batches on its own background thread.
        if LANGFUSE_ENABLED:
            if dec_args or dec_kwargs:
                return cast("Callable[..., object]", langfuse_observe(*dec_args, **dec_kwargs)(func))
            return cast("Callable[..., object]", langfuse_observe(func))

        # Branch 3: no-op
        return func

    # Bare form: @observable (function passed positionally, no decorator params)
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return _apply(args[0], (), {})

    # Parameterized form: @observable(...) -> returns a decorator
    def decorator(func: Callable) -> Callable:
        return _apply(func, args, kwargs)

    return decorator


def _flush_bounded(flush: Callable[[], object], timeout: float) -> None:
    """Run `flush` on a daemon thread, waiting at most `timeout` seconds for it.

    Failures are logged at debug level: a tracing backend that cannot flush is
    a diagnostics problem, never a reason to fail the caller or block exit.
    """

    finished = threading.Event()

    def _run() -> None:
        try:
            flush()
        except Exception as exc:  # noqa: BLE001 - optional diagnostics must not propagate
            logger.debug("Tracing flush failed: %s", type(exc).__name__)
        finally:
            finished.set()

    threading.Thread(target=_run, name="dana-observability-flush", daemon=True).start()
    finished.wait(timeout)


def _flush_active_backend() -> None:
    """Resolve and flush the backend inside the protected lifecycle worker."""
    if LANGSMITH_ENABLED:
        from langsmith import Client

        Client().flush(timeout=_FLUSH_TIMEOUT_S)
    elif LANGFUSE_ENABLED and OBSERVER is not None:
        OBSERVER.flush()


def _shutdown_active_backend() -> None:
    """Resolve and release the backend inside the protected lifecycle worker.

    Langfuse owns background exporter threads that `shutdown()` stops, so it
    must not simply be flushed. LangSmith keeps a process-wide run tree with no
    equivalent teardown, so flushing is the whole release.
    """
    if LANGSMITH_ENABLED:
        from langsmith import Client

        Client().flush(timeout=_FLUSH_TIMEOUT_S)
    elif LANGFUSE_ENABLED and OBSERVER is not None:
        OBSERVER.shutdown()


def flush_observability(timeout: float | None = None) -> None:
    """Deliver buffered traces from the active backend, bounded by `timeout`.

    Intended for explicit lifecycle boundaries (session end, CLI exit, ACP
    shutdown) rather than per-operation calls. No-op when no backend is enabled
    or after `shutdown_observability()`. Never raises.
    """
    if _shutdown_done.is_set() or not (LANGSMITH_ENABLED or LANGFUSE_ENABLED):
        return
    _flush_bounded(_flush_active_backend, _FLUSH_TIMEOUT_S if timeout is None else timeout)


def shutdown_observability(timeout: float | None = None) -> None:
    """Flush and release the active backend. Idempotent; never raises.

    Call once when the process stops tracing, not per session: the backend is
    process-wide.
    """
    if _shutdown_done.is_set():
        return
    # Set before releasing so a concurrent flush cannot touch a freed client.
    _shutdown_done.set()
    if not (LANGSMITH_ENABLED or LANGFUSE_ENABLED):
        return
    _flush_bounded(_shutdown_active_backend, _FLUSH_TIMEOUT_S if timeout is None else timeout)


async def aflush_observability(timeout: float | None = None) -> None:
    """Await `flush_observability()` off the event loop, bounded by `timeout`."""
    import asyncio

    await asyncio.to_thread(flush_observability, timeout)


async def ashutdown_observability(timeout: float | None = None) -> None:
    """Await `shutdown_observability()` off the event loop, bounded by `timeout`."""
    import asyncio

    await asyncio.to_thread(shutdown_observability, timeout)
