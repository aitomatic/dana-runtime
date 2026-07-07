"""
Observable decorator for tracking function calls with a tracing backend.

Dispatches to ONE tracing backend per process, selected at decoration time:
  1. LangSmith  — when `langsmith` is importable AND
                  (`LANGSMITH_TRACING=true` OR `DANA_LANGSMITH_ENABLED` truthy)
  2. Langfuse   — when `langfuse` is importable AND `LANGFUSE_ENABLED` truthy
  3. no-op      — default (decorator returns the function unchanged)

Backends are EXCLUSIVE: LangSmith takes precedence over Langfuse. Toggle the
active backend via environment variables, not by editing call sites.

LangSmith relies on its SDK's background-thread batching (no per-call flush).
Langfuse preserves its existing flush-after-each-call behavior.
"""

from collections.abc import Callable
import functools
import inspect
import os
from typing import cast


# --- Langfuse (existing backend) ---
try:
    from langfuse import Langfuse
    from langfuse import observe as langfuse_observe
except ModuleNotFoundError:
    Langfuse = None

    def langfuse_observe(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return args[0]
        return decorator


# --- LangSmith (alternative backend) ---
try:
    from langsmith import traceable as langsmith_traceable
except ModuleNotFoundError:
    langsmith_traceable = None


_TRUTHY = ("true", "1", "yes")
# LangSmith run_type vocabulary. langfuse `as_type` has no 1:1 mapping.
_VALID_RUN_TYPES = {"chain", "llm", "tool", "prompt", "retriever"}

# Enablement is read ONCE at module load. Toggling requires a process restart
# (or `importlib.reload(dana.common.observable)` in tests).
LANGFUSE_ENABLED = Langfuse is not None and os.getenv("LANGFUSE_ENABLED", "false").lower() in _TRUTHY
LANGSMITH_ENABLED = langsmith_traceable is not None and (
    os.getenv("LANGSMITH_TRACING", "false").lower() == "true" or os.getenv("DANA_LANGSMITH_ENABLED", "false").lower() in _TRUTHY
)

# Langfuse client singleton (used only on the langfuse branch for per-call flush).
if LANGFUSE_ENABLED:
    assert Langfuse is not None  # LANGFUSE_ENABLED requires the import to have succeeded
    OBSERVER = Langfuse()
else:
    OBSERVER = None


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
    return ls


def _langfuse_wrap(execute_function: Callable, func: Callable) -> Callable:
    """Wrap a langfuse-observed callable with post-invocation flush (sync + async)."""
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*wrapper_args, **wrapper_kwargs):
            result = await execute_function(*wrapper_args, **wrapper_kwargs)
            if OBSERVER:
                OBSERVER.flush()
            return result

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*wrapper_args, **wrapper_kwargs):
        result = execute_function(*wrapper_args, **wrapper_kwargs)
        if OBSERVER:
            OBSERVER.flush()
        return result

    return wrapper


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
        # Branch 1: LangSmith (precedence). traceable is sync+async native; no flush.
        # Positional decorator params are dropped (none used by any call site);
        # only translated kwargs are forwarded. LANGSMITH_ENABLED guarantees the import succeeded.
        if LANGSMITH_ENABLED:
            traceable = cast("Callable[..., Callable]", langsmith_traceable)
            return traceable(**_langsmith_kwargs(dec_kwargs))(func)

        # Branch 2: Langfuse. Preserve existing flush-after-each-call behavior.
        if LANGFUSE_ENABLED:
            if dec_args or dec_kwargs:
                execute_function = langfuse_observe(*dec_args, **dec_kwargs)(func)
            else:
                execute_function = langfuse_observe(func)
            return _langfuse_wrap(cast("Callable[..., object]", execute_function), func)

        # Branch 3: no-op
        return func

    # Bare form: @observable (function passed positionally, no decorator params)
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return _apply(args[0], (), {})

    # Parameterized form: @observable(...) -> returns a decorator
    def decorator(func: Callable) -> Callable:
        return _apply(func, args, kwargs)

    return decorator
