"""Best-effort release of a provider SDK stream's HTTP response.

Shared by the OpenAI-compatible and Gemini streaming paths. A plain ``async for``
never runs the SDK stream's ``__aexit__``, so a consumer that stops early
(cancellation, host-side break, mid-stream error) would hold the response open
until garbage collection, and for Gemini the same is true of an async generator
abandoned mid-flight.

Failures are logged, never raised: teardown must not mask the real result of the
turn, and the payload is deliberately absent from the log for the same reason.
"""

import inspect
from typing import Any

import structlog


logger = structlog.get_logger()


async def close_stream(stream: Any, *, event: str = "stream_close_failed") -> None:
    """Close ``stream``, awaiting an async closer when the SDK exposes one.

    Handles both SDK shapes: ``aclose()`` (async generators, google-genai) and
    ``close()`` (may be sync or async, as in the OpenAI SDK). No-op when the object
    exposes neither.
    """
    closer = getattr(stream, "aclose", None) or getattr(stream, "close", None)
    if closer is None:
        return
    try:
        result = closer()
        if inspect.isawaitable(result):
            await result
    except Exception as e:  # pragma: no cover - teardown must not raise
        logger.warning(event, error_type=type(e).__name__)
