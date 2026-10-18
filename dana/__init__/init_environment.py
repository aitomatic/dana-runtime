"""
Dana Agent - Domain-Aware Neurosymbolic Agents

This package provides the core agent framework for building and managing
specialized AI agents with domain-specific knowledge and capabilities.
"""

import importlib.util
import sys
import types

from dotenv import find_dotenv, load_dotenv


def _module_available(name: str) -> bool:
    """Report whether `name` can be imported, without importing it.

    Probing with `import name` would pay that module's full import cost on every
    `import dana` (langfuse alone is ~200 ms) purely to decide whether to install
    a shim. `find_spec` answers the same question for ~0.1 ms.

    An entry already in `sys.modules` counts as available and short-circuits:
    `find_spec` raises ValueError when that entry has `__spec__ is None`, which
    is how the shims below (and embedder-injected stubs) appear.
    """
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _redact_exception_locals() -> None:
    """Stop structlog tracebacks from rendering frame locals.

    structlog 25 defaults to ``ConsoleRenderer(exception_formatter=RichTracebackFormatter(show_locals=True))``,
    so any ``logger.error(..., exc_info=True)`` prints every local variable of every
    frame. Dana locals routinely hold full ``llm_messages``, provider replay payloads
    (``reasoning_items``/``encrypted_content``), API keys, and prompts, so a single
    error log can write private request material to a terminal, CI log, or transcript.

    The renderer keeps its traceback, colors, and frame context; only the values are
    withheld. The configured processor chain is walked directly, because
    ``ConsoleRenderer.get_active`` raises when no console renderer is configured and
    this must stay a no-op for embedders that swap in their own logging stack.

    Matched by class/module name rather than by importing ``structlog.dev``, which
    pulls in ``rich`` (~90 ms) that many processes never otherwise load.
    """
    import structlog

    get_config = getattr(structlog, "get_config", None)
    if get_config is None:
        # structlog absent, or replaced by the no-op shim installed above.
        return

    for processor in get_config()["processors"]:
        cls = type(processor)
        if cls.__name__ != "ConsoleRenderer" or not cls.__module__.startswith("structlog"):
            continue
        formatter = getattr(processor, "exception_formatter", None)
        if formatter is None:
            continue
        # RichTracebackFormatter is a dataclass; show_locals is the value-bearing flag.
        if type(formatter).__name__ == "RichTracebackFormatter" and getattr(formatter, "show_locals", False):
            formatter.show_locals = False


def _install_structlog_shim() -> None:
    if _module_available("structlog"):
        return

    import logging

    shim = types.ModuleType("structlog")

    class _BoundLogger:
        def __init__(self, logger: logging.Logger) -> None:
            self._logger = logger

        def bind(self, **_kwargs: object) -> "_BoundLogger":
            return self

        def _format(self, message: str, **kwargs: object) -> str:
            if not kwargs:
                return message
            return f"{message} {kwargs}"

        def debug(self, message: str, **kwargs: object) -> None:
            self._logger.debug(self._format(message, **kwargs))

        def info(self, message: str, **kwargs: object) -> None:
            self._logger.info(self._format(message, **kwargs))

        def warning(self, message: str, **kwargs: object) -> None:
            self._logger.warning(self._format(message, **kwargs))

        def error(self, message: str, **kwargs: object) -> None:
            self._logger.error(self._format(message, **kwargs))

    def get_logger(name: str | None = None) -> _BoundLogger:
        logging.basicConfig(level=logging.INFO)
        return _BoundLogger(logging.getLogger(name or "dana"))

    def configure(*_args: object, **_kwargs: object) -> None:
        return None

    def make_filtering_bound_logger(_level: int) -> type[_BoundLogger]:
        return _BoundLogger

    shim.get_logger = get_logger
    shim.configure = configure
    shim.make_filtering_bound_logger = make_filtering_bound_logger

    sys.modules["structlog"] = shim


def init_environment(verbose: bool = False):
    """Load environment variables from .env file.

    Args:
        verbose: If True, print status messages. Default is False (quiet).
    """
    _install_structlog_shim()
    _redact_exception_locals()
    dotenv_path = find_dotenv()
    if verbose:
        print(f"Loading environment variables from {dotenv_path}")
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        load_dotenv()
