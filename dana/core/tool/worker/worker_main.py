"""Worker entry point — runs as a subprocess reading JSON-line requests from stdin.

The worker:
1. Suppresses all logging (logs go to stderr, not stdout).
2. Creates its own process group (for group-level cleanup).
3. Reads JSON-line requests from stdin.
4. Imports the requested module and calls the specified callable.
5. Writes JSON-line responses to stdout.
6. Handles timeouts via a watchdog thread.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import signal
import sys
import threading
import traceback
from typing import Any


def _suppress_logging() -> None:
    """Redirect all logging to stderr so stdout stays clean for JSON-line IPC."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    # Add a stderr handler so logs don't go to stdout
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    root.addHandler(stderr_handler)
    root.setLevel(logging.WARNING)

    # Also suppress structlog
    try:
        import structlog

        structlog.configure(
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        # Redirect structlog to stderr
        structlog_logger = logging.getLogger("structlog")
        structlog_logger.handlers = []
        structlog_logger.addHandler(stderr_handler)
    except ImportError:
        pass


def _resolve_callable(module_name: str, callable_path: str) -> Any:
    """Resolve a callable from a module by dotted path.

    ``module_name``   — e.g. ``"os"``
    ``callable_path`` — e.g. ``"path.join"`` or ``"MyClass.method"``
    """
    module = importlib.import_module(module_name)
    parts = callable_path.split(".")
    obj = module
    for part in parts:
        obj = getattr(obj, part)
    return obj


def _execute(request: dict[str, Any]) -> dict[str, Any]:
    """Execute a single request and return the response dict."""
    request_id = request["request_id"]
    module_name = request["module"]
    callable_path = request["callable"]
    args = request.get("args", [])
    kwargs = request.get("kwargs", {})
    timeout_ms = request.get("timeout_ms", 0)

    try:
        fn = _resolve_callable(module_name, callable_path)
    except (ImportError, AttributeError) as exc:
        return {
            "type": "response",
            "request_id": request_id,
            "status": "error",
            "result": f"Failed to resolve {module_name}:{callable_path}: {exc}",
        }

    # Execute with optional timeout
    result: Any = None
    error: str | None = None
    cancelled = False

    if timeout_ms > 0:
        # Use a timer-based approach for timeout
        completed = threading.Event()
        timeout_occurred = threading.Event()
        thread_result: list[Any] = []
        thread_error: list[str] = []

        def _run() -> None:
            try:
                val = fn(*args, **kwargs)
                if not timeout_occurred.is_set():
                    thread_result.append(val)
            except Exception as exc:
                if not timeout_occurred.is_set():
                    thread_error.append(f"{type(exc).__name__}: {exc}")
            finally:
                completed.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        timed_out = not completed.wait(timeout_ms / 1000.0)
        if timed_out:
            timeout_occurred.set()
            cancelled = True
            result = f"Execution timed out after {timeout_ms}ms"
        elif thread_error:
            error = thread_error[0]
        else:
            result = thread_result[0] if thread_result else None
    else:
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    if cancelled:
        status = "cancelled"
    elif error:
        status = "error"
        result = error
    else:
        status = "success"

    return {
        "type": "response",
        "request_id": request_id,
        "status": status,
        "result": result,
    }


def worker_entry_point() -> None:
    """Main entry point for the isolated worker subprocess.

    Reads JSON-line requests from stdin, executes them, and writes
    JSON-line responses to stdout. Stdin EOF terminates the worker.
    """
    # Suppress logging — stdout is reserved for JSON-line IPC
    _suppress_logging()

    # Create our own process group for group-level cleanup
    try:
        os.setpgid(0, 0)
    except PermissionError:
        # Already in a different process group (e.g. in tests)
        pass

    # Ignore SIGINT in the worker — parent handles cancellation
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            # Malformed input — write error and continue
            response = {
                "type": "response",
                "request_id": "unknown",
                "status": "error",
                "result": f"Invalid JSON: {line[:200]}",
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            continue

        response = _execute(request)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    worker_entry_point()
