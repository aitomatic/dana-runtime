"""Process manager for isolated worker subprocesses (D2).

Manages the lifecycle of isolated worker subprocesses:
- Spawn with process-group isolation
- Send requests / receive responses over stdin/stdout pipes
- Timeout enforcement
- Process-group cleanup on cancel, crash, or context exit
- Leak detection (track all owned PIDs)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import structlog

from dana.core.tool.worker.ipc import (
    WorkerRequest,
    WorkerResponse,
    WorkerStatus,
    decode_response,
    encode_request,
)


logger = structlog.get_logger()


class WorkerProcessManager:
    """Manages an isolated worker subprocess for unsafe mutating tools.

    Each manager owns exactly one worker subprocess. The worker runs in its
    own process group (set by ``worker_entry_point``). On cleanup, the entire
    process group is killed to prevent orphaned children.

    Thread-safe: uses a lock around subprocess I/O.
    """

    def __init__(self, worker_id: str = "default") -> None:
        self._worker_id = worker_id
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._owned_pids: set[int] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the worker subprocess.

        The worker runs ``dana.core.tool.worker.worker_main:worker_entry_point``
        as a subprocess with its own process group.
        """
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import logging, os, sys; "
                    "logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True); "
                    "os.environ['DANA_LOG_LEVEL'] = 'WARNING'; "
                    "os.environ['STRUCTLOG_LOG_LEVEL'] = 'WARNING'; "
                    "# Redirect structlog to stderr before any dana imports\n"
                    "import structlog; "
                    "structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr)); "
                    "from dana.core.tool.worker.worker_main import worker_entry_point; "
                    "worker_entry_point()",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Start in a new process group for group-level cleanup
                start_new_session=True,
            )
            self._started = True
            self._owned_pids.add(self._process.pid)
            logger.info(
                "worker_started",
                worker_id=self._worker_id,
                pid=self._process.pid,
            )

    def is_alive(self) -> bool:
        """Check if the worker subprocess is still running."""
        proc = self._process
        if proc is None:
            return False
        return proc.poll() is None

    def close(self, timeout: float = 5.0) -> None:
        """Gracefully shut down the worker and clean up its process group.

        Sends SIGTERM to the process group, waits ``timeout`` seconds,
        then sends SIGKILL if still alive.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            proc = self._process
            if proc is None:
                return

            try:
                pgid = os.getpgid(proc.pid)
                # Send SIGTERM to the entire process group
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Force kill
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait()
            except (ProcessLookupError, PermissionError, OSError):
                # Process already gone or we lack permission
                pass
            finally:
                self._owned_pids.discard(proc.pid)
                self._process = None
                self._started = False

    # ------------------------------------------------------------------
    # Request / Response
    # ------------------------------------------------------------------

    def send_request(self, request: WorkerRequest) -> WorkerResponse:
        """Send a request to the worker and wait for the response.

        Raises:
            RuntimeError: if the worker is not running or has crashed.
        """
        self.start()
        proc = self._process
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("Worker process is not running")

        request_line = encode_request(request) + "\n"

        with self._lock:
            # Check if process is still alive
            if proc.poll() is not None:
                self._handle_crash(proc)
                return WorkerResponse(
                    request_id=request.request_id,
                    status=WorkerStatus.CRASHED,
                    result=f"Worker process exited with code {proc.returncode}",
                )

            try:
                proc.stdin.write(request_line.encode("utf-8"))
                proc.stdin.flush()
            except BrokenPipeError:
                self._handle_crash(proc)
                return WorkerResponse(
                    request_id=request.request_id,
                    status=WorkerStatus.CRASHED,
                    result="Worker stdin pipe broken",
                )

        # Read response (with timeout)
        response = self._read_response(request.request_id, request.timeout_ms)
        return response

    def _read_response(self, request_id: str, timeout_ms: int) -> WorkerResponse:
        """Read a JSON-line response from the worker's stdout.

        Uses a timeout if specified. Returns a CANCELLED or CRASHED response
        if the read fails or times out.
        """
        proc = self._process
        if proc is None or proc.stdout is None:
            return WorkerResponse(
                request_id=request_id,
                status=WorkerStatus.CRASHED,
                result="Worker process not available",
            )

        # When no timeout, do a blocking readline
        if timeout_ms <= 0:
            try:
                raw = proc.stdout.readline()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                raw = raw.strip()
                if not raw:
                    return WorkerResponse(
                        request_id=request_id,
                        status=WorkerStatus.CRASHED,
                        result="Worker produced no output",
                    )
                return decode_response(raw)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                return WorkerResponse(
                    request_id=request_id,
                    status=WorkerStatus.CRASHED,
                    result=f"Invalid response from worker: {exc}",
                )

        deadline = time.monotonic() + (timeout_ms / 1000.0)
        buffer: list[str] = []

        while time.monotonic() < deadline:
            # Check if process died
            if proc.poll() is not None:
                # Read any remaining output
                remaining = self._drain_stdout(proc)
                if remaining:
                    buffer.append(remaining)
                break

            # Try to read a line (non-blocking-ish)
            remaining_timeout = deadline - time.monotonic()
            if remaining_timeout <= 0:
                break
            line = self._read_line_timeout(proc.stdout, remaining_timeout)
            if line is not None:
                buffer.append(line)
                break

            # Short sleep to avoid busy-wait
            time.sleep(0.01)

        if not buffer:
            # Timeout — kill the process group
            self._kill_process_group()
            return WorkerResponse(
                request_id=request_id,
                status=WorkerStatus.CANCELLED,
                result=f"Request timed out after {timeout_ms}ms",
            )

        raw = "".join(buffer).strip()
        if not raw:
            return WorkerResponse(
                request_id=request_id,
                status=WorkerStatus.CRASHED,
                result="Worker produced no output",
            )

        try:
            return decode_response(raw)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            return WorkerResponse(
                request_id=request_id,
                status=WorkerStatus.CRASHED,
                result=f"Invalid response from worker: {exc}",
            )

    @staticmethod
    def _read_line_timeout(stream: Any, timeout: float) -> str | None:
        """Try to read a line from a stream with a timeout.

        Returns None if the timeout expires before a complete line is available.
        """
        if timeout <= 0:
            return None

        import selectors

        sel = selectors.DefaultSelector()
        try:
            sel.register(stream, selectors.EVENT_READ)
            events = sel.select(timeout=timeout)
            if events:
                line = stream.readline()
                if isinstance(line, bytes):
                    return line.decode("utf-8")
                return line
        except (OSError, ValueError):
            pass
        finally:
            sel.close()
        return None

    @staticmethod
    def _drain_stdout(proc: subprocess.Popen) -> str:
        """Drain any remaining stdout from a terminated process."""
        if proc.stdout is None:
            return ""
        try:
            remaining = proc.stdout.read()
            if isinstance(remaining, bytes):
                return remaining.decode("utf-8")
            return remaining
        except OSError:
            return ""

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _handle_crash(self, proc: subprocess.Popen) -> None:
        """Handle a crashed worker process — log and clean up."""
        stderr_output = ""
        if proc.stderr:
            try:
                stderr_output = proc.stderr.read()
                if isinstance(stderr_output, bytes):
                    stderr_output = stderr_output.decode("utf-8", errors="replace")
            except OSError:
                pass

        logger.error(
            "worker_crashed",
            worker_id=self._worker_id,
            pid=proc.pid,
            returncode=proc.returncode,
            stderr=stderr_output[:2000],
        )
        self._owned_pids.discard(proc.pid)

    def kill_process_group(self) -> None:
        """Kill the entire process group of the worker.

        Public API for external callers (e.g. ToolExecutionEngine.cancel).
        """
        self._kill_process_group()

    def _kill_process_group(self) -> None:
        """Kill the entire process group of the worker."""
        proc = self._process
        if proc is None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=2.0)
        except (ProcessLookupError, PermissionError, OSError, subprocess.TimeoutExpired):
            pass
        finally:
            self._owned_pids.discard(proc.pid)
            self._process = None
            self._started = False

    # ------------------------------------------------------------------
    # Leak detection
    # ------------------------------------------------------------------

    @property
    def owned_pids(self) -> set[int]:
        """Return the set of PIDs owned by this manager."""
        return set(self._owned_pids)

    def assert_no_leaks(self) -> None:
        """Assert that no owned subprocesses are still alive.

        Raises:
            RuntimeError: if any owned PID is still running.
        """
        alive = []
        for pid in list(self._owned_pids):
            try:
                # Sending signal 0 checks if the process exists
                os.kill(pid, 0)
                alive.append(pid)
            except (ProcessLookupError, PermissionError):
                self._owned_pids.discard(pid)

        if alive:
            raise RuntimeError(f"Subprocess leak detected: PIDs {alive} are still running (worker_id={self._worker_id})")

    def __enter__(self) -> WorkerProcessManager:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
