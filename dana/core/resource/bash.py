"""
Bash Resource - Shell command execution for Dana agents.

Executes shell commands in a persistent session with configurable timeout
and optional background execution. Output is captured and returned to the agent.
"""

import asyncio
from dataclasses import dataclass, field
import os
import subprocess
import threading
import time
from typing import Any
import uuid

from structlog import get_logger

from dana.common.protocols.war import tool_use
from dana.core.resource.base_resource import BaseResource
from dana.core.tool.offload import blocking_io, current_physical_work


logger = get_logger()

# Constants
DEFAULT_TIMEOUT_MS = 120000  # 2 minutes
MAX_TIMEOUT_MS = 600000  # 10 minutes
MAX_OUTPUT_CHARS = 30000

#: Slice of the blocking wait between cooperative-cancellation checks. The wait is
#: served in slices so a cancelled dispatch stops waiting within a bounded time
#: (rather than holding an admission permit for the full 10-minute maximum)
#: *without* killing the background command, which is still the model's process.
#: ``Popen.communicate`` preserves partial reads across a retry, so slicing is safe.
WAIT_POLL_INTERVAL_S = 0.1

#: Drain budget for a *non-blocking* drain of a process that is already reaped.
#: Output already at end-of-file costs none of it (the selector reports the pipes
#: ready at once and ``wait`` returns immediately for a reaped process); the budget
#: only bounds the case where a descendant inherited the pipes and keeps them open,
#: so a poll can never turn into a wait.
NOWAIT_DRAIN_TIMEOUT_S = 0.01


@dataclass
class BackgroundTask:
    """Represents a background shell task.

    State is shared between the event loop (background start) and the offload
    worker threads that wait on it, so every field below is guarded by ``lock``.
    The lock is *per task* (not one lock for the whole resource) so waiting on one
    task never blocks polling another.
    """

    task_id: str
    process: subprocess.Popen[str]
    command: str
    description: str
    stdout_buffer: list[str] = field(default_factory=list)
    stderr_buffer: list[str] = field(default_factory=list)
    #: True once the process's pipes have reached end-of-file and the final output is
    #: recorded. A killed command whose descendant still holds the pipes keeps this
    #: ``False`` on purpose: its output stream is still open, so a later poll finishes
    #: the drain and recovers what the command wrote instead of stranding it.
    completed: bool = False
    #: The terminated process's exit code, recorded as soon as it is known: by a drain
    #: that reached end-of-file, or by a kill that delivered a signal. Stays ``None``
    #: while the process is still alive. A set code with ``completed`` False means the
    #: process is finished but a descendant is still holding its output stream open.
    exit_code: int | None = None
    #: Guards this task's mutable state (buffers, completed, exit_code) so a
    #: blocking wait and a concurrent poll/kill cannot tear it (I7).
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: Serializes *waiters* only. Two threads calling ``communicate`` on one
    #: ``Popen`` concurrently would split its output between them, so a second
    #: waiter queues here; a killer is not a waiter, so ``kill_task`` can still
    #: unblock a wait in progress.
    wait_lock: threading.Lock = field(default_factory=threading.Lock)


class BashResource(BaseResource):
    """
    Execute shell commands in a persistent session.

    Provides command execution capabilities with:
    - Configurable timeout (default 2 minutes, max 10 minutes)
    - Background execution support
    - Persistent working directory across commands
    - Output truncation for large outputs

    Security notes:
    - Commands are executed with the same permissions as the Dana process
    - Be cautious with user-provided command strings
    - Consider sandboxing for untrusted inputs
    """

    def __init__(
        self,
        working_directory: str | None = None,
        resource_id: str = "bash",
        **kwargs: Any,
    ):
        """
        Initialize the Bash resource.

        Args:
            working_directory: Initial working directory for commands.
                              Defaults to current working directory.
            resource_id: Resource identifier.
            **kwargs: Additional arguments passed to BaseResource.
        """
        super().__init__(resource_type="bash", resource_id=resource_id, **kwargs)
        self._working_directory = working_directory or os.getcwd()
        self._background_tasks: dict[str, BackgroundTask] = {}
        #: Guards the task registry (membership only; per-task state has its own
        #: lock). Both are ``threading`` locks because blocking waits run on the
        #: offload pool, not on the event loop thread.
        self._tasks_lock = threading.Lock()
        self._env = os.environ.copy()

    def _task_for(self, task_id: str) -> BackgroundTask | None:
        """Look up a background task under the registry lock."""
        with self._tasks_lock:
            return self._background_tasks.get(task_id)

    @tool_use
    async def execute(
        self,
        command: str,
        description: str = "",
        timeout: int | None = None,
        run_in_background: bool = False,
    ) -> dict[str, Any]:
        """
        Execute a shell command.

        Args:
            command: Shell command to execute. Use && to chain dependent commands,
                    ; to chain independent commands. Quote paths with spaces.
            description: Human-readable description of what the command does.
                        Keep brief (5-10 words) for simple commands, add more
                        context for complex piped commands or obscure flags.
            timeout: Timeout in milliseconds. Default 120000 (2 min), max 600000 (10 min).
            run_in_background: If True, run command in background and return immediately.
                              Use get_task_output() to retrieve results later.

        Returns:
            Dict with keys:
            - success: Whether command executed successfully
            - stdout: Command standard output (truncated if > 30000 chars)
            - stderr: Command standard error output
            - exit_code: Process exit code (0 = success)
            - task_id: (background only) ID for retrieving output later
            - truncated: True if output was truncated

        Examples:
            # Simple command
            execute("git status", description="Show working tree status")

            # Chained commands
            execute("npm install && npm test", description="Install deps and run tests")

            # Background execution
            execute("npm run dev", run_in_background=True, description="Start dev server")
        """
        # Validate timeout
        timeout_ms = timeout or DEFAULT_TIMEOUT_MS
        timeout_ms = min(timeout_ms, MAX_TIMEOUT_MS)
        timeout_seconds = timeout_ms / 1000

        logger.info(
            "bash_execute",
            command=command[:100],
            description=description,
            timeout_ms=timeout_ms,
            background=run_in_background,
            cwd=self._working_directory,
        )

        if run_in_background:
            # Background mode is already non-blocking (Popen), no await needed
            return self._execute_background(command, description)
        else:
            return await self._execute_foreground(command, timeout_seconds)

    async def _execute_foreground(
        self,
        command: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Execute command in foreground and wait for completion."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_directory,
                env=self._env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                logger.warning("bash_timeout", command=command[:100], timeout_seconds=timeout_seconds)
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout_seconds} seconds",
                    "exit_code": -1,
                    "truncated": False,
                }

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            truncated = False

            # Truncate output if too large
            if len(stdout) > MAX_OUTPUT_CHARS:
                stdout = stdout[:MAX_OUTPUT_CHARS] + f"\n\n... (truncated, {len(stdout_bytes)} total chars)"
                truncated = True

            if len(stderr) > MAX_OUTPUT_CHARS:
                stderr = stderr[:MAX_OUTPUT_CHARS] + f"\n\n... (truncated, {len(stderr_bytes)} total chars)"
                truncated = True

            logger.info(
                "bash_completed",
                exit_code=process.returncode,
                stdout_len=len(stdout_bytes),
                stderr_len=len(stderr_bytes),
                truncated=truncated,
            )

            return {
                "success": process.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": process.returncode,
                "truncated": truncated,
            }

        except Exception as e:
            logger.error("bash_error", command=command[:100], error=str(e))
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "truncated": False,
            }

    def _execute_background(
        self,
        command: str,
        description: str,
    ) -> dict[str, Any]:
        """Execute command in background and return task ID."""
        task_id = str(uuid.uuid4())[:8]

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._working_directory,
                env=self._env,
            )

            task = BackgroundTask(
                task_id=task_id,
                process=process,
                command=command,
                description=description,
            )
            with self._tasks_lock:
                self._background_tasks[task_id] = task

            logger.info("bash_background_started", task_id=task_id, command=command[:100])

            return {
                "success": True,
                "task_id": task_id,
                "message": f"Command started in background. Use get_task_output(task_id='{task_id}') to retrieve results.",
            }

        except Exception as e:
            logger.error("bash_background_error", command=command[:100], error=str(e))
            return {
                "success": False,
                "task_id": None,
                "message": f"Failed to start background command: {e}",
            }

    @tool_use
    @blocking_io
    def get_task_output(
        self,
        task_id: str,
        wait: bool = True,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve output from a background task.

        Args:
            task_id: Task ID returned by execute() with run_in_background=True
            wait: If True, wait for the command's output stream to close (see
                ``timeout``). If False, report what is already available without
                waiting: a shell that has been reaped is drained within a tiny
                budget, but a descendant that still holds its pipes is left alone.
            timeout: Max wait time in milliseconds, including any queued waiter
                (default 30000, max 600000)

        Returns:
            Dict with keys:
            - success: Whether retrieval succeeded
            - status: "running", "completed", or "failed"
            - stdout: Current or final stdout
            - stderr: Current or final stderr
            - exit_code: Exit code if completed, None if running
            - cancelled: True (only when the wait itself was cancelled) — the
              command keeps running, so poll again with ``wait=False``
        """
        task = self._task_for(task_id)
        if task is None:
            return {
                "success": False,
                "status": "not_found",
                "stdout": "",
                "stderr": f"Task '{task_id}' not found",
                "exit_code": None,
            }

        with task.lock:
            drained = task.completed

        if drained:
            return self._completed_result(task)

        if not wait:
            # A poll never waits for the output stream to close. A running command
            # is reported as-is: reading its pipes would make the poll a wait. A
            # shell the OS has already reaped *is* drained, within the tiny
            # :data:`NOWAIT_DRAIN_TIMEOUT_S` budget, because its stream is normally
            # at end-of-file and its output must not be stranded (a descendant that
            # inherited the pipes, as in ``sleep 300 & echo hi``, keeps them open;
            # then the poll reports the terminal code it has, records no final
            # output yet, and loses nothing: a later poll resumes the same drain).
            with task.lock:
                reaped = task.process.poll() is not None
            if not reaped:
                return self._running_result(task)
            return self._wait_for_task(task, task_id, NOWAIT_DRAIN_TIMEOUT_S, poll=True)

        # One deadline covers acquiring the waiter lock and draining the process.
        timeout_ms = timeout or 30000
        timeout_ms = min(timeout_ms, MAX_TIMEOUT_MS)
        timeout_seconds = timeout_ms / 1000

        return self._wait_for_task(task, task_id, timeout_seconds)

    def _completed_result(self, task: BackgroundTask) -> dict[str, Any]:
        """The recorded final output of a command whose pipes have been drained."""
        with task.lock:
            return {
                "success": True,
                "status": "completed",
                "stdout": "".join(task.stdout_buffer),
                "stderr": "".join(task.stderr_buffer),
                "exit_code": task.exit_code,
            }

    def _running_result(
        self,
        task: BackgroundTask,
        *,
        note: str | None = None,
        cancelled: bool = False,
    ) -> dict[str, Any]:
        """The status of a command whose output has not been fully drained.

        Reading the recorded buffers takes ``task.lock`` for that read only: the
        caller holds no lock and is not waiting on a pipe. ``exit_code`` is reported
        once the *process* is terminal even though the stream is still open (a killed
        command, or a shell whose descendant holds the pipes), so a finished process is
        never described as merely running; it stays ``None`` while the process lives.
        """
        with task.lock:
            stdout = "".join(task.stdout_buffer)
            stderr = "".join(task.stderr_buffer)
            exit_code = task.exit_code
            # ``returncode`` is set once the process is reaped and is ``None`` while it
            # runs, so this never invents a status for a live command.
            reaped_code = task.process.returncode

        result = {
            "success": True,
            "status": "running",
            "stdout": stdout,
            "stderr": stderr if note is None else stderr + note,
            "exit_code": exit_code if exit_code is not None else reaped_code,
        }
        if cancelled:
            result["cancelled"] = True
        return result

    def _wait_for_task(
        self,
        task: BackgroundTask,
        task_id: str,
        timeout_seconds: float,
        *,
        poll: bool = False,
    ) -> dict[str, Any]:
        """Drain the process within one cancellable deadline and return its status.

        This is the only code path that touches a task's pipes, shared by waits and
        polls. ``task.lock`` is taken for brief state reads and writes only: the
        process wait happens under ``task.wait_lock`` *alone*, so a poll or a wait
        can never stall :meth:`list_tasks`, which runs on the event loop and needs
        ``task.lock``.

        Both lock acquisition and ``communicate`` use
        :data:`WAIT_POLL_INTERVAL_S` slices. A queued waiter can therefore time out
        or honour its own cancellation without waiting for another caller's
        up-to-ten-minute process wait. Only the lock holder drains or records
        output, so a timeout or cancellation never splits or duplicates the
        command's output: buffers are appended only after a drain really succeeded.

        ``poll`` bounds the whole path by :data:`NOWAIT_DRAIN_TIMEOUT_S`, never
        queues for the drain lock and never reports cancellation or a timeout note,
        because a poll is a status read rather than a wait.

        The command is never killed here: abandoning a wait is not abandoning the
        model's process. ``Popen.communicate`` accumulates partial reads across
        timeout retries, so the next caller finishes them losslessly.
        """
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        acquired = False
        try:
            while True:
                if not poll:
                    claim = current_physical_work()
                    if claim is not None and claim.cancel.is_set():
                        logger.info("bash_wait_cancelled", task_id=task_id)
                        return self._running_result(
                            task,
                            note="\n(wait cancelled; command still running)",
                            cancelled=True,
                        )

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    note = None if poll else f"\n(still running after {timeout_seconds}s wait)"
                    return self._running_result(task, note=note)

                if not acquired:
                    if poll:
                        # A poll must not queue behind another caller's drain.
                        acquired = task.wait_lock.acquire(blocking=False)
                        if not acquired:
                            return self._running_result(task)
                    else:
                        acquired = task.wait_lock.acquire(timeout=min(WAIT_POLL_INTERVAL_S, remaining))
                        if not acquired:
                            continue
                    with task.lock:
                        already_drained = task.completed
                    if already_drained:
                        return self._completed_result(task)
                    # Recheck cancellation and the shared deadline after waiting
                    # for the lock, before starting any process wait.
                    continue

                try:
                    stdout, stderr = task.process.communicate(timeout=min(WAIT_POLL_INTERVAL_S, remaining))
                except subprocess.TimeoutExpired:
                    continue
                with task.lock:
                    task.stdout_buffer.append(stdout)
                    task.stderr_buffer.append(stderr)
                    task.completed = True
                    task.exit_code = task.process.returncode
                    exit_code = task.exit_code
                logger.info("bash_background_completed", task_id=task_id, exit_code=exit_code)
                return self._completed_result(task)
        finally:
            if acquired:
                task.wait_lock.release()

    @tool_use
    @blocking_io
    def kill_task(self, task_id: str) -> dict[str, Any]:
        """Terminate a running background task.

        Marked offload-eligible (I7): the SIGTERM wait can block for seconds, which
        must not stall the event loop. Nothing loop-bound is touched — the process is
        a plain ``subprocess.Popen`` and the task state is guarded by threading locks.

        Args:
            task_id: Task ID to terminate

        Returns:
            Dict with success status and message
        """
        task = self._task_for(task_id)
        if task is None:
            return {
                "success": False,
                "message": f"Task '{task_id}' not found",
            }

        with task.lock:
            drained = task.completed
            reaped = task.process.poll() is not None
        if drained or reaped:
            # Nothing left to signal. A reaped process is already terminal, and if a
            # descendant still holds its stream open the recorded (signal) code stands
            # while a later poll finishes the drain.
            return {
                "success": True,
                "message": f"Task '{task_id}' already completed",
            }

        exit_code = -15  # SIGTERM
        try:
            task.process.terminate()
        except ProcessLookupError:
            # Already exited between the check above and the signal: record what it
            # produced instead of reporting a failure.
            pass
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to kill task: {e}",
            }

        forced = False
        try:
            task.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            task.process.kill()
            try:
                task.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            exit_code = -9  # SIGKILL
            forced = True

        self._drain_killed_task(task, task_id, exit_code)
        logger.info("bash_task_killed", task_id=task_id, forced=forced)
        return {
            "success": True,
            "message": f"Task '{task_id}' killed (force)" if forced else f"Task '{task_id}' terminated",
        }

    def _drain_killed_task(self, task: BackgroundTask, task_id: str, exit_code: int) -> None:
        """Record a killed command's terminal status, draining its output when bounded.

        A kill must never turn into an unbounded wait: the drain is bounded
        (:data:`NOWAIT_DRAIN_TIMEOUT_S`) and never queues behind a waiter that already
        owns the drain lock. Two outcomes are therefore possible and are kept distinct:

        * the drain reached end-of-file. Everything the command wrote is recorded and
          ``completed`` is set, so the drain is finished for good;
        * the drain did not (a descendant still holds the pipes, or another caller owns
          the lock). ``completed`` stays ``False`` and the tail is left unread; the
          signal exit code is still recorded, and a later poll or listing finishes the
          drain and records whatever the command had written. Marking the task drained
          here would strand that output permanently, because the stream really is still
          open and no caller would ever read it again.
        """
        outcome = self._wait_for_task(task, task_id, NOWAIT_DRAIN_TIMEOUT_S, poll=True)
        with task.lock:
            if task.completed:
                # The drain above finished; keep the recorded code, which is the real
                # one when the command had already exited on its own.
                return
            if task.exit_code is None and outcome["status"] == "completed":
                task.exit_code = outcome["exit_code"]
            if task.process.poll() is not None:
                # The process is gone, so the signal code is authoritative even though a
                # descendant is still holding the stream open.
                task.exit_code = exit_code

    @tool_use
    def list_tasks(self) -> dict[str, Any]:
        """
        List all background tasks.

        Runs on the event loop, so it never waits: for each task it takes
        ``task.lock`` for a brief read only. A command the OS has already reaped is
        drained through the same bounded, non-queuing path a poll uses, so a listing
        can never strand a finished command's output.

        ``status`` reports whether the output stream is finished: "completed" once the
        final output has been drained and recorded, "running" while a descendant still
        holds the stream open. ``exit_code`` is the process's own status, reported as
        soon as it is known; a finished process with ``status`` "running" means only
        that a descendant keeps the stream open, and the output recorded so far remains
        readable through ``get_task_output``.

        Returns:
            Dict with list of tasks and their statuses
        """
        tasks = []
        with self._tasks_lock:
            snapshot = list(self._background_tasks.items())
        for task_id, task in snapshot:
            with task.lock:
                drained = task.completed
                # Only a reaped process is worth draining here; a running one would
                # make this a wait.
                reaped = None if drained else task.process.poll() is not None
                exit_code = task.exit_code

            if reaped:
                outcome = self._wait_for_task(task, task_id, NOWAIT_DRAIN_TIMEOUT_S, poll=True)
                if outcome["status"] == "completed":
                    drained = True
                    exit_code = outcome["exit_code"]
                else:
                    # Undrained: the process may still be terminal (killed, or reaped
                    # with a descendant holding the stream), so report the code it has.
                    with task.lock:
                        exit_code = task.exit_code if task.exit_code is not None else task.process.returncode

            tasks.append(
                {
                    "task_id": task_id,
                    "command": task.command[:50] + "..." if len(task.command) > 50 else task.command,
                    "description": task.description,
                    "status": "completed" if drained else "running",
                    "exit_code": exit_code,
                }
            )

        return {
            "success": True,
            "tasks": tasks,
            "count": len(tasks),
        }

    def set_working_directory(self, path: str) -> bool:
        """
        Change the working directory for future commands.

        Args:
            path: New working directory path

        Returns:
            True if directory exists and was set, False otherwise
        """
        if os.path.isdir(path):
            self._working_directory = os.path.abspath(path)
            logger.info("bash_cwd_changed", new_cwd=self._working_directory)
            return True
        return False

    @property
    def working_directory(self) -> str:
        """Get the current working directory."""
        return self._working_directory
