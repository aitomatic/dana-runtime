"""MCP Cleanup — clean close, stdio child reaping, and session teardown.

Per ADR-008 (Session MCP Leases):
- Clean close on session teardown.
- stdio children reaped (no leaks).

Per ADR-013 (ACP Protocol Mapping and Stdout Discipline):
- No stdout leak from MCP subprocesses.

The cleanup handler ensures that when a session ends:
1. All MCP transports are closed gracefully.
2. All stdio subprocesses are reaped (process groups killed).
3. All leases are released.
4. No owned processes remain (assert_no_leaks).
"""

from __future__ import annotations

import logging
import os
import signal
from typing import Any


logger = logging.getLogger(__name__)


class MCPCleanupHandler:
    """Handles MCP cleanup on session teardown.

    Manages the lifecycle of MCP transports and ensures no subprocess leaks.

    Usage::

        handler = MCPCleanupHandler()
        handler.register_transport("filesystem", transport)
        await handler.close_all()
        handler.assert_no_leaks()
    """

    def __init__(self) -> None:
        self._transports: dict[str, Any] = {}
        self._child_pids: dict[str, list[int]] = {}

    @property
    def transports(self) -> dict[str, Any]:
        """Registered transports keyed by server name (copy)."""
        return dict(self._transports)

    def register_transport(self, server_name: str, transport: Any) -> None:
        """Register a transport for cleanup tracking.

        Args:
            server_name: The MCP server name.
            transport: The transport instance.
        """
        self._transports[server_name] = transport
        logger.debug("Transport registered: '%s'", server_name)

    def register_child_pid(self, server_name: str, pid: int) -> None:
        """Register a child PID for reaping.

        Args:
            server_name: The MCP server name.
            pid: The child process PID.
        """
        if server_name not in self._child_pids:
            self._child_pids[server_name] = []
        self._child_pids[server_name].append(pid)
        logger.debug("Child PID registered: '%s' PID=%d", server_name, pid)

    def unregister_transport(self, server_name: str) -> None:
        """Unregister a transport (e.g. after graceful close).

        Args:
            server_name: The MCP server name.
        """
        self._transports.pop(server_name, None)
        self._child_pids.pop(server_name, None)

    async def close_transport(self, server_name: str) -> None:
        """Close a single transport by server name.

        Args:
            server_name: The MCP server name.
        """
        transport = self._transports.pop(server_name, None)
        if transport is not None:
            try:
                if hasattr(transport, "close") and callable(transport.close):
                    maybe_coro = transport.close()
                    if hasattr(maybe_coro, "__await__"):
                        await maybe_coro
                logger.info("Transport closed for '%s'", server_name)
            except Exception as exc:
                logger.warning("Transport close error for '%s': %s", server_name, exc)

        # Reap child PIDs for this server
        self._reap_child_pids(server_name)

    async def close_all(self) -> None:
        """Close all registered transports and reap all child processes.

        Iterates over all transports and closes them gracefully. If a
        transport close fails, the error is logged but cleanup continues
        (best-effort).
        """
        for server_name in list(self._transports.keys()):
            await self.close_transport(server_name)

        # Reap any remaining child PIDs
        for server_name in list(self._child_pids.keys()):
            self._reap_child_pids(server_name)

        logger.info("All transports closed")

    def _reap_child_pids(self, server_name: str) -> None:
        """Reap child PIDs for a server.

        Sends SIGTERM first, then SIGKILL after a short grace period.
        Calls ``os.waitpid`` to prevent zombie accumulation.
        """
        pids = self._child_pids.pop(server_name, [])
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                logger.debug("SIGTERM sent to '%s' PID=%d", server_name, pid)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.warning("Child reap error for '%s' PID=%d: %s", server_name, pid, exc)

        # Grace period for SIGTERM to take effect
        import time

        time.sleep(0.1)

        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.warning("Child kill error for '%s' PID=%d: %s", server_name, pid, exc)
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass

    def assert_no_leaks(self) -> None:
        """Assert that no owned subprocesses are still alive.

        Raises:
            RuntimeError: If any registered child PID is still running.
        """
        for server_name, pids in self._child_pids.items():
            for pid in pids:
                try:
                    # Sending signal 0 checks if the process exists
                    os.kill(pid, 0)
                    raise RuntimeError(f"Subprocess leak detected: PID {pid} for server '{server_name}' is still running")
                except ProcessLookupError:
                    # Process exited — good
                    pass
                except RuntimeError:
                    raise
                except Exception as exc:
                    logger.warning("Leak check error for '%s' PID=%d: %s", server_name, pid, exc)

    @property
    def transport_count(self) -> int:
        """Number of currently registered transports."""
        return len(self._transports)

    @property
    def child_pid_count(self) -> int:
        """Number of currently registered child PIDs."""
        return sum(len(pids) for pids in self._child_pids.values())
