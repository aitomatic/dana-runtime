"""Entry point for the Dana ACP stdio agent.

Run with::

    dana-acp           # console script
    python -m dana.apps.acp

Stdout is reserved exclusively for JSON-RPC frames. All diagnostics
(structlog, logging) go to stderr so they never corrupt the protocol stream.
"""

from __future__ import annotations

import asyncio
import logging
import sys


def configure_stderr_logging() -> None:
    """Route ALL logging to stderr — stdout is JSON-RPC frames only."""
    import structlog

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    # Redirect structlog to stderr for ACP mode only — scoped here so we
    # never touch structlog behavior for the REPL, CLIs, or other consumers.
    #
    # WriteLoggerFactory, not PrintLoggerFactory: PrintLogger.msg() resolves its
    # destination through structlog's module-global `stdout`, which is bound the
    # first time structlog is imported. An ACP host that redirects sys.stdout
    # (for example `contextlib.redirect_stdout(sys.stderr)`) before that import
    # makes every later diagnostic land on the protocol stream. WriteLogger
    # binds the stream at construction, so stdout purity no longer depends on
    # import order.
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.WriteLoggerFactory(file=sys.stderr),
    )


def main() -> None:
    """Synchronous entry point — answers --help/--version, then runs the async agent."""
    from dana.apps.cli_flags import standard_parser

    standard_parser(
        "dana-acp",
        "Dana ACP agent — JSON-RPC over stdio (stdout is reserved for protocol frames)",
    ).parse_args()

    configure_stderr_logging()
    asyncio.run(main_async())


async def main_async() -> None:
    """Run the DanaACPAgent over ACP stdio JSON-RPC."""
    import acp

    from dana.apps.acp.agent import DanaACPAgent
    from dana.common.observable import ashutdown_observability

    agent = DanaACPAgent()
    try:
        await acp.run_agent(agent, use_unstable_protocol=True)
    finally:
        # Release session, permission, and journal state even when the transport
        # fails mid-connection; then flush and stop the process-wide tracing
        # backend. Both are bounded, so a wedged store or exporter cannot hold
        # the process open.
        try:
            await agent.aclose()
        finally:
            await ashutdown_observability()


if __name__ == "__main__":
    main()
