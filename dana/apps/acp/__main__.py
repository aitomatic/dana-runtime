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
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
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

    agent = DanaACPAgent()
    await acp.run_agent(agent, use_unstable_protocol=True)


if __name__ == "__main__":
    main()
