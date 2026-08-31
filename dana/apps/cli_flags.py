"""Shared --help/--version flag parsing for Dana CLI entrypoints.

Stdlib argparse only. Every console-script main() builds its parser via
``standard_parser`` and calls ``parse_args()`` FIRST, so ``--help`` and
``--version`` answer before any agent is constructed. Unknown flags get the
argparse usage error (exit code 2), never swallowed into a conversation.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib import metadata
from pathlib import Path


def _version() -> str:
    """Installed package version, falling back to pyproject.toml for source checkouts."""
    try:
        return metadata.version("dana")
    except metadata.PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            if line.startswith("version"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
        return "unknown"


def standard_parser(
    prog: str,
    description: str = "",
    setup: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.ArgumentParser:
    """Build a parser with the standard ``--help``/``--version`` flags.

    ``setup(parser)`` registers program-specific flags (e.g. REPL's --verbose)
    before parsing.
    """
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--version", action="version", version=f"{prog} {_version()}")
    if setup is not None:
        setup(parser)
    return parser
