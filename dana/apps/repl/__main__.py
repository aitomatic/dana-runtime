#!/usr/bin/env python3
"""
Adana REPL - Entry Point

This module serves as the entry point for the Adana interactive REPL.
"""

import sys

from dana.apps.cli_flags import standard_parser


def main():
    """Main entry point for the Adana REPL."""

    def setup(parser):
        parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (default: quiet)")

    args = standard_parser(
        "dana-agent-repl",
        "Adana Interactive REPL",
        setup=setup,
    ).parse_args()

    try:
        # Load .env files (override existing env vars)
        from dotenv import find_dotenv, load_dotenv

        dotenv_path = find_dotenv()
        if dotenv_path:
            load_dotenv(dotenv_path, override=True)
        else:
            load_dotenv(override=True)

        from dana.apps.repl.repl_app import AdanaREPLApp

        app = AdanaREPLApp(verbose=args.verbose)
        app.run()

    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
    except Exception as e:
        print(f"Error starting Adana REPL: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
