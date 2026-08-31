#!/usr/bin/env python3
"""
Dana Conversational Agent - Entry Point

Dana is a conversational agent that can manage and orchestrate other agents,
resources, and workflows through natural conversation.
"""

import sys

from dana.apps.cli_flags import standard_parser


def main():
    """Main entry point for the Dana conversational agent."""
    standard_parser(
        "dana-agent",
        "Dana conversational agent — manages agents, resources, and workflows via conversation",
    ).parse_args()

    try:
        # Load .env files manually
        from dotenv import find_dotenv, load_dotenv

        dotenv_path = find_dotenv()
        if dotenv_path:
            load_dotenv(dotenv_path, override=True)
        else:
            load_dotenv(override=True)

        from dana.apps.dana.dana_app import DanaApp

        app = DanaApp()
        app.run()

    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
    except Exception as e:
        print(f"Error starting Dana: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
