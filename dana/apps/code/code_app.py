"""Dana Code Application.

D7 (Sprint 3): the CLI is a host adapter over the host-neutral
:class:`~dana.core.session.agent_session.AgentSession` STAR core (Option B —
in-process, NOT routed through ACP). It constructs an ``AgentSession`` backed
by the Session Journal, drives async turns via ``session.prompt()``, and
renders the resulting ``HostEvent`` stream through ``RichCLIRenderer``.

Rollback: ``DANA_CODE_AGENTSESSION_ENABLED=0`` reverts to the legacy
``DanaCodingAgent`` + renderer-as-Notifiable path, unchanged.
"""

import asyncio
import contextlib
import importlib.metadata
import logging
import os
import sys

from dotenv import find_dotenv, load_dotenv
import structlog


def _load_env():
    """Load environment variables from .env file (overrides existing env vars)."""
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
    else:
        load_dotenv(override=True)


_load_env()

from dana.cli.rich_cli_renderer import RichCLIRenderer


try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    PromptSession = None  # type: ignore
    FileHistory = None  # type: ignore
    Style = None  # type: ignore


def _agentsession_enabled() -> bool:
    """Whether the AgentSession path is active (default on).

    ``DANA_CODE_AGENTSESSION_ENABLED=0`` selects the legacy DanaCodingAgent path.
    """
    return os.environ.get("DANA_CODE_AGENTSESSION_ENABLED", "1") != "0"


class DanaCodeApp:
    """Dana Code - Interactive coding agent with rich CLI.

    Two execution paths, selected at startup by ``DANA_CODE_AGENTSESSION_ENABLED``:

    - **AgentSession path (default):** the CLI is a host adapter over
      ``AgentSession``; turns are async and render via the HostEvent bridge.
    - **Legacy path:** ``DanaCodingAgent`` driven synchronously through the
      renderer's ``Notifiable`` interface (pre-D7 behavior).
    """

    def __init__(self):
        """Initialize the Dana Code application."""
        if sys.platform == "win32":
            term = os.environ.get("TERM", "")
            if term in ["xterm-256color", "xterm-color"] and not os.environ.get("WT_SESSION"):
                os.environ["PROMPT_TOOLKIT_NO_CONSOLE"] = "1"

        # Suppress debug logging
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        )

        # Legacy path state
        self.agent = None
        # AgentSession path state
        self.agent_session = None
        self._repo = None  # keep the journal repository alive for the session
        # D7.3: permission policy state (AgentSession path)
        self._grant_store = None
        self._permission_adapter = None
        self._grant_db = None

        self.renderer = None
        self._prompt_session = None

        if PROMPT_TOOLKIT_AVAILABLE and FileHistory and PromptSession:
            from pathlib import Path

            history_dir = Path.home() / ".adana"
            history_dir.mkdir(exist_ok=True)
            history_file = history_dir / "dana_code_history.txt"

            try:
                self._prompt_session = PromptSession(
                    history=FileHistory(str(history_file)),
                    style=self._get_style(),
                )
            except Exception as e:
                if "NoConsoleScreenBufferError" in str(e) or "console" in str(e).lower():
                    self._prompt_session = None
                else:
                    raise

    def _get_style(self):
        """Get the prompt_toolkit style."""
        if PROMPT_TOOLKIT_AVAILABLE and Style:
            return Style.from_dict(
                {
                    "prompt": "#00aa00 bold",
                }
            )
        return None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        """Run the interactive loop.

        Selects the AgentSession path (default) or the legacy DanaCodingAgent
        path based on ``DANA_CODE_AGENTSESSION_ENABLED``.
        """
        if _agentsession_enabled():
            asyncio.run(self._run_agentsession())
        else:
            self._run_legacy()

    # ------------------------------------------------------------------
    # AgentSession path (D7 — Option B, in-process)
    # ------------------------------------------------------------------

    async def _run_agentsession(self) -> None:
        """Async REPL over AgentSession; renders the HostEvent stream."""
        await self._initialize_session()

        try:
            while True:
                try:
                    user_input = await self._aread_input()

                    if not user_input.strip():
                        continue

                    if user_input.strip().lower() in ["exit", "quit", "bye", "/exit"]:
                        print("\nGoodbye!")
                        break

                    if user_input.strip().startswith("/"):
                        if await self._handle_command_async(user_input.strip()):
                            continue
                        else:
                            break

                    await self._converse_async(user_input)

                except (KeyboardInterrupt, asyncio.CancelledError):
                    # Ctrl-C between turns (at the input prompt) → clear and
                    # resume with a fresh prompt. Mid-turn Ctrl-C is absorbed
                    # inside ``_converse_async`` (cooperative cancel). A second
                    # SIGINT is force-raised by asyncio.run's Runner → exit.
                    continue
                except EOFError:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
                    print("Type /help for commands or /exit to quit.")
        finally:
            await self._close_repo()

    async def _close_repo(self) -> None:
        """Close the journal repository so aiosqlite releases its connection.

        Without this, ``asyncio.run`` shutdown can hang on the abandoned
        aiosqlite worker thread (the "Event loop is closed" errors are the
        symptom). Called from ``_run_agentsession``'s ``finally``. Also closes
        the in-memory permission grant db (D7.3) so it does not leak a worker
        thread on exit.
        """
        if self._repo is not None:
            with contextlib.suppress(Exception):
                await self._repo.close()
            self._repo = None
        if self._grant_db is not None:
            with contextlib.suppress(Exception):
                await self._grant_db.close()
            self._grant_db = None

    async def _initialize_session(self) -> None:
        """Construct an AgentSession backed by the Session Journal.

        Mirrors how ``DanaACPAgent`` builds a session (journal path, owner
        scope, SESSION_CREATED fact, agent factory). ``AgentSession`` is the
        only module reached into here — no STAR core types (ADR-001).
        """
        from datetime import UTC, datetime
        from uuid import uuid4

        from dana.core.session.agent_session import AgentSession
        from dana.core.session.journal.models import SessionRecord
        from dana.core.session.journal.sqlite import SQLiteJournalRepository
        from dana.core.session.models import FactType, JournalFact, OwnerScope

        llm_provider = os.environ.get("DANA_LLM_PROVIDER", "openai")
        model = os.environ.get("DANA_MODEL", "gpt-5")

        journal_path = os.path.expanduser(os.environ.get("DANA_CODE_JOURNAL", os.environ.get("DANA_ACP_JOURNAL", "~/.dana/journal.db")))
        os.makedirs(os.path.dirname(journal_path) or ".", exist_ok=True)
        repo = await SQLiteJournalRepository.open(journal_path)
        self._repo = repo

        owner_id = os.environ.get("USER", "local")
        cwd = os.getcwd()
        scope = OwnerScope(owner_id=owner_id, workspace=cwd)
        session_id = str(uuid4())

        record = SessionRecord.new(session_id, scope)
        init_facts = [
            JournalFact(
                fact_id=str(uuid4()),
                owner_scope=scope,
                session_id=session_id,
                sequence=1,
                fact_type=FactType.SESSION_CREATED,
                timestamp=datetime.now(UTC),
                correlation_id=str(uuid4()),
                causation_id=None,
                schema_version=1,
                payload={},
            ),
        ]
        await repo.create_session(record, init_facts)

        session = AgentSession(
            owner_scope=scope,
            session_id=session_id,
            repository=repo,
        )
        self.agent_session = session

        # D7.3: wire the permission policy (evaluator + grant store) — parity
        # with DanaACPAgent.new_session. Gate by DANA_CODE_PERMISSION_PREFLIGHT.
        self._grant_store = None
        self._permission_adapter = None
        from dana.config.code_capabilities import permission_preflight_enabled

        if permission_preflight_enabled():
            import aiosqlite

            from dana.apps.code.permissions import CLIPermissionAdapter
            from dana.core.policy.evaluator import PolicyEvaluator
            from dana.core.policy.hard_policy import create_default_hard_policy
            from dana.core.policy.modes import PermissionMode
            from dana.core.policy.store_schema import POLICY_SQLITE_DDL
            from dana.core.policy.store_sqlite import SQLiteGrantStore

            grant_db = await aiosqlite.connect(":memory:")
            grant_db.row_factory = aiosqlite.Row
            for stmt in POLICY_SQLITE_DDL:
                await grant_db.execute(stmt)
            await grant_db.commit()
            self._grant_db = grant_db
            grant_store = SQLiteGrantStore(grant_db)
            evaluator = PolicyEvaluator(create_default_hard_policy(), grant_store, PermissionMode.DEFAULT)
            session.set_policy_evaluator(evaluator)
            self._grant_store = grant_store
            self._permission_adapter = CLIPermissionAdapter(evaluator, grant_store, scope)

        self.renderer = RichCLIRenderer(verbose=True, show_tool_calls=True)
        self._print_banner(llm_provider, model)

    async def _aread_input(self) -> str:
        """Read one line of input asynchronously.

        Uses prompt_toolkit's ``prompt_async`` when available; otherwise falls
        back to blocking ``input()`` off-thread.
        """
        if PROMPT_TOOLKIT_AVAILABLE and self._prompt_session:
            return await self._prompt_session.prompt_async("❯ ")
        return await asyncio.to_thread(input, "❯ ")

    async def _converse_async(self, message: str) -> None:
        """Run one turn through AgentSession, rendering the HostEvent stream.

        ``AgentSession.prompt()`` serializes turns: a conflicting prompt raises
        ``SessionBusy`` (caught here). The renderer consumes each HostEvent via
        the D7.2 bridge (``handle_host_event``).
        """
        from dana.core.session.agent_session import SessionBusy, TextBlock

        assert self.agent_session is not None
        assert self.renderer is not None

        blocks = [TextBlock(text=message)]
        gen = self.agent_session.prompt(blocks)
        try:
            async for event in gen:
                self.renderer.handle_host_event(event)
        except SessionBusy:
            print("\n⏳ A turn is already in progress. Please wait for it to finish.\n")
        except (KeyboardInterrupt, asyncio.CancelledError):
            # Ctrl-C mid-turn → cooperative cancel (ADR-005). prompt() catches
            # the cancellation internally and terminalizes the turn as
            # TURN_CANCELLED — a truthful terminal fact rendered by D7.2. If the
            # cancellation propagated here, set the cancel event and drain any
            # remaining events so the terminal is rendered, then RESUME the
            # REPL (absorb the cancel — do not re-raise / exit). Closing the
            # generator ensures its ``async with`` lock releases so the next
            # turn is never stuck-busy.
            with contextlib.suppress(RuntimeError, Exception):
                await self.agent_session.cancel()
            with contextlib.suppress(Exception):
                async for event in gen:
                    self.renderer.handle_host_event(event)
            with contextlib.suppress(Exception):
                await gen.aclose()
            print("\n⏹ Turn interrupted.\n")

    # ------------------------------------------------------------------
    # Legacy path (DANA_CODE_AGENTSESSION_ENABLED=0)
    # ------------------------------------------------------------------

    def _run_legacy(self) -> None:
        """Pre-D7 synchronous REPL over DanaCodingAgent (rollback path)."""
        self._initialize_legacy_agent()

        while True:
            try:
                if PROMPT_TOOLKIT_AVAILABLE and self._prompt_session:
                    user_input = self._prompt_session.prompt("❯ ")
                else:
                    user_input = input("❯ ")

                if not user_input.strip():
                    continue

                if user_input.strip().lower() in ["exit", "quit", "bye", "/exit"]:
                    print("\nGoodbye!")
                    break

                if user_input.strip().startswith("/"):
                    if self._handle_command_legacy(user_input.strip()):
                        continue
                    else:
                        break

                self._converse_legacy(user_input)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Type /help for commands or /exit to quit.")

    def _initialize_legacy_agent(self):
        """Initialize DanaCodingAgent with RichCLIRenderer (rollback path)."""
        # Lazy import: STAR core is reached into ONLY on the legacy rollback path.
        from dana.core.agent.builtin_agents.dana_coding_agent import DanaCodingAgent

        llm_provider = os.environ.get("DANA_LLM_PROVIDER", "openai")
        model = os.environ.get("DANA_MODEL", "gpt-5")

        self.agent = DanaCodingAgent(
            agent_id="dana-code",
            agent_type="dana_coding_agent",
            llm_provider=llm_provider,
            model=model,
        )

        self.renderer = RichCLIRenderer(verbose=True, show_tool_calls=True)
        self.agent.with_notifiable(self.renderer)

        self._print_banner(llm_provider, model)

    def _converse_legacy(self, message: str):
        """Send a message to the legacy agent and display the response."""
        assert self.agent is not None
        assert self.renderer is not None

        try:
            traces = self.agent.query(message=message)
            response = traces.get("response", "")

            # Only print response if renderer is not in verbose mode
            # (verbose renderer already prints the final response)
            if response and not self.renderer.verbose:
                try:
                    from rich.markdown import Markdown

                    self.renderer.console.print()
                    self.renderer.console.print(Markdown(str(response)))
                    self.renderer.console.print()
                except ImportError:
                    print(f"\n{response}\n")

        except Exception as e:
            print(f"\nError: {e}\n")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _print_banner(self, provider: str, model: str) -> None:
        """Print a Rich-formatted startup banner."""
        from rich.console import Console
        from rich.text import Text

        try:
            version = importlib.metadata.version("dana-agent")
        except importlib.metadata.PackageNotFoundError:
            version = "dev"

        cwd = os.getcwd().replace(os.path.expanduser("~"), "~")

        console = Console()
        banner = Text()
        banner.append(f"\n  Dana Code v{version}\n", style="bold")
        banner.append(f"  {provider} · {model}\n", style="dim")
        banner.append(f"  {cwd}\n", style="dim")
        console.print(banner)

    async def _handle_command_async(self, command: str) -> bool:
        """AgentSession-path slash commands (delegates to dana.apps.code.commands).

        Returns True to continue, False to exit.
        """
        from dana.apps.code import commands as cmds

        cmd = command[1:].lower().strip()
        assert self.renderer is not None

        if cmd == "help":
            print(cmds.HELP_TEXT)
            return True
        if cmd == "compact":
            print(cmds.compact_toggle(self))
            return True
        if cmd == "status":
            print(cmds.status_lines(self))
            return True
        if cmd == "permissions":
            print(await cmds.list_permissions_async(self))
            return True
        if cmd == "reset":
            print(await cmds.reset_session(self))
            return True
        if cmd == "model" or cmd.startswith("model "):
            print(await cmds.switch_model(self, cmd))
            return True
        print(f"\nUnknown command: {command}")
        print("Type /help for available commands.\n")
        return True

    def _handle_command_legacy(self, command: str) -> bool:
        """Legacy-path slash commands (sync subset; /model + /permissions are
        AgentSession-only)."""
        from dana.apps.code import commands as cmds

        cmd = command[1:].lower().strip()
        assert self.renderer is not None

        if cmd == "help":
            print(cmds.HELP_TEXT)
            return True
        if cmd == "compact":
            print(cmds.compact_toggle(self))
            return True
        if cmd == "status":
            print(cmds.status_lines(self))
            return True
        if cmd == "reset":
            assert self.agent is not None
            self.agent._timeline.timeline.clear()
            print("\nConversation history reset.\n")
            return True
        if cmd in ("model", "permissions") or cmd.startswith("model "):
            print("\n/model and /permissions are available on the AgentSession path only.\n")
            return True
        print(f"\nUnknown command: {command}")
        print("Type /help for available commands.\n")
        return True
