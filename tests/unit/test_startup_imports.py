"""Behavior tests for lazy startup imports and CLI boot cost.

These assert observable import behavior (which names resolve, which SDKs load,
which process streams stay clean) rather than reading source text, so a refactor
that keeps the behavior keeps passing.
"""

from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter, with tracing env vars cleared."""
    env = {
        k: v for k, v in __import__("os").environ.items() if k not in ("LANGFUSE_ENABLED", "LANGSMITH_TRACING", "DANA_LANGSMITH_ENABLED")
    }
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_import_dana_does_not_load_provider_sdks():
    """`import dana` must not import the LLM client libraries.

    They cost ~0.6 s and a caller may only need `dana.config` or a CLI parser.
    """
    result = _run(
        "import sys\n"
        "import dana\n"
        "for mod in ('anthropic', 'google.genai', 'langfuse'):\n"
        "    assert mod not in sys.modules, f'{mod} imported eagerly by import dana'\n"
        "print('clean')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_import_dana_apps_cli_flags_does_not_load_agent_stack():
    """The --help/--version path must not build the agent runtime.

    `dana.apps.cli_flags` is imported by every console script before parsing
    argv, so anything it pulls in is paid even when only `--help` is requested.
    """
    result = _run(
        "import sys\n"
        "import dana.apps.cli_flags\n"
        "for mod in ('dana.core.agent.star_agent', 'dana.common.llm.llm', 'anthropic'):\n"
        "    assert mod not in sys.modules, f'{mod} imported on the --help path'\n"
        "print('clean')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_lazy_top_level_exports_resolve():
    """Deferring the exports must not change what `from dana import ...` gives."""
    result = _run(
        "import dana\n"
        "from dana import LLM, LLMMessage, LLMResponse, STARAgent\n"
        "assert LLM.__module__ == 'dana.common.llm.llm', LLM.__module__\n"
        "assert LLMMessage.__module__ == 'dana.common.llm.types', LLMMessage.__module__\n"
        "assert LLMResponse.__module__ == 'dana.common.llm.types', LLMResponse.__module__\n"
        "assert STARAgent.__module__ == 'dana.core.agent.star_agent', STARAgent.__module__\n"
        "assert sorted(dana.__all__) == ['LLM', 'LLMMessage', 'LLMResponse', 'STARAgent']\n"
        "assert dana.STARAgent is STARAgent\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_unknown_top_level_attribute_raises_attribute_error():
    """A missing name must still raise AttributeError (hasattr/getattr contracts)."""
    result = _run(
        "import dana\n"
        "assert not hasattr(dana, 'DefinitelyNotAnExport')\n"
        "try:\n"
        "    dana.DefinitelyNotAnExport\n"
        "except AttributeError:\n"
        "    print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_agent_package_exports_resolve_and_do_not_recurse():
    """`from dana.core.agent import ...` resolves every name, and importing the
    repository layer (which the agent package reaches through the timeline) does
    not fail on a half-initialized package."""
    result = _run(
        "from dana.core.agent import BaseAgent, BaseSTARAgent, STARAgent, AssistantAgent\n"
        "assert BaseAgent.__module__ == 'dana.core.agent.base_agent'\n"
        "assert STARAgent.__module__ == 'dana.core.agent.star_agent'\n"
        "assert AssistantAgent.__module__ == 'dana.core.agent.assistant_agent'\n"
        "assert BaseSTARAgent.__module__ == 'dana.core.agent.base_star_agent'\n"
        "import dana.repositories\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_runtime_package_exports_resolve():
    """The runtime package still exposes its documented names."""
    result = _run(
        "from dana.core.runtime import AgentRuntime, ParsedResponse, RuntimeRegistry\n"
        "assert AgentRuntime.__module__ == 'dana.core.runtime.base'\n"
        "assert ParsedResponse.__module__ == 'dana.core.runtime.protocols'\n"
        "assert RuntimeRegistry.__module__ == 'dana.core.runtime.selector'\n"
        "import dana.core.runtime as r\n"
        "assert sorted(r.__all__) == sorted(n for n in dir(r) if n in r.__all__)\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_langfuse_prompt_repository_import_does_not_load_sdk():
    """The repository class must be importable without the Langfuse SDK, and
    importing it must not drag the SDK in (it is on the `import dana` path)."""
    result = _run(
        "import sys\n"
        "from dana.repositories import LangfusePromptRepository\n"
        "assert LangfusePromptRepository.__module__ == 'dana.repositories.langfuse_repository'\n"
        "assert 'langfuse' not in sys.modules, 'SDK imported eagerly'\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_langfuse_client_creation_raises_import_error_without_sdk():
    """A missing SDK fails with an ImportError naming the install extra."""
    result = _run(
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'langfuse':\n"
        "            raise ModuleNotFoundError('blocked for test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "from dana.repositories.langfuse_repository import _get_langfuse_client\n"
        "from dana.config.storage_config import LangfuseStorageConfig\n"
        "try:\n"
        "    _get_langfuse_client(LangfuseStorageConfig(public_key='k', secret_key='s'))\n"
        "except ImportError as exc:\n"
        "    msg = str(exc)\n"
        "    assert 'langfuse' in msg, msg\n"
        "    assert 'dana[observability]' in msg, msg\n"
        "    print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_acp_configure_stderr_logging_keeps_stdout_clean():
    """ACP diagnostics must never reach stdout, even when the embedding process
    redirected stdout onto stderr before structlog was first imported.

    That ordering used to poison structlog's module-global stdout, so every later
    diagnostic landed on the JSON-RPC channel.
    """
    result = _run(
        "import sys\n"
        "real_out, real_err = sys.stdout, sys.stderr\n"
        "sys.stdout = real_err\n"
        "import dana.apps.acp.__main__ as m\n"
        "sys.stdout, sys.stderr = real_out, real_err\n"
        "m.configure_stderr_logging()\n"
        "import structlog\n"
        "structlog.get_logger('acp').info('SENTINEL_DIAGNOSTIC', agent='a1')\n"
        "sys.stdout.flush()\n"
    )
    assert result.returncode == 0, result.stderr
    assert "SENTINEL_DIAGNOSTIC" not in result.stdout, "diagnostic leaked onto the protocol stream"
    assert "SENTINEL_DIAGNOSTIC" in result.stderr, "diagnostic was dropped instead of routed to stderr"
