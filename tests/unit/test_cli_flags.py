"""D8 — CLI trust-signal smoke tests for all six console scripts.

Subprocess-based: each entrypoint must answer --help/--version (exit 0)
before constructing any agent, reject unknown flags with a usage error
(exit != 0), keep bare-invocation behavior, and create nothing under an
empty HOME on the --help path.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


# Console scripts live next to the interpreter of the running test session.
BIN_DIR = Path(sys.executable).parent

# (console script, module for python -m fallback)
ENTRYPOINTS = [
    ("dana-agent", "dana.apps.dana"),
    ("dana-agent-repl", "dana.apps.repl"),
    ("dana-code", "dana.apps.code"),
    ("dana-memory", None),  # entry module is dana.lib.memory.cli:main, no __main__.py
    ("dana-init", "dana.apps.init"),
    ("dana-acp", "dana.apps.acp"),
]


def _pyproject_version() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise AssertionError("version not found in pyproject.toml")


def _run(script: str, module: str | None, *args: str, env: dict | None = None, stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run the console script, falling back to `python -m` if not installed."""
    exe = BIN_DIR / script
    cmd = [str(exe), *args] if exe.is_file() else [sys.executable, "-m", module, *args]
    assert module is not None or exe.is_file(), f"{script} not installed and no module fallback"
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        env=run_env,
        input=stdin,
        cwd=str(Path(__file__).resolve().parents[2]),
    )


@pytest.mark.parametrize("script,module", ENTRYPOINTS)
def test_help_exits_zero_with_usage(script: str, module: str | None):
    result = _run(script, module, "--help")
    assert result.returncode == 0, f"{script} --help rc={result.returncode}\n{result.stderr}"
    assert "usage" in result.stdout.lower(), f"{script} --help stdout:\n{result.stdout}"
    # Help must never construct an agent or enter a REPL/conversation
    assert "Goodbye" not in result.stdout


@pytest.mark.parametrize("script,module", ENTRYPOINTS)
def test_version_exits_zero_with_pyproject_version(script: str, module: str | None):
    result = _run(script, module, "--version")
    assert result.returncode == 0, f"{script} --version rc={result.returncode}\n{result.stderr}"
    assert _pyproject_version() in result.stdout, f"{script} --version stdout:\n{result.stdout}"
    assert result.stdout.strip().endswith(_pyproject_version())


@pytest.mark.parametrize("script,module", ENTRYPOINTS)
def test_bogus_flag_exits_nonzero_with_usage(script: str, module: str | None):
    result = _run(script, module, "--bogus-flag")
    assert result.returncode != 0, f"{script} --bogus-flag rc=0 (swallowed into conversation?)\n{result.stdout}"
    assert "usage" in result.stderr.lower(), f"{script} --bogus-flag stderr:\n{result.stderr}"


def test_help_creates_nothing_under_empty_home(tmp_path: Path):
    empty_home = tmp_path / "empty-home"
    empty_home.mkdir()
    result = _run("dana-code", "dana.apps.code", "--help", env={"HOME": str(empty_home)})
    assert result.returncode == 0, result.stderr
    leftovers = [p.name for p in empty_home.iterdir()]
    assert leftovers == [], f"--help created files under empty HOME: {leftovers}"


def test_bare_dana_code_still_enters_repl_and_exits_on_eof():
    """Bare invocation preserved: dana-code with immediate EOF prints its banner and Goodbye."""
    result = _run("dana-code", "dana.apps.code", stdin="")
    assert result.returncode == 0, f"rc={result.returncode}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    assert "Dana Code" in result.stdout
    assert "Goodbye" in result.stdout


def test_bare_dana_memory_still_errors_on_missing_subcommand():
    """Bare invocation preserved: dana-memory without a subcommand is a usage error."""
    result = _run("dana-memory", None)
    assert result.returncode != 0
    assert "usage" in result.stderr.lower()
