"""Isolate simulated terminal capabilities from the test runner's terminal."""

import pytest


@pytest.fixture(autouse=True)
def terminal_environment(monkeypatch):
    # Rich ignores explicit console widths under TERM=dumb. Each test supplies
    # its own Console to choose color support and dimensions.
    monkeypatch.setenv("TERM", "xterm-256color")
    for name in ("NO_COLOR", "FORCE_COLOR", "COLUMNS", "LINES"):
        monkeypatch.delenv(name, raising=False)
