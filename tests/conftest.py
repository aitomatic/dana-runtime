"""
Pytest configuration for LLM tests
"""

import asyncio
from collections.abc import Generator
import os
import sys

import pytest


# --- Environment guard (D8) -------------------------------------------------
# Running bare `pytest` (e.g. via a pyenv shim) silently uses a foreign
# interpreter whose `mcp` may be too old, producing confusing ImportErrors and
# failures instead of a clear message. All test entrypoints (Makefile targets)
# go through `uv run`, which resolves this project's venv. Fail fast and loud
# if the active environment cannot provide the `mcp` API dana requires.
try:
    from mcp.client.streamable_http import streamable_http_client  # noqa: F401
except Exception as exc:
    try:
        import mcp

        mcp_origin = str(mcp.__file__)
    except Exception:
        mcp_origin = "not importable"
    raise RuntimeError(
        "Tests are running against a foreign Python environment.\n"
        f"  interpreter: {sys.executable}\n"
        f"  mcp resolved to: {mcp_origin}\n"
        f"  failure: {exc}\n"
        "Run the suite through the project environment instead:\n"
        "  uv sync && make test      (or)\n"
        "  uv run pytest tests/"
    ) from exc
# ----------------------------------------------------------------------------


# Disable Langfuse for all tests to prevent DuplicateFilter issues
# This must be done before any imports that might trigger Langfuse initialization
os.environ["LANGFUSE_ENABLED"] = "false"

# Skip harness tests by default (they require mock LLM infrastructure)
# Run them explicitly with: pytest tests/harness/ -v
collect_ignore_glob = ["harness/*"]


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "functional: marks tests as functional tests")
    config.addinivalue_line("markers", "regression: marks tests as regression tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "provider: marks tests for specific providers")
    config.addinivalue_line("markers", "live: marks tests as live tests that involve live resources (LLMs)")
    config.addinivalue_line("markers", "requires_api_keys: marks tests as requiring API keys (skip in CI without keys)")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--live", action="store_true", default=False, help="Run live tests that involve live resources (LLMs)")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if not config.getoption("--live"):
        # If --live flag is not provided, skip live tests
        skip_live = pytest.mark.skip(reason="Live tests require --live flag")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)
    else:
        # If --live flag is provided, only run live tests
        skip_non_live = pytest.mark.skip(reason="Only live tests are run with --live flag")
        for item in items:
            if "live" not in item.keywords:
                item.add_marker(skip_non_live)
