"""Tests for @observable backend dispatch: LangSmith vs Langfuse vs no-op.

Module-level constants (LANGSMITH_ENABLED, LANGFUSE_ENABLED) freeze at import,
so every test reloads `dana.common.observable` after mutating env / mocks.
No real network calls: backends are monkeypatched.
"""

import asyncio
import builtins
import importlib
import sys

import pytest


ENV_VARS = ("LANGSMITH_TRACING", "DANA_LANGSMITH_ENABLED", "LANGFUSE_ENABLED")


def _reload_observable():
    import dana.common.observable as obs

    importlib.reload(obs)
    return obs


@pytest.fixture
def clean_env(monkeypatch):
    """Strip tracing env vars so each test starts from a known-disabled baseline."""
    for k in ENV_VARS:
        monkeypatch.delenv(k, raising=False)


def _install_backend_mocks(monkeypatch, *, traceable=None, observe=None):
    """Patch the real langsmith/langsmith attributes observable binds at import."""
    if traceable is not None:
        monkeypatch.setattr("langsmith.traceable", traceable)
    if observe is not None:
        monkeypatch.setattr("langfuse.observe", observe)
    # Langfuse() must be callable when LANGFUSE_ENABLED; MagicMock satisfies flush().
    from unittest.mock import MagicMock

    monkeypatch.setattr("langfuse.Langfuse", lambda *a, **k: MagicMock(flush=lambda: None))


def test_langsmith_takes_precedence_over_langfuse(monkeypatch, clean_env):
    """F1+F2: both env flags set -> only langsmith.traceable applied, langfuse.observe skipped."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")

    traceable_calls: list = []
    observe_calls: list = []

    def fake_traceable(*a, **k):
        traceable_calls.append((a, k))
        return lambda f: f

    def fake_observe(*a, **k):
        observe_calls.append((a, k))
        return lambda f: f

    _install_backend_mocks(monkeypatch, traceable=fake_traceable, observe=fake_observe)
    obs = _reload_observable()

    assert obs.LANGSMITH_ENABLED is True
    assert obs.LANGFUSE_ENABLED is True  # both on, but langsmith wins (exclusive)

    @obs.observable(name="x")
    def f():
        return 1

    assert f() == 1
    assert traceable_calls, "langsmith.traceable must be applied"
    assert not observe_calls, "langfuse.observe must NOT be applied when langsmith active"


def test_langfuse_only_when_langsmith_unset(monkeypatch, clean_env):
    """F3: langfuse path taken when langsmith env unset."""
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")

    observe_calls: list = []

    def fake_observe(*a, **k):
        observe_calls.append((a, k))
        return lambda f: f

    _install_backend_mocks(monkeypatch, observe=fake_observe)
    obs = _reload_observable()

    assert obs.LANGSMITH_ENABLED is False
    assert obs.LANGFUSE_ENABLED is True

    @obs.observable(name="x")
    def f():
        return 2

    assert f() == 2
    assert observe_calls, "langfuse.observe must be applied"


def test_dana_namespace_alias_enables_langsmith(monkeypatch, clean_env):
    """F1 variant: DANA_LANGSMITH_ENABLED truthy values all enable langsmith."""
    for val in ("true", "1", "yes"):
        monkeypatch.setenv("DANA_LANGSMITH_ENABLED", val)
        _install_backend_mocks(monkeypatch, traceable=lambda *a, **k: lambda f: f)
        obs = _reload_observable()
        assert obs.LANGSMITH_ENABLED is True, f"DANA_LANGSMITH_ENABLED={val!r} should enable"


def test_noop_returns_function_unchanged(monkeypatch, clean_env):
    """F4: neither backend enabled -> decorator returns the SAME function object (identity)."""
    _install_backend_mocks(monkeypatch, traceable=lambda *a, **k: lambda f: f, observe=lambda *a, **k: lambda f: f)
    obs = _reload_observable()

    assert obs.LANGSMITH_ENABLED is False
    assert obs.LANGFUSE_ENABLED is False

    def f():
        return 7

    wrapped = obs.observable(name="x")(f)
    assert wrapped is f, "no-op branch must return the function unchanged (identity)"
    assert wrapped() == 7


def test_kwarg_adapter_as_type_generation_maps_to_chain(monkeypatch, clean_env):
    """F7: langfuse as_type='generation' (no langsmith equiv) -> run_type='chain'."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    captured: dict = {}

    def fake_traceable(*a, **k):
        captured.update(k)
        return lambda f: f

    _install_backend_mocks(monkeypatch, traceable=fake_traceable)
    obs = _reload_observable()

    @obs.observable(name="x", as_type="generation", tags=["t"])
    def f():
        pass

    assert captured["run_type"] == "chain"
    assert captured["name"] == "x"
    assert captured["tags"] == ["t"]


def test_kwarg_adapter_session_user_folded_into_metadata(monkeypatch, clean_env):
    """F7: session_id/user_id merged into metadata (langsmith has no direct equivalent)."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    captured: dict = {}

    def fake_traceable(*a, **k):
        captured.update(k)
        return lambda f: f

    _install_backend_mocks(monkeypatch, traceable=fake_traceable)
    obs = _reload_observable()

    @obs.observable(name="x", session_id="s1", user_id="u1", metadata={"k": "v"})
    def f():
        pass

    assert captured["metadata"] == {"k": "v", "session_id": "s1", "user_id": "u1"}


def test_bare_observable_form_routes_through_langsmith(monkeypatch, clean_env):
    """F5: @observable (no parens) routes through langsmith without double-call."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    traceable_calls: list = []

    def fake_traceable(*a, **k):
        traceable_calls.append((a, k))
        if a and callable(a[0]):
            return a[0]  # bare form returns decorated fn directly
        return lambda f: f

    _install_backend_mocks(monkeypatch, traceable=fake_traceable)
    obs = _reload_observable()

    @obs.observable
    def f():
        return 9

    assert f() == 9
    # Bare form: traceable invoked exactly once with the function (no double-call bug)
    assert len(traceable_calls) == 1


def test_async_function_traced_in_langsmith_branch(monkeypatch, clean_env):
    """F6: async def handled correctly under langsmith (traceable is async-native)."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    traceable_calls: list = []

    def fake_traceable(*a, **k):
        traceable_calls.append((a, k))
        return lambda f: f

    _install_backend_mocks(monkeypatch, traceable=fake_traceable)
    obs = _reload_observable()

    @obs.observable(name="x")
    async def f():
        return 42

    assert asyncio.run(f()) == 42
    assert len(traceable_calls) == 1


def test_shim_installs_passthrough_when_langsmith_missing(monkeypatch, clean_env):
    """F8: _install_langsmith_shim() registers a working passthrough when import fails."""
    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "langsmith", raising=False)

    def blocking_import(name, *args, **kwargs):
        # Block only the first import (forces the shim path); once the shim
        # registers langsmith in sys.modules, defer to the real importer.
        if name == "langsmith" and "langsmith" not in sys.modules:
            raise ModuleNotFoundError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    ie_module = importlib.import_module("dana.__init__.init_environment")
    ie_module._install_langsmith_shim()  # must not raise; must install shim

    import langsmith

    assert callable(langsmith.traceable)

    @langsmith.traceable
    def g():
        return 3

    assert g() == 3  # passthrough, no instrumentation

    @langsmith.traceable(name="x")
    def h():
        return 4

    assert h() == 4  # parameterized passthrough form


def test_langfuse_branch_flushes_after_each_call(monkeypatch, clean_env):
    """SC7 regression: LANGFUSE_ENABLED path must call OBSERVER.flush() after each
    invocation (sync, bare, async). Locks the byte-equivalence invariant the
    `_langfuse_wrap` refactor could silently break."""
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    flush_count = {"n": 0}

    class _FakeClient:
        def flush(self):
            flush_count["n"] += 1

    monkeypatch.setattr("langfuse.Langfuse", lambda *a, **k: _FakeClient())

    def fake_observe(*a, **k):
        # Mirror langfuse.observe dual-form: bare -> observed fn; parameterized -> decorator
        if len(a) == 1 and not k and callable(a[0]):
            return a[0]
        return lambda f: f

    monkeypatch.setattr("langfuse.observe", fake_observe)
    obs = _reload_observable()
    assert obs.OBSERVER is not None

    @obs.observable(name="sync")
    def f_sync():
        return 1

    assert f_sync() == 1
    assert flush_count["n"] == 1, "sync parameterized must flush once"

    @obs.observable
    def f_bare():
        return 2

    assert f_bare() == 2
    assert flush_count["n"] == 2, "bare form must flush once"

    @obs.observable(name="async")
    async def f_async():
        return 3

    assert asyncio.run(f_async()) == 3
    assert flush_count["n"] == 3, "async must flush once"


def test_langsmith_exclusive_over_langfuse_under_async(monkeypatch, clean_env):
    """Exclusivity holds under async: both env set -> langsmith wins, langfuse never called."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    observe_calls: list = []

    def fake_observe(*a, **k):
        observe_calls.append((a, k))
        return lambda f: f

    _install_backend_mocks(monkeypatch, traceable=lambda *a, **k: lambda f: f, observe=fake_observe)
    obs = _reload_observable()

    @obs.observable(name="x")
    async def f():
        return 5

    assert asyncio.run(f()) == 5
    assert not observe_calls, "langfuse.observe must not be called under async either"
