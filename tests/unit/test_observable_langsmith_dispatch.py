"""Tests for @observable backend dispatch: LangSmith vs Langfuse vs no-op.

Module-level constants (LANGSMITH_ENABLED, LANGFUSE_ENABLED) freeze at import,
so every test reloads `dana.common.observable` after mutating env / mocks.
No real network calls: backends are monkeypatched.
"""

import asyncio
import importlib
import json
import os
import subprocess
import sys
import textwrap
import threading
import types

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
    if "langsmith" not in sys.modules:
        shim = types.ModuleType("langsmith")
        shim.traceable = lambda *a, **k: lambda f: f
        monkeypatch.setitem(sys.modules, "langsmith", shim)
    if "langfuse" not in sys.modules:
        shim = types.ModuleType("langfuse")
        shim.observe = lambda *a, **k: lambda f: f
        shim.Langfuse = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "langfuse", shim)
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


@pytest.mark.parametrize("enabled_flags", [(flag,) for flag in ENV_VARS] + [ENV_VARS])
@pytest.mark.parametrize("boundary", ["lifecycle", "acp", "code"])
def test_enabled_tracing_without_sdks_stays_disabled(enabled_flags, boundary):
    """Missing optional SDKs stay disabled, including at the real CLI boundaries."""
    code = textwrap.dedent(
        """
        import asyncio
        import importlib
        import runpy
        import socket
        import sys
        import types

        class BlockSDKs:
            def find_spec(self, name, path=None, target=None):
                if name.split(".", 1)[0] in ("langsmith", "langfuse"):
                    raise ModuleNotFoundError("optional SDK absent", name=name)
                return None

        def deny_network(*args, **kwargs):
            raise AssertionError("network disabled in observability regression")

        sys.meta_path.insert(0, BlockSDKs())
        socket.socket.connect = deny_network
        socket.create_connection = deny_network
        import dana.common.observable as obs

        assert obs.LANGSMITH_ENABLED is False
        assert obs.LANGFUSE_ENABLED is False
        assert obs.OBSERVER is None

        def completed():
            return "completed"

        assert obs.observable(completed) is completed
        assert obs.observable(name="completed")(completed) is completed
        boundary = sys.argv[1]
        if boundary == "acp":
            sys.argv = ["dana-acp"]
            runpy.run_module("dana.apps.acp", run_name="__main__")
        elif boundary == "code":
            completed_calls = []
            module = types.ModuleType("dana.apps.code.code_app")
            class LocalApp:
                def run(self):
                    completed_calls.append("completed")
            module.DanaCodeApp = LocalApp
            sys.modules[module.__name__] = module
            sys.argv = ["dana-code"]
            entry = importlib.import_module("dana.apps.code.__main__")
            assert entry.main() is None
            assert completed_calls == ["completed"]
        else:
            obs.flush_observability(timeout=0.05)
            asyncio.run(obs.aflush_observability(timeout=0.05))
            obs.shutdown_observability(timeout=0.05)
            asyncio.run(obs.ashutdown_observability(timeout=0.05))

        assert "langsmith" not in sys.modules
        assert "langfuse" not in sys.modules
        print("clean shutdown")
        """
    )
    env = dict(os.environ, PYTHON_DOTENV_DISABLED="1")
    env.update({flag: "false" for flag in ENV_VARS})
    env.update({flag: "true" for flag in enabled_flags})
    result = subprocess.run([sys.executable, "-c", code, boundary], input="", capture_output=True, text=True, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "clean shutdown" in result.stdout


def test_langfuse_branch_does_not_flush_per_call(monkeypatch, clean_env):
    """Per-span network flush is gone: the SDK batches on its own background
    thread. A synchronous OBSERVER.flush() after every decorated call put a
    network round trip on the critical path of every nested span."""
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    calls = {"flush": 0, "shutdown": 0}

    class _FakeClient:
        def flush(self):
            calls["flush"] += 1

        def shutdown(self):
            calls["shutdown"] += 1

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

    @obs.observable
    def f_bare():
        return 2

    @obs.observable(name="async")
    async def f_async():
        return 3

    assert f_sync() == 1
    assert f_bare() == 2
    assert asyncio.run(f_async()) == 3
    assert calls["flush"] == 0, "decorated calls must not flush per call"

    obs.flush_observability()
    assert calls["flush"] == 1, "explicit lifecycle flush must reach the client"

    obs.shutdown_observability()
    assert calls["shutdown"] == 1, "shutdown must release the client"

    obs.shutdown_observability()
    assert calls["shutdown"] == 1, "shutdown must be idempotent"
    obs.flush_observability()
    assert calls["flush"] == 1, "flush after shutdown must not touch a released client"


def test_lifecycle_helpers_noop_when_no_backend(monkeypatch, clean_env):
    """With tracing off, lifecycle calls are no-ops and import no SDK."""
    obs = _reload_observable()
    assert obs.LANGFUSE_ENABLED is False
    assert obs.LANGSMITH_ENABLED is False
    obs.flush_observability()
    obs.shutdown_observability()
    asyncio.run(obs.aflush_observability())
    asyncio.run(obs.ashutdown_observability())


def test_flush_is_bounded_when_backend_hangs(monkeypatch, clean_env):
    """A wedged exporter cannot hold the process open: the flush runs off-thread
    and we wait at most `timeout`."""
    import time as _time

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    release = threading.Event()

    class _HangingClient:
        def flush(self):
            release.wait(5)

        def shutdown(self):
            release.wait(5)

    monkeypatch.setattr("langfuse.Langfuse", lambda *a, **k: _HangingClient())
    obs = _reload_observable()

    started = _time.monotonic()
    obs.flush_observability(timeout=0.1)
    elapsed = _time.monotonic() - started
    release.set()

    assert elapsed < 2.0, f"flush must respect its timeout, took {elapsed:.2f}s"


@pytest.mark.parametrize(
    "lifecycle",
    ["flush_observability", "shutdown_observability", "aflush_observability", "ashutdown_observability"],
)
@pytest.mark.parametrize("failure_stage", ["resolution", "construction", "flush"])
def test_langsmith_lifecycle_failures_are_type_only_and_off_thread(monkeypatch, clean_env, caplog, lifecycle, failure_stage):
    """Backend resolution and execution share the bounded, payload-free worker."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    secret = "private-provider-payload-must-not-be-logged"
    workers = []

    class SensitiveFailure(RuntimeError):
        def __str__(self):
            raise AssertionError("diagnostics must not stringify backend exceptions")

    def fail():
        workers.append(threading.current_thread())
        if failure_stage == "resolution":
            raise ImportError(secret)
        raise SensitiveFailure(secret)

    class Client:
        def __init__(self):
            if failure_stage == "construction":
                fail()

        def flush(self, **kwargs):
            fail()

    backend = types.ModuleType("langsmith")
    backend.traceable = lambda *a, **k: lambda f: f
    run_trees = types.ModuleType("langsmith.run_trees")
    if failure_stage == "resolution":

        def missing_resolver(name):
            if name == "get_cached_client":
                fail()
            raise AttributeError(name)

        run_trees.__getattr__ = missing_resolver
    else:

        def get_cached_client(**kwargs):
            if failure_stage == "construction":
                fail()
            return Client()

        run_trees.get_cached_client = get_cached_client
    backend.run_trees = run_trees
    monkeypatch.setitem(sys.modules, "langsmith", backend)
    monkeypatch.setitem(sys.modules, "langsmith.run_trees", run_trees)
    obs = _reload_observable()

    with caplog.at_level("DEBUG", logger=obs.__name__):
        result = getattr(obs, lifecycle)(timeout=0.1)
        if lifecycle.startswith("a"):
            asyncio.run(result)

    assert workers
    assert all(worker.daemon and worker.name == "dana-observability-flush" for worker in workers)
    records = [record for record in caplog.records if record.name == obs.__name__]
    expected_type = "ImportError" if failure_stage == "resolution" else "SensitiveFailure"
    assert len(records) == 1
    assert records[0].getMessage() == f"Tracing flush failed: {expected_type}"
    assert records[0].args == (expected_type,)
    assert records[0].exc_info is None
    assert secret not in caplog.text


def test_langsmith_flush_targets_the_cached_tracing_client(monkeypatch, clean_env):
    """Flush and shutdown must reach the client `@traceable` queued runs on.

    ``RunTree.client`` resolves through ``langsmith.run_trees.get_cached_client()``,
    so that instance holds the buffered runs. Building a fresh ``Client()`` instead
    flushes nothing and silently drops the traces the process asked to deliver.
    """
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    events = {"flushes": 0, "resolutions": 0}

    class CachedClient:
        def flush(self, **kwargs):
            events["flushes"] += 1

    cached = CachedClient()

    class FreshClient:
        def __init__(self):  # pragma: no cover - constructing one is the bug
            raise AssertionError("lifecycle must not construct a fresh tracing client")

    backend = types.ModuleType("langsmith")
    backend.traceable = lambda *a, **k: lambda f: f
    backend.Client = FreshClient
    run_trees = types.ModuleType("langsmith.run_trees")

    def get_cached_client(**kwargs):
        events["resolutions"] += 1
        return cached

    run_trees.get_cached_client = get_cached_client
    backend.run_trees = run_trees
    monkeypatch.setitem(sys.modules, "langsmith", backend)
    monkeypatch.setitem(sys.modules, "langsmith.run_trees", run_trees)
    obs = _reload_observable()

    assert obs.LANGSMITH_ENABLED is True

    obs.flush_observability(timeout=0.5)
    assert events == {"flushes": 1, "resolutions": 1}, "flush must hit the cached client exactly once"

    obs.shutdown_observability(timeout=0.5)
    assert events["flushes"] == 2, "shutdown must deliver the same cached client's buffered runs"
    assert events["resolutions"] == 2


def test_shutdown_bounds_langsmith_client_construction(monkeypatch, clean_env):
    """Client resolution/construction cannot block a process-boundary shutdown."""
    import time as _time

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    release = threading.Event()
    started = threading.Event()
    finished = threading.Event()

    class Client:
        def flush(self, **kwargs):
            finished.set()

    backend = types.ModuleType("langsmith")
    backend.traceable = lambda *a, **k: lambda f: f
    run_trees = types.ModuleType("langsmith.run_trees")

    def get_cached_client(**kwargs):
        started.set()
        release.wait(5)
        return Client()

    run_trees.get_cached_client = get_cached_client
    backend.run_trees = run_trees
    monkeypatch.setitem(sys.modules, "langsmith", backend)
    monkeypatch.setitem(sys.modules, "langsmith.run_trees", run_trees)
    obs = _reload_observable()

    try:
        before = _time.monotonic()
        obs.shutdown_observability(timeout=0.05)
        elapsed = _time.monotonic() - before
        assert started.is_set()
        assert not finished.is_set()
        assert elapsed < 1.0, f"shutdown must respect its timeout, took {elapsed:.2f}s"
    finally:
        release.set()
        assert finished.wait(2), "release the isolated daemon before another test reloads the module"


def test_observable_does_not_import_sdk_when_disabled():
    """Tracing disabled: importing `dana.common.observable` must not pull either
    SDK, so a process that never traces pays nothing for them at startup.

    Run in a fresh interpreter: the property is about a clean import, and
    deleting/reimporting the module in-process leaves a stale attribute on the
    `dana.common` package.
    """
    import subprocess

    code = (
        "import sys\n"
        "import dana.common.observable as obs\n"
        "assert obs.LANGFUSE_ENABLED is False, obs.LANGFUSE_ENABLED\n"
        "assert obs.LANGSMITH_ENABLED is False, obs.LANGSMITH_ENABLED\n"
        "assert 'langfuse' not in sys.modules, 'langfuse SDK imported while disabled'\n"
        "assert 'langsmith' not in sys.modules, 'langsmith SDK imported while disabled'\n"
        "print('clean')\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ENV_VARS}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


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


def test_langsmith_processors_sanitize_cyclic_timeline_inputs(monkeypatch, clean_env):
    """LangSmith must not receive raw Dana timelines; SDK JSON serialization
    recurses through live agent/timeline graphs and can crash the caller.
    """
    from dana.core.timeline.timeline import Timeline, TimelineEntry, TimelineEntryType

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    captured: dict = {}

    def fake_traceable(*a, **k):
        captured.update(k)
        return lambda f: f

    _install_backend_mocks(monkeypatch, traceable=fake_traceable)
    obs = _reload_observable()

    @obs.observable(name="x")
    def f(timeline):
        return timeline

    timeline = Timeline()
    entry = TimelineEntry(TimelineEntryType.USER_MESSAGE, "hello")
    entry.metadata["timeline"] = timeline
    timeline.add_entry(entry)

    assert f(timeline) is timeline
    safe_inputs = captured["process_inputs"]({"timeline": timeline, "entry": entry})
    safe_outputs = captured["process_outputs"](timeline)

    json.dumps(safe_inputs)
    json.dumps(safe_outputs)
    assert safe_inputs["timeline"]["__class__"] == "Timeline"
    assert safe_inputs["timeline"]["entry_count"] == 1
    assert safe_inputs["entry"]["metadata"]["timeline"]["__class__"] == "Timeline"
    assert safe_outputs["output"]["__class__"] == "Timeline"
