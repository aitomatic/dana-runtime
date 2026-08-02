"""S4 Extension discovery — scenarios from sprint/plans/S4-extension-discovery.md.

Uses tmp dirs for global/project extension locations and a minimal concrete
BaseSTARAgent (so ``agent.load_extensions()`` / ``agent.on`` are exercised for
real). Each test maps to a plan row (T4.1..T4.8).
"""

from __future__ import annotations

from pathlib import Path

from dana.core.agent.base_star_agent import BaseSTARAgent
from dana.core.ext.event_bus import Event
from dana.core.ext.events import SESSION_RELOAD, TOOL_CALL
from dana.core.ext.extensions import ExtensionManager


# ---------------------------------------------------------------------------
# Minimal concrete agent (instantiable; phase bodies irrelevant to M4)
# ---------------------------------------------------------------------------


class _ExtAgent(BaseSTARAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="ext-test", auto_register=False)

    def _see(self, trace_inputs):
        return {"trace_percepts": dict(trace_inputs)}

    def _think(self, trace_percepts):
        return {"trace_thoughts": dict(trace_percepts)}

    def _act(self, trace_thoughts):
        return {"trace_outputs": dict(trace_thoughts)}

    def _reflect(self, trace_outputs):
        return {"trace_learning": dict(trace_outputs)}

    async def _think_async(self, trace_percepts):
        return {"trace_thoughts": dict(trace_percepts)}

    async def _act_async(self, trace_thoughts):
        return {"trace_outputs": dict(trace_thoughts)}


def _make_manager(agent: _ExtAgent, global_dir: Path, project_dir: Path, *, trust: bool = False) -> ExtensionManager:
    return ExtensionManager(agent, global_dir=global_dir, project_dir=project_dir, trust_project=trust)


# ---------------------------------------------------------------------------
# T4.1 — global extension loads + binds a handler that fires
# ---------------------------------------------------------------------------


_RECORDER = """
from dana.core.ext.events import TOOL_CALL
def setup(agent):
    agent.calls = []
    agent.on(TOOL_CALL, lambda e: agent.calls.append("fired"))
"""


def test_t41_global_extension_loads_and_binds(tmp_path):
    gdir = tmp_path / "global"
    gdir.mkdir()
    (gdir / "rec.py").write_text(_RECORDER)
    agent = _ExtAgent()
    mgr = _make_manager(agent, gdir, tmp_path / "project")

    report = mgr.load_all()

    assert [p.name for p in report.loaded] == ["rec.py"]
    agent.event_bus.emit_sync(Event(TOOL_CALL, {"tool_call_id": "t1", "operation": object()}))
    assert agent.calls == ["fired"]


# ---------------------------------------------------------------------------
# T4.2 / T4.3 — project-local is trust-gated
# ---------------------------------------------------------------------------


def test_t42_project_extension_not_loaded_without_trust(tmp_path):
    pdir = tmp_path / "project"
    pdir.mkdir()
    (pdir / "p.py").write_text("def setup(agent):\n    agent.on('ping', lambda e: None)\n")
    agent = _ExtAgent()
    mgr = _make_manager(agent, tmp_path / "global", pdir, trust=False)

    report = mgr.load_all()

    assert report.loaded == []  # project not trusted → nothing


def test_t43_project_extension_loaded_with_trust(tmp_path):
    pdir = tmp_path / "project"
    pdir.mkdir()
    (pdir / "p.py").write_text("def setup(agent):\n    agent.seen_project = True\n")
    agent = _ExtAgent()
    mgr = _make_manager(agent, tmp_path / "global", pdir, trust=True)

    report = mgr.load_all()

    assert [p.name for p in report.loaded] == ["p.py"]
    assert agent.seen_project is True


# ---------------------------------------------------------------------------
# T4.4 — bad extensions are isolated (skip + warn), good ones still load
# ---------------------------------------------------------------------------


def test_t44_bad_extensions_isolated(tmp_path):
    gdir = tmp_path / "global"
    gdir.mkdir()
    (gdir / "good.py").write_text("def setup(agent):\n    agent.good_loaded = True\n")
    (gdir / "syntax_err.py").write_text("def setup(agent:\n    pass\n")  # SyntaxError
    (gdir / "no_setup.py").write_text("X = 1\n")  # no setup callable
    (gdir / "raises.py").write_text("def setup(agent):\n    raise RuntimeError('boom')\n")
    agent = _ExtAgent()
    mgr = _make_manager(agent, gdir, tmp_path / "project")

    report = mgr.load_all()

    assert [p.name for p in report.loaded] == ["good.py"]
    failed_names = sorted(p.name for p, _ in report.failed)
    assert failed_names == ["no_setup.py", "raises.py", "syntax_err.py"]
    assert agent.good_loaded is True  # the good one still loaded despite siblings failing


# ---------------------------------------------------------------------------
# T4.5 — reload: old handlers unsubscribed, new active, SESSION_RELOAD emitted
# ---------------------------------------------------------------------------


def test_t45_reload_rebinds_handlers(tmp_path):
    gdir = tmp_path / "global"
    gdir.mkdir()
    ext = gdir / "ext.py"
    ext.write_text("def setup(agent):\n    agent.calls = []\n    agent.on('ping', lambda e: agent.calls.append('v1'))\n")
    agent = _ExtAgent()
    mgr = _make_manager(agent, gdir, tmp_path / "project")
    mgr.load_all()
    agent.event_bus.emit_sync(Event("ping", {}))
    assert agent.calls == ["v1"]
    assert len(agent.event_bus.handlers("ping")) == 1

    # rewrite the extension to a new handler
    ext.write_text("def setup(agent):\n    agent.calls = []\n    agent.on('ping', lambda e: agent.calls.append('v2'))\n")

    seen_reload: list = []
    agent.event_bus.subscribe(SESSION_RELOAD, lambda e: seen_reload.append(e.payload))
    report = mgr.reload_all()

    agent.event_bus.emit_sync(Event("ping", {}))
    assert agent.calls == ["v2"]  # new handler, old one unsubscribed
    assert len(agent.event_bus.handlers("ping")) == 1  # no duplicate handler
    assert len(seen_reload) == 1  # SESSION_RELOAD emitted once
    assert [p.name for p in report.loaded] == ["ext.py"]


# ---------------------------------------------------------------------------
# T4.6 — dedup: same resolved file in global+project loads once
# ---------------------------------------------------------------------------


def test_t46_dedup_same_resolved_path(tmp_path):
    # project dir is a symlink-ish alias: put the SAME file path reachable via both
    # by making global dir == project dir
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "once.py").write_text("def setup(agent):\n    agent.count = getattr(agent, 'count', 0) + 1\n")
    agent = _ExtAgent()
    mgr = _make_manager(agent, shared, shared, trust=True)  # both point to same dir

    report = mgr.load_all()

    assert len(report.loaded) == 1  # dedup by resolved path → single load
    assert agent.count == 1


# ---------------------------------------------------------------------------
# T4.7 — no load_extensions call → no extension handlers, no crash
# ---------------------------------------------------------------------------


def test_t47_no_load_no_crash(tmp_path):
    agent = _ExtAgent()
    # agent constructed, extensions never loaded
    assert agent.event_bus.handlers(TOOL_CALL) == []
    agent.event_bus.emit_sync(Event(TOOL_CALL, {}))  # no handlers → no-op, no crash


# ---------------------------------------------------------------------------
# T4.8 — agent.on alias == event_bus.subscribe (returns unsubscribe handle)
# ---------------------------------------------------------------------------


def test_t48_on_alias_subscribes(tmp_path):
    agent = _ExtAgent()
    seen: list = []
    unsub = agent.on("ping", lambda e: seen.append(e.payload["n"]))

    agent.event_bus.emit_sync(Event("ping", {"n": 1}))
    assert seen == [1]
    unsub()
    agent.event_bus.emit_sync(Event("ping", {"n": 2}))
    assert seen == [1]  # unsubscribed → no further fire


# ---------------------------------------------------------------------------
# Adversarial fix-regression tests (Stage 3)
# ---------------------------------------------------------------------------


def test_fix_failing_setup_rolls_back_partial_handlers(tmp_path):
    """[Finding A] A setup that subscribes then raises must roll back its
    handlers — no leak across reloads. After load, the partial handler must
    NOT fire; after reload it still must NOT fire (no accumulation)."""
    gdir = tmp_path / "global"
    gdir.mkdir()
    (gdir / "leak.py").write_text(
        "def setup(agent):\n"
        "    agent.on('ping', lambda e: agent.calls.append('leaked'))\n"
        "    raise RuntimeError('setup fails after subscribing')\n"
    )
    agent = _ExtAgent()
    agent.calls = []
    mgr = _make_manager(agent, gdir, tmp_path / "project")
    report = mgr.load_all()

    assert report.loaded == [] and [p.name for p, _ in report.failed] == ["leak.py"]
    agent.event_bus.emit_sync(Event("ping", {}))
    assert agent.calls == []  # rolled back → partial handler did NOT fire
    assert agent.event_bus.handlers("ping") == []

    # reload must not accumulate leaked handlers either
    mgr.reload_all()
    agent.event_bus.emit_sync(Event("ping", {}))
    assert agent.calls == []
    assert agent.event_bus.handlers("ping") == []


def test_fix_reload_does_not_leak_sys_modules(tmp_path):
    """[Finding B] Repeated reload must not accumulate stale module entries in
    sys.modules (each load used a fresh counter-based name; old ones leaked)."""
    import sys

    gdir = tmp_path / "global"
    gdir.mkdir()
    (gdir / "x.py").write_text("def setup(agent):\n    agent.on('ping', lambda e: None)\n")
    agent = _ExtAgent()
    mgr = _make_manager(agent, gdir, tmp_path / "project")

    before = sum(1 for n in sys.modules if n.startswith("_dana_ext_"))
    for _ in range(5):
        mgr.reload_all()
    after = sum(1 for n in sys.modules if n.startswith("_dana_ext_"))

    assert after == before + 1  # exactly one live module, not 5
