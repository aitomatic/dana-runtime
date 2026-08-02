"""Extension auto-discovery + hot reload (M4).

Drop-in Python extensions discovered from two locations and bound to an agent's
``EventBus`` via a ``setup(agent)`` factory. See
sprint/plans/S4-extension-discovery.md.

Contract::

    # ~/.dana/extensions/log_tool.py
    from dana.core.ext.events import TOOL_CALL
    def setup(agent):
        agent.on(TOOL_CALL, lambda e: print("tool:", e.payload["operation"].tool_identity.name))

Discovery locations:
- **Global** ``~/.dana/extensions/*.py`` — the user's own, always loaded.
- **Project** ``.dana/extensions/*.py`` — loaded only when
  ``DANA_TRUST_PROJECT_EXTENSIONS=1`` (or ``trust_project=True``); code that is
  not the user's own is gated behind an explicit trust flag (borrow Pi
  ``project_trust``).

A bad extension (syntax error, missing/raising ``setup``) is logged, skipped,
and does NOT abort the rest. Hot reload unsubscribes the previously tracked
handlers, re-discovers, and re-loads — MUST run at idle, never concurrent with
a turn (S1 Finding A: mutating the handler set during ``emit`` races).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import importlib.util
import itertools
import logging
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

from dana.core.ext.event_bus import Event
from dana.core.ext.events import SESSION_RELOAD


if TYPE_CHECKING:
    from dana.core.agent.base_star_agent import BaseSTARAgent

logger = logging.getLogger(__name__)


@dataclass
class LoadReport:
    """Outcome of a load/reload pass."""

    loaded: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.loaded) or bool(self.failed)


@dataclass
class _LoadedExt:
    path: Path
    module: ModuleType
    unsubs: list[Callable[[], None]]


class ExtensionManager:
    """Per-agent extension discovery + hot reload. Owned via ``agent.extensions``."""

    def __init__(
        self,
        agent: BaseSTARAgent,
        *,
        global_dir: Path | None = None,
        project_dir: Path | None = None,
        trust_project: bool | None = None,
    ) -> None:
        self.agent = agent
        self._global_dir = global_dir if global_dir is not None else Path.home() / ".dana" / "extensions"
        self._project_dir = project_dir if project_dir is not None else Path.cwd() / ".dana" / "extensions"
        self._trust_project = trust_project if trust_project is not None else os.environ.get("DANA_TRUST_PROJECT_EXTENSIONS") == "1"
        self._loaded: dict[Path, _LoadedExt] = {}
        self._counter = itertools.count()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> list[Path]:
        """Return de-duplicated (by resolved path) ``*.py`` files to load.

        Global dir always; project dir only when trusted. Sorted for stable order.
        """
        paths: list[Path] = []
        seen: set[Path] = set()

        def collect(directory: Path) -> None:
            if not directory.is_dir():
                return
            for candidate in sorted(directory.glob("*.py")):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(candidate)

        collect(self._global_dir)
        if self._trust_project:
            collect(self._project_dir)
        return paths

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_all(self) -> LoadReport:
        """Discover + load every file. Bad files are skipped + logged, not fatal."""
        report = LoadReport()
        for path in self.discover():
            try:
                module = self._exec_module(path)
            except Exception as exc:  # SyntaxError, ImportError, ...
                logger.warning("extension import failed (%s): %s", path, exc)
                report.failed.append((path, f"import: {exc}"))
                continue
            ext = self._run_setup(module, path)
            if ext is None:
                report.failed.append((path, "setup"))
                continue
            self._loaded[ext.path.resolve()] = ext
            report.loaded.append(path)
        if report.loaded:
            logger.info("extensions loaded: %d, failed: %d", len(report.loaded), len(report.failed))
        return report

    def reload_all(self) -> LoadReport:
        """Unsubscribe tracked handlers, re-discover, re-load, emit ``SESSION_RELOAD``.

        MUST be called at idle (not concurrent with a turn/emit) — unsubscribing
        handlers during ``emit`` races the handler set (S1 Finding A).
        """
        for ext in self._loaded.values():
            for unsub in ext.unsubs:
                try:
                    unsub()
                except Exception:  # pragma: no cover - defensive
                    logger.warning("extension unsubscribe failed: %s", ext.path, exc_info=True)
            # drop the old module so repeated reloads don't leak sys.modules entries
            sys.modules.pop(getattr(ext.module, "__name__", None), None)
        self._loaded.clear()
        report = self.load_all()
        try:
            self.agent.event_bus.emit_sync(
                Event(
                    SESSION_RELOAD,
                    {"loaded": [str(p) for p in report.loaded], "failed": [str(p) for p, _ in report.failed]},
                )
            )
        except Exception:  # pragma: no cover - never block reload on notify
            logger.warning("session_reload emit failed", exc_info=True)
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _exec_module(self, path: Path) -> ModuleType:
        """Execute the file as a fresh module. Raises on syntax/import error.

        Reads source + ``compile`` + ``exec`` directly instead of
        ``spec.loader.exec_module`` so hot-reload ALWAYS reads fresh source. The
        default ``SourceFileLoader`` consults/writes a ``.pyc`` cache keyed on
        ``(mtime, source_size)`` — an edit that keeps the same byte size within
        the same second (e.g. ``"v1"`` → ``"v2"``) is a cache hit and would exec
        stale code, breaking reload semantics.
        """
        name = f"_dana_ext_{path.stem}_{next(self._counter)}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None:
            raise ImportError(f"cannot create module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            source = path.read_text()
            code = compile(source, str(path), "exec")
            exec(code, module.__dict__)
        except Exception:
            sys.modules.pop(name, None)
            raise
        return module

    def _run_setup(self, module: ModuleType, path: Path) -> _LoadedExt | None:
        """Find + call ``setup(agent)``, capturing subscriptions for clean reload.

        Subscriptions made during ``setup`` are captured by temporarily wrapping
        ``bus.subscribe`` (restored in ``finally``). A raising ``setup`` is logged
        and returns None (any partial handlers stay until a later reload). A
        missing/non-callable ``setup`` is treated the same.
        """
        setup = getattr(module, "setup", None)
        if not callable(setup):
            logger.warning("extension %s: no callable setup(agent); skipping", path)
            return None

        unsubs: list[Callable[[], None]] = []
        bus = self.agent.event_bus
        real_subscribe = bus.subscribe

        def recording_subscribe(event_type: str, handler: Any) -> Any:
            unsub = real_subscribe(event_type, handler)
            unsubs.append(unsub)
            return unsub

        bus.subscribe = recording_subscribe  # type: ignore[method-assign]
        try:
            try:
                setup(self.agent)
            except Exception as exc:
                # Transactional rollback: a setup that registered handlers before
                # raising must not leak them (each reload would re-leak). Undo the
                # partial registrations, then treat the extension as failed.
                logger.warning("extension %s: setup raised: %s; rolling back", path, exc)
                for unsub in unsubs:
                    try:
                        unsub()
                    except Exception:  # pragma: no cover - defensive
                        logger.warning("extension rollback unsubscribe failed: %s", path, exc_info=True)
                return None
            return _LoadedExt(path=path, module=module, unsubs=unsubs)
        finally:
            bus.subscribe = real_subscribe  # type: ignore[method-assign]
