"""Tests for ``TimelineRepositoryDefaultsMixin`` — the backward-compat shim
that lets external repos satisfy ``TimelineRepositoryProtocol.list_sessions``
without implementing session discovery."""

from dana.repositories import TimelineRepositoryDefaultsMixin


def test_list_sessions_default_returns_empty():
    assert TimelineRepositoryDefaultsMixin().list_sessions() == []


def test_list_sessions_default_ignores_prefix():
    assert TimelineRepositoryDefaultsMixin().list_sessions(prefix="anything") == []


def test_inherited_subclass_gets_empty_default():
    """Subclasses inheriting the mixin get the no-op by default."""

    class _ExternalRepo(TimelineRepositoryDefaultsMixin):
        pass

    assert _ExternalRepo().list_sessions() == []
    assert _ExternalRepo().list_sessions(prefix="sess") == []


def test_subclass_can_override():
    """Subclasses can override to implement real listing."""

    class _ExternalRepo(TimelineRepositoryDefaultsMixin):
        def list_sessions(self, prefix: str = "") -> list[str]:
            return ["real-session"] if "real-session".startswith(prefix) else []

    assert _ExternalRepo().list_sessions() == ["real-session"]
    assert _ExternalRepo().list_sessions(prefix="real") == ["real-session"]
    assert _ExternalRepo().list_sessions(prefix="nope") == []
