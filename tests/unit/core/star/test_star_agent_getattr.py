"""Regression tests: STARAgent magic __getattr__ removal.

Unknown attributes must raise AttributeError (not silently become converse()
calls / REPL launches), while the documented surface (aquery, converse,
aquery_stream, aquery_text_stream) stays intact. See Sprint 4 story D8.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from dana.core.agent.star_agent import STARAgent


def _make_agent() -> STARAgent:
    """Create a minimal STARAgent without registry or LLM side-effects."""
    return STARAgent(
        agent_type="test-agent",
        auto_register=False,
        enable_skills=False,
        enable_web_search=False,
        enable_code_execution=False,
        enable_assistant=False,
        compress_timeline=False,
    )


class TestMagicGetattrRemoved:
    """Unknown attributes behave like normal Python attributes."""

    def test_unknown_method_raises_attribute_error(self):
        agent = _make_agent()
        with pytest.raises(AttributeError, match="some_unknown_method"):
            agent.some_unknown_method("arg")

    def test_unknown_attribute_access_raises_attribute_error(self):
        agent = _make_agent()
        with pytest.raises(AttributeError):
            agent.hi_how_are_you  # noqa: B018  (attribute access is the test)

    def test_hasattr_unknown_attribute_is_false(self):
        agent = _make_agent()
        assert hasattr(agent, "definitely_not_real") is False

    def test_missing_private_attr_not_swallowed(self):
        """A missing private attr still raises (no magic stub returned)."""
        agent = _make_agent()
        with pytest.raises(AttributeError):
            agent._no_such_private_attr  # noqa: B018

    def test_documented_surface_is_callable(self):
        agent = _make_agent()
        for name in ("aquery", "converse", "aconverse", "aquery_stream", "aquery_text_stream"):
            assert callable(getattr(agent, name)), name


class TestDocumentedSurfaceStillWorks:
    """The documented query surface works without network access.

    aquery_stream/aquery_text_stream flows are covered by
    tests/unit/core/star/test_star_agent_streaming.py.
    """

    def test_converse_delegates_to_communicator(self):
        """converse() forwards to the communicator; no REPL is launched."""
        agent = _make_agent()
        with patch.object(agent, "_communicator") as comm:
            agent.converse(initial_message="hi")
        comm.converse.assert_called_once_with(initial_message="hi", session_id=None)

    @pytest.mark.asyncio
    async def test_aquery_returns_dict(self, tmp_path):
        """Mocked STAR phases: aquery completes and returns a dict result."""
        from dana.config.storage_config import FileStorageConfig
        from dana.core.agent.base_star_agent import EXIT_STAR_LOOP_FLAG
        from dana.repositories.local_file_repository import LocalTimelineRepository
        from dana.repositories.repository_factory import RepositoryFactory, RepositoryType

        factory = RepositoryFactory()
        factory.register(RepositoryType.TIMELINE, LocalTimelineRepository, FileStorageConfig(workspace_folder=str(tmp_path)))
        agent = _make_agent()
        agent._repository_factory = factory

        expected = {"answer": "ok", EXIT_STAR_LOOP_FLAG: True}
        with (
            patch.object(agent, "_see", return_value={"trace_percepts": {}}),
            patch.object(agent, "_think_async", new=AsyncMock(return_value={"trace_thoughts": {}})),
            patch.object(agent, "_act_async", new=AsyncMock(return_value={"trace_outputs": expected})),
        ):
            result = await agent.aquery(message="hello")

        assert isinstance(result, dict)
        assert result == expected
