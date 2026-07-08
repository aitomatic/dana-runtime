from __future__ import annotations

import builtins
from typing import Any

import pytest

from dana.core.agent.components.communicator import Communicator


class _FakeAgent:
    agent_type = "dana-librarian"
    object_id = "dana-librarian"
    available_agents: list[Any] = []
    available_resources: list[Any] = []
    available_workflows: list[Any] = []

    def __init__(self) -> None:
        self.aquery_calls: list[dict[str, Any]] = []
        self.query_called = False

    async def aquery(self, **kwargs: Any) -> dict[str, Any]:
        self.aquery_calls.append(kwargs)
        return {"response": "Rows: 1"}

    def query(self, **_kwargs: Any) -> dict[str, Any]:
        self.query_called = True
        raise AssertionError("Communicator.converse must use aquery()")


@pytest.mark.asyncio
async def test_converse_sync_wrapper_uses_async_agent_api_from_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _FakeAgent()
    inputs = iter(["/quit"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    Communicator(agent).converse(initial_message="show rows", session_id="sess:one")

    assert agent.aquery_calls == [{"message": "show rows", "session_id": "sess:one"}]
    assert not agent.query_called
