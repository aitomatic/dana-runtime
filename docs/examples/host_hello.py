"""Hello-world host. Live LLM by default; DANA_MOCK_LLM=1 prints a canned reply."""

import asyncio
import os

from dana.core.runtime.protocols import StreamEvent, StreamEventType
from dana.core.session import AgentSession, TextBlock


class MockAgent:
    async def aquery_stream(self, *, message=None, **kwargs):
        yield StreamEvent(StreamEventType.TEXT_DELTA, "Hello from mock dana!", 0)
        yield StreamEvent(StreamEventType.DONE, None, 0)


async def main():
    session = await AgentSession.create(agent_factory=MockAgent if os.environ.get("DANA_MOCK_LLM") in ("1", "true") else None)
    [print(e.text) async for e in session.prompt([TextBlock(text="hello")]) if e.event_type.name == "ASSISTANT_CONTENT_FINAL"]
    await session._repository.close()


asyncio.run(main())
