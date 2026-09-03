"""Known event-name constants for the dana event bus.

Use these constants instead of raw strings to avoid typos. The bus itself is
string-keyed, so adding a new event type only requires adding a constant here.
S1 ships lifecycle + tool + session names; later milestones emit them.
"""

from __future__ import annotations


# STAR lifecycle (emitted by S2)
SEE_END = "see_end"
THINK_END = "think_end"
ACT_END = "act_end"
REFLECT_END = "reflect_end"

# Tool execution (emitted by M3)
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"

# Session lifecycle (emitted by S4 / agent lifecycle)
SESSION_START = "session_start"
SESSION_RELOAD = "session_reload"
SESSION_SHUTDOWN = "session_shutdown"
