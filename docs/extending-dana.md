# Extending Dana (v2.0 Extensibility Backbone)

Dana v2.0 ships an **intercept-capable EventBus** as its extensibility backbone.
You can observe, modify, or block almost anything in the agent loop by
registering a handler — either programmatically or as a drop-in extension file.
No subclassing required.

> Substrate: `dana/core/ext/` · shipped milestones M1 (EventBus) → M2 (STAR
> loop) → M3 (Tool engine) → M4 (Extension discovery).

## The 30-second version

Drop a Python file into `~/.dana/extensions/`:

```python
# ~/.dana/extensions/log_tools.py
from dana.core.ext.events import TOOL_CALL

def setup(agent):
    def _log(event):
        op = event.payload["operation"]
        print(f"tool called: {op.tool_identity.name} args={dict(op.arguments)}")
    agent.on(TOOL_CALL, _log)
```

Start Dana — every tool call is now logged. Edit the file, call `/reload`, and
the new handler takes effect immediately. Done.

## What you can hook

| Event          | Emitted by      | Payload                                  | Typical use             |
|----------------|-----------------|------------------------------------------|-------------------------|
| `see_end`      | STAR loop (M2)  | `{"result": <see trace>}`                | Observe/modify percepts |
| `think_end`    | STAR loop (M2)  | `{"result": <think trace>}`              | Inspect reasoning       |
| `act_end`      | STAR loop (M2)  | `{"result": <act trace>}`                | Post-act hook           |
| `reflect_end`  | STAR loop (M2)  | `{"result": <learning trace>}`           | Learning observer       |
| `tool_call`    | Tool engine (M3)| `{"tool_call_id", "operation"}`          | **Block/modify a call** |
| `tool_result`  | Tool engine (M3)| `{"tool_call_id", "operation", "result"}`| Rewrite a result        |
| `session_reload` | Discovery (M4)| `{"loaded", "failed"}`                   | React to `/reload`      |

Constant names live in `dana.core.ext.events` — import them to avoid typos.

## The handler contract

A handler is a plain callable `(event) -> dict | None`:

- **Return `None`** → pass-through (just observing).
- **Return `{"block": True, "reason": "..."}`** → stop the action (a blocked
  `tool_call` is not executed; a blocked STAR phase exits the loop cleanly).
- **Return `{"modify": <dict>}`** → replace the value:
  - `tool_call`: `{"modify": {"arguments": {...}}}` rewrites the call's args.
  - `tool_result` / STAR phases: `{"modify": <new_result>}` replaces the result.

Multiple handlers run in subscribe order; **first non-`None` wins** (later
handlers for that event are skipped). A handler that raises is caught, logged,
and treated as `None` — it can never crash the agent loop.

## Three concrete examples

### 1. Guard: block `rm -rf` and protect `.env` (deny-only policy)

```python
# ~/.dana/extensions/guard.py
from dana.core.ext.events import TOOL_CALL
from dana.core.ext.permission import PermissionPolicy

def setup(agent):
    policy = PermissionPolicy()
    policy.deny(lambda op: "rm -rf blocked"
                if op.tool_identity.name == "bash_tool"
                and "rm -rf" in str(op.arguments.get("command", "")) else None)
    policy.deny(lambda op: "protected path"
                if op.tool_identity.name in ("write", "edit")
                and op.arguments.get("path", "") == ".env" else None)
    agent.on(TOOL_CALL, policy.on_tool_call)
```

A blocked call returns a `policy_block` tool_result with the reason — the agent
sees *why* it was denied.

### 2. Rewrite a tool's arguments on the fly

```python
from dana.core.ext.events import TOOL_CALL

def setup(agent):
    def force_safe_search(event):
        args = dict(event.payload["operation"].arguments)
        if event.payload["operation"].tool_identity.name == "web_search":
            args["safe"] = True
            return {"modify": {"arguments": args}}
        return None
    agent.on(TOOL_CALL, force_safe_search)
```

### 3. Observe every STAR phase in order

```python
from dana.core.ext.events import SEE_END, THINK_END, ACT_END, REFLECT_END

def setup(agent):
    for evt, name in [(SEE_END,"see"),(THINK_END,"think"),(ACT_END,"act"),(REFLECT_END,"reflect")]:
        agent.on(evt, lambda e, n=name: print(f"[{n}]"))
```

## Programmatic API (no file needed)

You don't need the discovery layer — register handlers directly on any agent:

```python
agent = STARAgent(...)
unsub = agent.on(TOOL_CALL, my_handler)   # returns an unsubscribe callable
# ...
unsub()                                    # remove the handler
```

`agent.on(event, handler)` is a thin alias for `agent.event_bus.subscribe(...)`.
The bus is **per-agent** (correct session scope, never a global).

## Extension discovery & hot reload

| Location                  | When loaded                  |
|---------------------------|------------------------------|
| `~/.dana/extensions/*.py` | Always (your home = trusted) |
| `.dana/extensions/*.py`   | Only if `DANA_TRUST_PROJECT_EXTENSIONS=1` |

- Each file must define `setup(agent)`. Files with syntax errors, a missing
  `setup`, or a raising `setup` are **skipped and warned** — they never crash
  startup, and other extensions still load.
- `agent.load_extensions()` discovers + loads (call it after constructing the
  agent; it is not auto-called).
- `agent.reload_extensions()` unsubscribes the old handlers, re-reads every file
  from disk, re-binds, and emits `session_reload`. Call it at idle (not during a
  turn). A failing `setup` rolls back any handlers it registered partway.

```python
agent.load_extensions()     # startup
# ... edit ~/.dana/extensions/guard.py ...
agent.reload_extensions()   # new rules live immediately
```

## Rules of the road

- **Never mutate `event.payload`** — return a `{"modify": ...}` dict instead.
  (`Operation.arguments` is read-only and enforces this.)
- **Subscribe at setup time**, never mid-turn. The bus is not thread-safe for
  concurrent subscribe/unsubscribe against `emit`.
- **Only literal `True` blocks** — `{"block": True}`. (Truthy non-bools pass
  through, so `{"block": "false"}` does *not* block.)
- Project-local extensions run **arbitrary Python** with no sandbox in v0.1 —
  only enable `DANA_TRUST_PROJECT_EXTENSIONS` for repos you trust.

## Where to look next

- `dana/core/ext/event_bus.py` — the bus contract (first-wins, never-raises).
- `dana/core/ext/permission.py` — `PermissionPolicy` (deny-only rules).
- `dana/core/ext/guard.py` — a bundled reference guard.
- `sprint/plans/S{1..4}-*.md` — the design + decisions behind each milestone.
