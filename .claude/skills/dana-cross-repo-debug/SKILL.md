---
name: dana-cross-repo-debug
description: Debug dana-runtime bugs that surface through the sibling dana-os / dana-console integration. Use this skill whenever the user mentions console-bug, mise run console:dev, localhost:5273, Playground, dana-librarian, LangSmith/Langfuse tracing behavior through the console, or a bug where dana-os appears to use the wrong dana-runtime revision. It guides reproducing through tmux/UI/API, checking backend logs, binding dana-os to the local dana-runtime checkout, fixing runtime code, and verifying the cross-repo path end to end.
category: debugging
keywords: [dana-runtime, dana-os, dana-console, tmux, cross-repo, langsmith, playground]
metadata:
  author: dana-runtime
  version: "1.0.0"
---

# Dana Cross-Repo Debug

Use this skill when a bug is reported in dana-runtime but only appears when the sibling dana-os console is running. The important trap is that dana-os may pin `dana` to a remote git revision, so a fix in `/Users/lam/Desktop/repos/dana-runtime` is invisible until the dana-os workspace points to the local checkout or to a commit containing the fix.

## Repos And Surfaces

Assume these local paths unless the user says otherwise:

- Runtime repo: `/Users/lam/Desktop/repos/dana-runtime`
- Console/integration repo: `/Users/lam/Desktop/repos/dana-os-dana-odb-librarian`
- tmux session: `console-bug`
- Dev command: `mise run console:dev`
- Frontend: `http://localhost:5273/`
- Backend: `http://localhost:8100/`
- UI route: `http://localhost:5273/` -> `Playground` -> `@dana-librarian`
- Runtime librarian agent package: `/Users/lam/Desktop/repos/dana-os-dana-odb-librarian/dana-agents/dana-runtime-librarian`

## Working Rule

Debug the path the user actually runs. Do not stop after a unit test if the bug was reported through the console. Confirm:

1. The console backend imports `dana` from the intended checkout.
2. The failing request reaches dana-runtime code.
3. The tmux log no longer shows the original failure after the fix.

## Start With The Live Console

Inspect the tmux pane before changing code:

```bash
tmux capture-pane -t console-bug -p -S -240
tmux list-panes -t console-bug -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_path} #{pane_current_command}'
```

Check whether the servers are already running:

```bash
lsof -nP -iTCP:8100 -sTCP:LISTEN || true
lsof -nP -iTCP:5273 -sTCP:LISTEN || true
curl -I --max-time 5 http://localhost:5273/ || true
curl -sS -m 5 http://localhost:8100/docs | head || true
```

If `console:dev` refuses because port `8100` is in use, identify the holder. If it is a leaked uvicorn reload pair from the same console session, stop it before retrying. Prefer gentle termination first; use `kill -9` only for a stuck leaked process.

```bash
lsof -nP -iTCP:8100 -sTCP:LISTEN
kill <pid> <pid>
sleep 2
lsof -nP -iTCP:8100 -sTCP:LISTEN || true
```

Start or restart the console in the existing tmux session:

```bash
tmux send-keys -t console-bug 'mise run console:dev' Enter
sleep 8
tmux capture-pane -t console-bug -p -S -120
```

## Confirm Which Runtime Dana-OS Uses

This is the most common source of false verification. `uv run` can reinstall the remote pinned `dana` package at startup.

From the dana-os repo, inspect the source pin:

```bash
sed -n '1,80p' /Users/lam/Desktop/repos/dana-os-dana-odb-librarian/pyproject.toml
git -C /Users/lam/Desktop/repos/dana-os-dana-odb-librarian diff -- pyproject.toml uv.lock
```

Confirm the active interpreter imports local runtime files:

```bash
/Users/lam/Desktop/repos/dana-os-dana-odb-librarian/.venv/bin/python - <<'PY'
import dana
import dana.common.observable
print(dana.__file__)
print(dana.common.observable.__file__)
PY
```

Expected during local debugging:

```text
/Users/lam/Desktop/repos/dana-runtime/dana/__init__.py
/Users/lam/Desktop/repos/dana-runtime/dana/common/observable.py
```

## Bind Dana-OS To Local Dana-Runtime

If dana-os points at a remote git source:

```toml
dana = { git = "https://github.com/aitomatic/dana-runtime.git", rev = "..." }
```

For local debugging, change only the `dana` source to:

```toml
dana = { path = "../dana-runtime", editable = true }
```

Then refresh the lock in dana-os:

```bash
cd /Users/lam/Desktop/repos/dana-os-dana-odb-librarian
uv lock
```

Restart `console-bug`. Watch the startup log: it should reinstall `dana` from the local path, not fetch the remote git revision.

If the user wants a commit-bound integration instead of local editable debugging, first commit the dana-runtime fix, then pin dana-os back to:

```toml
dana = { git = "https://github.com/aitomatic/dana-runtime.git", rev = "<fix-commit-hash>" }
```

Run `uv lock` after changing the pin. Do not invent a commit hash before the fix is committed and available.

## Reproduce Through The Backend First

Use the backend API for fast, deterministic reproduction. It exercises the same backend path the Playground uses, without browser automation noise.

Find the registered librarian package:

```bash
curl -sS -m 20 http://localhost:8100/api/sessions \
  -H 'content-type: application/json' \
  --data '{"operation":"agent.find","args":{},"mode":"live"}' \
  | jq '.result.agents[] | select(.name|contains("librarian"))'
```

Check readiness and mission id:

```bash
curl -sS -m 20 http://localhost:8100/api/sessions \
  -H 'content-type: application/json' \
  --data '{"operation":"agent.describe","args":{"agent_dir":"/Users/lam/Desktop/repos/dana-os-dana-odb-librarian/dana-agents/dana-runtime-librarian"},"mode":"live"}' \
  | jq '.result | {name,operation,mission_id,readiness,errors}'
```

Run the crashing turn:

```bash
curl -sS -m 90 http://localhost:8100/api/sessions \
  -H 'content-type: application/json' \
  --data '{"operation":"agent.ask","args":{"mission_id":"mission:bms-librarian:primary","message":"model dump exclude_none=True mode=json"},"mode":"live"}' \
  | jq '{invocation_id, ok:(.result.answer? != null), detail, answer:(.result.answer? // null)}'
```

Immediately inspect logs:

```bash
tmux capture-pane -t console-bug -p -S -160
```

Classify the first real failure, not the final wrapper error. For example:

- LangSmith serializer failures mention `langsmith/_internal/_serde.py`, `serialize_run_dict`, `Timeline.__repr__`, or `CompressedTimeline`.
- Wrong runtime source shows paths under `.venv/lib/python.../site-packages/dana` rather than `/Users/lam/Desktop/repos/dana-runtime/dana`.
- Event loop failures such as `Cannot run the event loop while another loop is running` may be secondary if they occur after another exception is already unwinding.

## Use The UI When Needed

Use the UI when the bug depends on selection state, readiness gating, mention insertion, websocket streaming, or visual behavior.

Manual path:

1. Open `http://localhost:5273/`.
2. Click `Playground` in the left rail.
3. In the composer, type `@dana-librarian`.
4. Select the suggestion if needed.
5. Send the prompt.
6. Watch `tmux capture-pane -t console-bug -p -S -160`.

For automation, use the repo's browser/devtools skill or a focused Puppeteer script. Prefer one script that keeps a single browser session open over many tiny commands, because session state can be lost between helper invocations.

## Fix In Dana-Runtime

Once the failure is traced to dana-runtime, edit the runtime repo. Use the existing code structure and add a regression test near the failing subsystem.

Common locations for this integration path:

- `dana/common/observable.py`: tracing backend dispatch.
- `dana/common/trace_sanitizer.py`: safe trace payload conversion.
- `dana/common/llm/llm.py`: sync and async LLM boundaries.
- `dana/core/llm/llm_caller.py`: provider invocation, retry, reactive compaction.
- `dana/core/runtime/base.py`: prompt build, native tools, runtime dispatch.
- `dana/core/timeline/*`: timeline and compressed timeline behavior.

After editing, run focused verification in dana-runtime:

```bash
cd /Users/lam/Desktop/repos/dana-runtime
uv run python -m py_compile <touched-python-files>
uv run ruff check <touched-python-files-and-tests>
uv run python -m pytest <focused-tests> -q
```

If the fix is about cross-repo behavior, finish with the live backend call and tmux log check. The final proof is the console operation returning HTTP 200 and the tmux pane showing normal runtime logs instead of the original error.

## Verification Checklist

Before reporting success, verify all that apply:

- `pyproject.toml` in dana-os points to local `../dana-runtime` for local debugging, or to a real fix commit hash for pinned integration.
- `uv.lock` in dana-os reflects the same source.
- The dana-os venv imports `dana` from `/Users/lam/Desktop/repos/dana-runtime/dana`.
- `mise run console:dev` is running in `console-bug`.
- `http://localhost:5273/` and `http://localhost:8100/` respond.
- `agent.describe` for dana-runtime-librarian reports readiness `ready`.
- `agent.ask` reproducer returns an `invocation_id` and `result.answer`.
- `tmux capture-pane` does not show the original crash after the fresh request.
- Focused dana-runtime tests and compile/lint checks pass.

## Report Format

Keep the report concise:

```markdown
Root cause: <one sentence>
Fix: <files changed and behavior>
Cross-repo binding: <local editable path or commit hash>
Verification:
- <command>: <result>
- <tmux/UI/API proof>: <result>
Unresolved questions: <only if any>
```

## Common Pitfalls

- Do not trust a successful local dana-runtime test if dana-os still pins a remote git revision.
- Do not keep restarting `console:dev` without checking leaked `8100` listeners.
- Do not treat old tmux scrollback as current. Restart cleanly or compare timestamps.
- Do not debug browser selectors before proving the backend operation fails.
- Do not commit dana-os dependency changes accidentally if they were only for local debugging.
