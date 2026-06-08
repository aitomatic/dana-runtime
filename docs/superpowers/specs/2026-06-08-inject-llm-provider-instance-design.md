# Design: Inject an `LLMProvider` instance through STARAgent → Runtime → call site

**Date:** 2026-06-08
**Branch:** `develop`
**Status:** Approved — ready for implementation plan

## Problem

`STARAgent` can only be told *which* LLM to use via `llm_provider: str` + `model: str`. The
provider is then (re)built from `.env` deep inside the stack. Callers that have already
constructed a `dana.common.llm.providers` instance (custom `base_url`, pre-authed client,
non-env credentials, a shared/pooled client) cannot hand it in. Goal: pass a pre-built
`LLMProvider` **instance** from the agent layer down to the actual API call site, and allow
re-pointing it at runtime.

## Key finding: instance injection is mostly already wired

The polymorphism exists one layer down; the agent ctor is the only true gap.

- `LLM.__init__(provider: str | LLMProvider, model=None)` — already accepts an instance
  (`dana/common/llm/llm.py`). An `LLMProvider` instance sets `provider_name="custom"`.
- `AgentRuntime.set_llm(llm)` — already fans the LLM to its `LLMCaller`
  (`dana/core/runtime/base.py:150`).
- `RLMResource.__init__` builds `LLM(provider=..., model=...)` (`rlm_resource.py:157`) — the
  `provider=` arg already accepts an instance.
- Providers expose `self.model` and `self.name` (e.g. `openai_compatible_base.py:294`,
  `anthropic.py:212`), so name+model can be derived from an instance.

## Why runtime + LLMCaller need no new params

Normalization (`LLMProvider → LLM`) happens **once**, at the agent boundary in
`_apply_llm_provider`. Runtime and `LLMCaller` keep speaking their existing `LLM` currency
(`llm=` ctor arg + `set_llm`). Pushing the raw provider instance two layers deeper would
duplicate the wrapping in three places — DRY violation for zero gain.

`LLMCaller._resolve_llm()` (the real call site, `llm_caller.py:496`) resolves in priority order:

1. `self._llm` (set via `set_llm`) — **short-circuits everything**
2. `agent.llm_client` — read lazily only when `self._llm` is None
3. build fresh `LLM(provider=self._provider, model=self._model)` from name strings

Therefore `runtime.set_llm(llm)` → `LLMCaller.set_llm(llm)` sets priority-#1, so `.create()` uses
exactly the injected provider. **The explicit `set_llm` is mandatory** for the mid-session
re-point: priority #2 (`agent.llm_client`) is shadowed once the caller has cached a `self._llm`
from a prior call, so setting `agent._llm_client` alone would silently no-op an in-flight caller.

## The three injection sinks

An injected provider must reach **all three**, or split-brain results (agent reads the
injected provider while the actual call site silently builds a different one from `.env`):

| Sink | Today | Mutation surface |
|------|-------|------------------|
| `STARAgent._llm_client` | lazy `LLM(provider=str, model=str)` | direct assign |
| `runtime._llm_caller._llm` | `LLMCaller` builds own from name/model | `runtime.set_llm(llm)` — **exists** |
| `LTMemory → RLMResource._llm` | `LLM(provider=str, model=str)` | new thin `llm` passthrough + `set_llm` |

## Chosen approach: dedicated `llm_provider_instance` param + central fan-out method

Add a **new** param `llm_provider_instance: LLMProvider | None` rather than widening
`llm_provider` to `str | LLMProvider`. Rationale:

- `llm_provider` already means a **name string** at 51 call sites (`llm_provider="openai"`).
  Redefining it as an instance would break them; keeping it string-typed is backward-compatible.
- Distinct named params (no `str | LLMProvider` union) read unambiguously — the caller's intent
  is explicit at the call site, no isinstance branching to reason about.

Rejected: (1) widen `llm_provider` to a union — breaks/obscures the 51 existing string callers;
(2) accept a pre-built `LLM` only — caller wants to pass the provider, not pre-wrap it.

### Parameter table

| Param | Type | Role |
|-------|------|------|
| `llm_provider` | `str \| None` | provider name (legacy, unchanged) |
| `model` | `str \| None` | model name (legacy, unchanged) |
| `llm_provider_instance` | `LLMProvider \| None` | pre-built instance — **wins when set** |

### Spine: a single fan-out method

Both the constructor and the public runtime setter call this. It is the only place that
knows about all three sinks.

```python
def _apply_llm_provider(self, llm_provider_instance=None, llm_provider=None, model=None):
    # normalize → LLM (instance wins)
    if llm_provider_instance is not None:
        if llm_provider is not None or model is not None:
            logger.debug("llm_provider_instance set; ignoring llm_provider/model args")
        llm = LLM(provider=llm_provider_instance)   # provider_name → "custom"; model from instance
    else:                                            # legacy name/model path
        llm = LLM(provider=llm_provider, model=model)

    self._llm_client = llm
    if self._runtime is not None:
        self._runtime.set_llm(llm)                   # → LLMCaller.set_llm
    if self._ltmemory is not None:
        self._ltmemory.set_llm(llm)                  # new thin setter → RLMResource.set_llm

def set_llm_provider(self, llm_provider_instance=None, llm_provider=None, model=None):
    """Re-point this agent (and its runtime + LTMemory) at a new provider/LLM.

    Pass `llm_provider_instance` (an LLMProvider) for instance injection, or
    `llm_provider` (name str) + `model` for the legacy path. Instance wins.
    """
    self._apply_llm_provider(llm_provider_instance, llm_provider, model)
```

### Constructor branch (instance wins)

```python
if llm_provider_instance is not None:
    name  = getattr(llm_provider_instance, "name", None) or "custom"
    model = getattr(llm_provider_instance, "model", None)   # instance wins; `model` arg ignored
else:
    name  = llm_provider or config_manager.get_first_available_provider() or "anthropic"

if runtime is None:
    runtime = RuntimeRegistry.select_codec_runtime(provider=name, model=model, codec=codec)
self._runtime = runtime
# ... build self._ltmemory (if ltmemory_path) ...
self._apply_llm_provider(llm_provider_instance, llm_provider, model)
```

`name` is cosmetic in this path: `select_codec_runtime` returns `CodecRuntimeWith[out]NativeToolUse`
based on the codec, not provider-specific runtimes, and `set_llm` overrides whatever LLM the
runtime built. The string is metadata only.

### Signature changes

- `STARAgent.__init__`: add `llm_provider_instance: LLMProvider | None = None`.
  `llm_provider: str | None` and `model: str | None` stay unchanged (backward-compatible).
- New public `STARAgent.set_llm_provider(llm_provider_instance=None, llm_provider=None, model=None)`.
- `LTMemory.__init__`: add `llm: LLM | None = None`; when present, pass to `RLMResource`
  instead of `llm_provider`/`llm_model`. New `LTMemory.set_llm(llm)` → `RLMResource.set_llm`.
- `RLMResource.__init__`: add `llm: LLM | None = None`; when present, `self._llm = llm` and
  skip the `LLM(provider=str, model=str)` build. New `RLMResource.set_llm(llm)` →
  `self._llm = llm`.

### Ordering constraint

`_apply_llm_provider` pushes into `self._runtime` and `self._ltmemory`, so it MUST run after
both are constructed. The lazy `llm_client` property remains as a fallback, but the ctor now
resolves eagerly via the fan-out.

## Edge cases

- **Instance + pre-built `runtime`:** instance wins — `set_llm` mutates the passed runtime.
  Not an error (per decision).
- **`llm_provider`/`model` + `llm_provider_instance`:** the string args are ignored (instance
  binds its own model). `logger.debug`, not a raise.
- **`set_llm_provider` mid-session:** re-points all three sinks; in-flight calls hold their own
  `llm` ref, so no torn state.
- **Legacy `llm_provider="openai"` string:** unchanged — `llm_provider_instance is None` path.

## Testing

1. `STARAgent(llm_provider_instance=OpenAIProvider(base_url=..., model=...))` → assert
   `agent.llm_client.provider is instance` **and** `runtime._llm_caller._llm is agent.llm_client`
   (proves no split-brain).
2. `set_llm_provider(llm_provider_instance=other)` → both sinks now reference `other`.
3. `ltmemory_path` set + injected provider → `RLMResource._llm` uses the injected provider.
4. Regression: `llm_provider="openai"` string path behaves exactly as before.

## Out of scope

- Fallback-provider (`ProviderConfig`) wiring in `LLMCaller` — unchanged.
- Streaming mixin — uses the same `runtime`/`llm_client`, no separate sink.

## Unresolved questions

- None blocking. `set_llm_provider` accepts both the instance and the legacy name+model args, so
  re-point-by-name is supported. Provider `.name` collisions are inert (`select_codec_runtime`
  branches on codec, not name) — no action.
