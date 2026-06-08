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

## The three injection sinks

An injected provider must reach **all three**, or split-brain results (agent reads the
injected provider while the actual call site silently builds a different one from `.env`):

| Sink | Today | Mutation surface |
|------|-------|------------------|
| `STARAgent._llm_client` | lazy `LLM(provider=str, model=str)` | direct assign |
| `runtime._llm_caller._llm` | `LLMCaller` builds own from name/model | `runtime.set_llm(llm)` — **exists** |
| `LTMemory → RLMResource._llm` | `LLM(provider=str, model=str)` | new thin `llm` passthrough + `set_llm` |

## Chosen approach (A): overload `llm_provider` + central fan-out method

Rejected alternatives: (B) separate `provider_instance=` param — two params for one job,
`model` ambiguity, not DRY with `LLM.__init__`; (C) accept a pre-built `LLM` only — caller
wants to pass the provider, not pre-wrap it.

Approach A widens the existing `llm_provider` param to `str | LLMProvider`, mirroring the
`str | LLMProvider` overload `LLM.__init__` already exposes, so the agent ctor reads
consistently with the layer beneath it.

### Spine: a single fan-out method

Both the constructor and the public runtime setter call this. It is the only place that
knows about all three sinks.

```python
def _apply_llm_provider(self, provider, model=None):
    # normalize → LLM
    if isinstance(provider, LLM):
        llm = provider
    elif isinstance(provider, LLMProvider):
        llm = LLM(provider=provider)            # provider_name → "custom"; model from instance
    else:                                        # str | None (legacy path)
        llm = LLM(provider=provider, model=model)

    self._llm_client = llm
    if self._runtime is not None:
        self._runtime.set_llm(llm)               # → LLMCaller.set_llm
    if self._ltmemory is not None:
        self._ltmemory.set_llm(llm)              # new thin setter → RLMResource.set_llm

def set_llm_provider(self, provider, model=None):   # public — runtime re-point
    """Re-point this agent (and its runtime + LTMemory) at a new provider/LLM.

    `provider` may be a provider name (str), an LLMProvider instance, or a pre-built LLM.
    """
    self._apply_llm_provider(provider, model)
```

### Constructor branch (instance wins)

```python
if isinstance(llm_provider, LLMProvider):
    name  = getattr(llm_provider, "name", None) or "custom"
    model = getattr(llm_provider, "model", None)   # instance wins; `model` arg ignored
else:
    name  = llm_provider or config_manager.get_first_available_provider() or "anthropic"

if runtime is None:
    runtime = RuntimeRegistry.select_codec_runtime(provider=name, model=model, codec=codec)
self._runtime = runtime
# ... build self._ltmemory (if ltmemory_path) ...
self._apply_llm_provider(
    llm_provider if isinstance(llm_provider, LLMProvider) else name,
    model,
)
```

`name` is cosmetic in this path: `select_codec_runtime` returns `CodecRuntimeWith[out]NativeToolUse`
based on the codec, not provider-specific runtimes, and `set_llm` overrides whatever LLM the
runtime built. The string is metadata only.

### Signature changes

- `STARAgent.__init__`: `llm_provider: str | None` → `llm_provider: str | LLMProvider | None`.
- New public `STARAgent.set_llm_provider(provider, model=None)`.
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
- **`model` arg + instance:** `model` ignored (instance binds its own model). `logger.debug`,
  not a raise.
- **`set_llm_provider` mid-session:** re-points all three sinks; in-flight calls hold their own
  `llm` ref, so no torn state.
- **Legacy `llm_provider="openai"` string:** unchanged — flows through the `else` branch.

## Testing

1. Inject `OpenAIProvider(base_url=..., model=...)` → assert `agent.llm_client.provider is instance`
   **and** `runtime._llm_caller._llm is agent.llm_client` (proves no split-brain).
2. `set_llm_provider(other_instance)` → both sinks now reference `other_instance`.
3. `ltmemory_path` set + injected provider → `RLMResource._llm` uses the injected provider.
4. Regression: `llm_provider="openai"` string path behaves exactly as before.

## Out of scope

- Fallback-provider (`ProviderConfig`) wiring in `LLMCaller` — unchanged.
- Streaming mixin — uses the same `runtime`/`llm_client`, no separate sink.

## Unresolved questions

- Should `set_llm_provider` accept a bare `str` name too (re-point by name)? Spec assumes yes
  (cheap, same fan-out) — confirm during implementation.
- Provider `name` collision: if a custom instance's `.name` matches a registered provider name,
  `select_codec_runtime` still only branches on codec, so no behavioral impact — noted, no action.
