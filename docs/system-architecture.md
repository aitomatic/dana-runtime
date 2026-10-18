# System Architecture

**Version:** 0.1.1 | **Last Updated:** 2026-03-21

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Applications Layer                        │
│  (adana, adana-repl, dana-code, dana-init, dana-cli)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    Agent Layer                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  STARAgent (STAR Loop Implementation)                   ││
│  │  ├─ Communicator (LLM interface)                        ││
│  │  ├─ State (Conversation state)                          ││
│  │  ├─ Learner (Learning from interactions)               ││
│  │  └─ Observer (Introspection & metrics)                 ││
│  └─────────────────────────────────────────────────────────┘│
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                 Core Systems Layer                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Resource System      Timeline         Workflows        ││
│  │  ├─ BashResource      ├─ Entry        ├─ BaseWorkflow  ││
│  │  ├─ FileIOResource    ├─ Compressor   ├─ Executor      ││
│  │  ├─ SearchResource    └─ Serializer   └─ Validation    ││
│  │  └─ CustomResources                                     ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Memory          Prompts        Skills        Reminder  ││
│  │  ├─ STMemory     ├─ Builder     ├─ Registry  └─ Context ││
│  │  ├─ LTMemory     └─ API         └─ Executor ││
│  └─────────────────────────────────────────────────────────┘│
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│              LLM Abstraction Layer                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  CodecRuntimeBase (Unified interface)                   ││
│  │  ├─ NativeToolsCodec (Claude, GPT-4 Turbo)            ││
│  │  └─ CSXMLCodec (Non-native tool use)                   ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Provider Implementations                               ││
│  │  ├─ OpenAI           ├─ Azure OpenAI                   ││
│  │  ├─ Anthropic        ├─ Gemini                         ││
│  │  └─ Anthropic-Like   └─ Local (LLaMA/Ollama)          ││
│  └─────────────────────────────────────────────────────────┘│
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│            Data Persistence & Infrastructure                │
│  ├─ Repository Layer (File, Database adapters)             │
│  ├─ Configuration (config.json, environment)               │
│  ├─ Logging (structlog)                                    │
│  └─ Utilities (common functions, helpers)                  │
└─────────────────────────────────────────────────────────────┘
```

## STAR Agent Pattern (Execution Flow)

```
User Input
    │
    ▼
┌─────────────────────┐
│  SEE (Perceive)     │
│  • Parse intent     │
│  • Analyze context  │
│  • Load memory      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  THINK (Reason)     │
│  • Prompt building  │
│  • LLM inference    │
│  • Tool planning    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ACT (Execute)      │
│  • Call resources   │
│  • Collect results  │
│  • Update timeline  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  REFLECT (Learn)    │
│  • Analyze outcome  │
│  • Update memory    │
│  • Metrics capture  │
└────────┬────────────┘
         │
         ▼
    Response
```

## Component Interaction Diagram

```
STARAgent
├─ Communicator (LLM Communication)
│  ├─ Runtime (Provider abstraction)
│  │  └─ Codec (Tool schema conversion)
│  │      ├─ NativeToolsCodec
│  │      └─ CSXMLCodec
│  ├─ Prompt Builder
│  │  ├─ System prompt
│  │  ├─ Tool schemas
│  │  └─ Conversation history
│  └─ LLM Providers
│      ├─ OpenAI
│      ├─ Anthropic
│      ├─ Gemini
│      └─ Others
│
├─ State (Conversation State)
│  ├─ Timeline (History)
│  │  ├─ User entries
│  │  ├─ Assistant entries
│  │  ├─ Tool results
│  │  └─ Compression logic
│  ├─ Short-term memory
│  └─ Tool cache
│
├─ Learner (Learning System)
│  ├─ Long-term memory
│  │  ├─ Lessons
│  │  ├─ Episodes
│  │  ├─ Facts
│  │  └─ Patterns
│  └─ Feedback processing
│
└─ Observer (Metrics & Introspection)
   ├─ Token tracking
   ├─ Performance metrics
   ├─ Error logging
   └─ Debug information
```

## Data Flow: Message Processing

```
User Message
    │
    ▼
Timeline.add_entry(UserEntry)
    │
    ▼
check_should_compress()
    │
    ├─ Yes → Timeline.compress()
    │        (LLM-based summarization)
    │
    ▼
Prompt.build(
    system_prompt,
    timeline_entries,
    tool_schemas,
    memory_context
)
    │
    ▼
Runtime.complete(
    messages,
    tools
) ─ Handles provider-specific format
    │
    ├─ OpenAI: native tools
    ├─ Anthropic: native tools
    └─ Others: XML-based tools
    │
    ▼
Parse Response (ToolCalls + Text)
    │
    ├─ Tool Calls
    │  │
    │  └─ for each tool_call:
    │     │
    │     ├─ ResourceRegistry.get(tool_name)
    │     ├─ Execute with parameters
    │     └─ Timeline.add_entry(ToolResultEntry)
    │
    ├─ Text Content
    │  │
    │  └─ Timeline.add_entry(AssistantEntry)
    │
    ▼
Optional Refinement (if tool errors or incomplete)
    │
    ▼
Learner.process_interaction(
    context,
    outcome,
    feedback
)
    │
    ▼
Return Response to User
```

Default prompt context contains the date, timezone, and local user name. It
does not perform IP geolocation. Applications supplying site or location context
can override `AgentRuntime._get_runtime_context`; explicitly selecting the
`location` field retains the existing opt-in lookup behavior.

## Resource Execution System

```
STARAgent
    │
    └─ Resource Registry (Global)
       │
       ├─ BashResource
       │  ├─ execute(command)
       │  └─ Tool Schema: {name, description, parameters}
       │
       ├─ FileIOResource
       │  ├─ read(path)
       │  ├─ write(path, content)
       │  └─ list_dir(path)
       │
       ├─ FileEditResource
       │  ├─ edit_file(path, old, new)
       │  └─ Shows diff before applying
       │
       ├─ SearchResource
       │  ├─ search(query, num_results)
       │  └─ Returns structured results
       │
       ├─ WebResearchResource
       │  ├─ research(topic, sources)
       │  ├─ HTML extraction
       │  ├─ Content synthesis
       │  └─ Multi-source aggregation
       │
       ├─ TaskResource
       │  ├─ task(description, prompt, subagent_type, resume)
       │  └─ task_output(task_id)
       │
       ├─ TodoResource
       │  ├─ add(title)
       │  ├─ complete(id)
       │  └─ list()
       │
       ├─ SkillResource
       │  ├─ execute_skill(name, args)
       │  └─ Load from .claude/skills/
       │
       ├─ MCPResource
       │  ├─ Call MCP servers
       │  ├─ Server discovery
       │  └─ Protocol handling
       │
       └─ CustomResources (User-defined)
          └─ Extend BaseResource
             ├─ Annotated methods = tools
             └─ Auto schema generation
```

`AgentRuntime` keys native tool schemas by agent, membership revision, and the
current underlying provider identity, so changing capabilities inside one LLM
wrapper also refreshes them. Bound dispatch registries have an independent
agent/revision key. STAR's `with_*`, `add_*`, successful `remove_*`, and changed
LLM binding invalidate metadata and rendered codec prompts without discarding
prompt-template overrides. Stable warm refreshes reuse the metadata.
`AgentSession` keys the merged `ToolCatalog` by the runtime catalog revision and
MCP wiring identity; replacing MCP wiring refreshes its schemas and dispatch map.

Tool discovery checks decorator markers with static attribute lookup before
binding declared callables. It does not evaluate unrelated public properties,
which can construct clients or render prompts. Normal Python attribute binding
is preserved for functions installed directly on an instance, so their first
required argument remains in the model-visible schema.
Native tool dispatch resolves bound resources for the invoking agent, not a
runtime-wide most recently used agent. That owner remains explicit across
permission/EventBus waits, so overlapping invocations on a shared runtime do not
redirect a resumed call to another agent's resources. Registry rebuilds publish
a complete fresh mapping instead of clearing a mapping held by an active call.
The owner, build revision, and mapping form one cache entry, captured once by a
reader and replaced atomically by a builder or reset. Standalone executors retain
the zero-argument registry-getter contract; the runtime supplies its
invocation-aware getter through an explicit keyword dependency.

## Workflow Execution

```
WorkflowExecutor
    │
    ├─ Parse workflow definition
    │  (steps, conditions, parallel tasks)
    │
    ├─ For each step:
    │  │
    │  ├─ Validate inputs
    │  ├─ Execute step (CallableWorkflow)
    │  ├─ Collect outputs
    │  ├─ Handle errors/retries
    │  └─ Pass outputs to next step
    │
    ├─ Conditional branches
    │  ├─ Evaluate condition
    │  └─ Route to correct step
    │
    ├─ Parallel execution
    │  ├─ Run concurrent steps
    │  └─ Gather results
    │
    ▼
Return aggregated results
```

## LLM Provider Architecture

```
Runtime (Abstract Interface)
    │
    ├─ CodecRuntimeBase
    │  │
    │  ├─ __init__(codec: ToolCodec, ...)
    │  └─ async complete(messages, tools) -> Response
    │
    ├─ DefaultRuntime
    │  ├─ Try providers in priority order
    │  └─ Fallback on provider failure
    │
    ├─ ProviderSpecificRuntime (OpenAI, Anthropic, etc.)
    │  └─ Provider-specific optimizations
    │
    └─ Codec System
       │
       ├─ NativeToolsCodec
       │  ├─ Anthropic native tools
       │  ├─ OpenAI function calling
       │  └─ No XML conversion needed
       │
       └─ CSXMLCodec
          ├─ Convert tools to XML format
          ├─ Parse XML responses
          └─ For non-native providers
```

`LLMCaller` owns request retries and the logical-request deadline
(`DANA_LLM_REQUEST_DEADLINE_SECONDS`, default 900 seconds; zero disables it).
An attempt is capped by both its provider timeout and the remaining request
budget. The retry deadline applies until the first stream chunk; an established
stream uses a fresh idle timeout for each pull rather than expiring solely
because total stream duration exceeds the pre-first-chunk budget. SDK-owned
retries are disabled. A stream retries only before any chunk has been emitted.

**Retry ownership change:** STAR no longer automatically replays an entire
Situation/Thought/Action/Result cycle. `LLMCaller` owns bounded provider-request
retries. Replaying a full cycle could execute already completed tools again;
request retries never replay those effects. Operators relying on the old
whole-cycle retry behavior must now handle the recorded terminal failure or
submit a new turn explicitly.

Streaming completion carriers preserve provider-reported finish status,
usage/cache/reasoning counters, response identity, and compatible protected
replay state. Unknown finish status remains unknown; truncation and refusal
must not become successful final answers. Typed provider errors survive the
LLM facade. Owners call `LLM.aclose()` and `Embedder.aclose()` on the owning
event loop; shutdown also joins cleanup of providers retired by model switches.
Rejected provider switches leave the active client unchanged and close the
unused replacement; rejected factory-created embedding clients are also closed.
An incomplete response takes precedence over any included function calls.
Responses API error events propagate as typed errors; Gemini SDK finish enums
are normalized to wire values before the STAR completion guard.

Sampling omission and explicit zero are different caller policies. Runtime
construction preserves that provenance through `LLMCaller`: absent streaming
temperature leaves the provider default intact, while explicit zero is not
silently omitted. Provider model restrictions still apply independently.

Named provider selection and assignment through `llm_client` transfer ownership
to STAR; `llm_provider_instance` is borrowed. Replacement retires the previous
owned client without closing borrowed clients. Agent shutdown closes internally
created child agents, not caller-supplied children attached with `with_agents`
or `add_agent`. The caller remains responsible for borrowed clients and children.
Ownership follows the facade's current provider: switching from a borrowed
provider to a newly constructed provider makes the replacement agent-owned.
Reinjecting the same pool preserves ownership without retiring the installed
pool; displaced facades transfer pending cleanup to the surviving facade.
`STARAgent.close()` is non-blocking inside an event loop. Await `aclose()` to
finish owned-provider and owned-child cleanup, including earlier retirements
when the current provider is borrowed. Concurrent callers join a dedicated
teardown task, not another caller's lifetime; cancelling a joiner does not
cancel that cleanup.

The public `dana.common.llm` exports load lazily. Importing message types does
not initialize unused provider or embedding modules. Optional tracing SDKs
remain disabled when absent. Lifecycle import, construction, and flushing run
inside the bounded shutdown worker; failure diagnostics include exception types,
not provider payloads.

## Provider Configuration & Priority

```
config.json
{
  "llm_providers": {
    "openai": {
      "priority": 100,
      "api_key": "${OPENAI_API_KEY}",
      "models": ["gpt-4.1", "gpt-4.1-mini", "o3", "o4-mini"]
    },
    "anthropic": {
      "priority": 90,
      "api_key": "${ANTHROPIC_API_KEY}",
      "models": ["claude-sonnet-4-6", "claude-opus-4-6"]
    },
    "gemini": {
      "priority": 85,
      "api_key": "${GOOGLE_API_KEY}",
      "models": ["gemini-2.5-flash", "gemini-2.5-pro"]
    },
    "azure": {
      "priority": 40,
      "api_key": "${AZURE_OPENAI_KEY}",
      "endpoint": "${AZURE_ENDPOINT}"
    }
  }
}
```

DefaultRuntime tries providers in priority order (high to low).

## Memory System Architecture

```
STARAgent
    │
    ├─ Short-Term Memory (STMemory)
    │  ├─ Per-session cache
    │  ├─ Fast retrieval
    │  └─ Location: dana/core/memory/
    │
    └─ Long-Term Memory (LTMemory)
       ├─ Persistent markdown files
       ├─ 4 memory types:
       │  ├─ Lesson (Key learnings)
       │  ├─ Episode (Significant interactions)
       │  ├─ Fact (Factual information)
       │  └─ Pattern (Observed patterns)
       │
       ├─ Storage: ~/.dana/memory/ or configured path
       ├─ Retrieval: Embedding-based search (optional)
       └─ Location: dana/lib/memory/
```

## Prompt Building Pipeline

```
User Message + Context
    │
    ├─ PromptBuilder.build()
    │  │
    │  ├─ 1. System Prompt
    │  │  └─ Model instructions, capabilities
    │  │
    │  ├─ 2. Memory Context
    │  │  ├─ STMemory retrieval
    │  │  └─ LTMemory search results
    │  │
    │  ├─ 3. Tool Schemas
    │  │  ├─ Names, descriptions
    │  │  └─ Parameter schemas (JSON)
    │  │
    │  ├─ 4. Conversation Timeline
    │  │  ├─ Recent entries (uncompressed)
    │  │  └─ Older entries (if compressed)
    │  │
    │  └─ 5. Reminder Context
    │     └─ Dynamic context injected
    │
    ▼
Final Prompt → LLM
```

## Token Management & Compression

```
Timeline.add_entry(entry)
    │
    ├─ Update token count
    │
    └─ Check compression threshold
       │
       ├─ If tokens < threshold * max_tokens: OK
       │
       └─ If tokens >= threshold * max_tokens:
          │
          ├─ Compress older entries
          │  │
          │  └─ LLM-based summarization
          │     ├─ Summarize 3-5 oldest entries
          │     ├─ Replace with summary
          │     └─ Maintain token budget
          │
          └─ Continue conversation
```

`CompressedTimeline` maintains a cached token contribution for each native
message and folds only changed contributions into the total. Tool calls do not
permanently force full-history retokenization. Shared mutable payloads keep
dependency snapshots so in-place caller edits invalidate the affected message;
an escaped public message reference also requires attribute-change checks.
Retained dependency comparisons can still scale with mutable history size.
This is incremental token estimation, not a constant-time guarantee for
arbitrary caller-mutated histories.
Snapshot retention also consumes memory proportional to aliased payloads.
Preserving public mutation visibility can cost more than recounting on some
workloads; plain-history benchmark gains must not be generalized to this path.

Runtime-context replacement, hydration, shrink, and compaction keep the message
cache aligned with the live projection. Hydrating caller-owned dictionaries
copies their metadata mapping before installing internal entry backlinks, so
the caller's original payload remains JSON-serializable.

The default compression trigger is 150000 tokens, resolved through
`DANA_COMPACT_TRIGGER_TOKENS`. The context-window budget is separate from that
trigger; an explicit `max_context_tokens` does not redefine when compression
should run. Compression size depends on the retained entries and summary, not
a guaranteed fixed reduction ratio.

### Compaction Parity Upgrades (Phases 1–4)

The compression pipeline now has three additional layers for parity with
OpenClaude-style engines:

1. **Single-knob heuristic trigger (P3)** — `DANA_COMPACT_TRIGGER_TOKENS`
   (default 150000, clamp `[8k, 2M]`) gates `needs_compression()`. Optional
   `system_tokens_fn` / `tools_tokens_fn` callbacks fold system-prompt and
   tools-schema size into the estimate. Always `len(str)/4`.
2. **Cheap client-side shrink (P6)** — `cheap_shrink_tool_results()` stubs
   old `tool_result` bodies to `"[cleared for context budget]"` while
   preserving `tool_call_id`. Opt-in via
   `CompressedTimelineConfig.enable_cheap_shrink_tool_results`. A predictive
   gate skips shrink when it cannot close the token gap alone — prevents
   vacuous summaries over stubs.
3. **Reactive compact + circuit breaker (P2)** — `PromptTooLongError`
   raised by providers is caught in `llm_caller._invoke_llm_sync/async`.
   The sync path uses `timeline.reactive_compact(attempt)`; the async path
   awaits `timeline.reactive_compact_async(attempt)`. Both retain progressive
   drops, forward-orphan pruning, full summary and exponential backoff.
   After 3 consecutive failures the circuit opens;
   cooldown `DANA_CIRCUIT_COOLDOWN_SECONDS` (default 300s) plus half-open
   probe provide automatic recovery. Kill switch via
   `DANA_DISABLE_REACTIVE_COMPACT=1`.
   Async compaction propagates cancellation to the awaited summarizer rather
   than abandoning a worker that can later overwrite the timeline. A sync-only
   callback must be joined before ownership releases, so its remaining physical
   duration can overrun the logical request deadline.

**Provider PTL mapping:**
| Provider | Detection |
| --- | --- |
| Anthropic / Anthropic-like | `BadRequestError` body `type="invalid_request_error"` + `"prompt is too long"` in message |
| OpenAI / Azure / Moonshot | `APIStatusError` body `code="context_length_exceeded"` |
| Gemini | No SDK error — post-hoc WARNING log on `finish_reason=="MAX_TOKENS"` (reactive compact unavailable; tune `DANA_COMPACT_TRIGGER_TOKENS` conservatively) |

**Telemetry:** `dana/core/timeline/telemetry.py` exposes
`CompressionLogFields` TypedDict allowlist. An AST-based unit test asserts
log `extra={...}` keys stay within the allowlist (no prompt-content leakage).

## Error Handling & Recovery

```
Resource Execution
    │
    ├─ Try: execute tool
    │  │
    │  ├─ Success → Return result
    │  │
    │  └─ Error:
    │     │
    │     ├─ Catch specific exception
    │     │
    │     ├─ Log error with context
    │     │
    │     ├─ Decide on retry (transient vs permanent)
    │     │  ├─ Transient (network): retry with backoff
    │     │  └─ Permanent (invalid): fail immediately
    │     │
    │     └─ Return error to agent
    │        └─ Agent may retry with different params
    │
    └─ Timeline.add_entry(ToolErrorEntry)
```

## Streaming Architecture

```
Agent.aquery_stream(message=…)
    │
    ├─ STAR loop streaming (_see/_think/_act, StreamEvent per phase)
    │  │
    │  └─ Runtime LLM streaming (llm_caller.call_llm_stream)
    │     (Server-Sent Events or chunked)
    │
    ├─ Token-by-token yield
    │
    ├─ Collect full response
    │
    ├─ Parse tool calls if present
    │
    └─ Timeline.add_entry(AssistantEntry)
```

`AgentSession.prompt()` forwards assistant text as provisional host deltas
before the provider finishes. Reasoning remains a distinct event. Text before
a tool call is narration, not a committed final answer. Final content and the
turn terminal fact commit atomically; cancellation or failure retains partial
host output without admitting it as a successful model-facing final answer.

The session owns stream pulls, closes nested iterators, and resolves in-flight
journal commits before acknowledging cancellation. Hosts must exhaust or close
the prompt iterator before calling `AgentSession.aclose()`. A session created
with an injected repository borrows it; `AgentSession.create()` owns and closes
the repository it opens. ACP and CLI close sessions before host-owned storage.

## Concurrency Model

- **Bounded admission**: Permission decisions run before admission, so an
  unanswered prompt holds no shared capacity. Leaf tools and nested tasks use
  separate limits; `with_agents` query dispatches use the nested limit too.
- **Effect ordering**: Conflicting declared effects retain call order, including
  file targets redirected by interception. MCP `readOnlyHint=True` calls can
  overlap; unannotated MCP calls serialize within their server, not globally.
  Catalogs and session lifecycle hooks resolve against the invoking agent.
- **Blocking tools**: Synchronous callables explicitly marked `@blocking_io`
  run in the bounded worker pool. Unmarked synchronous Python stays on the
  event loop; it is not assumed thread-safe. Async file tools may explicitly
  offload their blocking cores through `run_blocking`; async signatures do not
  imply inline disk I/O. Cancellation signals cooperative waits and retains
  admission/effect ownership until physical work exits. A non-cooperative
  callable can therefore delay cancellation; Python threads cannot safely be
  forcibly terminated.
  Root executors close only their owned pool; a direct-call fallback pool lives
  until process exit. Awaited worker joins run off-loop. File-edit path locks
  are weakly indexed, with strong ownership through each holder or waiter, so
  completed paths do not accumulate registry entries.
- **Bash pipe ownership**: Process exit and output completion are separate.
  Descendants may keep inherited pipes open after the shell exits. Status stays
  `running` until output drains, while `exit_code` can report the terminal process.
  Pipe draining never holds the task-state lock across a wait.
- **Nested progress**: A nested call can lend its held slot to a descendant to
  avoid self-deadlock. An ancestor cannot release that capacity while borrowed
  descendant work remains active.
- **Task state**: `TaskResource` constructs a fresh child from its registered
  factory for each new task. Cancellation records `cancelled` before propagating
  `CancelledError`; `task_output(task_id)` then returns `Status: cancelled`
  rather than leaving an exited child permanently `running`. The existing
  session handle remains available for an explicit resume.
- **MCP request identities**: stdio and HTTP transports pass a
  `correlating_write_stream(write_stream)` wrapper to the supported
  `ClientSession(read_stream, write_stream)` constructor. Context-local call
  correlation records the SDK-assigned request ID without mutating SDK internals,
  including overlapping calls to the same tool. Integrations constructing an MCP
  SDK session directly must wrap their write stream to get this correlation.
- **MCP dispatch boundary**: Harness names retain their server prefix for
  discovery, permissions and effect ordering. The remote request uses the
  server's unprefixed tool name. Session catalogs retain discovered read-only
  annotations; nested transport contexts close in reverse acquisition order.

## Extension Points

### 1. Custom Resources
```python
class MyResource(BaseResource):
    async def my_tool(self, param: str) -> str:
        return f"result: {param}"
```

### 2. Custom Workflows
```python
class MyWorkflow(BaseWorkflow):
    async def execute(self, context):
        result1 = await self.step1()
        result2 = await self.step2(result1)
        return result2
```

### 3. Custom Memory Adapters
```python
class DatabaseMemory(LTMemory):
    async def save(self, memory):
        await db.insert(memory)
```

### 4. Custom Providers
Implement provider interface + add to config.json

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Message processing | <5s avg | Depends on LLM |
| Tool execution | Variable | Depends on tool |
| Timeline compression | 1-2s | LLM-based |
| Memory retrieval | <100ms | Embedding search |
| Resource lookup | <1ms | Hash registry |
| Streaming first token | 1-3s | LLM latency |

## Security Architecture

- **Input Validation**: Pydantic models for all inputs
- **Code Execution**: Python sandbox with restricted builtins
- **Environment Secrets**: Never logged, loaded from .env
- **Tool Filtering**: Only allowed resources accessible
- **Command Execution**: Bash sandboxing where possible

## Session Journal Architecture (D1: Durable Dana Conversation)

The Session Journal is the **sole durable authority** for journal-managed Dana
sessions. Input and turn-start facts are committed before model execution.
Streamed assistant text is provisional and can be journaled in buffered chunks;
final assistant content and the successful terminal fact are committed together
before completion is returned. Restarted hosts rebuild the conversation from
committed final facts and compatible derived checkpoints, not provisional chunks.

```
┌────────────────────┐ JSON-RPC (stdio) ┌──────────────────────────────┐
│  Host (ACP client) │ ◀──────────────▶ │  DanaACPAgent                │
│  - dana-console    │                  │  └─ AgentSession (1 per sess)│
│  - sessionStorage    │                  │     ├─ STARAgent            │
└────────────────────┘                  │     ├─ JournalRepository ──┐ │
        ▲                                │     └─ ProtectedStateCodec│ │
        │ session_update                 └──────────────────────────┼─┘
        │ (HostEvent stream)                                       │
┌───────┴─────────────┐   append/read           ┌──────────────────▼───┐
│ Conversation View   │ ◀─────────────────────  │  Session Journal     │
│ Host Event View     │   projectors             │  (SQLite / Postgres) │
│ Trace View          │                          │  - session_journals  │
└─────────────────────┘                          │  - session_facts     │
                                                  │  - projection_checkpts│
                                                  └──────────────────────┘
```

### Core components

- **`AgentSession`** (`dana/core/session/agent_session.py`) — the
  host-neutral orchestrator. Owns one agent, one owner/workspace scope, the
  session journal identity and version, and the active turn. Serializes
  mutations: only one active turn per session; a conflicting `prompt`
  raises `SessionBusy`. All turn lifecycle facts are journaled before,
  during, and after the model call.
- **`JournalRepository`** protocol (`dana/core/session/journal/protocol.py`)
  — the backend-agnostic persistence contract. Two reference
  implementations:
  - **`SQLiteJournalRepository`** — on-disk SQLite (WAL mode, foreign keys
    enforced). Default for local and single-host deployments.
  - **`PostgresJournalRepository`** — PostgreSQL via asyncpg, JSONB-typed
    payloads, owner-scoped primary/foreign keys. Recommended for shared and
    multi-tenant deployments.
- **`DanaACPAgent`** (`dana/apps/acp/agent.py`) — the ACP protocol façade.
  Translates `initialize`, `session/new`, `session/load`, `session/resume`,
  `session/prompt`, and `session/cancel` into `AgentSession` operations and
  streams `HostEvent`s back as ACP `session_update` notifications.

### Data model

- **`OwnerScope`** (`owner_id` + `workspace`) — the immutable tenant
  principal. Every journal operation is scoped by it; cross-owner access is
  impossible at the data layer.
- **`JournalFact`** — the durable, stored form of a journal fact after
  persistence assigns identity. Each fact carries a `sequence` (1..N within
  a session), a typed `FactType`, a `correlation_id` (groups a turn), a
  `causation_id` (links causes), a JSON-safe `payload`, and an envelope-
  encrypted `protected_payload` for provider replay material.
- **`FactType`** (D1 text-only conversation set): `SESSION_CREATED`,
  `SESSION_LOADED`, `SESSION_RESUMED`, `TURN_STARTED`, `USER_CONTENT_FINAL`,
  `ASSISTANT_CONTENT_CHUNK`, `ASSISTANT_CONTENT_FINAL`, `TURN_COMPLETED`,
  `TURN_INTERRUPTED`, `TURN_ERROR`, `TURN_CANCELLED`,
  `LEGACY_TIMELINE_MIGRATED`.
- **`ProjectionCheckpoint`** — a named cursor + opaque JSON blob saved by a
  projection. Stored OUT-OF-BAND of journal facts: writing one never changes
  a fact and never advances the session version.

### Projections (views)

Projections are pure functions over the fact stream; they never mutate the
journal and always rebuild deterministically from facts + checkpoint.

- **Conversation View** (`ConversationProjector`) — the user/assistant
  message list. Excludes partial assistant output from interrupted turns
  (partial output is retained in the Host Event View); surfaces an
  interruption observation to the next model turn.
- **Host Event View** (`HostEventProjector`) — the ACP `session_update`
  stream, replayed in order on `session/load` so the host UI shows the
  pre-crash conversation immediately on restart.
- **Trace View** — reasoning/observability projection (planned beyond D1).

### Optimistic concurrency

`append` declares the `expected_version` it observed; if the durable
version differs, `JournalConflict` is raised and no facts are persisted.
The SQLite adapter uses `BEGIN IMMEDIATE`; the Postgres adapter uses the
equivalent row-level lock. This is the contract that lets multiple
subprocesses share a journal safely.

Ordinary turns fold facts after the retained projection cursor instead of
rereading the full journal. Owner/sequence-bound checkpoints are derived caches,
not authority: reopen can rebuild from facts. The retained model working set
and checkpoint preserve the actual plain or compressed context, including
summary messages, completed tool groups, and compatible replay state. External
writers remain visible through the journal version and sequence boundary.
Restoration copies the committed snapshot before decoding, so a failed turn's
mutable content, metadata, or tool arguments cannot corrupt recovery state.
Checkpoint entries are decoded inside the rejection boundary; malformed caches
fall back to the intact journal rather than preventing session load.

`AgentSession` marks its agent as journal-managed before hydration. That agent
does not also write a whole-history legacy timeline snapshot after each turn.
Standalone STAR query/stream/resume callers retain legacy snapshot durability;
the journal ownership flag is independent of protected replay ownership.
Internal session checkpoint/replay reads borrow the working native messages
synchronously and never expose those mutable objects to external callers.

### Protected state

Provider Replay State (e.g. OpenAI `encrypted_content`, reasoning items)
required for continuity is **never** placed in the regular `payload`; it is
envelope-encrypted (AES-GCM + HKDF + AAD) and carried only in
`protected_payload`. The encryption key is sourced from
`DANA_SESSION_STATE_KEY` via `EnvProtectedStateKeyProvider`. A payload-
sanitization layer rejects secret-bearing keys before persistence.

### Migration and rollback

- **Legacy Timeline → Journal** (`migrate_legacy_timeline`) — imports
  legacy `TimelineEntry` sessions into the journal. Idempotent via a
  content-addressed SHA-256 stored on a `LEGACY_TIMELINE_MIGRATED` marker
  fact. Text-only in D1 (`USER_MESSAGE` and `AGENT_RESPONSE` are converted;
  thoughts, tools, summaries, and ephemeral context are skipped).
- **Compatibility projection** (`journal_facts_to_timeline_entries`) —
  projects journal facts back to the legacy Timeline shape behind the
  `DANA_SESSION_JOURNAL_AUTHORITY=0` rollback flag, so legacy readers keep
  working after the journal becomes the sole authority.
- **Crash recovery** (`recover_interrupted_turns`) — detects started-but-
  unterminated turns and appends a typed `TURN_INTERRUPTED` fact for each.
  Idempotent; invoked automatically on every `session/load`.

### Operational health

`dana.core.session.health.check_journal_health(repository, scope)` returns a
**redacted** aggregate report — only counts and booleans, never owner_ids,
session_ids, payloads, or protected payloads. Covers database connectivity,
session counts by status, interrupted-turn detection, projection lag
tracking, and legacy migration marker counts. See
[`docs/acp-configuration.md`](acp-configuration.md#health-checks) for usage.

### Reference

- Design spec: `plans/` (see *Durable Dana Conversation* / *ACP AgentSession*).
- Storage, migration, rollback, key rotation runbook:
  [`docs/session-journal-storage.md`](session-journal-storage.md).
- ACP host configuration: [`docs/acp-configuration.md`](acp-configuration.md).

---

**Version:** 0.1.1 | **Last Updated:** 2026-03-21
