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
       │  ├─ create(title, description)
       │  ├─ update(id, status)
       │  └─ list(filter)
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

**Default Settings:**
- `max_tokens`: 4096
- `compression_threshold`: 0.8 (compress at 80% usage)
- Compression reduces tokens to ~60% of original

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
   raised by providers is caught in `llm_caller._invoke_llm_sync/async`,
   which calls `timeline.reactive_compact(attempt)` (drop 5→10→20 oldest
   kept entries + forward-orphan pruning + full summary) with exponential
   backoff 1s/3s. After 3 consecutive failures the circuit opens;
   cooldown `DANA_CIRCUIT_COOLDOWN_SECONDS` (default 300s) plus half-open
   probe provide automatic recovery. Kill switch via
   `DANA_DISABLE_REACTIVE_COMPACT=1`.

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
Agent.stream_response(messages)
    │
    ├─ Runtime.stream_complete(messages, tools)
    │  │
    │  └─ Provider-specific streaming
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

## Concurrency Model

- **Async throughout**: All I/O is non-blocking
- **Tool execution**: Sequential by default (preserves order)
- **Multiple agents**: Can run concurrently with asyncio.gather()
- **Web research**: Concurrent HTTP requests within single research task

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

The Session Journal is the **sole durable authority** for Dana agent
sessions. Every turn — input, streamed chunks, terminal — is appended as a
typed, immutable `JournalFact` *before* the model is invoked and *before*
the response is returned. A host restart that calls `session/load` sees the
full conversation replayed before the call returns.

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
