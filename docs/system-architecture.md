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

---

**Version:** 0.1.1 | **Last Updated:** 2026-03-21
