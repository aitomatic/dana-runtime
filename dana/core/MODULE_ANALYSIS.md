# Dana Core Module Analysis

## 1. Project Overview

**Project Type:** Agent Framework / AI Agent Library
**Primary Language:** Python 3.10+
**Architecture Pattern:** STAR (See-Think-Act-Reflect) Pattern with Composition-based Components

The `dana.core` module is the core component of the Dana/OpenDXA agentic architecture. It provides foundational building blocks for creating conversational AI agents with resource management, workflow execution, memory systems, and multi-agent coordination capabilities.

### Tech Stack
- **Runtime:** Python 3.10+ with async/await support
- **LLM Integration:** Multi-provider support (Anthropic, OpenAI)
- **Logging:** structlog for structured logging
- **Web/HTTP:** requests for web operations
- **Concurrency:** threading, asyncio

---

## 2. Detailed Directory Structure Analysis

```
dana_agent/dana/core/
├── __init__.py                     # Module entry point, lazy-loads STARAgent
├── global_registry.py              # Multi-agent/resource/workflow discovery
├── agent/                          # Agent subsystem (STAR pattern implementation)
│   ├── __init__.py
│   ├── base_agent.py               # Base agent with identity & registry
│   ├── base_star_agent.py          # STAR loop contract definition
│   ├── star_agent.py               # Main STARAgent implementation
│   ├── timeline.py                 # Conversation timeline management
│   ├── xml_utils.py                # XML parsing utilities
│   └── components/                 # Composable agent components
│       ├── communicator.py         # Interactive conversation handler
│       ├── event_log_api.py        # Event logging for observers
│       ├── learner.py              # 4-phase learning system
│       ├── observer.py             # Environment observation protocol
│       ├── prompt_engineer.py      # Legacy prompt handling
│       ├── python_sandbox.py       # Safe code execution
│       ├── state.py                # Agent state management
│       ├── tool_caller.py          # Tool execution
│       └── tool_schema.py          # OpenAI-compatible tool schemas
├── context/                        # Context building for LLM
│   ├── builder.py                  # Token-budget-aware context assembly
│   └── context.py                  # Context dataclass
├── memory/                         # Memory systems
│   ├── ltmemory.py                 # Long-term persistent memory (RLM)
│   └── stmemory.py                 # Short-term session memory
├── reflection/                     # Learning reflection
│   └── reflection.py               # 4-phase memory distillation
├── resource/                       # External capability resources
│   ├── base_resource.py            # Base resource class
│   └── simple_search.py            # DuckDuckGo web search resource
├── runtime/                        # Agent runtime implementations
│   ├── __init__.py                 # AgentRuntime abstract base
│   ├── default.py                  # JSON-output runtime (recommended)
│   └── legacy.py                   # XML-based runtime (deprecated)
├── workflow/                       # Workflow orchestration
│   ├── base_workflow.py            # Base workflow with composition
│   ├── callable_workflow.py        # Function-to-workflow wrapper
│   ├── validation.py               # Workflow validation
│   └── workflow_executor.py        # SA-loop deterministic executor
├── knowledge/                      # Prompt and codec management
│   └── prompts/
│       ├── prompt_api.py           # Prompt template management
│       ├── prompt_engineer/        # Component-specific prompts
│       └── codecs/                 # Output format codecs
│           ├── abstract_codec.py   # Codec interface
│           └── xml_format.py       # XML/CSXML codec
└── skills/                         # Claude Code integration
    └── claude_code_skills.py       # Subprocess skill execution
```

---

## 3. File-by-File Breakdown

### Core Application Files

| File | Lines | Purpose |
|------|-------|---------|
| `star_agent.py` | ~1100 | Main STARAgent implementation with STAR loop, timeline compression, learning integration |
| `base_star_agent.py` | ~272 | Abstract STAR pattern contract (See-Think-Act-Reflect) |
| `base_agent.py` | ~177 | Base agent identity, registry, resource/workflow management |
| `timeline.py` | ~761 | Chronological conversation management with token-aware compression |
| `global_registry.py` | ~325 | Thread-safe multi-agent, resource, workflow discovery |

### Component Files

| File | Lines | Purpose |
|------|-------|---------|
| `communicator.py` | ~433 | Interactive CLI conversation with command handlers |
| `learner.py` | ~913 | 4-phase STAR learning (acquisitive, episodic, integrative, retentive) |
| `state.py` | ~66 | Dataclass for agent state management |
| `observer.py` | ~61 | Protocol for environment sensors (IoT, HVAC, etc.) |

### Runtime Files

| File | Lines | Purpose |
|------|-------|---------|
| `default.py` | ~778 | JSON-output runtime with native tool support |
| `legacy.py` | ~112 | Deprecated XML-based runtime |
| `__init__.py` | ~62 | AgentRuntime ABC and ParsedResponse dataclass |

### Memory Files

| File | Lines | Purpose |
|------|-------|---------|
| `ltmemory.py` | ~204 | Markdown-based persistent memory with RLM queries |
| `stmemory.py` | ~177 | Bounded session timeline with token estimation |
| `reflection.py` | ~182 | 4-phase memory distillation via LLM |

### Workflow Files

| File | Lines | Purpose |
|------|-------|---------|
| `base_workflow.py` | ~407 | Composable workflows with `|` operator |
| `workflow_executor.py` | ~344 | Deterministic SA-loop execution with retry |

### Resource Files

| File | Lines | Purpose |
|------|-------|---------|
| `base_resource.py` | ~76 | Base resource with registry integration |
| `simple_search.py` | ~326 | DuckDuckGo web search (no API key required) |
| `claude_code_skills.py` | ~461 | Claude Code subprocess skill execution |

---

## 4. API Endpoints Analysis

This is a library module, not a web API. However, it exposes the following programmatic interfaces:

### Agent API
```python
# Main entry point
agent = STARAgent(
    agent_type="my-agent",
    llm_provider="anthropic",
    model="claude-sonnet-4-20250514",
    enable_web_search=True,
    ltmemory_path="./memories/",
)

# Synchronous query
result = agent.query(message="Hello", session_id="abc123")

# Async query
result = await agent.aquery(message="Hello")

# Interactive conversation
agent.converse(initial_message="Hi")

# Add capabilities
agent.with_resources(MyResource())
agent.with_workflows(MyWorkflow())
agent.with_agents(SubAgent())
```

### Resource API
```python
class MyResource(BaseResource):
    @tool_use
    def my_method(self, param: str) -> dict:
        """Method exposed to LLM as tool."""
        return {"result": param}
```

### Workflow API
```python
class MyWorkflow(BaseWorkflow):
    def _do_execute(self, **kwargs) -> dict:
        return {"result": "done"}

# Workflow composition
pipeline = workflow1 | workflow2 | transform_func
result = pipeline.execute(data="input")
```

---

## 5. Architecture Deep Dive

### STAR Pattern (See-Think-Act-Reflect)

The core architecture implements a cognitive loop inspired by human reasoning:

```
┌─────────────────────────────────────────────────────────────┐
│                      STAR LOOP                              │
│                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────┐ │
│  │   SEE   │──▶│  THINK  │──▶│   ACT   │──▶│   REFLECT   │ │
│  └─────────┘   └─────────┘   └─────────┘   └─────────────┘ │
│       │             │             │               │        │
│  Perceive      LLM Call      Execute          Learning     │
│  inputs        + Parse        Tools            Phases      │
│                                                             │
│  Loop continues until done=true or max iterations          │
└─────────────────────────────────────────────────────────────┘
```

### Phase Details

| Phase | Input | Output | Description |
|-------|-------|--------|-------------|
| **SEE** | User message, tool results | Timeline percepts | Perceives inputs, updates timeline |
| **THINK** | Timeline context | Response + tool calls | LLM reasoning via runtime |
| **ACT** | Tool calls | Tool results | Executes tools via runtime |
| **REFLECT** | Trace outputs | Learning artifacts | 4-phase learning (async) |

### Learning Phases

```
ACQUISITIVE ──▶ EPISODIC ──▶ INTEGRATIVE ──▶ RETENTIVE
(per-loop)    (per-query)   (per-session)   (long-term)
```

### Runtime Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AgentRuntime (ABC)                      │
│  build_prompt() │ call_llm() │ parse_response() │ execute_tools() │
└─────────────────────────────────────────────────────────────┘
            │                               │
   ┌────────┴────────┐            ┌────────┴────────┐
   │ DefaultRuntime  │            │  LegacyRuntime  │
   │  (JSON output)  │            │  (XML output)   │
   │  [RECOMMENDED]  │            │  [DEPRECATED]   │
   └─────────────────┘            └─────────────────┘
```

### Component Composition

```
STARAgent
├── _communicator (Communicator)     # Interactive conversation
├── _state (State)                   # State management
├── _learner (Learner)               # Learning phases
├── _timeline (Timeline)             # Conversation history
├── _runtime (AgentRuntime)          # LLM/tool execution
├── _ltmemory (LTMemory)             # Long-term memory (optional)
├── _event_log (EventLogAPI)         # Event logging (optional)
├── _resources []                    # External capabilities
├── _workflows []                    # Orchestrated processes
└── _agents []                       # Sub-agents
```

---

## 6. Environment & Setup Analysis

### Required Dependencies
- `structlog` - Structured logging
- `requests` - HTTP operations
- `dana.common` - Common utilities (LLM, protocols)
- `dana.repositories` - Storage abstraction

### Optional Dependencies
- `ddgs` or `duckduckgo-search` - Web search
- Claude Code CLI - Skill execution

### Configuration
```python
from dana.common.config import config_manager

# LLM provider selection (auto-detects from environment)
provider = config_manager.get_first_available_provider()

# Provider-specific model
model = config_manager.get_provider_default_model(provider)
```

### Environment Variables
- `ANTHROPIC_API_KEY` - Anthropic API key
- `OPENAI_API_KEY` - OpenAI API key
- `USER` / `USERNAME` - User identification for context

---

## 7. Technology Stack Breakdown

### Core Technologies
| Category | Technology | Purpose |
|----------|------------|---------|
| Language | Python 3.10+ | Type hints, async/await |
| Async | asyncio | Non-blocking operations |
| Concurrency | threading | Background learning |
| Logging | structlog | Structured JSON logging |

### LLM Integration
| Provider | Support | Features |
|----------|---------|----------|
| Anthropic | Full | Default provider, native tools |
| OpenAI | Full | JSON mode, native tools |
| Others | Via adapters | Extensible |

### Storage
| Type | Implementation | Purpose |
|------|----------------|---------|
| Timeline | Repository pattern | Conversation persistence |
| Learning | Repository pattern | Learning artifacts |
| LTMemory | Markdown files | Human-readable memory |

---

## 8. Visual Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DANA CORE ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   User/Caller   │
                              └────────┬────────┘
                                       │ query() / aquery()
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STARAgent                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          STAR LOOP                                     │  │
│  │  ┌───────┐    ┌─────────┐    ┌───────┐    ┌──────────────────────┐   │  │
│  │  │  SEE  │───▶│  THINK  │───▶│  ACT  │───▶│      REFLECT         │   │  │
│  │  │       │    │         │    │       │    │ (async background)   │   │  │
│  │  └───┬───┘    └────┬────┘    └───┬───┘    └──────────────────────┘   │  │
│  └──────┼─────────────┼─────────────┼───────────────────────────────────┘  │
│         │             │             │                                       │
│         │             ▼             │                                       │
│         │     ┌───────────────┐     │                                       │
│         │     │ AgentRuntime  │     │                                       │
│         │     │ ┌───────────┐ │     │                                       │
│         │     │ │build_prompt│ │     │                                       │
│         │     │ │ call_llm  │ │     │                                       │
│         │     │ │parse_resp │ │     │                                       │
│         │     │ │exec_tools │ │     │                                       │
│         │     │ └───────────┘ │     │                                       │
│         │     └───────┬───────┘     │                                       │
│         │             │             │                                       │
│         ▼             ▼             ▼                                       │
│  ┌───────────┐  ┌───────────┐  ┌───────────────────────────────────────┐   │
│  │ Timeline  │  │    LLM    │  │           Tool Execution              │   │
│  │           │  │ Provider  │  │  ┌──────────┬──────────┬──────────┐  │   │
│  │ Entries:  │  │ (Anthropic│  │  │Resources │Workflows │Sub-Agents│  │   │
│  │ -User msg │  │  OpenAI)  │  │  └──────────┴──────────┴──────────┘  │   │
│  │ -Agent rsp│  └───────────┘  └───────────────────────────────────────┘   │
│  │ -Tool call│                                                              │
│  │ -Tool rslt│                                                              │
│  └───────────┘                                                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Memory Systems                                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐    │   │
│  │  │  STMemory    │  │  LTMemory    │  │      Reflection        │    │   │
│  │  │ (session)    │  │ (persistent) │  │  4-phase distillation  │    │   │
│  │  └──────────────┘  └──────────────┘  └────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │ Global Registry │
                              │  ┌───────────┐  │
                              │  │  Agents   │  │
                              │  │ Resources │  │
                              │  │ Workflows │  │
                              │  └───────────┘  │
                              └─────────────────┘
```

### Data Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ SEE Phase                                                   │
│  1. Add user message to Timeline                            │
│  2. Process previous tool results (if any)                  │
│  3. Mark latest user message                                │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ THINK Phase                                                 │
│  1. Compress timeline if needed (LLM summarization)         │
│  2. Build LLM messages via runtime.build_prompt()           │
│  3. Call LLM via runtime.call_llm()                         │
│  4. Parse response via runtime.parse_response()             │
│  5. Extract: done, reasoning, response, tool_calls          │
│  6. Add thoughts/response to Timeline                       │
│  7. Retry up to 3x on format errors                         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ACT Phase                                                   │
│  1. Execute tool calls via runtime.execute_tools()          │
│  2. Route to: agents, resources, or workflows               │
│  3. Add tool results to Timeline                            │
│  4. Add multi-step progress reminders                       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ REFLECT Phase (async, non-blocking)                         │
│  1. ACQUISITIVE: Per-loop learning                          │
│  2. Store insights via Learner                              │
│  3. RETENTIVE: Persist to LTMemory (if available)           │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
              ┌────────────────────────┐
              │  done=true?            │
              │  max iterations?       │
              │  error?                │
              └────────────┬───────────┘
                    NO     │     YES
              ┌────────────┴───────────┐
              │                        │
              ▼                        ▼
         Loop back              Return response
         to SEE                 to caller
```

---

## 9. Key Insights & Recommendations

### Code Quality Assessment

**Strengths:**
- Clean separation of concerns via composition
- Observable pattern for debugging and monitoring
- Thread-safe registry for multi-agent coordination
- Async support throughout the stack
- Flexible runtime abstraction for different LLM outputs

**Areas for Improvement:**
- `star_agent.py` is large (~1100 lines) - could benefit from further decomposition
- Some duplication between sync and async THINK/ACT methods
- Legacy runtime code adds maintenance burden

### Security Considerations

1. **Tool Execution:** Tools execute arbitrary code paths - validate inputs
2. **Web Search:** `simple_search.py` fetches arbitrary URLs - sanitize outputs
3. **Claude Skills:** Executes subprocess with `--dangerously-skip-permissions`
4. **LTMemory:** Stores to file system - validate paths

### Performance Optimization Opportunities

1. **Timeline Compression:** Already implemented with LLM summarization
2. **Parallel Tool Execution:** `execute_tools_async` supports this
3. **Context Caching:** `ContextBuilder` has token budget management
4. **IP Geolocation:** Already cached per session

### Maintainability Suggestions

1. **Deprecation:** Remove `LegacyRuntime` in next major version
2. **Type Hints:** Some methods lack return type hints
3. **Documentation:** Consider adding more inline examples
4. **Testing:** Each component could have dedicated test files

### Architectural Recommendations

1. **Plugin System:** Consider formalizing resource/workflow discovery
2. **Event Bus:** Replace threading-based learning with async event system
3. **Metrics:** Add prometheus-style metrics for observability
4. **Configuration:** Consider pydantic-settings for validation

---

## 10. Usage Examples

### Basic Agent

```python
from dana.core import STARAgent

agent = STARAgent(
    agent_type="assistant",
    llm_provider="anthropic",
    model="claude-sonnet-4-20250514",
)

result = agent.query(message="What is 2 + 2?")
print(result["response"])
```

### Agent with Resources

```python
from dana.core import STARAgent
from dana.core.resource import BaseResource
from dana.common.protocols.war import tool_use

class Calculator(BaseResource):
    @tool_use
    def add(self, a: int, b: int) -> dict:
        """Add two numbers."""
        return {"result": a + b}

agent = STARAgent(agent_type="math")
agent.with_resources(Calculator(resource_id="calc"))

result = agent.query(message="Add 5 and 3")
```

### Agent with Memory

```python
agent = STARAgent(
    agent_type="memory-agent",
    ltmemory_path="./my_memories/",
)

# Memories persist across sessions
agent.query(message="Remember that my favorite color is blue")
# Later...
agent.query(message="What is my favorite color?")
```

### Workflow Composition

```python
from dana.core.workflow import BaseWorkflow

class FetchWorkflow(BaseWorkflow):
    def _do_execute(self, url: str, **kwargs) -> dict:
        return {"content": requests.get(url).text}

class ParseWorkflow(BaseWorkflow):
    def _do_execute(self, content: str, **kwargs) -> dict:
        return {"parsed": content.upper()}

# Compose with | operator
pipeline = FetchWorkflow() | ParseWorkflow()
result = pipeline.execute(url="https://example.com")
```

---

## 11. Module Statistics

| Metric | Value |
|--------|-------|
| Total Files | 46 |
| Total Directories | 14 |
| Python Files | 44 |
| Approx. Lines of Code | ~8,500 |
| Core Components | 6 (agent, context, memory, reflection, resource, workflow) |
| Abstract Base Classes | 4 (AgentRuntime, AbstractCodec, BasePromptEngineer, ObserverProtocol) |

---

*Generated: 2026-01-19*
*Module: dana_agent/dana/core*
*Version: Based on develop branch*
