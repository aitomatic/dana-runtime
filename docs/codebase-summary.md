# Dana Runtime - Codebase Summary

**Version:** 0.1.1 | **License:** MIT | **Python:** >=3.12

## Project Overview

Dana is a Python agentic runtime implementing the **STAR pattern (See-Think-Act-Reflect)** for building conversational AI agents. It provides a complete framework for LLM-powered agents with multi-provider support, tool execution, conversation timeline management, and workflow composition.

## Codebase Structure

**Total:** 210 Python files, ~38,662 LOC

```
dana/
├── __init__/              # Environment initialization
├── apps/                  # CLI applications (5 entry points)
├── cli/                   # Terminal UI components
├── common/                # Shared protocols, LLM providers, schemas
├── config/                # Configuration management
├── core/                  # Core runtime & agent logic
├── lib/                   # Web research, memory, MCP client
├── repositories/          # Data persistence layer
└── specs/                 # Design specifications (markdown)
```

### Component Breakdown

| Module | Files | LOC | Purpose |
|--------|-------|-----|---------|
| **core/** | 90 | 20,001 | Agent runtime, tools, timeline, prompts |
| **common/** | 48 | 6,242 | LLM providers, schemas, base classes |
| **lib/** | 33 | 7,741 | Web research, memory, MCP, embeddings |
| **apps/** | 15 | 1,594 | CLI apps & entry points |
| **cli/** | 12 | 1,555 | Rich terminal UI components |
| **repositories/** | 5 | 1,249 | Data persistence adapters |
| **config/** | 2 | 69 | Configuration structures |
| **specs/** | 27 | — | Design docs & architecture specs |

## Core Architecture

### Agent System (dana/core/agent/)
**4,728 LOC** - Main orchestrator for conversational agents

- **STARAgent**: Primary class implementing See-Think-Act-Reflect loop
- **Hierarchy**: BaseAgent → BaseSTARAgent → STARAgent → AssistantAgent
- **Composition**: Communicator, State, Learner, Observer components
- **Features**:
  - Streaming response support
  - Token-aware context management
  - Multi-turn conversation handling
  - Python sandbox execution

**Key Files:**
- `star_agent.py` (8,827 tokens) - Main STAR implementation
- `base_agent.py` - Base agent protocol
- `communicator.py` - LLM communication
- `state.py` - Conversation state management
- `learner.py` - Learning from interactions
- `observer.py` - Observation & introspection
- `python_sandbox.py` - Safe code execution

### Timeline System (dana/core/timeline/)
**3,172 LOC** - Conversation history with compression

- **Entry Types**: 9 types (user, assistant, tool_result, summary, etc.)
- **Token Awareness**: Auto-compression at 80% context threshold
- **Summarization**: LLM-based history compression
- **Serialization**: JSON persistence

**Key Files:**
- `timeline.py` - Main timeline class
- `entry.py` - Entry type definitions
- `compressor.py` - History compression logic

### Resource System (dana/core/resource/)
**2,795 LOC** - Tool execution framework with auto-registration

**Built-in Resources:**
- **BashResource** - Execute shell commands
- **FileIOResource** - Read/write files
- **FileEditResource** - Edit file contents
- **SearchResource** - Web search
- **TaskResource** - Task management
- **TodoResource** - Todo lists
- **SkillResource** - Claude Code skills

**Features:**
- Auto-registration on instantiation
- Tool schema generation
- Result parsing & formatting

**Key Files:**
- `base_resource.py` - Resource base class
- `bash_resource.py`, `file_io_resource.py`, etc. - Implementations
- `resource_registry.py` - Global registry

### Runtime & LLM Integration (dana/core/runtime/ + dana/common/llm/)
**~6,500 LOC combined** - Provider-agnostic LLM abstraction

**Runtime Hierarchy:**
- AgentRuntime (base protocol)
- DefaultRuntime (uses all providers)
- AnthropicRuntime, OpenAIRuntime, GeminiRuntime, AzureRuntime

**LLM Providers:**
- **Anthropic** (386 LOC) - Claude models
- **OpenAI** (320 LOC) - GPT models
- **Google Gemini** (296 LOC) - Gemini models
- **Azure** (180 LOC) - Azure OpenAI
- **Anthropic-Like** - Custom endpoints

**Codec System:** Native tool use vs. CSXMLCodec for non-native support

### Workflow System (dana/core/workflow/)
**1,252 LOC** - Composition engine for multi-step workflows

- **BaseWorkflow**: Abstract base for composable workflows
- **CallableWorkflow**: Executes composition of steps
- **WorkflowExecutor**: Manages workflow execution
- **Validation**: Input/output validation

### Memory Systems
**~4,000 LOC combined**

**Short-Term Memory (STMemory)**
- Per-session conversation cache
- Location: `dana/core/memory/`

**Long-Term Memory (LTMemory)**
- Markdown-based persistence
- Memory Types: lesson, episode, fact, pattern
- Location: `dana/lib/memory/`

### Web Research (dana/lib/resources/web_research/)
**3,200+ LOC** - Complete research pipeline

- HTML extraction & cleaning
- URL fetching & caching
- Content parsing
- Multi-source synthesis
- Search result integration

**Key Files:**
- `web_researcher.py` - Main orchestrator
- `extractors/` - Content extraction
- `parsers/` - Format-specific parsing

### MCP Client (dana/lib/resources/mcp/)
**1,236 LOC** - Model Context Protocol implementation

- Server discovery
- Tool registration
- Message protocol handling
- SSE event streaming

### Prompt System (dana/core/prompt/)
**~1,500 LOC** - Prompt building & generation

- **PromptBuilder**: Constructs prompts from components
- **PromptAPI**: High-level interface
- Codec-specific prompt generation
- Environment info injection

### Reminder System (dana/core/reminder/)
**538 LOC** - Context injection before LLM calls

- Lazy evaluation
- Dynamic context population
- Pre-call hooks

### Skills System (dana/core/skills/)
**1,573 LOC** - Claude Code skills integration

- Skill discovery from `.claude/skills/`
- Skill execution framework
- Dana-specific skill utilities

### CLI Components (dana/cli/)
**1,555 LOC** - Rich terminal UI

**Components:**
- **RichCLIRenderer** - Main rendering engine
- **HintBar** - Contextual hints
- **ProgressTracker** - Progress indicators
- **StreamDisplay** - Token streaming
- **SubagentCard** - Subagent status
- **ToolCard** - Tool execution
- **Spinner** - Loading animations

## Applications (Entry Points)

| Command | Location | Purpose |
|---------|----------|---------|
| `adana` | `dana/apps/dana/` | Main conversational agent |
| `adana-repl` | `dana/apps/repl/` | Interactive Python REPL |
| `dana-code` | `dana/apps/code/` | Coding agent with UI |
| `dana-init` | `dana/apps/init/` | Config initialization |
| `dana-cli` | `dana/apps/cli/` | CLI client |

## Configuration

**File:** `config.json` (user home or current directory)

```json
{
  "llm_providers": {
    "openai": { "priority": 100, "models": ["gpt-4.1", "gpt-4.1-mini"] },
    "anthropic": { "priority": 90, "models": ["claude-sonnet-4-6"] },
    "gemini": { "priority": 85, "models": ["gemini-2.5-flash"] }
  }
}
```

## Key Design Patterns

1. **STAR Agent Pattern** - Structured reasoning loop
2. **Composition over Inheritance** - STARAgent composes subsystems
3. **Protocol-Based Design** - Extensible abstractions
4. **Provider-Agnostic** - Codec-based LLM abstraction
5. **Auto-Registration** - Resources/workflows self-register
6. **Global Registry** - Centralized agent/resource/workflow registry

## Data Persistence

**Repository Layer** (`dana/repositories/`)
- Abstract interfaces for persistence
- Implementations: file-based, database adapters

## Testing Infrastructure

**105 test files** across 8 categories:
- Unit tests
- Integration tests
- Functional tests
- Live tests (against real LLMs)
- Regression tests
- Stress tests

**Testing Tools:**
- pytest with asyncio support
- Mock LLM support (DANA_MOCK_LLM=true)
- Commands: `make test`, `make test-live`, `make test-cov`

## Quality Standards

**Linting:** Ruff (line-length 140, py312)
**Type Checking:** MyPy
**Formatting:** Ruff (integrated)
**Hooks:** Pre-commit checks

## Dependencies

**Core:** openai, anthropic, google-genai, httpx, python-dotenv, structlog, pydantic, requests, beautifulsoup4, html2text, rich

**Optional Groups:**
- web: lxml, selenium
- local: llama-stack, ollama
- data: pandas, matplotlib
- memory: lancedb, sentence-transformers
- knowledge: rdflib, rank-bm25
- observability: langfuse, langsmith (alternative tracing backend, exclusive with langfuse)

### Tracing Backends (exclusive)

`@observable` (`dana/common/observable.py`) dispatches to ONE tracing backend per process, selected at decoration time. Toggle via env, not code.

| Precedence | Trigger | Backend |
|------------|---------|---------|
| 1 (highest) | `LANGSMITH_TRACING=true` OR `DANA_LANGSMITH_ENABLED=true|1|yes` (+ `langsmith` installed) | LangSmith `@traceable` (background-thread batching, no per-call flush) |
| 2 | `LANGFUSE_ENABLED=true|1|yes` (+ `langfuse` installed) | Langfuse `@observe` + per-call flush |
| 3 (default) | neither set | no-op passthrough |

**LangSmith env vars:** `LANGSMITH_TRACING` (enable, case-sensitive `true`), `DANA_LANGSMITH_ENABLED` (dana-namespace alias), `LANGSMITH_API_KEY` (required for emission), `LANGSMITH_PROJECT` / `LANGSMITH_ENDPOINT` (optional).

If both `LANGSMITH_TRACING` and `LANGFUSE_ENABLED` are set, LangSmith wins. Install via `pip install dana[observability]`.

**Silent no-op caveat:** if `LANGSMITH_TRACING=true` (or `DANA_LANGSMITH_ENABLED`) is set but the `langsmith` package is not installed, an import shim makes `@observable` a silent no-op (no error, no traces). The same applies to `LANGFUSE_ENABLED=true` without `langfuse`. Always install the backend package when enabling its env flag to avoid silent misconfiguration.

## Key Files by Importance

| File | LOC | Purpose |
|------|-----|---------|
| `core/agent/star_agent.py` | 450+ | STAR loop implementation |
| `core/resource/base_resource.py` | 200+ | Resource framework |
| `core/timeline/timeline.py` | 300+ | Conversation management |
| `common/llm/llm.py` | 350+ | LLM abstraction |
| `lib/memory/long_term_memory.py` | 400+ | LT memory system |
| `lib/resources/web_research/web_researcher.py` | 500+ | Research pipeline |

## Extensibility Points

1. **Custom Agents** - Extend BaseAgent
2. **Custom Resources** - Extend BaseResource
3. **Custom Workflows** - Extend BaseWorkflow
4. **Custom Providers** - Implement provider interface
5. **Custom Memory** - Implement memory protocols

## Environment Variables

**Required:**
- `OPENAI_API_KEY` (or other LLM provider keys)

**Optional:**
- `DANA_MOCK_LLM` - Enable mock LLM for testing
- `DANA_DEBUG` - Enable debug logging
- `DANA_CONFIG_PATH` - Custom config location

## Running the Project

```bash
# Install dependencies
uv sync

# Run tests
make test

# Start main agent
adana

# Start REPL
adana-repl

# Start coding agent
dana-code
```

---

**Last Updated:** 2026-03-21 | **Version:** 0.1.1
