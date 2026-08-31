# Dana Runtime

**STAR Pattern Agent Framework for Python**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version 0.2.0](https://img.shields.io/badge/version-0.2.0-green.svg)](#version-history)

Dana is a Python agentic runtime implementing the **STAR pattern (See-Think-Act-Reflect)** for building conversational AI agents. It provides multi-provider LLM support, extensible tool resources, timeline-based context management with automatic compression, and a set of CLI applications.

## Features

### 🧠 STAR Agent Pattern
Structured reasoning loop for transparent, explainable agent behavior:
- **See** - Perceive user intent and context
- **Think** - Reason about response using available resources
- **Act** - Execute tools and retrieve information
- **Reflect** - Learn from outcomes and update memory

### 🔌 Multi-Provider LLM Support
- **OpenAI** - gpt-4.1, gpt-4.1-mini, o3, o4-mini
- **Anthropic** - claude-sonnet, claude-opus, claude-haiku
- **Google Gemini** - gemini-2.5-flash, gemini-2.5-pro
- **Azure OpenAI** - Full compatibility
- **Local Models** - LLaMA Stack, Ollama
- **Custom Endpoints** - Anthropic-like protocol support

### 🛠️ Built-in Resources
- **BashResource** - Execute shell commands
- **FileIOResource** - Read/write files
- **FileEditResource** - Edit files with diffs
- **SearchResource** - Web search integration
- **TaskResource** - Task management
- **TodoResource** - Todo list operations
- **SkillResource** - Claude Code skills (gated by `DANA_CLAUDE_SKILLS=1`)
- **CodeExecutionResource** - Sandboxed Python execution
- **Web research pipeline** - search, fetch, extract, synthesize resources

### 📝 Timeline Management
- Chronological conversation history
- Token-aware automatic compression
- LLM-based history summarization
- Serializable persistence per session

### 🧠 Memory Systems
- **Short-Term Memory** - Per-session caching
- **Long-Term Memory** - Persistent markdown storage
- Memory types: lessons, episodes, facts, patterns

## ✨ CLI Applications

All six entrypoints answer `--help` and `--version`:

| Command | Purpose |
|---------|---------|
| `dana-agent` | Interactive conversational agent |
| `dana-agent-repl` | Interactive Python REPL with Dana imported |
| `dana-code` | Coding-focused agent with rich UI |
| `dana-memory` | Memory store inspection (`--json` for machine-readable output) |
| `dana-init` | Bootstrap config setup |
| `dana-acp` | Agent Client Protocol (ACP) server |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/aitomatic/dana-runtime.git
cd dana-runtime

# Install dependencies
uv sync
```

Requires Python >= 3.11.

### Configure an LLM provider

Dana reads provider keys from environment variables (or a `.env` file at the
repo root, loaded automatically):

```bash
# Required (at least one provider)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or bootstrap interactively:

```bash
dana-init
```

### First Agent

```python
import asyncio

from dana.core.agent import STARAgent


async def main():
    agent = STARAgent(model="gpt-4.1")
    response = await agent.aquery(message="What time is it?")
    print(response["response"])


asyncio.run(main())
```

`aquery` returns a dict with keys such as `response`, `reasoning`,
`tool_calls`, and `done`. This sample requires an LLM API key.

### Embedding: AgentSession hello world

For host applications (durable conversation, turn events, journalling), use
`AgentSession` instead of driving `STARAgent` directly. A 15-line working
example lives at [docs/examples/host_hello.py](docs/examples/host_hello.py):

```python
session = await AgentSession.create()
[print(e.text) async for e in session.prompt([TextBlock(text="hello")]) if e.event_type.name == "ASSISTANT_CONTENT_FINAL"]
```

Run it with `uv run python docs/examples/host_hello.py` (live LLM; set
`DANA_MOCK_LLM=1` for a canned reply — the mock switch is a property of that
example script, not a library feature).

### Run Interactive Agents

```bash
dana-agent        # main conversational agent
dana-agent-repl   # Python REPL with Dana imported
dana-code         # coding agent
```

## Configuration

### config.json

Dana ships defaults in `dana/config.json`; override per-user at
`~/.dana/config.json` or `./config.json`, or point `DANA_CONFIG_PATH` at a
custom location. Provider entries use this shape:

```json
{
  "llm": {
    "providers": {
      "openai": {
        "name": "OpenAI",
        "priority": 100,
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4.1",
        "models": ["gpt-4.1", "gpt-4.1-mini", "o3", "o4-mini"]
      }
    }
  }
}
```

### Environment Variables

```bash
# Required (at least one provider)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional
export DANA_CONFIG_PATH="/path/to/config.json"  # Custom config location
```

## Architecture

Dana uses a **layered architecture** with clear separation of concerns:

```
Applications (CLI)
    ↓
Agent Layer (STARAgent + Components)
    ↓
Core Systems (Resources, Timeline, Workflows)
    ↓
LLM Abstraction (Providers, Codecs)
    ↓
Data Persistence & Infrastructure
```

**Key Components:**
- **STARAgent** - Main orchestrator with streaming support
- **AgentSession** - Host-facing conversational session with durable journal
- **Resource System** - Tool execution framework with auto-registration
- **Timeline** - Conversation history with compression
- **Runtime** - Provider-agnostic LLM abstraction
- **Workflow** - Multi-step composition engine

For detailed architecture, see [docs/system-architecture.md](docs/system-architecture.md).

## Usage Examples

All examples below require an LLM API key unless noted.

### Basic Agent

```python
from dana.core.agent import STARAgent

agent = STARAgent(model="gpt-4.1")
response = await agent.aquery(message="Summarize Python features")
print(response["response"])
```

### Streaming Responses

```python
import asyncio

from dana.core.agent import STARAgent


async def main():
    agent = STARAgent(model="gpt-4.1")
    async for event in agent.aquery_stream(message="Write a poem"):
        print(event.event_type.name, event.data)


asyncio.run(main())
```

Emits `THINKING`, `TEXT_DELTA`, and `DONE` events. For a raw
text-only stream, hosts can use `aquery_text_stream(message=...,
cancel_event=...)` after adding the user message to the timeline (see
[docs/examples/host_hello.py](docs/examples/host_hello.py) for the preferred
session-level path).

### Custom Resources

```python
from dana.core.resource import BaseResource


class MyResource(BaseResource):
    """Custom resource for your domain."""

    async def my_tool(self, param: str) -> str:
        return f"Processed: {param}"


# Auto-registers with the global registry on instantiation
my_resource = MyResource()

# Use in agent
agent = STARAgent(model="gpt-4.1")
response = await agent.aquery(message="Call my_tool with 'hello'")
```

### Web Research

```python
from dana.lib.agents.web_research import WebResearchAgent

research = WebResearchAgent()
result = await research.aquery(message="Research Python 3.12 features")
print(result["response"])
```

### Workflows

```python
from dana.core.workflow import BaseWorkflow


class ResearchWorkflow(BaseWorkflow):
    """Research a topic and report."""

    async def execute(self, topic: str):
        return {"topic": topic}


workflow = ResearchWorkflow(workflow_id="research")
```

## Development

### Setup Development Environment

```bash
# Install dev dependencies
uv sync

# Run tests
make test

# Run unit tests only
make test-unit

# Run linting
make lint

# Format code
make format

# Auto-fix lint issues
make fix
```

### Testing

```bash
make test          # All tests (excludes live)
make test-unit     # Unit tests only
make test-live     # Live tests (requires API keys)
make test-cov      # Coverage report
```

### Code Quality

**Tools:**
- **Ruff** - Linting & formatting (line-length 140)
- **Pytest** - Testing framework

**Standards:**
- Type hints required on all functions
- All tests must pass before commit
- Follow [code-standards.md](docs/code-standards.md)

## Documentation

- **[Project Overview & PDR](docs/project-overview-pdr.md)** - Vision, goals, requirements
- **[Codebase Summary](docs/codebase-summary.md)** - Module structure and organization
- **[Code Standards](docs/code-standards.md)** - Coding conventions and patterns
- **[System Architecture](docs/system-architecture.md)** - Architecture diagrams and data flows
- **[Branching Strategy](docs/branching-strategy.md)** - Git branching model and release flow
- **[Project Roadmap](docs/project-roadmap.md)** - Development timeline and milestones
- **[Extending Dana](docs/extending-dana.md)** - Adding agents, resources, workflows

## API Reference

### STARAgent

```python
agent = STARAgent(
    model: str | None,               # e.g. "gpt-4.1"
    llm_provider: str | None,        # e.g. "openai", "anthropic"
    max_context_tokens: int = 4000,  # Timeline context budget
    enable_web_search: bool = False,     # search() + fetch_url(), no API key
    enable_code_execution: bool = False, # sandboxed Python execution
    enable_skills: bool = True,          # Claude Code skills (DANA_CLAUDE_SKILLS=1 gates the scan)
)

# Query agent (async; returns a dict)
response = await agent.aquery(message: str)

# Stream events (THINKING / TEXT_DELTA / DONE)
async for event in agent.aquery_stream(message: str): ...

# Ephemeral replacement for this agent instance (no repository write)
agent.override_system_prompt_template("You are a domain specialist.")

# Only codec runtimes can persist the replacement to their prompt repository
agent.override_system_prompt_template(
    "You are a persistent domain specialist.",
    persist=True,
)

# Conversation state
state = agent.get_state()                    # dict
summary = agent.get_timeline_summary()      # str
```

`persist=False` is the default: the override is ephemeral, scoped to the agent
instance, and never written to the prompt repository. `persist=True` is supported
only by codec runtimes and writes to their configured prompt repository; base
runtimes raise `NotImplementedError`. The template fully replaces, rather than
extends, the default system prompt, so retain every required tool-usage and
output-format instruction in the replacement.

### Custom Resources

```python
from dana.core.resource import BaseResource


class CustomResource(BaseResource):
    async def my_tool(self, param: str) -> str:
        """Tool docstring becomes tool description."""
        return result


# Auto-registers on instantiation
resource = CustomResource()
```

## Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/my-feature`)
3. Make changes following [code-standards.md](docs/code-standards.md)
4. Run tests (`make test`)
5. Commit with clear message
6. Push to fork
7. Create pull request

## License

MIT License — see [LICENSE](LICENSE). Copyright (c) 2026 Dana Contributors.

## Citation

```bibtex
@software{dana-runtime,
  title={Dana: Domain-Aware Neurosymbolic Agents},
  author={Aitomatic, Inc.},
  year={2026},
  url={https://github.com/aitomatic/dana-runtime}
}
```

## Support

- **Documentation:** [docs/](docs/) directory
- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions

## Version History

- **0.2.0** - CLI entrypoints (`dana-agent`, `dana-agent-repl`, `dana-code`, `dana-memory`, `dana-init`, `dana-acp`) with `--help`/`--version`; `AgentSession` host API
- **0.1.1** (2026-03-21) - Stable, production-ready
- **0.1.0** (2026-03-01) - Initial release

## Acknowledgments

Dana is developed by [Aitomatic, Inc.](https://aitomatic.com) with contributions from the open-source community.

---

**Quick Links:** [Docs](docs/) | [Examples](docs/examples/) | [API Reference](#api-reference) | [Contributing](#contributing) | [License](LICENSE)
