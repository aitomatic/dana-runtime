# Dana Runtime

**STAR Pattern Agent Framework for Python**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version 0.1.1](https://img.shields.io/badge/version-0.1.1-green.svg)](#release-status)

Dana is a production-ready Python agentic runtime implementing the **STAR pattern (See-Think-Act-Reflect)** for building sophisticated conversational AI agents. It provides multi-provider LLM support, extensible tool frameworks, intelligent context management, and rich terminal interfaces.

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

### 🛠️ Built-in Resources (9)
- **BashResource** - Execute shell commands
- **FileIOResource** - Read/write files
- **FileEditResource** - Edit files with diffs
- **SearchResource** - Web search integration
- **TaskResource** - Task management
- **TodoResource** - Todo list operations
- **SkillResource** - Claude Code skills
- **WebResearchResource** - Advanced web research
- **MCPResource** - Model Context Protocol support

### 📝 Intelligent Timeline Management
- Chronological conversation history with 9 entry types
- Token-aware automatic compression at 80% threshold
- LLM-based history summarization
- Serializable persistence

### 🧠 Memory Systems
- **Short-Term Memory** - Per-session caching
- **Long-Term Memory** - Persistent markdown storage
- Memory types: lessons, episodes, facts, patterns

### 🌐 Web Research Pipeline
- HTML extraction and cleaning
- Multi-source synthesis
- URL caching
- Search result aggregation

### ✨ CLI Applications
| Command | Purpose |
|---------|---------|
| `adana` | Interactive conversational agent |
| `adana-repl` | Interactive Python REPL |
| `dana-code` | Coding-focused agent with rich UI |
| `dana-init` | Bootstrap config setup |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/aitomatic/dana-runtime.git
cd dana-runtime

# Install dependencies
uv sync

# Initialize configuration
dana-init
```

### First Agent

```python
from dana.core.agent import STARAgent

# Create agent
agent = STARAgent(model="gpt-4.1")

# Process message
response = await agent.process("What time is it?")
print(response)
```

### Run Interactive Agent

```bash
# Start main conversational agent
adana

# Start Python REPL with Dana imported
adana-repl

# Start coding agent
dana-code
```

## Configuration

### config.json

Create `~/.dana/config.json` or `./config.json`:

```json
{
  "llm_providers": {
    "openai": {
      "priority": 100,
      "api_key": "${OPENAI_API_KEY}",
      "models": ["gpt-4.1", "gpt-4.1-mini"]
    },
    "anthropic": {
      "priority": 90,
      "api_key": "${ANTHROPIC_API_KEY}",
      "models": ["claude-sonnet-4-6"]
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
export DANA_DEBUG=true          # Enable debug logging
export DANA_CONFIG_PATH="/path" # Custom config location
```

### Initialize Config

```bash
dana-init
# Interactive setup for API keys and preferences
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
- **Resource System** - Tool execution framework with auto-registration
- **Timeline** - Conversation history with compression
- **Runtime** - Provider-agnostic LLM abstraction
- **Memory** - Short-term and long-term persistence
- **Workflow** - Multi-step composition engine

For detailed architecture, see [docs/system-architecture.md](docs/system-architecture.md).

## Usage Examples

### Basic Agent

```python
from dana.core.agent import STARAgent

agent = STARAgent(model="gpt-4.1")
response = await agent.process("Summarize Python features")
print(response)
```

### With Custom Resources

```python
from dana.core.agent import STARAgent
from dana.core.resource import BaseResource

class MyResource(BaseResource):
    """Custom resource for your domain."""

    async def my_tool(self, param: str) -> str:
        return f"Processed: {param}"

# Register (auto-registers on instantiation)
my_resource = MyResource()

# Use in agent
agent = STARAgent(model="gpt-4.1")
response = await agent.process("Call my_tool with 'hello'")
```

### Streaming Responses

```python
agent = STARAgent(model="gpt-4.1")

async for token in agent.stream_response("Write a poem"):
    print(token, end="", flush=True)
print()
```

### Web Research

```python
from dana.lib.resources.web_research import WebResearchResource

research = WebResearchResource()
result = await research.research(
    topic="Python 3.12 features",
    num_sources=3
)
print(result)
```

### Workflows

```python
from dana.core.workflow import BaseWorkflow, WorkflowExecutor

class ResearchWorkflow(BaseWorkflow):
    async def execute(self, topic: str):
        # Step 1: Research
        research_result = await self.research(topic)

        # Step 2: Summarize
        summary = await self.summarize(research_result)

        # Step 3: Generate report
        report = await self.generate_report(summary)

        return report

executor = WorkflowExecutor(ResearchWorkflow())
result = await executor.run(topic="AI trends")
```

## Development

### Setup Development Environment

```bash
# Install dev dependencies
uv sync

# Run tests
make test

# Run linting
make lint

# Format code
make format

# Type checking
make type-check
```

### Testing

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run with coverage
make test-cov

# Run against live LLMs
make test-live

# Mock LLM testing
DANA_MOCK_LLM=true make test
```

### Code Quality

**Tools:**
- **Ruff** - Linting & formatting (line-length 140)
- **MyPy** - Type checking
- **Pytest** - Testing framework

**Standards:**
- Type hints required on all functions
- >70% code coverage target
- All tests must pass before commit
- Follow [code-standards.md](docs/code-standards.md)

## Documentation

- **[Project Overview & PDR](docs/project-overview-pdr.md)** - Vision, goals, requirements
- **[Codebase Summary](docs/codebase-summary.md)** - Module structure and organization
- **[Code Standards](docs/code-standards.md)** - Coding conventions and patterns
- **[System Architecture](docs/system-architecture.md)** - Architecture diagrams and data flows
- **[Project Roadmap](docs/project-roadmap.md)** - Development timeline and milestones

## API Reference

### STARAgent

```python
agent = STARAgent(
    model: str,                      # e.g., "gpt-4.1"
    system_prompt: Optional[str],    # Custom system prompt
    tools: Optional[list[str]],      # Enabled tool names
    max_tokens: int = 4096,          # Context limit
    compression_threshold: float = 0.8  # Auto-compress at %
)

# Process message
response = await agent.process(message: str) -> str

# Stream response
async for token in agent.stream_response(message: str):
    # Handle token

# Access conversation history
timeline = agent.state.timeline
messages = await timeline.get_entries()
```

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

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Message processing | <5s avg | Depends on LLM |
| Tool execution | Variable | Depends on tool |
| Timeline compression | 1-2s | LLM-based |
| Streaming first token | 1-3s | LLM latency |

**Optimization Tips:**
- Use `stream_response()` for better perceived latency
- Enable memory caching for repeated queries
- Batch multiple requests when possible
- Use compression threshold of 0.7 for faster responses

## Troubleshooting

### API Key Issues

```python
# Verify API key is set
import os
print(os.getenv("OPENAI_API_KEY"))

# Or use config.json
dana-init  # Re-run setup
```

### Timeout Issues

```python
# Increase timeout in agent creation
agent = STARAgent(
    model="gpt-4.1",
    timeout=30  # seconds
)
```

### Debug Mode

```bash
# Enable debug logging
export DANA_DEBUG=true
adana

# Or in code
import structlog
structlog.configure(
    processors=[structlog.processors.JSONRenderer()]
)
```

### Memory Issues

```python
# Reduce context limit
agent = STARAgent(model="gpt-4.1", max_tokens=2048)

# Or adjust compression
agent.state.timeline.compression_threshold = 0.6
```

## Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/my-feature`)
3. Make changes following [code-standards.md](docs/code-standards.md)
4. Run tests (`make test`)
5. Commit with clear message
6. Push to fork
7. Create pull request

**Development Rules:**
- Type hints required
- >70% test coverage
- All tests passing
- Ruff linting compliant
- MyPy type checking passes

## License

MIT License - see [LICENSE](LICENSE) file

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
- **Email:** support@aitomatic.com

## Roadmap

**Current Phase:** Enhancement (Q2 2026)

| Phase | Timeline | Status |
|-------|----------|--------|
| 1. Foundation | Q1 2026 | ✅ Complete |
| 2. Enhancement | Q2 2026 | ⚠️ In Progress |
| 3. Extended Features | Q3 2026 | 🔜 Planned |
| 4. Production Hardening | Q4 2026 | 🔜 Planned |

See [project-roadmap.md](docs/project-roadmap.md) for detailed timeline.

## Version History

- **0.1.1** (2026-03-21) - Stable, production-ready
- **0.1.0** (2026-03-01) - Initial release

## Acknowledgments

Dana is developed by [Aitomatic, Inc.](https://aitomatic.com) with contributions from the open-source community.

---

**Quick Links:** [Docs](docs/) | [Examples](#usage-examples) | [API Reference](#api-reference) | [Contributing](#contributing) | [License](LICENSE)

**Latest Release:** 0.1.1 | **Last Updated:** 2026-03-21
