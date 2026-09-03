# Code Standards & Guidelines

**Version:** 0.1.1 | **Last Updated:** 2026-03-21

## Core Principles

**YAGNI - KISS - DRY**
- You Aren't Gonna Need It: Don't add features "just in case"
- Keep It Simple, Stupid: Prefer clarity over cleverness
- Don't Repeat Yourself: Extract common patterns into reusable components

## File Organization

### Directory Structure

```
dana/
├── __init__/              # Initialization & environment setup
├── apps/                  # CLI applications (5 entry points)
├── cli/                   # Terminal UI components
├── common/
│   ├── llm/               # LLM abstraction & providers
│   ├── protocols/         # Base protocols & interfaces
│   ├── resource/          # Resource-related protocols
│   ├── schemas/           # Data models & schemas
│   └── utils/             # Shared utilities
├── config/                # Configuration structures
├── core/
│   ├── agent/             # Agent implementations
│   ├── resource/          # Built-in resources
│   ├── timeline/          # Conversation timeline
│   ├── workflow/          # Workflow composition
│   ├── memory/            # Short-term memory
│   ├── prompt/            # Prompt building
│   ├── reminder/          # Context injection
│   ├── skills/            # Claude Code skills
│   └── knowledge/         # Knowledge base systems
├── lib/
│   ├── memory/            # Long-term memory
│   └── resources/         # Extended resources (web, MCP)
└── repositories/          # Data persistence adapters
```

### File Naming Conventions

**Python Files:**
- Use `kebab-case` with descriptive names
- Examples: `star_agent.py`, `base_resource.py`, `web_researcher.py`
- Avoid abbreviations unless universally recognized

**Configuration Files:**
- `config.json` - User configuration
- `pyproject.toml` - Package metadata
- `.env` - Environment variables (never commit)

**Test Files:**
- `test_*.py` - Unit tests
- `test_*/` - Test subdirectories
- Same structure as main codebase

## Code Style

### Python Style Guide

**Language Version:** Python 3.12+

**Linting & Formatting:**
- **Tool:** Ruff
- **Line Length:** 140 characters
- **Commands:**
  ```bash
  ruff check .              # Lint
  ruff format .             # Format
  ruff check . --fix        # Lint & fix
  ```

**Imports:**
- Standard library first
- Third-party packages second
- Local imports last
- Use absolute imports: `from dana.core.agent import STARAgent`

```python
import asyncio
from typing import Any, Optional
from pathlib import Path

import structlog
from pydantic import BaseModel

from dana.core.agent import STARAgent
from dana.common.llm import LLMProvider
```

### Type Hints

**Requirement:** Full type hints on all functions/methods

```python
async def process_message(
    self,
    message: str,
    tools: Optional[list[str]] = None
) -> dict[str, Any]:
    """Process a message through the agent."""
    pass
```

**Guidelines:**
- Use modern syntax: `list[str]` not `List[str]`
- Use `Optional[T]` for nullable values
- Use `Any` sparingly; be specific
- Use `Protocol` for structural typing
- Return types always required

### Docstrings

**Format:** Google-style docstrings

```python
def create_agent(
    model: str,
    tools: Optional[list[str]] = None,
    system_prompt: Optional[str] = None
) -> STARAgent:
    """Create a STAR agent instance.

    Args:
        model: Model identifier (e.g., 'gpt-4.1')
        tools: List of tool names to enable
        system_prompt: Custom system prompt

    Returns:
        Configured STARAgent instance

    Raises:
        ValueError: If model not found in config
    """
    pass
```

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `STARAgent`, `BaseResource` |
| Functions | snake_case | `create_agent()`, `process_message()` |
| Constants | UPPER_SNAKE_CASE | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Private | Leading `_` | `_internal_method()` |
| Protected | Single `_` | `_base_method()` |
| Properties | snake_case | `@property agent_state` |

### Code Organization Within Files

**Order:**
1. Module docstring
2. Imports (standard, third-party, local)
3. Constants
4. Helper functions (module-level)
5. Class definitions
6. Main execution (`if __name__ == "__main__"`)

```python
"""Timeline management for conversation history."""

import asyncio
from datetime import datetime
from typing import Any

import structlog

from dana.core.timeline.entry import TimelineEntry

logger = structlog.get_logger(__name__)

MAX_TOKENS = 4096
DEFAULT_COMPRESSION_THRESHOLD = 0.8


async def compress_timeline(entries: list[TimelineEntry]) -> list[TimelineEntry]:
    """Compress timeline entries."""
    pass


class Timeline:
    """Manages conversation history with automatic compression."""

    def __init__(self, max_tokens: int = MAX_TOKENS):
        self.max_tokens = max_tokens

    async def add_entry(self, entry: TimelineEntry) -> None:
        """Add entry to timeline."""
        pass
```

## Architecture Patterns

### 1. Composition Over Inheritance

**Pattern:** STARAgent composes subsystems rather than inheriting

```python
class STARAgent(BaseSTARAgent):
    """STAR agent using composition."""

    def __init__(self, ...):
        self.communicator = Communicator(...)
        self.state = State(...)
        self.learner = Learner(...)
        self.observer = Observer(...)
```

**Benefits:**
- Flexible subsystem swapping
- Clearer dependencies
- Easier testing

### 2. Protocol-Based Design

**Pattern:** Use protocols for extensibility

```python
from typing import Protocol

class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def complete(self, messages: list[dict]) -> str:
        """Generate completion."""
        ...

class OpenAIProvider:
    """OpenAI implementation."""

    async def complete(self, messages: list[dict]) -> str:
        return await self.client.chat.completions.create(...)
```

**Benefits:**
- No base class required
- Structural typing
- Extensible without modification

### 3. Auto-Registration

**Pattern:** Resources/agents auto-register on creation

```python
class BaseResource:
    """Base resource with auto-registration."""

    _registry: dict[str, type[BaseResource]] = {}

    def __init_subclass__(cls):
        """Register subclass automatically."""
        cls._registry[cls.__name__] = cls

class BashResource(BaseResource):
    """Auto-registers on import."""
    pass
```

### 4. Provider Pattern

**Pattern:** Codec-based LLM abstraction

```python
class CodecRuntimeBase:
    """Base runtime with codec support."""

    def __init__(self, codec: ToolCodec, ...):
        self.codec = codec

    async def complete(self, messages, tools):
        # Codec handles native vs XML tool use
        return await self.codec.complete(messages, tools)
```

### 5. Resource Framework

**Pattern:** Extensible tool execution

```python
class BaseResource:
    """Base class for all resources."""

    @property
    def schema(self) -> dict[str, Any]:
        """Generate tool schema from methods."""
        return self._generate_schema()

    def get_tools(self) -> list[Tool]:
        """Get executable tools."""
        pass

class CustomResource(BaseResource):
    """Custom resource implementation."""

    async def my_tool(self, arg: str) -> str:
        """Tool method - auto-exposed."""
        return f"Result: {arg}"
```

## Testing Requirements

### Test Structure

**Location:** `tests/` with same structure as `dana/`

```
tests/
├── unit/
├── integration/
├── functional/
├── live/               # Against real LLMs
├── regression/
├── stress/
└── conftest.py
```

### Test Guidelines

**Coverage:** Minimum 70%

```bash
make test-cov           # Run with coverage report
```

**Test Patterns:**

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_agent_processing():
    """Test agent message processing."""
    agent = STARAgent(model="test")
    result = await agent.aquery(message="Hello")
    assert isinstance(result, dict)

@pytest.mark.asyncio
async def test_resource_with_mock():
    """Test resource with mocked LLM."""
    with patch("dana.common.llm.LLM") as mock_llm:
        mock_llm.return_value = AsyncMock()
        resource = CustomResource()
        await resource.execute("test")
        mock_llm.assert_called()
```

**Markers:**
- `@pytest.mark.live` - Requires real LLM
- `@pytest.mark.asyncio` - Async test
- `@pytest.mark.windows_console` - Windows-specific

**Mock LLM:**
```bash
DANA_MOCK_LLM=true make test
```

## Error Handling

### Exception Hierarchy

```python
class DanaException(Exception):
    """Base exception for Dana."""
    pass

class AgentError(DanaException):
    """Agent execution error."""
    pass

class ResourceError(DanaException):
    """Resource operation error."""
    pass

class LLMError(DanaException):
    """LLM provider error."""
    pass
```

### Error Handling Pattern

```python
async def execute_resource(self, resource: str, **kwargs) -> Any:
    """Execute resource with error handling."""
    try:
        resource_obj = self.registry.get(resource)
        if not resource_obj:
            raise ResourceError(f"Resource not found: {resource}")
        return await resource_obj.execute(**kwargs)
    except ResourceError:
        raise
    except Exception as e:
        logger.error("resource_execution_failed", error=str(e))
        raise ResourceError(f"Execution failed: {e}") from e
```

## Async/Await Patterns

**Requirement:** All I/O operations must be async

```python
async def fetch_data(self, url: str) -> str:
    """Fetch data from URL - must be async."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text

async def handle_messages(self, messages: list[str]):
    """Process multiple messages concurrently."""
    tasks = [self.handle(msg) for msg in messages]
    return await asyncio.gather(*tasks)
```

## Logging Standards

**Tool:** structlog

```python
import structlog

logger = structlog.get_logger(__name__)

# Structured logging
logger.info(
    "agent_executed",
    agent_id=agent.id,
    model=agent.model,
    tokens_used=tokens,
    duration=elapsed_ms
)

# Debug logs (disable in production)
logger.debug("internal_state", state=agent_state)

# Errors with context
logger.error(
    "execution_failed",
    error=str(e),
    resource=resource_name,
    exc_info=True
)
```

**Naming:** Use snake_case event names

## Configuration Management

**File:** `dana/config/storage_config.py`

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings from environment."""

    openai_api_key: str
    anthropic_api_key: str
    max_tokens: int = 4096
    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

**Usage:**
```python
settings = Settings()
agent = STARAgent(model="gpt-4.1", api_key=settings.openai_api_key)
```

## Performance Guidelines

### Token Management

```python
class Timeline:
    """Timeline with token awareness."""

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.compression_threshold = 0.8

    async def should_compress(self) -> bool:
        """Check if compression needed."""
        used = self.calculate_tokens()
        return used >= self.max_tokens * self.compression_threshold
```

### Streaming Support

```python
async def stream_response(
    self,
    messages: list[dict]
) -> AsyncIterator[str]:
    """Stream response tokens."""
    async with self.llm.stream(messages) as stream:
        async for chunk in stream:
            yield chunk.content
```

### Caching Strategies

```python
from functools import lru_cache

class WebResearcher:
    """Web research with URL caching."""

    @lru_cache(maxsize=256)
    async def fetch_url(self, url: str) -> str:
        """Fetch with cache."""
        return await self._fetch(url)
```

## Code Review Checklist

Before submitting code, verify:

- [ ] Type hints on all functions/methods
- [ ] Docstrings with Args/Returns/Raises
- [ ] Error handling with specific exceptions
- [ ] Async/await for all I/O operations
- [ ] Tests with >70% coverage
- [ ] No hardcoded secrets/credentials
- [ ] Logging at appropriate levels
- [ ] Follows naming conventions
- [ ] No unnecessary complexity
- [ ] Ruff linting passes
- [ ] MyPy type checking passes

## File Size Guidelines

**Target:** Keep files under 200 LOC

**When to split:**
- File exceeds 200 lines
- Multiple distinct responsibilities
- Testability suffers from size
- Readability decreased

**Splitting Strategy:**
1. Identify logical boundaries
2. Extract into separate module
3. Use composition or inheritance
4. Update imports in original file

**Example:**

```python
# agent.py - 250 LOC → split into:
# agent.py (100 LOC) - Main class
# agent_communicator.py (80 LOC) - Communicator component
# agent_state.py (70 LOC) - State management
```

## Security Guidelines

### Environment Variables

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

# Access with defaults
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set")
```

### Input Validation

```python
from pydantic import BaseModel, Field

class ToolCall(BaseModel):
    """Validated tool call."""

    name: str = Field(..., min_length=1, pattern=r"^[a-z_]+$")
    args: dict = Field(default_factory=dict)

    @field_validator("name")
    def validate_name(cls, v: str) -> str:
        """Validate tool name."""
        if not v.islower():
            raise ValueError("Tool name must be lowercase")
        return v
```

### Safe Code Execution

```python
class PythonSandbox:
    """Safe Python execution environment."""

    async def execute(self, code: str) -> str:
        """Execute code in isolated environment."""
        # Never use eval() - always use sandbox
        return await self._run_in_subprocess(code)
```

## Common Pitfalls to Avoid

1. **Blocking I/O in Async** - Always use async libraries (httpx, asyncpg, etc.)
2. **Hardcoded Values** - Use configuration/environment variables
3. **Missing Error Handling** - Catch and log specific exceptions
4. **Weak Type Hints** - Use specific types, avoid `Any` when possible
5. **Oversized Functions** - Keep functions focused, extract helpers
6. **Missing Tests** - Aim for >70% coverage
7. **Unclear Names** - Use descriptive names for clarity
8. **Premature Optimization** - Optimize only when profiled

## Resources

- **Python:** https://pep8.org/
- **Type Hints:** https://peps.python.org/pep-0484/
- **Async:** https://docs.python.org/3/library/asyncio.html
- **Testing:** https://docs.pytest.org/
- **Logging:** https://www.structlog.org/

---

**Version:** 0.1.1 | **Last Updated:** 2026-03-21
