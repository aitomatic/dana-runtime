# Dana Core Skills Module Analysis

## 1. Project Overview

**Project Type:** Resource Plugin / Integration Library
**Primary Language:** Python 3.10+
**Architecture Pattern:** Resource-based Plugin with Subprocess Execution
**Module Purpose:** Bridge between Dana agents and Claude Code CLI skills

The `dana.core.skills` module provides a resource that discovers and executes Claude Code skills via subprocess. It enables Dana agents to leverage the extensive skill ecosystem available in the Claude Code CLI without requiring direct integration.

### Tech Stack
- **Runtime:** Python 3.10+ with type hints
- **Subprocess:** Python subprocess module for CLI execution
- **Platform Support:** Cross-platform (macOS, Linux, Windows) with macOS-specific keychain integration
- **Dependencies:**
  - `dana.common.protocols.war` - Tool use decorator
  - `dana.core.resource` - Base resource class

---

## 2. Detailed Directory Structure Analysis

```
dana_agent/dana/core/skills/
├── __init__.py              # Module entry point, exports ClaudeCodeSkills
└── claude_code_skills.py    # Main implementation (461 lines)
```

### Directory Purpose
This is a **single-purpose module** focused on:
1. **Skill Discovery** - Scanning `~/.claude/skills/` for available skills
2. **Skill Exposure** - Making skills available to agents via `@tool_use` decorator
3. **Skill Execution** - Running Claude Code CLI subprocess with proper environment

### Connection to Other Parts
```
dana.core.skills
     │
     ├──▶ dana.core.resource.BaseResource (inherits)
     │
     ├──▶ dana.common.protocols.war.tool_use (decorator)
     │
     └──▶ ~/.claude/skills/ (external filesystem)
          └── <skill_name>/SKILL.md
```

---

## 3. File-by-File Breakdown

### Core Application Files

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 11 | Module entry point, docstring, exports |
| `claude_code_skills.py` | 461 | Full implementation of ClaudeCodeSkills resource |

### `__init__.py` Details
- **Purpose:** Package initialization and public API definition
- **Exports:** `ClaudeCodeSkills` class only
- **Docstring:** Explains skills as "ontological elements" - composable capabilities

### `claude_code_skills.py` Details

**Class: `ClaudeCodeSkills`**
- Inherits from `BaseResource`
- Implements `@tool_use` decorated methods for agent access

**Key Methods:**

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__` | 30 | Initialize with optional skill filtering |
| `_check_claude_available` | 10 | Verify Claude CLI is installed |
| `_discover_skills` | 20 | Scan skills directory for SKILL.md files |
| `_parse_skill_description` | 15 | Extract description from SKILL.md |
| `_filter_skills` | 5 | Filter to requested skills only |
| `_build_execution_env` | 10 | Prepare environment variables |
| `_build_command` | 20 | Construct CLI command with flags |
| `_home_writable` | 15 | Check if home dir is writable |
| `_sync_claude_config_dir` | 20 | Copy config to writable location |
| `_sync_keychain_credentials` | 30 | macOS keychain credential handling |
| `execute` | 50 | Main tool method - run a skill |

---

## 4. API Endpoints Analysis

This is a **library resource**, not a web API. It exposes programmatic interfaces:

### Tool Method: `execute()`

```python
@tool_use
def execute(self, task: str, context: str = "") -> dict:
    """Execute a task using Claude Code skills.

    Args:
        task: What you want done. Include output file path if creating files.
        context: Relevant information from the conversation that the skill needs.

    Returns:
        dict with:
        - success (bool): Whether the task completed successfully
        - output (str): Output from Claude Code
        - error (str): Error message if failed
    """
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `enabled` | `bool` | Whether Claude CLI is available and skills were discovered |
| `skills` | `list[dict]` | List of available/filtered skills |
| `all_skills` | `list[dict]` | All discovered skills before filtering |
| `disable_session_persistence` | `bool` | Session persistence setting |

---

## 5. Architecture Deep Dive

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Dana STARAgent                              │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Resources Collection                        │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │  │
│  │  │ SimpleWebSearch │  │ ClaudeCodeSkills│  │ CustomResource│  │  │
│  │  └─────────────────┘  └────────┬────────┘  └───────────────┘  │  │
│  └─────────────────────────────────┼─────────────────────────────┘  │
└─────────────────────────────────────┼───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ClaudeCodeSkills Resource                       │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Initialization                                                 │  │
│  │  1. Check Claude CLI available                                 │  │
│  │  2. Discover skills from ~/.claude/skills/                     │  │
│  │  3. Filter to requested skills (optional)                      │  │
│  │  4. Generate dynamic docstring with skill list                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ execute(task, context)                                         │  │
│  │  1. Build prompt with context                                  │  │
│  │  2. Create output directory                                    │  │
│  │  3. Build environment (strip API key)                          │  │
│  │  4. Build CLI command with flags                               │  │
│  │  5. Run subprocess                                             │  │
│  │  6. Retry with API key if auth error                           │  │
│  │  7. Return result dict                                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Claude Code CLI                              │
│                                                                     │
│  claude --dangerously-skip-permissions [--no-session-persistence]   │
│         -p "Context: ... Task: ..."                                 │
│                                                                     │
│  Working Directory: ./skill_output/                                 │
│  Environment: CLAUDE_CODE_DISABLE_ATTACHMENTS=1                     │
│               CLAUDE_CODE_IDE_SKIP_AUTO_INSTALL=true                │
└─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ~/.claude/skills/                               │
│                                                                     │
│  ├── pptx/                                                          │
│  │   └── SKILL.md     "Create PowerPoint presentations"            │
│  ├── xlsx/                                                          │
│  │   └── SKILL.md     "Create Excel spreadsheets"                  │
│  ├── pdf/                                                           │
│  │   └── SKILL.md     "Generate PDF documents"                     │
│  └── docx/                                                          │
│      └── SKILL.md     "Create Word documents"                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Request Lifecycle

```
Agent decides to use skill
         │
         ▼
┌─────────────────────────────┐
│ 1. LLM calls execute()      │
│    with task + context      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 2. Build execution context  │
│    - Create output dir      │
│    - Build environment      │
│    - Handle config dir      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 3. Subprocess execution     │
│    - claude CLI call        │
│    - Capture stdout/stderr  │
│    - Timeout handling       │
└──────────────┬──────────────┘
               │
      ┌────────┴────────┐
      │ Success?        │
      └───┬─────────┬───┘
          │         │
     YES  │         │  NO (auth error)
          │         │
          │         ▼
          │    ┌────────────────────┐
          │    │ 4. Retry with      │
          │    │    ANTHROPIC_API_KEY│
          │    └─────────┬──────────┘
          │              │
          ▼              ▼
┌─────────────────────────────┐
│ 5. Return result dict       │
│    {success, output, error} │
└─────────────────────────────┘
```

### Key Design Patterns

1. **Resource Pattern** - Extends `BaseResource` for registry integration
2. **Factory Pattern** - Dynamic docstring generation based on discovered skills
3. **Adapter Pattern** - Bridges Dana agents to Claude Code CLI
4. **Retry Pattern** - Automatic retry with API key on auth failures

---

## 6. Environment & Setup Analysis

### Required Environment

| Requirement | Purpose |
|-------------|---------|
| Claude Code CLI | `npm install -g @anthropic-ai/claude-code` |
| Skills Directory | `~/.claude/skills/` with SKILL.md files |
| Write Access | Output directory (default: `./skill_output/`) |

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Optional | Fallback authentication |
| `USER` / `USERNAME` | Auto | Keychain account (macOS) |

### Internal Environment Setup

The module sets these for the subprocess:
```python
CLAUDE_CODE_DISABLE_ATTACHMENTS=1
CLAUDE_CODE_IDE_SKIP_AUTO_INSTALL=true
CHOKIDAR_USEPOLLING=1
CHOKIDAR_INTERVAL=500
WATCHPACK_POLLING=true
```

### Installation

```python
# Basic usage - all discovered skills
from dana.core.skills import ClaudeCodeSkills

skills = ClaudeCodeSkills()

# Filtered skills - document specialist
skills = ClaudeCodeSkills(skills=["pptx", "docx", "pdf"])

# Custom configuration
skills = ClaudeCodeSkills(
    skills_dir="~/my-skills",
    output_dir="./output",
    timeout=600,
    disable_session_persistence=True,
)
```

---

## 7. Technology Stack Breakdown

### Runtime Environment
- **Python 3.10+** with type hints
- **subprocess** module for CLI execution
- **pathlib** for path handling
- **shutil** for file operations
- **hashlib** for config dir naming

### Platform-Specific Features

| Platform | Feature | Implementation |
|----------|---------|----------------|
| macOS | Keychain integration | `security` CLI commands |
| Unix | FD limit increase | `resource.setrlimit()` |
| Windows | None | Basic subprocess only |

### External Dependencies

| Dependency | Type | Purpose |
|------------|------|---------|
| Claude Code CLI | External | Skill execution engine |
| `~/.claude/skills/` | Filesystem | Skill definitions |
| `~/.claude.json` | Config | Claude Code configuration |
| Keychain (macOS) | System | Credential storage |

---

## 8. Visual Architecture Diagram

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER / AGENT                               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ClaudeCodeSkills                               │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Properties                                                     │ │
│  │  ├── enabled: bool                                              │ │
│  │  ├── skills: list[dict]                                         │ │
│  │  ├── all_skills: list[dict]                                     │ │
│  │  └── disable_session_persistence: bool                          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Methods                                                        │ │
│  │  ├── execute(task, context) -> dict      [@tool_use]            │ │
│  │  ├── _check_claude_available() -> bool                          │ │
│  │  ├── _discover_skills() -> list[dict]                           │ │
│  │  ├── _parse_skill_description(path) -> str                      │ │
│  │  ├── _filter_skills(names) -> list[dict]                        │ │
│  │  ├── _build_execution_env() -> tuple[dict, str|None]            │ │
│  │  ├── _build_command(prompt, path, env) -> list[str]             │ │
│  │  ├── _home_writable() -> bool                                   │ │
│  │  ├── _sync_claude_config_dir(target) -> None                    │ │
│  │  ├── _sync_keychain_credentials(target) -> None   [macOS]       │ │
│  │  ├── _get_preexec_fn() -> Callable|None           [Unix]        │ │
│  │  ├── _run_claude_subprocess(...) -> CompletedProcess            │ │
│  │  ├── _extract_error_output(result) -> str                       │ │
│  │  ├── _should_retry_with_api_key(result, key) -> bool            │ │
│  │  ├── _build_result_dict(result) -> dict                         │ │
│  │  └── get_execute_docstring() -> str                             │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  BaseResource   │  │  ~/.claude/     │  │  Claude Code    │
│  (inheritance)  │  │  skills/        │  │  CLI            │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                       DISCOVERY FLOW                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ~/.claude/skills/                                               │
│       │                                                          │
│       ├── pptx/SKILL.md ─────┐                                   │
│       ├── xlsx/SKILL.md ─────┼──▶ _discover_skills()             │
│       ├── docx/SKILL.md ─────┤         │                         │
│       └── pdf/SKILL.md ──────┘         ▼                         │
│                              ┌─────────────────┐                 │
│                              │ skills: [       │                 │
│                              │   {name, desc}, │                 │
│                              │   {name, desc}, │                 │
│                              │   ...           │                 │
│                              │ ]               │                 │
│                              └─────────────────┘                 │
│                                      │                           │
│                                      ▼                           │
│                              ┌─────────────────┐                 │
│                              │ Dynamic docstring│                │
│                              │ for execute()   │                 │
│                              └─────────────────┘                 │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                       EXECUTION FLOW                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  execute(task="Create slides", context="Topic: AI")              │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────┐                     │
│  │ Prompt: "Context: Topic: AI             │                     │
│  │         Task: Create slides"            │                     │
│  └────────────────────┬────────────────────┘                     │
│                       │                                          │
│                       ▼                                          │
│  ┌─────────────────────────────────────────┐                     │
│  │ Environment:                            │                     │
│  │   - Strip ANTHROPIC_API_KEY (for retry) │                     │
│  │   - Add CLAUDE_CODE_* vars              │                     │
│  │   - Setup config dir if needed          │                     │
│  └────────────────────┬────────────────────┘                     │
│                       │                                          │
│                       ▼                                          │
│  ┌─────────────────────────────────────────┐                     │
│  │ Command:                                │                     │
│  │   claude --dangerously-skip-permissions │                     │
│  │          [--no-session-persistence]     │                     │
│  │          -p "<prompt>"                  │                     │
│  └────────────────────┬────────────────────┘                     │
│                       │                                          │
│                       ▼                                          │
│  ┌─────────────────────────────────────────┐                     │
│  │ subprocess.run(                         │                     │
│  │   cmd, env, cwd=./skill_output,         │                     │
│  │   timeout=300, preexec_fn              │                     │
│  │ )                                       │                     │
│  └────────────────────┬────────────────────┘                     │
│                       │                                          │
│            ┌──────────┴──────────┐                               │
│            │                     │                               │
│       SUCCESS               AUTH ERROR                           │
│            │                     │                               │
│            │                     ▼                               │
│            │         ┌─────────────────────┐                     │
│            │         │ Retry with          │                     │
│            │         │ ANTHROPIC_API_KEY   │                     │
│            │         └──────────┬──────────┘                     │
│            │                    │                                │
│            ▼                    ▼                                │
│  ┌─────────────────────────────────────────┐                     │
│  │ Return:                                 │                     │
│  │   {"success": True/False,               │                     │
│  │    "output": "...",                     │                     │
│  │    "error": "..."}                      │                     │
│  └─────────────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. Key Insights & Recommendations

### Code Quality Assessment

**Strengths:**
- Clean separation of concerns with small, focused methods
- Good error handling with retry mechanism
- Platform-aware implementation (macOS keychain, Unix FD limits)
- Dynamic docstring generation for LLM awareness
- Proper resource inheritance for registry integration

**Well-Structured:**
- Each method has a single responsibility
- Private methods prefixed with `_`
- Clear initialization flow
- Type hints throughout

### Potential Improvements

1. **Async Support**
   ```python
   # Current: synchronous only
   def execute(self, task: str, context: str = "") -> dict:

   # Suggested: add async variant
   async def aexecute(self, task: str, context: str = "") -> dict:
       # Use asyncio.subprocess
   ```

2. **Skill Validation**
   ```python
   # Add validation that skill exists before execution
   def execute(self, task: str, skill: str = None, context: str = "") -> dict:
       if skill and skill not in [s["name"] for s in self._skills]:
           return {"success": False, "error": f"Unknown skill: {skill}"}
   ```

3. **Caching**
   ```python
   # Cache skill discovery results
   @functools.lru_cache(maxsize=1)
   def _discover_skills(self) -> tuple[dict, ...]:  # tuple for hashability
   ```

### Security Considerations

| Risk | Mitigation | Status |
|------|------------|--------|
| Command injection | Prompt passed via `-p` flag, not shell | ✅ Mitigated |
| API key exposure | Key stripped from env, added only on retry | ✅ Mitigated |
| File system access | Operates in `./skill_output/` directory | ⚠️ User-controlled |
| Subprocess timeout | 300s default, configurable | ✅ Implemented |
| `--dangerously-skip-permissions` | **Required for automation** | ⚠️ Documented risk |

**Critical Note:** The `--dangerously-skip-permissions` flag bypasses Claude Code's permission system. This is necessary for automated execution but means:
- Claude Code can execute arbitrary file operations
- Network requests are unrestricted
- Code execution is possible

**Recommendation:** Document this clearly for users and consider sandboxing options.

### Performance Optimization Opportunities

1. **Skill Discovery Caching**
   - Currently rediscovers on every instantiation
   - Cache with file modification time checks

2. **Connection Pooling**
   - Each execution spawns a new process
   - Consider persistent Claude Code session (conflicts with `disable_session_persistence`)

3. **Parallel Execution**
   - Current: Sequential subprocess calls
   - Opportunity: `asyncio.gather()` for multiple skill calls

### Maintainability Suggestions

1. **Configuration Dataclass**
   ```python
   @dataclass
   class SkillsConfig:
       skills_dir: str = "~/.claude/skills"
       output_dir: str = "./skill_output"
       timeout: int = 300
       disable_session_persistence: bool = False
   ```

2. **Result Dataclass**
   ```python
   @dataclass
   class SkillResult:
       success: bool
       output: str
       error: str
   ```

3. **Logging**
   ```python
   import structlog
   logger = structlog.get_logger()

   # Add structured logging for debugging
   logger.info("executing_skill", task=task, context_length=len(context))
   ```

---

## 10. Usage Examples

### Basic Usage

```python
from dana.core.skills import ClaudeCodeSkills

# All available skills
skills = ClaudeCodeSkills()

# Execute a task
result = skills.execute(
    task="Create a 5-slide presentation about AI trends. Save to ./skill_output/ai_trends.pptx",
    context="Focus on: LLMs, multimodal AI, and agents"
)

if result["success"]:
    print(f"Created: {result['output']}")
else:
    print(f"Error: {result['error']}")
```

### Filtered Skills

```python
# Document-focused agent
doc_skills = ClaudeCodeSkills(
    skills=["pptx", "docx", "xlsx", "pdf"]
)

# Check available skills
print(doc_skills.skills)
# [{"name": "pptx", "description": "Create PowerPoint..."}, ...]
```

### With STARAgent

```python
from dana.core import STARAgent
from dana.core.skills import ClaudeCodeSkills

agent = STARAgent(agent_type="document-assistant")
agent.with_resources(ClaudeCodeSkills(skills=["pptx", "docx"]))

# Agent can now use skills
result = agent.query(
    message="Create a presentation about Q4 results with our revenue of $5.2M"
)
```

### Custom Configuration

```python
skills = ClaudeCodeSkills(
    skills_dir="~/my-company-skills",
    output_dir="/tmp/skill_output",
    timeout=600,  # 10 minutes
    disable_session_persistence=True,  # Fresh session each time
    resource_id="company-skills",
)
```

---

## 11. Module Statistics

| Metric | Value |
|--------|-------|
| Total Files | 2 |
| Total Lines | ~472 |
| Public Classes | 1 (`ClaudeCodeSkills`) |
| Public Methods | 1 (`execute`) |
| Private Methods | 16 |
| Properties | 4 |
| External Dependencies | 0 (stdlib only) |
| Platform-Specific Code | 2 sections (macOS, Unix) |

---

## 12. Skill Directory Format

### Expected Structure
```
~/.claude/skills/
├── skill-name/
│   ├── SKILL.md        # Required: skill definition
│   ├── README.md       # Optional: detailed docs
│   └── examples/       # Optional: example files
```

### SKILL.md Format
```markdown
# Skill Name

Brief description of what the skill does.
This is extracted as the skill description.

## Usage
...

## Examples
...
```

The first non-heading, non-empty line (or first heading content) becomes the skill description, truncated to 200 characters.

---

*Generated: 2026-01-19*
*Module: dana_agent/dana/core/skills*
*Version: Based on develop branch*
