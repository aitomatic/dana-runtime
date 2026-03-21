# Dana Runtime - Project Overview & PDR

**Version:** 0.1.1 | **Status:** Active Development | **Last Updated:** 2026-03-21

## Executive Summary

**Dana** (Domain-Aware Neurosymbolic Agents) is a production-ready Python agentic runtime by Aitomatic, Inc. It implements the **STAR pattern (See-Think-Act-Reflect)** for building sophisticated conversational AI agents with multi-provider LLM support, extensible tool frameworks, and intelligent context management.

## Vision & Goals

### Vision
Enable developers to build domain-aware AI agents that combine symbolic reasoning with neural capabilities, providing transparent, controllable, and explainable agent behavior.

### Primary Goals
1. **Multi-Provider LLM Support** - Seamless integration with OpenAI, Anthropic, Google, Azure, and local models
2. **Extensible Tool Framework** - Easy addition of custom tools and resources
3. **Intelligent Context Management** - Token-aware timeline compression and memory
4. **Production Ready** - Robust error handling, logging, and observability
5. **Developer Experience** - Rich CLI, REPL, comprehensive examples

## Core Features

### 1. STAR Agent Pattern
Conversational agents follow structured reasoning:
- **See** - Perceive user intent and context
- **Think** - Reason about response using resources/tools
- **Act** - Execute tools, retrieve information
- **Reflect** - Learn from outcomes, update state

### 2. Multi-Provider LLM Integration
- **OpenAI** - gpt-4.1, gpt-4.1-mini, o3, o4-mini
- **Anthropic** - claude-sonnet, claude-opus, claude-haiku
- **Google Gemini** - gemini-2.5-flash, gemini-2.5-pro
- **Azure OpenAI** - Full compatibility
- **Local Models** - LLaMA Stack, Ollama support
- **Custom Endpoints** - Anthropic-like protocol support

### 3. Tool Execution Framework
Built-in resources for common tasks:
- **BashResource** - Execute shell commands
- **FileIOResource** - Read/write files
- **FileEditResource** - Edit file contents with diff
- **SearchResource** - Web search integration
- **TaskResource** - Task management
- **TodoResource** - Todo list operations
- **SkillResource** - Claude Code skills
- **WebResearchResource** - Advanced web research with synthesis
- **MCPResource** - Model Context Protocol support

### 4. Conversation Timeline Management
- Chronological entry tracking (9 entry types)
- Token-aware automatic compression
- LLM-based conversation summarization
- Serializable history

### 5. Memory Systems
- **Short-Term Memory (STMemory)** - Per-session cache
- **Long-Term Memory (LTMemory)** - Persistent markdown storage
- Memory Types: lessons, episodes, facts, patterns

### 6. Web Research Capabilities
- HTML content extraction
- Multi-source synthesis
- URL caching
- Search result aggregation

### 7. CLI Applications
| App | Command | Purpose |
|-----|---------|---------|
| Dana Agent | `adana` | Interactive conversational agent |
| REPL | `adana-repl` | Interactive Python environment |
| Code Agent | `dana-code` | Coding-focused agent with UI |
| Init | `dana-init` | Bootstrap config setup |

## Product Requirements

### Functional Requirements

#### FR-1: Multi-Provider LLM Support
- Support ≥5 LLM providers with provider-agnostic abstraction
- Implement fallback mechanism when primary provider unavailable
- Handle provider-specific tool schemas (native vs XML)
- Status: ✅ Complete

#### FR-2: Extensible Resource Framework
- Enable custom resource creation via BaseResource extension
- Auto-registration of resources on instantiation
- Tool schema generation from resource methods
- Status: ✅ Complete

#### FR-3: Conversation Timeline Management
- Maintain chronological conversation history with 9 entry types
- Implement token-aware compression at 80% context threshold
- Support LLM-based history summarization
- Status: ✅ Complete

#### FR-4: STAR Agent Pattern
- Implement full See-Think-Act-Reflect loop
- Support streaming responses
- Enable tool call handling and result integration
- Status: ✅ Complete

#### FR-5: Workflow Composition
- Enable multi-step workflow definition and execution
- Support conditional execution and branching
- Implement workflow validation
- Status: ✅ Complete

#### FR-6: Web Research Integration
- HTML content extraction and parsing
- Multi-source synthesis
- Search result integration
- Status: ✅ Complete

#### FR-7: Memory Systems
- Implement short-term memory (per-session)
- Implement long-term memory (persistent)
- Support 4 memory types (lesson, episode, fact, pattern)
- Status: ⚠️ In Progress

#### FR-8: Observability & Logging
- Structured logging via structlog
- Debug mode support
- Performance metrics
- Status: ✅ Complete

### Non-Functional Requirements

#### NFR-1: Performance
- LLM response latency: <5s average
- Token compression at 80% threshold
- Streaming response support
- Status: ✅ Complete

#### NFR-2: Reliability
- Error handling for all resource operations
- Graceful degradation on provider failure
- Retry logic for transient failures
- Status: ✅ Complete

#### NFR-3: Maintainability
- Code coverage: >70%
- Type hints throughout codebase
- Comprehensive test suite (105 test files)
- Status: ✅ Complete

#### NFR-4: Extensibility
- Plugin architecture for resources/workflows
- Auto-registration system
- Protocol-based design
- Status: ✅ Complete

#### NFR-5: Security
- Safe code execution (Python sandbox)
- Input validation & sanitization
- Environment variable protection
- Status: ✅ Complete

#### NFR-6: Compatibility
- Python 3.12+ support
- Cross-platform (macOS, Linux, Windows)
- Async/await support throughout
- Status: ✅ Complete

## Architecture

### Layered Design

```
┌──────────────────────────────────┐
│  Applications (CLI Apps)          │ adana, adana-repl, dana-code
├──────────────────────────────────┤
│  Agent Layer                      │ STARAgent, Components
├──────────────────────────────────┤
│  Core Systems                     │ Timeline, Resources, Workflows
├──────────────────────────────────┤
│  LLM Abstraction                  │ Codec, Providers, Runtime
├──────────────────────────────────┤
│  Data Persistence                 │ Repository Layer, Memory
├──────────────────────────────────┤
│  Infrastructure                   │ Config, Logging, Utils
└──────────────────────────────────┘
```

### Key Design Patterns
1. **Composition** - STARAgent composes: Communicator, State, Learner, Observer
2. **Provider Pattern** - LLM abstraction via codec/runtime
3. **Resource Pattern** - Tool framework with auto-registration
4. **Protocol-Based** - Extensible via protocol definitions
5. **Global Registry** - Centralized resource/agent/workflow management

## Acceptance Criteria

### Core Functionality
- [x] STAR agent loop operational
- [x] Multi-provider LLM integration working
- [x] Tool execution framework functional
- [x] Timeline compression operational
- [x] Workflow composition implemented
- [x] Web research pipeline complete

### Quality Standards
- [x] >70% code coverage
- [x] Type hints throughout
- [x] All tests passing
- [x] Linting compliant (Ruff)
- [x] MyPy type checking passes

### Developer Experience
- [x] CLI applications functional
- [x] Configuration system working
- [x] Error messages clear
- [x] Documentation complete
- [x] Examples provided

### Production Readiness
- [x] Error handling robust
- [x] Logging structured
- [x] Observability integrated
- [x] Performance optimized
- [x] Security hardened

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| LLM Response Latency | <5s avg | ✅ Achieved |
| Code Coverage | >70% | ✅ >80% |
| Test Pass Rate | 100% | ✅ 100% |
| Supported Providers | 5+ | ✅ 6+ |
| Built-in Resources | 8+ | ✅ 9 |
| CLI Applications | 4+ | ✅ 5 |

## Roadmap

### Phase 1: Foundation (Complete ✅)
- Core agent infrastructure
- Multi-provider LLM support
- Basic resource framework
- Timeline management

### Phase 2: Enhancement (In Progress ⚠️)
- Memory system expansion
- Advanced web research
- Workflow optimization
- Performance tuning

### Phase 3: Extended Features (Planned 🔜)
- Additional LLM providers
- Domain-specific agents
- Advanced observability
- Distributed execution

### Phase 4: Production Hardening (Planned 🔜)
- Enterprise features
- Scale testing
- Security audit
- Performance optimization

## Technical Constraints

### Required
- Python 3.12+
- Async/await capable environment
- Network access for LLM APIs

### Recommended
- 2GB+ RAM for large context
- Bash shell for full functionality
- Modern terminal (256 color support)

### Optional
- LLaMA Stack or Ollama (for local models)
- Database adapters (for persistence)
- Redis (for distributed caching)

## Dependencies

**Core (Always):**
- openai, anthropic, google-genai, httpx, pydantic, requests, beautifulsoup4, html2text, rich, structlog

**Optional Groups:**
- web: lxml, selenium
- local: llama-stack, ollama
- data: pandas, matplotlib
- memory: lancedb, sentence-transformers
- knowledge: rdflib, rank-bm25
- observability: langfuse

## Versioning & Release Strategy

**Current:** 0.1.1
**Channel:** Stable (ready for production use)
**License:** MIT

**Release Process:**
1. Increment version in `pyproject.toml`
2. Update `CHANGELOG.md` with features/fixes
3. Create git tag
4. Build and publish to PyPI
5. Publish release notes

## Known Limitations

1. **Memory** - LT memory relies on markdown; consider DB adapter for scale
2. **Distributed** - Currently single-process; consider task queue for scale
3. **Streaming** - Some providers have limited streaming support
4. **Local Models** - Performance variable based on hardware

## Future Enhancements

1. **Distributed Execution** - Task queues for parallel agent execution
2. **Database Persistence** - SQL/NoSQL adapters for memory & timeline
3. **Advanced Observability** - Metrics, tracing, profiling
4. **Domain Specialization** - Healthcare, Finance, Legal agent templates
5. **Voice Integration** - Speech-to-text and text-to-speech
6. **Multi-agent Collaboration** - Agent-to-agent communication

---

**Approval Status:** Ready for Production
**Last Review:** 2026-03-21
**Next Review:** 2026-04-21
