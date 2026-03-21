# Dana Runtime Documentation

Complete documentation for the Dana STAR Pattern Agent Framework.

## Quick Navigation

### 🚀 Getting Started
- **[README.md](../README.md)** - Project overview, quick start, installation
  - 5-minute quick start
  - Feature highlights
  - Configuration setup
  - Usage examples

### 📚 Documentation Index

#### Project Level
1. **[project-overview-pdr.md](project-overview-pdr.md)** - Vision, goals, requirements
   - Executive summary
   - Core features (8)
   - Product requirements (8 FR + 6 NFR)
   - Success metrics
   - Known limitations
   - Status: Production ready

2. **[project-roadmap.md](project-roadmap.md)** - Development timeline & milestones
   - 4-phase development roadmap (Q1-Q4 2026)
   - Phase 1: Foundation (✅ Complete)
   - Phase 2: Enhancement (⚠️ In Progress)
   - Phase 3-4: Planned features
   - Issue tracking & blockers
   - Community goals

#### For Developers
3. **[codebase-summary.md](codebase-summary.md)** - Codebase structure & organization
   - 210 Python files, 38,662 LOC
   - 8 major modules with breakdown
   - Key design patterns
   - Testing infrastructure (105 tests)
   - Performance characteristics
   - Extensibility points

4. **[code-standards.md](code-standards.md)** - Coding conventions & quality standards
   - Python 3.12+ style guide (Ruff, MyPy)
   - Type hints requirements
   - Docstring format (Google-style)
   - Architecture patterns (5)
   - Testing guidelines (>70% coverage)
   - Security & performance
   - Code review checklist

#### Architecture & Design
5. **[system-architecture.md](system-architecture.md)** - System design & data flows
   - Layered architecture (5 layers)
   - STAR agent pattern execution flow
   - Component interactions
   - Message processing pipeline
   - Resource execution system
   - LLM provider abstraction
   - Memory system architecture
   - Extension points

## Document Overview

| Document | Purpose | Audience | Length |
|----------|---------|----------|--------|
| README.md | Project entry point | Everyone | 478 LOC |
| project-overview-pdr.md | Vision, goals, requirements | Product, Architects | 323 LOC |
| project-roadmap.md | Development timeline | Team, Managers | 410 LOC |
| codebase-summary.md | Code structure & organization | Developers | 313 LOC |
| code-standards.md | Quality standards | Developers | 630 LOC |
| system-architecture.md | Design & data flows | Architects, SR Developers | 534 LOC |

## Common Tasks

### I'm new to Dana
1. Start with [README.md](../README.md) - Get overview
2. Run quick start - `dana-init && adana`
3. Read [codebase-summary.md](codebase-summary.md) - Understand structure
4. Check [code-standards.md](code-standards.md) - Learn conventions

### I'm implementing a feature
1. Review [system-architecture.md](system-architecture.md) - Understand design
2. Follow [code-standards.md](code-standards.md) - Code quality
3. Check [project-roadmap.md](project-roadmap.md) - Milestones & priorities
4. Reference [codebase-summary.md](codebase-summary.md) - Module locations

### I'm reviewing code
1. Use [code-standards.md](code-standards.md) - Code review checklist
2. Verify [system-architecture.md](system-architecture.md) - Design compliance
3. Check test requirements - >70% coverage mandatory
4. Ensure type hints throughout

### I'm planning roadmap
1. Read [project-overview-pdr.md](project-overview-pdr.md) - Requirements
2. Review [project-roadmap.md](project-roadmap.md) - Phases & milestones
3. Check blockers & dependencies
4. Update progress monthly

## Key Statistics

**Codebase:**
- 210 Python files
- 38,662 lines of code
- 105 test files across 8 categories
- 8 major modules
- 9 built-in resources
- 6 LLM providers

**Documentation:**
- 6 primary documents
- 2,688 total lines
- All files <800 LOC
- Cross-referenced
- Version: 0.1.1

**Quality:**
- Type hints: 100%
- Code coverage: >70%
- Linting: Ruff
- Type checking: MyPy
- Pre-commit hooks: Enabled

## Documentation Standards

All documentation follows these standards:
- **Markdown** format with clear hierarchy
- **Tables & lists** for structured data (not prose)
- **Code examples** where relevant
- **Cross-references** between documents
- **Version numbers** on all files
- **Last updated** dates
- **Target audience** identified

## Updates & Maintenance

### Review Schedule
- **Monthly:** First Friday (9 AM)
- **Quarterly:** End of quarter review
- **Annual:** 2026-12-31 comprehensive review

### Update Triggers
- Major feature implementation
- Architecture changes
- Version releases
- Roadmap milestone completion
- Security updates

### Reporting
All changes documented in:
- `plans/reports/` - Documentation reports
- Git history - Version tracking
- [project-roadmap.md](project-roadmap.md) - Progress tracking

## Contributing to Documentation

### Process
1. Read [code-standards.md](code-standards.md) - Standards apply to docs too
2. Keep files under 800 LOC
3. Use clear headings & structure
4. Include code examples where helpful
5. Cross-reference related content
6. Update "Last Updated" date
7. Submit for review

### File Naming
- Use kebab-case: `my-feature-guide.md`
- Be descriptive: names should indicate content
- Location: `docs/` directory or subdirectory

## External Links

**Project Resources:**
- [GitHub Repository](https://github.com/aitomatic/dana-runtime)
- [PyPI Package](https://pypi.org/project/dana-runtime/)
- [Aitomatic Website](https://aitomatic.com)

**Dependencies & Tools:**
- [Python 3.12+](https://www.python.org/)
- [Ruff Linter](https://github.com/astral-sh/ruff)
- [MyPy Type Checker](http://mypy-lang.org/)
- [Pytest Testing Framework](https://pytest.org/)

## Support

- **Issues:** [GitHub Issues](https://github.com/aitomatic/dana-runtime/issues)
- **Discussions:** [GitHub Discussions](https://github.com/aitomatic/dana-runtime/discussions)
- **Email:** support@aitomatic.com

---

**Version:** 0.1.1 | **Last Updated:** 2026-03-21 | **Status:** Complete ✅
