# Dana-ODB: Vision & Goals

## Name

**Dana-ODB** — Ontological Database. The "O" signals ontology-first, distinguishing it from generic knowledge bases, graph DBs, or vector stores.

## Vision

Dana-ODB is the **semantic operating system for data and agent cognition**. It sits above all other data stores as the authoritative layer of meaning — storing not data, but the ontological relationships *among* data, sources, concepts, and agents.

It is to agent cognition what a relational schema is to a transactional system: the organizing principle that makes everything else coherent.

## Core Identity

- **Ontology-native** — the schema *is* the ontology; reasoning is first-class, not an add-on
- **Semantic federation layer** — sits above RDBMS, vector DBs, RAG stores, IoT, filesystems; owns the meaning, not the raw data
- **Agent-serving and agent-using** — serves AI agents as clients; also uses AI agents internally for autonomous maintenance
- **Self-bootstrapping** — primitives defined in terms of the system itself, progressively replaceable (analogous to a C compiler written in C)

## Goals

### Immediate (v0)
- Functional embedded deployment (SQLite-style: no server, links into host app)
- Stable core API with defined extension points
- Native ontological reasoning over stored knowledge
- Clean integration contract with dana-runtime (reference harness)

### Near-term
- Plugin-style connector architecture for external data sources
- Dana-Ontologist companion agent (first internal agent, Phase 1 bootstrap)
- Horizontal scalability via graph cluster topology

### Long-term
- Full self-hosting: internal logic progressively replaced by agents
- Large-scale distributed "mind" deployment
- Ecosystem of domain-specific ontologies and cognitive agents

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Ontology-first** | Meaning is structured, not inferred from flat text |
| **Embedded-first** | Must ship inside an agent package with zero infra overhead |
| **Stable API, evolving internals** | Core query/reasoning API stays stable; underlying tech can swap |
| **Harness-agnostic** | Not tied to dana-runtime; integration requires minor assembly |
| **Modular/plugin** | Each connector, reasoner, and backend is a swappable module |
| **YAGNI on primitives** | Start with the minimum that works; replace with agents over time |

## Relationship to Dana-Runtime

Dana-ODB and dana-runtime are **co-evolving siblings**:

- dana-runtime is the reference integration and the agent execution engine
- dana-odb is the semantic substrate that agents reason over
- They are designed together but kept architecturally clean and separable
- See [architecture-concepts.md](architecture-concepts.md) for the circular dependency and bootstrapping design challenge
