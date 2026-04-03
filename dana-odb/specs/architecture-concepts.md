# Dana-ODB: Architecture Concepts & Open Questions

## Positioning

Dana-ODB is the **semantic layer above all other stores**:

```
┌─────────────────────────────────────┐
│           Dana Agents               │
├─────────────────────────────────────┤
│           Dana-Runtime              │  ← agent execution harness
├─────────────────────────────────────┤
│           Dana-ODB                  │  ← ontology, reasoning, federation
├──────┬──────┬──────┬────────┬───────┤
│RDBMS │Vector│ RAG  │  IoT   │  FS   │  ← raw data stores (connectors)
└──────┴──────┴──────┴────────┴───────┘
```

Dana-ODB owns **meaning**, not raw data. Connectors are plugins.

---

## Core Architectural Properties

### Ontology-Native Reasoning
- Reasoning over the ontology is first-class, not a query add-on
- Forward chaining, pattern matching, SPARQL-like traversal
- Inference produces new ontological facts, not just query results

### Deployment Spectrum
- **Embedded** (Phase 0): single library, no server, ships inside agent packages
- **Clustered** (Phase N): horizontally scaled via graph topology
  - Dense local clusters (concept neighborhoods) = fast local queries
  - Sparse inter-cluster bridges = cross-domain reasoning paths
  - Sharding aligns with cluster boundaries naturally

### Plugin/Connector Architecture
- Each backend (RDBMS, vector, RAG, IoT, FS) is a swappable module
- Stable connector interface; internals can evolve independently
- Enables progressive addition of new source types

### Stable Core API
- Core query/reasoning API is stable and versioned
- Extensions added without breaking existing contracts
- Harness-agnostic: dana-runtime is the reference integration, not the only one
- "Some assembly required" for new integrations — not zero-config, but lightweight

### Implementation Language
- **Rust** — performance, safety, embeddability
- Enables both lightweight embedded mode and high-throughput distributed mode

---

## Key Design Challenge: Circular Dependency

Dana-ODB and dana-runtime have **bidirectional dependence**:

```
dana-runtime ──calls──→ dana-odb   (agents query/reason over ontology)
dana-odb ──uses──→ dana-runtime    (ODB runs agents internally for autonomous tasks)
```

This is intentional but requires careful management.

### Bootstrapping Sequence

| Phase | State |
|-------|-------|
| **Phase 0** | dana-odb is functional with no internal agents. All logic hand-coded in Rust. |
| **Phase 1** | dana-runtime built on top of dana-odb. Dana-Ontologist is first internal agent. |
| **Phase 2** | Internal ODB logic progressively replaced by agents via dana-runtime. |
| **Phase N** | Self-hosting: ODB reasons about and evolves itself via its own agent layer. |

### Design Constraints
- Dana-ODB **must be fully functional without agents** (Phase 0 baseline never breaks)
- The internal agent execution layer is **optional and pluggable**, not load-bearing from day one
- Clean, stable APIs on both sides manage the circular dependency via interfaces
- Bootstrapping sequence is explicit in the roadmap — not accidental

---

## The Dana-Ontologist

A permanent companion agent to the dana-odb layer:
- First internal agent target (Phase 1 bootstrap)
- Curates, audits, and evolves ontologies continuously
- Natural candidate for "eating our own dog food" — uses dana-odb to reason about dana-odb's own ontology

---

## Open Questions (to resolve in structured design phase)

1. **Query language** — SPARQL subset? Custom DSL? Both?
2. **Ontology format** — OWL? Custom? What's the wire format?
3. **Reasoning engine** — which inference strategies at Phase 0 (hand-coded Rust)?
4. **Sync/federation protocol** — how do embedded instances sync with shared ontologies?
5. **Connector interface contract** — what must a connector implement at minimum?
6. **dana-runtime integration contract** — what's the minimal "some assembly required" surface?
7. **Agent execution boundary** — where exactly does dana-odb hand off to dana-runtime?
8. **Versioning/migration** — how do ontologies evolve without breaking dependent agents?
9. **Security/access control** — who can modify the ontology? Per-agent namespaces?
10. **Observability** — how do agents (and humans) inspect reasoning traces?
