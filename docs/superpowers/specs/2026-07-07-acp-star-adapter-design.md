# ACP AgentSession and Session Journal - Design Spec

**Original date:** 2026-07-07
**Revised:** 2026-07-16
**Status:** Proposed revision for review
**Compatibility target:** `~/Desktop/repos/dana-os-docs-update/dana-console`
**Reference implementation:** `~/Desktop/repos/hermes-agent/acp_adapter` (read-only evidence, not a template)
**Domain language:** [`CONTEXT.md`](../../../CONTEXT.md)

## 1. Goal

Expose Dana as an ACP agent while using the integration to deepen the STAR
runtime rather than building policy and lifecycle behavior inside a protocol
adapter. The target architecture must support the full long-term capability
set, but delivery planning will stage it as independently releasable Console
workflows.

The first target is dana-console. Dana is selected through its existing Custom
ACP Agent provider configuration. Console changes remain limited to
compatibility wiring and small controls inside the existing Copilot surface.

## 2. Evidence

### dana-console contract

The current Console exercises:

- `initialize`, `session/new`, and text `session/prompt`
- `session/cancel`
- optional session modes and `session/set_mode`
- `session/update` for message, thought, tool, and mode updates
- `session/request_permission`
- strict draining of update handlers before the prompt response

It does not currently expose model selection, MCP configuration, attachments,
session history, or fork UI. Small model and attachment controls are acceptable.
New history and MCP-management workflows are planning decisions, not reasons to
remove the underlying capabilities from this design.

### Hermes findings

Hermes provides useful evidence for per-session ownership, transactional
persistence, ordered restoration, in-place compression, fail-closed approvals,
and subprocess process-group cleanup. It is not an execution journal:

- SQLite stores mutable session rows and message-shaped conversation rows.
- Normal turns append, but retry, rewind, and compression mutate active state.
- Compression summaries persist as ordinary message content, identified after
  restart through a text prefix rather than a typed checkpoint.
- ACP tool updates, permissions, cancellation, and streaming progress are live
  only and reconstructed approximately on resume.
- Most Python tools run in shared-process threads. Non-cooperative threads may
  continue after cancellation with unknown effects.

Dana borrows the proven operational mechanics while adopting typed journal
facts, explicit projection semantics, stable tool identity, and a
cancellation-first execution engine.

## 3. Design Principles

1. Slice delivery by complete user workflow, never by architectural layer.
2. ACP translates protocol; it does not own Dana behavior or policy.
3. One `AgentSession` owns one isolated STARAgent and one active turn.
4. The Session Journal is the sole durable authority for session history.
5. Conversation, host events, and traces are projections from the same facts.
6. Model presentation and invocation resolve through the same Tool Catalog.
7. Permission modes affect prompting, never hard policy.
8. Cancellation is truthful: requested, acknowledged, timed out, and
   effect-unknown are distinct outcomes.
9. Large payloads are referenced artifacts, not duplicated journal content.
10. SQLite and PostgreSQL implement the same journal contract.
11. Existing STAR behavior and stored sessions migrate incrementally.
12. YAGNI, KISS, and DRY apply inside each delivery, even though the target
    architecture remains complete.

## 4. Locked Decisions

| Decision | Resolution |
| --- | --- |
| Host interface | One deep, host-neutral `AgentSession` interface |
| Agent isolation | One STARAgent instance per session |
| Turn concurrency | One active writer per session; other sessions remain concurrent |
| Durable authority | Append-only Session Journal |
| Persistence adapters | SQLite and PostgreSQL, governed by one contract suite |
| Journal granularity | Semantic facts plus bounded content chunks and explicit finals |
| Retention | Facts retained until explicit session deletion |
| Large content | Access-controlled artifact references |
| Compression | Immutable, range-addressed Compression Checkpoints |
| Fork | Parent reference at a committed turn; no history copy |
| Crash recovery | Unterminated turn becomes Interrupted Turn; tool effects may be unknown |
| Reasoning | Thought Summaries are host-visible; Provider Replay State is protected |
| Tool discovery | Session-owned, versioned Tool Catalog |
| Tool identity | Stable provider-neutral identity; adapters generate aliases |
| Execution | Cancellation-first Tool Execution Engine |
| Threads | Cooperative only, with declared and tested cancellation latency |
| Hard cancellation | Killable worker/process group or acknowledged remote cancellation |
| Background work | Explicit Durable Job handoff; cascade before handoff |
| Permission model | Effect-based Operations, hard policy, modes, Policy Grants |
| Durable grants | Full allow/reject-always support with revocation |
| Autonomous workflow | Policy Preflight plus mandatory invocation enforcement |
| MCP ownership | Session MCP Leases; pooling is an internal optimization |
| Model selection | Configured provider/model combinations only |
| Multimodal | Text, image, embedded resource, and file resource |
| Migration | Shadow parity, authority cutover, bounded compatibility window |
| Console scope | Minimal compatibility changes and small existing-surface controls |

## 5. Architecture

```text
ACP adapter          CLI adapter          Future host adapter
     |                    |                       |
     +--------------- AgentSession ---------------+
                           |
                  one active turn/session
                           |
        +------------------+------------------+
        |                  |                  |
 Session Journal      Tool Catalog      Execution Policy
 sole authority       versioned         grants + modes
        |                  |                  |
        |           Tool Execution Engine ---+
        |           |- cooperative thread
        |           |- isolated worker
        |           |- remote cancellation
        |           `- durable job handoff
        |                  |
        |              MCP Leases
        |
        |- Conversation View -> STARAgent/model
        |- Host Event View   -> ACP/CLI
        `- Trace View        -> exporters
```

`AgentSession` is the only broad host-facing module. Journal adapters,
projection adapters, execution adapters, policy storage, and MCP transports are
internal seams justified by multiple real adapters. ACP types never enter STAR
core.

## 6. Core Modules

### 6.1 AgentSession

An AgentSession owns:

- one isolated STARAgent
- immutable owner and workspace scope
- a session journal identity and version
- the active turn and cancellation tree
- one Tool Catalog version per turn
- permission mode and policy context
- configured model and protected provider replay state
- MCP leases and durable job handoffs

It serializes mutations. A second prompt, model change, mode change, catalog
change, MCP mutation, or fork that conflicts with an active turn returns
`busy`. A terminal turn fact is appended only after required session mutation
is durable and owned work has completed or transferred ownership.

Reflection, memory updates, projections, and trace export may not mutate session
state after terminalization. Required work completes inside the turn; optional
work transfers to a Durable Job first.

### 6.2 Session Journal

The Session Journal is the sole durable authority. Journal facts include:

- session created, loaded, resumed, closed, archived, deleted, and forked
- turn started, content chunks, content final, terminal completion, error,
  cancellation, and interruption
- tool requested, authorized or denied, started, progress, result, failure,
  cancellation requested, cancellation acknowledged, and effect unknown
- permission request, user decision, grant reference, timeout, and disconnect
- model, mode, Tool Catalog, MCP lease, and session metadata changes
- Compression Checkpoints and projection progress
- Durable Job ownership transfer and terminal outcome

Each fact has immutable identity, owner scope, session identity, per-session
sequence, type, timestamp, correlation and causation identifiers, schema
version, sanitized payload, and optional artifact references.

#### Persistence contract

SQLite and PostgreSQL adapters implement equivalent domain semantics:

- append an ordered batch using the expected session version
- atomically advance session version and metadata
- read facts after a sequence
- read lineage and committed fork points
- list, archive, and purge sessions within Owner Scope
- manage projection checkpoints without changing journal facts
- reject conflicts rather than interleave two writers

PostgreSQL may use row or advisory locking, JSONB, indexes, partitioning, and
row-level security. SQLite may use WAL and `BEGIN IMMEDIATE`. Backend-specific
features do not leak into the journal interface. Real-database contract tests
are required for both.

#### Views

The Conversation View projects model-facing messages, the newest compatible
checkpoint, retained recent facts, tool results, and protected replay state. It
does not treat partial output from an Interrupted Turn as a completed response.

The Host Event View projects ordered text, thought, tool, permission, model,
mode, MCP, error, cancellation, and terminal updates. Partial interrupted text
remains visible.

The Trace View projects sanitized operational spans and metrics. Vendor
exporters consume this view rather than creating another source of truth.

#### Content flushing

Lifecycle facts are durable immediately. Text and Thought Summary streams use
bounded chunks or short flush intervals, followed by an explicit final fact.
The journal does not perform one database transaction per token.

#### Crash recovery

A turn with a start fact and no terminal fact is an Interrupted Turn. Recovery:

1. preserves partial output in Host Event View
2. excludes partial assistant output as a completed Conversation View message
3. marks started tools without terminal facts as effect unknown
4. appends a typed interruption recovery fact
5. gives the next model turn a concise interruption observation
6. never retries an unknown-effect operation automatically

#### Compression

A Compression Checkpoint:

- ends at a Committed Turn sequence
- covers an exact fact range
- stores summary content, retained-head policy, projection schema version, and
  summarizer provenance
- is immutable and may only be superseded
- affects Conversation View only
- never deletes or rewrites the facts it summarizes

#### Fork

A Session Fork references a parent session and committed parent sequence. The
child journal begins with lineage facts and reads inherited Conversation View
history through that reference. Forking during an active or interrupted turn is
not allowed. Parent history is not copied.

#### Protected state and artifacts

Host-visible Thought Summaries are sanitized journal facts. Hidden chain of
thought and secrets are not ordinary facts. Provider Replay State required for
continuity is encrypted and protected from host and trace projections by
default.

Images, files, oversized tool results, and restricted payloads live in an
authorized artifact store. Journal facts hold immutable hash, URI, media type,
size, and access metadata. Artifact retention is independent from journal-fact
retention; missing artifacts fail explicitly.

### 6.3 Tool Catalog

The session-owned Tool Catalog is the only source for model-visible schemas and
invocation targets. It absorbs existing reflection, named-tool registration,
resource scanning, workflow scanning, agent discovery, and MCP discovery.

Each entry declares:

- stable Tool Identity and source identity
- display name and provider alias rules
- input and output schemas
- normalized effect metadata
- cancellation capability and maximum cooperative latency
- invocation adapter and lifecycle requirements
- catalog version

Duplicate stable identities or provider aliases fail catalog construction.
Each turn pins one immutable version. Changes occur only between turns and
append a catalog-change fact.

### 6.4 Tool Execution Engine

Every tool call goes through one cancellation-first engine. It owns:

- schema validation and normalized Operation creation
- policy enforcement before start
- invocation identity, correlation, and journal lifecycle
- deadline and cancellation token
- result normalization, redaction, and artifact extraction
- cleanup callbacks, child ownership, and exactly one terminal result

Execution adapters are:

1. **Cooperative async:** cancellation propagates through task cancellation and
   explicit tokens.
2. **Cooperative thread:** decorated tools check an injected context, declare a
   maximum cancellation latency, and pass real cancellation contract tests.
3. **Isolated worker:** non-cooperative, blocking, untrusted, or side-effecting
   work runs behind a killable process or container boundary.
4. **Remote:** cancellation is terminal only after the remote system
   acknowledges it; otherwise effect disposition remains unknown.

The engine never reports `cancelled` merely because a future was abandoned.
Subprocesses use dedicated groups/jobs, graceful termination, bounded wait,
force kill, output drain, and reap. Cancellation cannot undo external effects
already committed before acknowledgement.

Child ownership defaults to `cascade`. `detach` requires successful Durable Job
handoff. `keep` is reserved for explicitly managed infrastructure and is never
the default for agent tools.

### 6.5 Execution Policy and Policy Grants

Policy evaluates normalized Operations rather than hard-coded tool names. An
Operation includes Tool Identity, effects, validated arguments, affected
locations, owner, workspace, and session context. Unknown effect metadata is
sensitive.

Decision precedence is:

```text
hard deny
-> durable reject grant
-> durable allow grant
-> permission mode
-> interactive prompt
-> fail-closed fallback
```

Mode semantics are fixed:

- `default`: safe reads proceed; sensitive operations prompt.
- `acceptEdits`: workspace writes proceed; execution, network, external
  mutation, and credential use still prompt.
- `bypassPermissions`: soft-policy operations proceed without prompting.
- hard policy applies in every mode.

Allow-always and reject-always decisions create revocable Policy Grants in the
policy store. The journal records the decision and grant reference; it is not
the mutable grant store. UI-created grants default to Owner Scope, workspace,
Tool Identity, effect, and location. Broader grants require explicit operator
provisioning. Timeout, disconnect, missing host capability, and cancellation
deny.

Autonomous workflows declare predictable Operations for Policy Preflight.
Preflight reports missing grants before work begins but never grants access or
replaces invocation-time enforcement for dynamic Operations.

### 6.6 MCP Leases

MCP integration uses the official protocol implementation rather than the
current ad hoc JSON-RPC clients. A session MCP Lease binds a validated server
descriptor and credential scope to the session.

- Dana-configured leases restore through durable references.
- Client-provided leases are session-scoped and never persist raw credentials.
- Discovered tools enter the session Tool Catalog.
- HTTP connections may be pooled internally when descriptors and credentials
  match.
- Stdio servers default to dedicated managed processes.
- Optional lease failure degrades that lease and updates the host.
- Required lease failure stops preflight or workflow start, not session load.
- Every MCP call uses normal policy, execution, cancellation, and journal paths.

No process-global MCP registry is allowed.

### 6.7 Model Catalog and Switching

The catalog exposes only configured provider/model combinations. A switch:

1. validates the configured target
2. builds the provider, model client, and compatible runtime before mutation
3. rebinds runtime, memory, prompt, tool-schema, and model-sensitive caches
4. preserves journal, Conversation View, Tool Catalog identity, policy, and MCP
   leases
5. includes protected replay state only when compatible
6. commits one model-change fact
7. leaves the old model untouched on any failure

Switching during an active turn returns `busy`.

### 6.8 Prompt Content and Artifacts

AgentSession accepts normalized text, image, embedded-resource, and
file-resource blocks. The core validates MIME type, size, workspace access,
model capability, and artifact authorization before a turn starts. ACP only
translates protocol blocks into this representation.

Audio and video remain part of the complete target backlog but require explicit
provider and Console workflows before advertisement.

## 7. ACP Mapping

| ACP method/surface | Dana mapping |
| --- | --- |
| `initialize` | Protocol version, capabilities, configured identity, auth status |
| `session/new` | Create AgentSession, initial model/mode, configured MCP leases |
| `session/prompt` | Normalize content and run one turn |
| `session/cancel` | Request turn cancellation and await truthful terminalization |
| `session/load` | Load journal and replay Host Event View before returning |
| `session/resume` | Restore active state and leases, then replay updates |
| `session/list` | Owner-scoped journal session listing |
| `session/fork` | Fork at a committed turn sequence |
| `session/set_mode` | Change Permission Mode outside an active turn |
| `session/set_model` | Atomic configured-model switch outside an active turn |
| `session/update` | Translate ordered Host Event View facts |
| `session/request_permission` | Host decision adapter for execution policy |

Stdout is reserved for JSON-RPC frames. Logs and full diagnostics go to stderr.
Prompt responses return only after every preceding update has been handled.

## 8. Console Compatibility

The Console baseline requires:

- Custom ACP command resolution and explicit environment allowlisting
- `initialize` then `session/new(cwd=...)`
- first-turn preamble compatibility
- text, thought, tool start, and tool update rendering
- permission options with stable IDs
- modes `default`, `acceptEdits`, and `bypassPermissions`
- cancellation that resolves pending permission requests first
- all session updates drained before `turn_end`

Minimal Console additions may retain the current session ID for process-restart
resume, show a model selector, and attach images/files. Conversation-list,
fork, and MCP-management surfaces remain planning decisions, while core and ACP
capabilities stay in the target design.

## 9. Delivery Decomposition

```text
D1 Durable conversation
|- D2 Cancellable tools
|  `- D3 Autonomous permission policy
|     `- D5 Configured MCP tools
|- D4 Configured model switching
`- D6 Images and file resources

Recommended release order: D1 -> D2 -> D3 -> D4 -> D5 -> D6
```

### D1. Durable Dana Conversation

**User outcome:** Dana streams a multi-turn text conversation and automatically
continues the same session after `dana-acp` restarts. Before this delivery,
Console cannot run Dana and reconnect creates a blank conversation.

**Demo:** Select Dana as Custom ACP Agent, provide a project fact, exchange
another turn, restart the ACP process, then ask Dana to recall the fact.

**ACP included:** `initialize`, `session/new`, text `session/prompt`, text
`session/update`, text-turn `session/cancel`, `session/load`, and
`session/resume`.

**Dana included:** Minimal AgentSession, single active turn, Session Journal,
Conversation and Host Event Views, SQLite/PostgreSQL adapters, protected replay
state, terminal turn facts, and legacy Timeline migration.

**Excluded from this delivery:** Tools, modes, model switching, MCP,
attachments, list/fork UI, and third-party factories.

**Changed areas:** STAR streaming, timeline persistence, repository factory,
new session/journal/view modules, `dana-acp`, and minimal Console session-ID
resume wiring.

**Dependency:** None.

**Uncertainty retired:** ACP streaming/update ordering, real database parity,
process-restart reconstruction, provider-compatible replay, and migration
idempotency.

**Acceptance:** First chunk arrives before completion; restart resumes context;
one terminal fact exists; concurrent same-session prompt returns `busy`; both
adapters project equivalent history; stdout contains ACP frames only.

**Automated tests:** Real SQLite and ephemeral PostgreSQL journal contracts,
projection parity, crash after input/during output, migration idempotency, ACP
subprocess framing, burst update ordering, writer conflict, and redaction.

**Real integration:** Real configured model through dana-console, including
ACP process kill/restart and contextual continuation.

**Rollback:** Disable journal authority, revert provider command, and read the
generated compatibility Timeline projection during the bounded rollback window.

**Documentation:** Architecture, storage setup, ACP configuration, migration,
rollback, and operational health.

**Effort/Risk:** XL / High.

**Not infrastructure-only:** The visible conversation survives a real process
restart.

**Go/No-go:** Both adapters pass; real Console restart succeeds; no stdout leak;
legacy parity has no unexplained differences.

### D2. Visible, Cancellable Tool Execution

**User outcome:** Tool calls appear as stable cards and Stop reaches a truthful
terminal outcome. Before this delivery, actions are invisible and cancellation
cannot prove underlying work stopped.

**Demo:** Start a long command, observe pending/in-progress, press Stop, see a
cancelled card and turn, and verify no owned process remains.

**ACP included:** Extend `session/prompt`, `session/cancel`, and
`session/update` with thought, tool-call, tool-update, result, and cancellation
states.

**Dana included:** Tool Catalog, Tool Identity, catalog versions, Tool Execution
Engine, cooperative decorator/latency contract, isolated worker, process-group
cleanup, remote acknowledgement, cancellation trees, Durable Jobs, and tool
journal facts.

**Excluded from this delivery:** Permission prompts, grants, MCP, rich
tool-specific rendering, and rollback of already-committed external effects.

**Changed areas:** Runtime discovery/schema generation, ToolExecutor path,
resource/workflow/agent registration, STAR ACT, Bash/process ownership,
streaming, and ACP translation.

**Dependency:** D1.

**Uncertainty retired:** Schema/dispatch agreement, legacy tool migration,
cooperative latency, worker isolation, and owned-work cleanup.

**Acceptance:** Collisions fail early; every call has one stable identity and
terminal fact; cancellation distinguishes acknowledged/timeout/unknown; unsafe
mutating tools use isolation; no owned subprocess leaks; turn terminal follows
all tool updates.

**Automated tests:** Catalog contracts, parallel same-name calls, cancellation
across queue/thread/worker/subprocess/remote/commit, kill escalation, crash
recovery, and Durable Job cascade/detach.

**Real integration:** Console Read/Bash/Edit runs, cancellation of long Bash and
isolated Python, OS process inspection, and journal verification.

**Rollback:** Feature flag selects the legacy executor for non-ACP hosts; Dana
ACP provider can be disabled independently.

**Documentation:** Tool migration, cancellation declarations, worker security,
Durable Jobs, and identity rules.

**Effort/Risk:** XL / High.

**Not infrastructure-only:** Users see progress and prove Stop terminates owned
work.

**Go/No-go:** Cancellation matrix passes without leaks; tool cards terminalize
once; existing tool behavior remains compatible.

### D3. Autonomous Permission Policy

**User outcome:** Users approve sensitive work, select modes, create durable
allow/reject grants, and run preflighted autonomous workflows without repeated
prompts.

**Demo:** In `default`, always-allow a workspace edit, reject a command once,
repeat the edit without a prompt, then run a declared workflow unattended.

**ACP included:** Mode state in `session/new`, `session/set_mode`,
`session/request_permission`, `current_mode_update`, and denied tool updates.
Options include allow once, always allow, reject once, and always reject.

**Dana included:** Operations, effect metadata, hard policy, Permission Modes,
Policy Grants, revocation, owner/workspace scoping, precedence, Policy Preflight,
fail-closed coordination, and permission/grant journal facts.

**Excluded from this delivery:** Implicit tenant-global grants, history-derived
grants, hard-policy bypass, and automatic grant widening.

**Changed areas:** Catalog metadata, execution enforcement, SQLite/PostgreSQL
policy stores, workflow manifests, AgentSession mode state, ACP permission
adapter, and existing Console permission/mode UI.

**Dependencies:** D1 and D2.

**Uncertainty retired:** Effect classification, grant safety, autonomous
workflow continuity, revocation, and cancellation during approval.

**Acceptance:** Hard deny wins; grants never cross scope; stale replies are
safe; timeout/disconnect/cancel denies; matching grants suppress only matching
prompts; revocation is immediate for the next Operation; preflight reports all
predictable missing grants.

**Automated tests:** Policy tables, grant matching/precedence, storage parity,
normalization, cross-owner isolation, modes, timeout/cancel races, preflight,
runtime enforcement, and journal redaction.

**Real integration:** All four permission choices and three modes in Console,
grant persistence across restart, and a real unattended workflow after
preflight.

**Rollback:** Disable durable-grant evaluation and return to default allow-once
prompts. Stored grants remain inactive; hard policy remains.

**Documentation:** Modes, grants, revocation, effects, preflight, operator
provisioning, and threat model.

**Effort/Risk:** L / High.

**Not infrastructure-only:** Users approve once and observe autonomous work
finish without repeated interruption.

**Go/No-go:** Security review, isolation tests, Console flows, cancellation
races, and autonomous workflow test all pass.

### D4. Configured Model Switching

**User outcome:** Console shows configured models and switches the active model
without losing conversation state.

**Demo:** Start with one provider, state a constraint, switch to another
configured provider, and continue using the same history and tools.

**ACP included:** Model state in `session/new`, `session/set_model`, and
`current_model_update`.

**Dana included:** Model Catalog, atomic provider/runtime construction,
provider-neutral Conversation View, replay compatibility, rebinding, cache
invalidation, and model-change facts.

**Excluded from this delivery:** Arbitrary IDs, automatic routing, mid-turn
switching, installation, and pricing UI.

**Changed areas:** Configuration, runtime selector, provider switching,
AgentSession mutation, replay projection, ACP mapping, and small Console selector.

**Dependency:** D1; integrates with D2.

**Uncertainty retired:** Runtime replacement, cross-provider history,
provider-specific replay, and atomic rebinding.

**Acceptance:** Only configured targets appear; failure preserves the old
model; history survives; incompatible protected state is excluded; model change
is journaled once.

**Automated tests:** Failure rollback, compatibility matrix, cache invalidation,
busy rejection, storage replay parity, and ACP translation.

**Real integration:** One conversation switched across two real configured
providers.

**Rollback:** Hide selector and pin startup model.

**Documentation:** Model configuration, compatibility, replay, and rollback.

**Effort/Risk:** M / Medium.

**Not infrastructure-only:** The user changes models and continues visibly.

**Go/No-go:** Real cross-provider continuation passes; failed switch causes no
partial mutation.

### D5. Configured MCP Tools

**User outcome:** Dana-configured MCP servers expose normal tools with policy,
cancellation, and Console lifecycle cards.

**Demo:** Configure a stdio filesystem MCP server, ask Dana to inspect the
workspace through it, approve the Operation, and cancel a long call.

**ACP included:** Reuse `session/new`, `session/prompt`, `session/cancel`,
`session/request_permission`, and tool updates. Client-provided MCP descriptors
remain supported by the target architecture but may be deferred in planning.

**Dana included:** Official MCP protocol, leases, handshake, capabilities,
`tools/list`, schema conversion, `tools/call`, stdio/HTTP adapters,
cancellation, required/optional restore, and namespaced Tool Identity.

**Excluded from this delivery:** Console MCP-management UI, unrestricted child
environment, process-global registry, and MCP prompts/resources not required by
the demonstrated workflow.

**Changed areas:** Replace duplicate MCP clients, configuration, catalog
adapter, remote execution adapter, policy effects, session restore and close.

**Dependencies:** D1-D3.

**Uncertainty retired:** Schema fidelity, cancellation acknowledgement, server
lifecycle, and dynamic catalog invalidation.

**Acceptance:** Real handshake/discovery; deterministic collisions; clean close;
optional failure degrades; required failure stops preflight; allowlisted
environment; exactly one terminal fact; stdio children reaped.

**Automated tests:** Fake-server protocol contract, real stdio subprocess, HTTP
failure/reconnect, schema edges, catalog invalidation, policy, and cleanup.

**Real integration:** Real configured MCP server invoked through Console.

**Rollback:** Disable MCP configuration loading; other tools remain available.

**Documentation:** Configuration, transports, security, lease requirement,
environment, and troubleshooting.

**Effort/Risk:** XL / High.

**Not infrastructure-only:** Users invoke a configured MCP tool visibly.

**Go/No-go:** Real server succeeds; cancellation leaks no owned process;
optional outage does not break restore.

### D6. Images and File Resources

**User outcome:** Users attach images and files that Dana validates, persists by
reference, and uses in a grounded response.

**Demo:** Attach equipment imagery and a configuration file, then ask Dana to
compare the observed state with the file.

**ACP included:** `session/prompt` text, image, embedded-resource, and
file-resource content. Image capability is advertised only when supported.

**Dana included:** Content normalization, MIME/size checks, workspace policy,
artifact references, multimodal Conversation View blocks, provider capability
validation, and independent artifact retention.

**Excluded from this delivery:** Audio/video, arbitrary URL fetch, unrestricted
file URIs, OCR pipeline, and attachment library UI.

**Changed areas:** STAR SEE/content admission, LLM content types, Conversation
View, artifact adapters, policy, ACP normalization, and Console attachment control.

**Dependencies:** D1 and D3.

**Uncertainty retired:** Provider block replay, embedded-byte durability,
artifact authorization, and model switching with unsupported media.

**Acceptance:** Blocks round-trip through both adapters; unsupported models fail
before turn start; traversal and oversized input fail; restart preserves
authorized attachments; missing artifacts fail explicitly.

**Automated tests:** Content matrix, MIME/size/path attacks, hash/deduplication,
restart replay, provider switching, missing/corrupt artifacts, and redaction.

**Real integration:** Real Console image and file prompt against a supporting
provider.

**Rollback:** Hide attachment control and reject non-text prompts.

**Documentation:** Limits, formats, model support, retention, and security.

**Effort/Risk:** L / Medium-High.

**Not infrastructure-only:** Users attach real content and receive a grounded
answer.

**Go/No-go:** Real provider, security, restart, and missing-artifact tests pass.

## 10. Complete Target Capabilities

The design retains these capabilities even when planning defers their delivery:

- trusted built-in and Python entry-point agent factories
- session create, list, load, resume, fork, archive, close, and delete
- client-provided session MCP leases
- configured model catalog and switching
- text, image, embedded, file, audio, and video content as provider support grows
- plan, usage, command, model, mode, MCP, and session host projections
- general ACP conformance beyond the first Console contract
- richer tool presentation adapters

Planning must mark deferral explicitly. It must not remove the architecture
seams or silently implement these capabilities inside ACP.

## 11. Testing Strategy

### Interface contracts

- SQLite and PostgreSQL journal behavior
- projection equivalence and rebuild
- policy grant storage and matching
- Tool Catalog schema/target agreement
- execution adapter cancellation and terminalization
- MCP transport and discovery
- artifact authorization and retention

### Fault and concurrency

- crash at every turn and tool lifecycle point
- optimistic writer conflicts
- interrupted permission and model changes
- stuck cooperative thread, worker, subprocess, remote, and MCP call
- forced kill and orphan detection
- projection lag and rebuild
- checkpoint failure and incompatible version
- unavailable provider, MCP server, database, and artifact store

### Compatibility

- dana-console fake-agent behavior and burst-update regression
- Custom ACP provider spawn and environment allowlist
- real streaming, restart resume, tools, modes, grants, models, MCP, and content
- update drain before prompt response
- stderr/stdout discipline

### Security

- Owner Scope isolation and PostgreSQL RLS integration
- hard policy dominance in every mode
- grant scope and revocation
- operation normalization and path traversal
- environment allowlists
- worker/process/container isolation
- MCP command/URL/credential policy
- protected provider state and secret redaction
- artifact access and deletion

## 12. Migration and Rollback

Migration has three authority phases:

1. **Shadow:** Existing Timeline is authoritative; journal receives shadow facts
   and parity is measured.
2. **Cutover:** Journal becomes authoritative; JSON is generated only as a
   compatibility projection.
3. **Retirement:** Compatibility writes stop after the rollback window; legacy
   files remain read-only archives.

First access imports a legacy session transactionally and records an idempotent
migration marker. No new path reads two authorities and chooses the newest.

Every delivery has an independent feature flag, provider selection, or adapter
rollback. Rollback never deletes journal facts, Policy Grants, or artifacts.

## 13. Recommended Order and Rationale

The thinnest viable first delivery is D1. It is larger internally than a
disposable ACP bridge, but it proves streaming and persistence through a
user-visible restart workflow.

D2 establishes truthful tool lifecycle and cancellation before permissions.
D3 then adds policy at the single execution point. D4 is lower risk than MCP
because Dana already has model-switching foundations. D5 proves the dynamic
catalog and remote execution adapters. D6 adds secured artifact-backed content
after persistence and policy are stable.

## 14. Changes From the Previous Spec

### Replaced

- mutable Timeline snapshots -> Session Journal plus Conversation View
- separate JSON SessionRepository -> journal session metadata
- horizontal kernel-first phases -> six Console-visible vertical deliveries
- direct/fallback tool registries -> one versioned Tool Catalog
- best-effort cancellation scope -> cancellation-first Tool Execution Engine
- ephemeral approval callback -> execution policy plus durable Policy Grants
- process-global MCP assumptions -> session MCP Leases

### Simplified

- AgentSession is the broad host interface; collaborators remain internal.
- Factory discovery is not required for the first Console workflow.
- ACP only translates Host Event View and host decisions.
- Rich projections exist in the target design without forcing immediate UI.

### Planning-phase deferrals

Planning may defer list/fork UI, client MCP UI, entry-point factories, audio,
video, rich plan/usage/command rendering, general ACP conformance, and
tool-specific presentation. Each deferral must preserve the target architecture
and name its later user workflow.

## 15. Unresolved Questions

- Exact Owner Scope and workspace identifiers supplied by the global platform.
- PostgreSQL RLS policy and artifact-store authorization integration.
- Isolated-worker technology and trusted tool reconstruction mechanism.
- Default maximum cooperative cancellation latency by tool effect class.
- Cross-provider replay compatibility for every configured provider pair.
- Production artifact backend and deletion policy.
- Exact Console persistence location for the resumed ACP session ID.
- Which complete target capabilities planning assigns to this program versus a
  later Console-owned program.
