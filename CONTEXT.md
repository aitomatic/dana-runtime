# Dana Agent Runtime

Dana runs stateful STAR agent conversations across terminal, ACP, and future hosts while preserving one coherent record of each session.

## Session History

**Session Journal**:
The sole durable authority for the ordered facts produced during one agent session. Facts remain available until explicit session deletion; large payloads may be retained separately as referenced artifacts.
_Avoid_: Timeline, execution log, transcript

**Owner Scope**:
The immutable tenant or principal scope that owns a Session Journal and its artifacts. Every session operation remains within this scope, including forks and projections.
_Avoid_: User field, optional tenant filter

**Journal Fact**:
An immutable, typed, and ordered statement about session activity. Content streams use bounded facts and an explicit final fact rather than treating each token as durable history.
_Avoid_: Event, log line, token delta

**Conversation View**:
The model-facing projection of a Session Journal, including the active compression checkpoint and retained recent conversation.
_Avoid_: Timeline snapshot, chat history

**Thought Summary**:
A sanitized reasoning update intentionally safe for host display and durable session history. It is distinct from hidden model reasoning and provider replay state.
_Avoid_: Chain of thought, raw reasoning

**Provider Replay State**:
Protected model-provider material required to continue a conversation faithfully. It is not host-visible session history or an ordinary Journal Fact.
_Avoid_: Thought, trace, reasoning log

**Compression Checkpoint**:
A typed, immutable Session Journal fact containing a summarized Conversation View for an exact committed sequence range. It records its projection version and provenance without replacing the facts it summarizes.
_Avoid_: Compact session, summary message

**Interrupted Turn**:
A turn that started but has no terminal Journal Fact. Partial output remains visible to hosts, while unfinished tool outcomes remain unknown and the Conversation View does not treat the partial answer as complete.
_Avoid_: Failed turn, cancelled turn

**Committed Turn**:
A turn closed by exactly one terminal Journal Fact. Only committed turn boundaries are valid fork points.
_Avoid_: Completed request

**Session Fork**:
A new Session Journal that inherits conversation history through an immutable reference to a parent session's committed turn. Parent facts are not copied into the child journal.
_Avoid_: Session copy, cloned transcript

## Tool Execution

**Tool Catalog**:
The versioned set of tools available to one session. A turn uses one immutable catalog version for both model presentation and invocation resolution.
_Avoid_: Global registry, tool list

**Tool Identity**:
The stable, provider-neutral identity of a tool within its source. Model-provider aliases and user-facing names may vary without changing journal or policy identity.
_Avoid_: Function name, display name, provider alias

**Operation**:
A normalized request to invoke a tool, described by its effects, validated arguments, and affected locations. Permission policy evaluates Operations rather than provider aliases or hard-coded tool names.
_Avoid_: Tool call dictionary, command

**Permission Mode**:
A session setting that controls when an otherwise permitted Operation requires user confirmation. Permission Modes never override hard policy.
_Avoid_: Security level, sandbox mode

**Policy Grant**:
A durable, revocable rule that allows or rejects matching Operations within an explicit Owner Scope, workspace, Tool Identity, effect, and location scope. Hard policy always overrides an allow grant.
_Avoid_: Remembered click, permission history, global wildcard

**Policy Preflight**:
A non-authorizing check that compares a workflow's declared Operations with hard policy and Policy Grants before execution. Dynamic Operations remain subject to invocation-time enforcement.
_Avoid_: Permission bypass, automatic approval

**Tool Execution Engine**:
The session-owned module that authorizes, runs, cancels, and terminalizes every tool invocation while recording its ordered Journal Facts.
_Avoid_: Tool wrapper, direct dispatch

**Cancellation Capability**:
A Tool Catalog declaration of how an invocation acknowledges cancellation. Cooperative thread tools declare and verify a maximum cancellation latency; hard cancellation requires a killable worker or acknowledged remote cancellation.
_Avoid_: Thread kill, best-effort stop

**Durable Job**:
Background work that has accepted ownership independently of its originating turn, with its own journal identity and cancellation handle. Work remains part of the parent cancellation tree until this handoff completes.
_Avoid_: Detached thread, fire-and-forget task

**MCP Lease**:
A session's scoped right to use one validated MCP server configuration and credential scope. Connection pooling is an internal optimization and does not change session ownership or tool visibility.
_Avoid_: Global MCP registration, shared server object
