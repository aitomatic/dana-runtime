# ACP Exposure and AgentSession Kernel - Design Spec

**Original date:** 2026-07-07
**Revised:** 2026-07-16
**Status:** Approved design
**Related:** [ACP protocol research report](../../../plans/reports/researcher-260707-2122-acp-protocol-research.md)

## 1. Goal

Expose registered `STARAgent` implementations as full Agent Client Protocol
(ACP) agents over JSON-RPC 2.0 on stdio. Use the work to add host-neutral
session, approval, MCP, cancellation, model-selection, and event capabilities
to Dana rather than hiding those capabilities inside the ACP adapter.

The first compatibility target is the ACP client in
`~/Desktop/repos/dana-os-docs-update/dana-console`, which already launches
Claude Code, Codex, and custom ACP agents.

## 2. Design Principles

1. ACP translates protocol; it does not own Dana policy or agent behavior.
2. One `AgentSession` owns one isolated `STARAgent` instance.
3. Core capabilities are reusable by ACP, CLI, and future gateway hosts.
4. Installed code is selected through trusted factory registration, never an
   arbitrary import path from an ACP request.
5. Persistent Dana configuration and session-scoped client configuration have
   different lifetimes and authority.
6. Permission bypass skips interactive prompts, not hard security policy.
7. Every turn has ordered events, one terminal outcome, bounded cancellation,
   and a durable snapshot.

## 3. Scope

### In scope

- ACP stdio entry point: `dana-acp`.
- Built-in and Python-entry-point agent factories selected by registered name.
- Full session lifecycle: create, list, load, resume, fork, close.
- Prompt, cancellation, session modes, model selection, and supported config
  options.
- Text, image, embedded-resource, and file-resource prompt content.
- Ordered text, thought, plan, tool, approval, usage, model, mode, error, and
  completion events.
- Core approval service with policy-based fallback.
- Core MCP manager that merges Dana-configured and client-provided servers.
- Durable session metadata through a repository protocol with a local JSON
  implementation.
- Existing STAR timeline persistence as the conversation source of truth.

### Out of scope

- Audio prompt content.
- Arbitrary client-supplied Python import paths or model identifiers.
- Persisting client-provided MCP server descriptors.
- Sharing one mutable STARAgent across concurrent ACP sessions.
- HTTP or remote ACP transport.
- Online model training or self-modifying policy.

## 4. Locked Decisions

| Decision | Resolution |
| --- | --- |
| Agent selection | Trusted registered factory name |
| Extensibility | Built-ins plus `dana.star_agents` Python entry points |
| Capability tier | Full ACP surface used by the target Console and SDK |
| Core boundary | Host-neutral `AgentSession` kernel |
| Agent isolation | One STARAgent per session |
| Persistence | Repository protocol plus local JSON metadata backend |
| Conversation state | Existing STAR timeline persistence |
| Approval fallback | Policy classifies; sensitive operations deny without host approval unless explicitly configured |
| Permission modes | `default`, `acceptEdits`, `bypassPermissions` |
| MCP sources | Persistent Dana configuration plus policy-gated session MCP |
| Model catalog | Only Dana-configured provider/model combinations |
| Model switch | Enhance and reuse `STARAgent.set_llm_provider()` |
| Prompt content | Text, images, embedded resources, file resources |
| Cancellation | Bounded hard cancellation with subprocess cleanup |
| Compatibility target | dana-console `CopilotSession` |

## 5. Architecture

```text
ACP stdio host       CLI host          Future gateway host
      |                  |                       |
      +---------- host-neutral commands/events -+
                             |
                    AgentSessionManager
             create/list/load/resume/fork/close
                             |
                       AgentSession
        +--------------------+--------------------+
        |                    |                    |
   STARAgent            EventBroker       CancellationScope
        |                    |                    |
   ApprovalService       MCPManager          ModelCatalog
        |                    |                    |
        +---------- SessionRepository ------------+
                             |
       AgentFactoryRegistry + Dana configuration
```

The ACP package depends on the session kernel. The session kernel may depend on
STAR public APIs and capability protocols. STAR core must not import ACP types.

## 6. Component Contracts

### AgentFactoryRegistry

Loads Dana built-ins and installed entry points from `dana.star_agents`. A
factory declares a stable ID, title, supported capabilities, and:

```python
def create(context: AgentCreationContext) -> STARAgent: ...
```

The initial built-ins are `star` and `coding`. Duplicate IDs, invalid factory
objects, and failing third-party entry points are isolated and reported without
preventing built-ins from loading.

### AgentSessionManager

Owns active sessions and coordinates the repository. It exposes create, list,
load, resume, fork, and close. Different sessions may run concurrently; one
session accepts only one active turn.

### AgentSession

Owns exactly one agent, workspace, turn lock, event broker, cancellation scope,
approval service, MCP manager, selected model, and permission mode. It is the
only object allowed to mutate that session's agent state.

### SessionRepository

Stores an atomic local JSON record containing:

- ACP session ID and STAR timeline/session ID
- factory ID and workspace
- selected configured provider/model
- permission mode
- persistent Dana MCP references
- creation and update timestamps

Client-provided MCP descriptors are deliberately excluded. Corrupt records are
quarantined and omitted from session listing.

### ApprovalService

Receives a host-neutral operation descriptor before sensitive execution. It
combines hard policy, workspace policy, operation classification, and session
mode, then returns allow, deny, or request-user-input.

- `default`: safe operations proceed; sensitive operations request approval.
- `acceptEdits`: workspace edits proceed; commands, network, and other
  sensitive operations still request approval.
- `bypassPermissions`: operations proceed without prompting only when hard
  policy permits them.

When a host cannot request approval, request-user-input resolves to deny unless
configuration explicitly supplies a narrower allow rule. Timeout, disconnect,
or cancellation also resolves to deny.

### MCPManager

Merges two sources:

1. Dana-configured servers: durable references restored with the session.
2. ACP client servers: policy-gated and scoped to the live session only.

Registration validates transport, executable or URL, arguments, environment,
workspace, and host policy before spawning or connecting. Untrusted stdio
commands and remote hosts require approval. MCP tools enter STAR through the
normal resource/tool registry and use stable namespacing to prevent collisions.

### ModelCatalog and model switching

The catalog exposes only configured provider/model combinations. ACP model
changes are serialized against the turn lock and use an enhanced
`STARAgent.set_llm_provider()` implementation. The switch must:

1. Validate the configured target.
2. Rebuild the LLM client.
3. Reselect the runtime when provider/runtime compatibility changes.
4. Rebind runtime and long-term-memory LLM sinks.
5. Invalidate system-prompt and model-sensitive caches.
6. Preserve timeline and session metadata.

### AgentEvent

The session kernel publishes typed events independent of ACP:

- message text and thought chunks
- plan updates
- tool start, progress, result, and denial
- approval required and resolved
- usage updates
- model and mode changes
- sanitized error
- terminal completion or cancellation

Each tool call has exactly one terminal tool event. Each turn has exactly one
terminal turn event.

## 7. ACP Protocol Mapping

The adapter uses the official Python `agent-client-protocol` SDK as an optional
dependency and reserves stdout exclusively for ACP JSON-RPC frames.

| ACP surface | Dana mapping |
| --- | --- |
| `initialize` | Version, capabilities, factory identity, auth methods |
| `session/new` | `AgentSessionManager.create` plus MCP merge |
| `session/list` | Durable repository listing |
| `session/load` | Reconstruct agent and replay persisted history |
| `session/resume` | Load or restore active session state |
| `session/fork` | Fork STAR timeline plus session metadata into a new ID |
| `session/prompt` | Normalize content and call `AgentSession.run_turn` |
| `session/cancel` | `CancellationScope.cancel` |
| `session/set_mode` | Approval mode transition outside active turn |
| `session/set_model` | Configured model switch outside active turn |
| `session/update` | Translate ordered `AgentEvent` values |
| `session/request_permission` | Host decision callback for `ApprovalService` |

The adapter advertises session list/load/resume/fork, images, models, and modes.
Authentication is reported from Dana's configured provider state; secrets are
not accepted as arbitrary ACP prompt data.

## 8. Turn Data Flow

1. ACP receives text, image, embedded-resource, or file-resource blocks.
2. The adapter normalizes and validates content, resource size, and workspace
   access before acquiring the session turn lock.
3. `AgentSession.run_turn` creates an event scope and calls STAR's streaming
   async path.
4. STAR emits host-neutral events directly from THINK and ACT boundaries. The
   design does not poll the timeline for tool state.
5. Before a sensitive tool executes, `ApprovalService` decides automatically
   or asks the ACP client through `session/request_permission`.
6. Approved tools execute; denied tools return typed results so STAR may recover
   or explain.
7. The ACP adapter translates queued events to `session/update` notifications.
8. On completion, error, or cancellation, the session flushes timeline and
   metadata, drains updates, emits one terminal event, and returns the ACP
   prompt response. The prompt response is always after its updates.

Model, mode, MCP, and fork mutations return `busy` while a turn is active.

## 9. Cancellation

Cancellation is a core capability, not only `asyncio.Task.cancel()`:

1. Signal a `CancellationScope` visible to LLM and tool execution.
2. Resolve pending approvals as denied/cancelled.
3. Cancel the active STAR task.
4. Ask owned tool and MCP subprocesses to terminate.
5. After a fixed grace period, kill remaining owned subprocesses.
6. Flush a cancelled session snapshot and emit the ACP cancelled stop reason.

Turn abandonment that leaves work running is not permitted.

## 10. Dana Console Compatibility

The target Console spawns an ACP process over stdio and requires:

- `initialize` followed by `session/new(cwd=...)`
- advertised session modes and live `session/set_mode`
- ordered message, thought, tool call, and tool call update notifications
- `session/request_permission` with allow/reject option IDs
- `session/cancel`
- all `session/update` handlers drained before the prompt response

Dana must preserve mode IDs `default`, `acceptEdits`, and
`bypassPermissions`. Richer plan, usage, model, session, and MCP support is
additive; the current Console may ignore update kinds it does not render.

## 11. Error Handling

- Unknown factory, model, mode, session, or unsupported content fails before a
  turn starts with a typed protocol error.
- A concurrent prompt on one session returns `busy`; other sessions continue.
- Approval timeout or disconnect denies the operation.
- MCP failure is reported without mutating persistent Dana configuration.
- Tool denial is a typed tool result, not an unhandled exception.
- Cancellation is bounded and owns subprocess cleanup.
- Agent failures emit sanitized client events; full traces go to stderr.
- Session writes are atomic; corrupt records are quarantined.
- Missing ACP optional dependencies produce an install hint and nonzero exit.
- No logs, warnings, tracebacks, or secrets may reach ACP stdout.

## 12. Testing Strategy

### Unit

- factory discovery, duplicate/failing entry points, and capability descriptors
- session repository round-trip, atomicity, quarantine, and fork metadata
- approval classification and all three modes, including hard denies
- configured model catalog and compatible/incompatible runtime switching
- MCP merge, namespacing, policy, environment filtering, and lifetime
- content normalization for text, image, embedded, and file resources
- event-to-ACP translation and terminal-event invariants

### Core contract

- every `AgentSession` lifecycle transition
- one active turn per session and cross-session concurrency
- one terminal tool event per call and one terminal event per turn
- cancellation during LLM, approval, sync tool, async tool, and MCP call
- durable restart, load, resume, list, and fork

### ACP integration

- initialize and advertised capabilities
- new/prompt/cancel/load/resume/list/fork
- modes, models, permissions, MCP, images, and resources
- update ordering before prompt response
- subprocess stdio framing and stderr discipline

### Compatibility and fault injection

- run against dana-console `CopilotSession` and its burst-update regression
- disconnect during approval
- stuck tool and forced process cleanup
- MCP spawn/connection failure
- corrupt session JSON and missing factory after restart
- real configured-model, filesystem-approval, and stdio-MCP smoke tests

### Security

- child environment allowlist and secret redaction
- workspace/file-resource boundaries
- MCP URL, command, argument, and environment policy
- hard denies remain effective in `bypassPermissions`
- no secrets in events, errors, logs, or stdout protocol frames

## 13. Delivery Boundaries

Implementation should be divided into independently verifiable increments:

1. AgentSession kernel, event types, factories, and durable repository.
2. Approval service, tool-executor hook, modes, and cancellation scope.
3. MCP manager and resource integration.
4. Model catalog and hardened provider switching.
5. ACP adapter and Console-compatible baseline.
6. Full session lifecycle, multimodal/resources, conformance, and hardening.

No phase may implement host-specific policy inside the ACP translator.
