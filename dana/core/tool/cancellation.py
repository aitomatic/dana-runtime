"""Cancellation trees — distinguishing acknowledged/timeout/unknown, kill escalation (D2).

Per ADR-005 (Cancellation-First Tool Execution Engine):
- Cancellation outcomes are requested/acknowledged/timed-out/effect-unknown — distinct
  and journaled.
- Cancellation cannot undo external effects already committed before acknowledgement.
- Child ownership defaults to ``cascade``; ``detach`` requires successful Durable Job
  handoff; ``keep`` reserved for explicitly managed infrastructure.

Per ADR-002 (Session Journal as Sole Durable Authority):
- Tool lifecycle facts are append-only; exactly one terminal fact per call.

Per ADR-011 (Crash Recovery as Interrupted Turn):
- Started tools without terminal facts are marked effect-unknown on crash recovery.
- Never auto-retry unknown-effect operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Cancellation outcome — the three distinct terminal outcomes
# ---------------------------------------------------------------------------


class CancellationOutcome(Enum):
    """The three distinct terminal outcomes of a cancellation request.

    Per ADR-005: these are distinct and journaled. Cancellation cannot undo
    external effects already committed before acknowledgement.
    """

    ACKNOWLEDGED = "acknowledged"
    """The tool acknowledged cancellation before producing a result or failure.
    The tool stopped within its declared max latency. External effects that
    committed before acknowledgement are durable and not undone."""

    TIMED_OUT = "timed_out"
    """The tool did not acknowledge cancellation within its declared max latency.
    For cooperative tools this is a contract violation; for isolated tools the
    worker process group was killed."""

    EFFECT_UNKNOWN = "effect_unknown"
    """The tool's terminal state cannot be determined — the process crashed,
    the worker vanished, or the journal was recovered mid-execution. Never
    auto-retry (ADR-011)."""


# ---------------------------------------------------------------------------
# Cancellation tree — hierarchical cancellation state
# ---------------------------------------------------------------------------


class ChildOwnership(Enum):
    """Ownership mode for child tools spawned by a parent.

    Per ADR-005:
    - ``CASCADE`` — default. Cancelling the parent cancels all children.
    - ``DETACH`` — the child becomes a Durable Job; cancellation does not
      propagate. Requires successful Durable Job handoff.
    - ``KEEP`` — reserved for explicitly managed infrastructure (e.g. long-running
      server processes). Not for general use.
    """

    CASCADE = "cascade"
    DETACH = "detach"
    KEEP = "keep"


@dataclass
class CancellationNode:
    """One node in the cancellation tree, representing a single tool call.

    Each node tracks its own cancellation state, its children (sub-tools it
    spawned), and the ownership mode that governs propagation.

    The tree is rooted at the turn-level tool call and grows as tools spawn
    sub-tools. Cancellation propagates from root to leaves according to
    ownership mode.
    """

    tool_call_id: str
    tool_name: str
    parent: CancellationNode | None = None
    children: list[CancellationNode] = field(default_factory=list)
    ownership: ChildOwnership = ChildOwnership.CASCADE
    outcome: CancellationOutcome | None = None
    cancelled_at: float | None = None  # monotonic timestamp
    acknowledged_at: float | None = None  # monotonic timestamp
    timed_out_at: float | None = None  # monotonic timestamp

    # ------------------------------------------------------------------
    # Tree navigation
    # ------------------------------------------------------------------

    def add_child(self, child: CancellationNode) -> None:
        """Add a child node under this one."""
        child.parent = self
        self.children.append(child)

    @property
    def root(self) -> CancellationNode:
        """Walk up to the root of this tree."""
        node: CancellationNode = self
        while node.parent is not None:
            node = node.parent
        return node

    @property
    def path_from_root(self) -> list[CancellationNode]:
        """Return the path from root to this node, inclusive."""
        path: list[CancellationNode] = []
        node: CancellationNode | None = self
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def find(self, tool_call_id: str) -> CancellationNode | None:
        """Find a node by tool_call_id in this subtree."""
        if self.tool_call_id == tool_call_id:
            return self
        for child in self.children:
            found = child.find(tool_call_id)
            if found is not None:
                return found
        return None

    def all_descendants(self) -> list[CancellationNode]:
        """Return all descendant nodes (recursive children)."""
        result: list[CancellationNode] = []
        for child in self.children:
            result.append(child)
            result.extend(child.all_descendants())
        return result

    # ------------------------------------------------------------------
    # Cancellation state
    # ------------------------------------------------------------------

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested on this node."""
        return self.cancelled_at is not None

    @property
    def is_terminal(self) -> bool:
        """Whether this node has a terminal cancellation outcome."""
        return self.outcome is not None

    def request_cancel(self, timestamp: float | None = None) -> None:
        """Request cancellation of this tool.

        Sets the cancellation timestamp. Does NOT propagate to children —
        that is the caller's responsibility via :meth:`propagate_cancel`.
        """
        import time

        self.cancelled_at = timestamp if timestamp is not None else time.monotonic()

    def acknowledge(self, timestamp: float | None = None) -> None:
        """Record that the tool acknowledged cancellation."""
        import time

        self.acknowledged_at = timestamp if timestamp is not None else time.monotonic()
        self.outcome = CancellationOutcome.ACKNOWLEDGED

    def mark_timed_out(self, timestamp: float | None = None) -> None:
        """Record that the tool timed out on cancellation."""
        import time

        self.timed_out_at = timestamp if timestamp is not None else time.monotonic()
        self.outcome = CancellationOutcome.TIMED_OUT

    def mark_effect_unknown(self) -> None:
        """Record that the tool's effect is unknown (crash recovery)."""
        self.outcome = CancellationOutcome.EFFECT_UNKNOWN

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate_cancel(self, timestamp: float | None = None) -> list[CancellationNode]:
        """Cancel this node and cascade to CASCADE children.

        Returns the list of nodes that were newly cancelled by this
        propagation (including self). DETACH children are NOT cancelled.
        KEEP children are NOT cancelled.

        This implements the cascade semantics from ADR-005: child ownership
        defaults to ``cascade``; ``detach`` requires successful Durable Job
        handoff; ``keep`` is reserved.
        """
        import time

        ts = timestamp if timestamp is not None else time.monotonic()
        affected: list[CancellationNode] = []

        if not self.is_cancelled:
            self.request_cancel(ts)
            affected.append(self)

        for child in self.children:
            if child.ownership == ChildOwnership.CASCADE:
                affected.extend(child.propagate_cancel(ts))
            # DETACH and KEEP children are not cancelled

        return affected

    def escalate(self, timestamp: float | None = None) -> list[CancellationNode]:
        """Escalate cancellation: force-kill all descendants regardless of ownership.

        Used when a DETACH child must be killed (e.g. the parent is being
        force-killed and the Durable Job handoff never completed). Returns
        the list of nodes affected.

        This is the "kill escalation" path — it overrides ownership for
        emergency cleanup.
        """
        import time

        ts = timestamp if timestamp is not None else time.monotonic()
        affected: list[CancellationNode] = []

        if not self.is_cancelled:
            self.request_cancel(ts)
            affected.append(self)

        for child in self.children:
            affected.extend(child.escalate(ts))

        return affected

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize this node to a JSON-safe dict."""
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "ownership": self.ownership.value,
            "outcome": self.outcome.value if self.outcome else None,
            "cancelled_at": self.cancelled_at,
            "acknowledged_at": self.acknowledged_at,
            "timed_out_at": self.timed_out_at,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], parent: CancellationNode | None = None) -> CancellationNode:
        """Deserialize a node from a dict."""
        node = cls(
            tool_call_id=data["tool_call_id"],
            tool_name=data["tool_name"],
            parent=parent,
            ownership=ChildOwnership(data["ownership"]),
            outcome=CancellationOutcome(data["outcome"]) if data.get("outcome") else None,
            cancelled_at=data.get("cancelled_at"),
            acknowledged_at=data.get("acknowledged_at"),
            timed_out_at=data.get("timed_out_at"),
        )
        for child_data in data.get("children", []):
            node.add_child(cls.from_dict(child_data, parent=node))
        return node


# ---------------------------------------------------------------------------
# CancellationTree — root-level container
# ---------------------------------------------------------------------------


class CancellationTree:
    """Root-level container for a turn's cancellation tree.

    Manages the forest of tool calls in a turn. Each top-level tool call
    is a root node; sub-tools are children of their parent.

    Provides:
    - ``find`` / ``get`` by tool_call_id
    - ``propagate_cancel`` with cascade semantics
    - ``escalate`` for force-kill
    - ``all_terminal`` check for ADR-002 compliance
    """

    def __init__(self) -> None:
        self._roots: list[CancellationNode] = []
        self._by_id: dict[str, CancellationNode] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        tool_call_id: str,
        tool_name: str,
        parent_id: str | None = None,
        ownership: ChildOwnership = ChildOwnership.CASCADE,
    ) -> CancellationNode:
        """Register a new tool call in the tree.

        Args:
            tool_call_id: Unique ID for this tool call.
            tool_name: Name of the tool being called.
            parent_id: If set, this tool is a child of the given parent.
            ownership: Ownership mode (default CASCADE).

        Returns:
            The newly created node.

        Raises:
            ValueError: If tool_call_id already exists or parent_id not found.
        """
        if tool_call_id in self._by_id:
            raise ValueError(f"Tool call {tool_call_id!r} already registered")

        node = CancellationNode(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            ownership=ownership,
        )

        if parent_id is not None:
            parent = self._by_id.get(parent_id)
            if parent is None:
                raise ValueError(f"Parent {parent_id!r} not found for tool {tool_call_id!r}")
            parent.add_child(node)
        else:
            self._roots.append(node)

        self._by_id[tool_call_id] = node
        return node

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, tool_call_id: str) -> CancellationNode | None:
        """Look up a node by tool_call_id."""
        return self._by_id.get(tool_call_id)

    def __contains__(self, tool_call_id: str) -> bool:
        return tool_call_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    @property
    def roots(self) -> list[CancellationNode]:
        """Return all root nodes (top-level tool calls)."""
        return list(self._roots)

    @property
    def all_nodes(self) -> list[CancellationNode]:
        """Return every node in the tree."""
        return list(self._by_id.values())

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self, tool_call_id: str) -> list[CancellationNode]:
        """Cancel a specific tool and cascade to its CASCADE children.

        Returns the list of nodes affected by this cancellation.

        Raises:
            ValueError: If tool_call_id is not found.
        """
        node = self._by_id.get(tool_call_id)
        if node is None:
            raise ValueError(f"Tool call {tool_call_id!r} not found in cancellation tree")
        return node.propagate_cancel()

    def escalate(self, tool_call_id: str) -> list[CancellationNode]:
        """Escalate cancellation on a tool, force-killing all descendants.

        Returns the list of nodes affected.

        Raises:
            ValueError: If tool_call_id is not found.
        """
        node = self._by_id.get(tool_call_id)
        if node is None:
            raise ValueError(f"Tool call {tool_call_id!r} not found in cancellation tree")
        return node.escalate()

    # ------------------------------------------------------------------
    # Terminal fact enforcement (ADR-002)
    # ------------------------------------------------------------------

    @property
    def all_terminal(self) -> bool:
        """Check whether every node has a terminal outcome.

        Per ADR-002: exactly one terminal fact per tool call. This property
        returns True when every registered node has an outcome set.
        """
        return all(node.is_terminal for node in self._by_id.values())

    def nodes_without_outcome(self) -> list[CancellationNode]:
        """Return all nodes that do not yet have a terminal outcome."""
        return [node for node in self._by_id.values() if not node.is_terminal]

    def mark_effect_unknown_for_all(self) -> list[CancellationNode]:
        """Mark all non-terminal nodes as EFFECT_UNKNOWN.

        Used during crash recovery (ADR-011): started tools without terminal
        facts are marked effect-unknown.
        """
        affected: list[CancellationNode] = []
        for node in self._by_id.values():
            if not node.is_terminal:
                node.mark_effect_unknown()
                affected.append(node)
        return affected

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire tree to a JSON-safe dict."""
        return {
            "roots": [r.to_dict() for r in self._roots],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CancellationTree:
        """Deserialize a tree from a dict."""
        tree = cls()
        for root_data in data.get("roots", []):
            node = CancellationNode.from_dict(root_data)
            tree._roots.append(node)
            # Rebuild the by_id index
            _rebuild_index(node, tree._by_id)
        return tree


def _rebuild_index(node: CancellationNode, index: dict[str, CancellationNode]) -> None:
    """Recursively rebuild the by_id index from a deserialized tree."""
    index[node.tool_call_id] = node
    for child in node.children:
        _rebuild_index(child, index)
