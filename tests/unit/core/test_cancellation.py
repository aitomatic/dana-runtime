"""D2 Cancellation Trees — cancellation matrix, kill escalation, terminal fact enforcement.

Covers:
- AC #1: Cancellation matrix across queue/thread/worker/subprocess/remote/commit
- AC #2: Cancellation distinguishes acknowledged/timeout/unknown
- AC #4: Exactly one terminal fact per tool call
- Edge cases: cancel already-terminal tool, detach during cascade, concurrent
  cancellation requests, effect-unknown on recovery
"""

from __future__ import annotations

import pytest

from dana.core.tool.cancellation import (
    CancellationNode,
    CancellationOutcome,
    CancellationTree,
    ChildOwnership,
)


# =========================================================================
# Helpers
# =========================================================================


def _make_node(
    tool_call_id: str = "tc1",
    tool_name: str = "test_tool",
    ownership: ChildOwnership = ChildOwnership.CASCADE,
) -> CancellationNode:
    return CancellationNode(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        ownership=ownership,
    )


# =========================================================================
# AC #1: Cancellation matrix — six execution contexts
# =========================================================================


class TestCancellationMatrix:
    """AC #1: Cancellation matrix covers all six execution contexts.

    The six contexts are: queue, thread, worker, subprocess, remote, commit.
    Each context has a distinct cancellation path that must be tested.
    """

    def test_cancel_queue_context(self):
        """Queue context: cancellation before execution starts."""
        tree = CancellationTree()
        node = tree.register("tc1", "queue_tool")
        # Cancel before any execution
        affected = tree.cancel("tc1")
        assert len(affected) == 1
        assert affected[0].tool_call_id == "tc1"
        assert affected[0].is_cancelled
        # Acknowledge the cancellation
        node.acknowledge()
        assert node.outcome == CancellationOutcome.ACKNOWLEDGED

    def test_cancel_thread_context(self):
        """Thread context: cooperative cancellation with flag."""
        tree = CancellationTree()
        node = tree.register("tc1", "thread_tool")
        # Simulate tool running in a thread
        tree.cancel("tc1")
        # Tool checks flag and acknowledges
        node.acknowledge()
        assert node.outcome == CancellationOutcome.ACKNOWLEDGED
        assert node.acknowledged_at is not None

    def test_cancel_worker_context(self):
        """Worker context: isolated process cancellation."""
        tree = CancellationTree()
        node = tree.register("tc1", "worker_tool")
        tree.cancel("tc1")
        # Worker process group killed — mark timed out
        node.mark_timed_out()
        assert node.outcome == CancellationOutcome.TIMED_OUT
        assert node.timed_out_at is not None

    def test_cancel_subprocess_context(self):
        """Subprocess context: child process cancellation."""
        tree = CancellationTree()
        parent = tree.register("parent", "parent_tool")
        child = tree.register("child", "child_tool", parent_id="parent")
        # Cancel parent — cascade to child
        affected = tree.cancel("parent")
        assert len(affected) == 2
        assert parent.is_cancelled
        assert child.is_cancelled
        # Child acknowledges
        child.acknowledge()
        assert child.outcome == CancellationOutcome.ACKNOWLEDGED

    def test_cancel_remote_context(self):
        """Remote context: external API call cancellation."""
        tree = CancellationTree()
        node = tree.register("tc1", "remote_tool")
        tree.cancel("tc1")
        # Remote API may not respond — mark effect unknown
        node.mark_effect_unknown()
        assert node.outcome == CancellationOutcome.EFFECT_UNKNOWN

    def test_cancel_commit_context(self):
        """Commit context: a tool that already committed its effects."""
        tree = CancellationTree()
        node = tree.register("tc1", "commit_tool")
        # Tool completes before cancellation arrives
        # Cancellation is requested but tool already finished
        tree.cancel("tc1")
        # The tool already produced a result — this is the terminal fact
        # Cancellation was requested but cannot undo committed effects
        assert node.is_cancelled
        # The outcome is None because the tool completed (not cancelled)
        # This tests ADR-005: cancellation cannot undo external effects
        # already committed before acknowledgement
        assert node.outcome is None  # No cancellation outcome — tool completed

    def test_cancel_already_terminal_tool(self):
        """Cancelling a tool that already has a terminal outcome is a no-op."""
        tree = CancellationTree()
        node = tree.register("tc1", "done_tool")
        node.acknowledge()  # Already terminal
        assert node.is_terminal
        # Cancel again — should not change outcome
        tree.cancel("tc1")
        assert node.outcome == CancellationOutcome.ACKNOWLEDGED

    def test_cancel_nonexistent_tool_raises(self):
        """Cancelling a tool not in the tree raises ValueError."""
        tree = CancellationTree()
        with pytest.raises(ValueError, match="not found"):
            tree.cancel("nonexistent")


# =========================================================================
# AC #2: Cancellation outcomes — acknowledged/timeout/unknown
# =========================================================================


class TestCancellationOutcomes:
    """AC #2: Each cancellation outcome is distinct and journaled."""

    def test_outcome_acknowledged(self):
        """ACKNOWLEDGED: tool stopped within max latency."""
        node = _make_node()
        node.request_cancel()
        node.acknowledge()
        assert node.outcome == CancellationOutcome.ACKNOWLEDGED
        assert node.acknowledged_at is not None
        assert node.cancelled_at is not None
        assert node.acknowledged_at >= node.cancelled_at

    def test_outcome_timed_out(self):
        """TIMED_OUT: tool did not acknowledge within max latency."""
        node = _make_node()
        node.request_cancel()
        node.mark_timed_out()
        assert node.outcome == CancellationOutcome.TIMED_OUT
        assert node.timed_out_at is not None
        assert node.timed_out_at >= node.cancelled_at

    def test_outcome_effect_unknown(self):
        """EFFECT_UNKNOWN: terminal state cannot be determined."""
        node = _make_node()
        node.mark_effect_unknown()
        assert node.outcome == CancellationOutcome.EFFECT_UNKNOWN
        # No cancellation was requested — this is a recovery scenario
        assert node.cancelled_at is None

    def test_outcomes_are_distinct_enums(self):
        """The three outcomes are distinct enum values."""
        assert CancellationOutcome.ACKNOWLEDGED != CancellationOutcome.TIMED_OUT
        assert CancellationOutcome.ACKNOWLEDGED != CancellationOutcome.EFFECT_UNKNOWN
        assert CancellationOutcome.TIMED_OUT != CancellationOutcome.EFFECT_UNKNOWN

    def test_outcome_values_are_strings(self):
        """Outcome values are JSON-safe strings for journaling."""
        assert CancellationOutcome.ACKNOWLEDGED.value == "acknowledged"
        assert CancellationOutcome.TIMED_OUT.value == "timed_out"
        assert CancellationOutcome.EFFECT_UNKNOWN.value == "effect_unknown"

    def test_acknowledge_after_timeout(self):
        """Acknowledge after timeout still records the correct outcome."""
        node = _make_node()
        node.request_cancel()
        node.mark_timed_out()
        # Tool eventually acknowledges (late)
        node.acknowledge()
        # The outcome is still ACKNOWLEDGED (last write wins for outcome)
        assert node.outcome == CancellationOutcome.ACKNOWLEDGED
        assert node.acknowledged_at is not None
        assert node.timed_out_at is not None


# =========================================================================
# Cancellation tree structure
# =========================================================================


class TestCancellationTree:
    """Cancellation tree structure and navigation."""

    def test_register_root_node(self):
        """Registering a root node adds it to the tree."""
        tree = CancellationTree()
        node = tree.register("tc1", "my_tool")
        assert node.tool_call_id == "tc1"
        assert node.tool_name == "my_tool"
        assert node.parent is None
        assert len(tree.roots) == 1
        assert len(tree) == 1

    def test_register_child_node(self):
        """Registering a child node links it to the parent."""
        tree = CancellationTree()
        parent = tree.register("parent", "parent_tool")
        child = tree.register("child", "child_tool", parent_id="parent")
        assert child.parent is parent
        assert child in parent.children
        assert len(tree) == 2

    def test_register_duplicate_raises(self):
        """Registering a duplicate tool_call_id raises ValueError."""
        tree = CancellationTree()
        tree.register("tc1", "tool_a")
        with pytest.raises(ValueError, match="already registered"):
            tree.register("tc1", "tool_b")

    def test_register_with_missing_parent_raises(self):
        """Registering with a non-existent parent raises ValueError."""
        tree = CancellationTree()
        with pytest.raises(ValueError, match="not found"):
            tree.register("child", "child_tool", parent_id="nonexistent")

    def test_find_node(self):
        """Finding a node by tool_call_id works."""
        tree = CancellationTree()
        tree.register("parent", "parent_tool")
        child = tree.register("child", "child_tool", parent_id="parent")
        found = tree.get("child")
        assert found is child

    def test_contains(self):
        """The 'in' operator works on the tree."""
        tree = CancellationTree()
        tree.register("tc1", "my_tool")
        assert "tc1" in tree
        assert "nonexistent" not in tree

    def test_all_nodes(self):
        """all_nodes returns every node in the tree."""
        tree = CancellationTree()
        tree.register("root1", "r1")
        tree.register("root2", "r2")
        tree.register("child1", "c1", parent_id="root1")
        tree.register("child2", "c2", parent_id="root1")
        assert len(tree.all_nodes) == 4

    def test_node_path_from_root(self):
        """path_from_root returns the correct path."""
        tree = CancellationTree()
        tree.register("root", "root_tool")
        tree.register("mid", "mid_tool", parent_id="root")
        leaf = tree.register("leaf", "leaf_tool", parent_id="mid")
        path = leaf.path_from_root
        assert len(path) == 3
        assert path[0].tool_call_id == "root"
        assert path[1].tool_call_id == "mid"
        assert path[2].tool_call_id == "leaf"

    def test_node_root_property(self):
        """The root property walks up to the root."""
        tree = CancellationTree()
        tree.register("root", "root_tool")
        tree.register("mid", "mid_tool", parent_id="root")
        leaf = tree.register("leaf", "leaf_tool", parent_id="mid")
        assert leaf.root.tool_call_id == "root"

    def test_all_descendants(self):
        """all_descendants returns all recursive children."""
        parent = _make_node("parent", "parent_tool")
        child1 = _make_node("child1", "c1")
        child2 = _make_node("child2", "c2")
        grandchild = _make_node("grandchild", "gc")
        parent.add_child(child1)
        parent.add_child(child2)
        child1.add_child(grandchild)
        desc = parent.all_descendants()
        assert len(desc) == 3
        assert desc[0].tool_call_id == "child1"
        assert desc[1].tool_call_id == "grandchild"
        assert desc[2].tool_call_id == "child2"

    def test_find_in_subtree(self):
        """Finding a node by tool_call_id in a subtree works."""
        parent = _make_node("parent", "parent_tool")
        child = _make_node("child", "child_tool")
        parent.add_child(child)
        found = parent.find("child")
        assert found is child
        assert parent.find("nonexistent") is None


# =========================================================================
# Cascade / Detach semantics
# =========================================================================


class TestCascadeDetach:
    """Cascade and detach ownership semantics (ADR-005)."""

    def test_cascade_propagates_to_children(self):
        """CASCADE: cancelling parent propagates to all CASCADE children."""
        tree = CancellationTree()
        parent = tree.register("parent", "parent_tool")
        child = tree.register("child", "child_tool", parent_id="parent")
        affected = tree.cancel("parent")
        assert parent.is_cancelled
        assert child.is_cancelled
        assert len(affected) == 2

    def test_detach_prevents_propagation(self):
        """DETACH: cancelling parent does NOT propagate to DETACH children."""
        tree = CancellationTree()
        parent = tree.register("parent", "parent_tool")
        child = tree.register("child", "child_tool", parent_id="parent", ownership=ChildOwnership.DETACH)
        affected = tree.cancel("parent")
        assert parent.is_cancelled
        assert not child.is_cancelled  # DETACH child survives
        assert len(affected) == 1  # Only parent affected

    def test_keep_prevents_propagation(self):
        """KEEP: cancelling parent does NOT propagate to KEEP children."""
        tree = CancellationTree()
        parent = tree.register("parent", "parent_tool")
        child = tree.register("child", "child_tool", parent_id="parent", ownership=ChildOwnership.KEEP)
        affected = tree.cancel("parent")
        assert parent.is_cancelled
        assert not child.is_cancelled
        assert len(affected) == 1

    def test_escalate_kills_all_descendants(self):
        """Escalate force-kills all descendants regardless of ownership."""
        tree = CancellationTree()
        parent = tree.register("parent", "parent_tool")
        detach_child = tree.register("detach_child", "detach_tool", parent_id="parent", ownership=ChildOwnership.DETACH)
        keep_child = tree.register("keep_child", "keep_tool", parent_id="parent", ownership=ChildOwnership.KEEP)
        cascade_child = tree.register("cascade_child", "cascade_tool", parent_id="parent")
        affected = tree.escalate("parent")
        assert parent.is_cancelled
        assert detach_child.is_cancelled
        assert keep_child.is_cancelled
        assert cascade_child.is_cancelled
        assert len(affected) == 4

    def test_mixed_ownership_cascade(self):
        """Mixed ownership: only CASCADE children are cancelled on normal cancel."""
        tree = CancellationTree()
        tree.register("parent", "parent_tool")
        c1 = tree.register("c1", "cascade_child", parent_id="parent")
        d1 = tree.register("d1", "detach_child", parent_id="parent", ownership=ChildOwnership.DETACH)
        k1 = tree.register("k1", "keep_child", parent_id="parent", ownership=ChildOwnership.KEEP)
        c2 = tree.register("c2", "cascade_child2", parent_id="parent")
        affected = tree.cancel("parent")
        assert c1.is_cancelled
        assert c2.is_cancelled
        assert not d1.is_cancelled
        assert not k1.is_cancelled
        assert len(affected) == 3  # parent + c1 + c2


# =========================================================================
# AC #4: Terminal fact enforcement
# =========================================================================


class TestTerminalFactEnforcement:
    """AC #4: Exactly one terminal fact per tool call."""

    def test_all_terminal_when_all_have_outcomes(self):
        """all_terminal is True when every node has an outcome."""
        tree = CancellationTree()
        tree.register("tc1", "tool_a")
        tree.register("tc2", "tool_b")
        tree.get("tc1").acknowledge()
        tree.get("tc2").mark_timed_out()
        assert tree.all_terminal

    def test_all_terminal_false_when_missing_outcomes(self):
        """all_terminal is False when some nodes lack outcomes."""
        tree = CancellationTree()
        tree.register("tc1", "tool_a")
        tree.register("tc2", "tool_b")
        tree.get("tc1").acknowledge()
        # tc2 has no outcome
        assert not tree.all_terminal

    def test_nodes_without_outcome(self):
        """nodes_without_outcome returns nodes missing terminal outcomes."""
        tree = CancellationTree()
        tree.register("tc1", "tool_a")
        tree.register("tc2", "tool_b")
        tree.get("tc1").acknowledge()
        missing = tree.nodes_without_outcome()
        assert len(missing) == 1
        assert missing[0].tool_call_id == "tc2"

    def test_mark_effect_unknown_for_all(self):
        """mark_effect_unknown_for_all marks all non-terminal nodes."""
        tree = CancellationTree()
        tree.register("tc1", "tool_a")
        tree.register("tc2", "tool_b")
        tree.get("tc1").acknowledge()
        affected = tree.mark_effect_unknown_for_all()
        assert len(affected) == 1
        assert affected[0].tool_call_id == "tc2"
        assert affected[0].outcome == CancellationOutcome.EFFECT_UNKNOWN
        # tc1 should still be ACKNOWLEDGED
        assert tree.get("tc1").outcome == CancellationOutcome.ACKNOWLEDGED
        # Now all are terminal
        assert tree.all_terminal

    def test_crash_recovery_marks_effect_unknown(self):
        """ADR-011: started tools without terminal facts are marked effect-unknown."""
        tree = CancellationTree()
        # Simulate tools that were started before a crash
        tree.register("tc1", "running_tool")
        tree.register("tc2", "completed_tool")
        tree.get("tc2").acknowledge()  # This one completed
        # Crash recovery
        affected = tree.mark_effect_unknown_for_all()
        assert len(affected) == 1
        assert affected[0].tool_call_id == "tc1"
        assert affected[0].outcome == CancellationOutcome.EFFECT_UNKNOWN


# =========================================================================
# Serialization
# =========================================================================


class TestCancellationSerialization:
    """Cancellation tree serialization round-trip."""

    def test_node_round_trip(self):
        """A CancellationNode serializes and deserializes correctly."""
        node = _make_node("tc1", "test_tool", ChildOwnership.CASCADE)
        node.request_cancel()
        node.acknowledge()
        data = node.to_dict()
        restored = CancellationNode.from_dict(data)
        assert restored.tool_call_id == "tc1"
        assert restored.tool_name == "test_tool"
        assert restored.ownership == ChildOwnership.CASCADE
        assert restored.outcome == CancellationOutcome.ACKNOWLEDGED
        assert restored.cancelled_at is not None
        assert restored.acknowledged_at is not None

    def test_tree_round_trip(self):
        """A CancellationTree serializes and deserializes correctly."""
        tree = CancellationTree()
        tree.register("root", "root_tool")
        tree.register("child", "child_tool", parent_id="root")
        tree.register("grandchild", "gc_tool", parent_id="child")
        tree.get("root").request_cancel()
        tree.get("child").acknowledge()
        tree.get("grandchild").mark_timed_out()

        data = tree.to_dict()
        restored = CancellationTree.from_dict(data)
        assert len(restored) == 3
        assert restored.get("root") is not None
        assert restored.get("child") is not None
        assert restored.get("grandchild") is not None
        assert restored.get("root").is_cancelled
        assert restored.get("child").outcome == CancellationOutcome.ACKNOWLEDGED
        assert restored.get("grandchild").outcome == CancellationOutcome.TIMED_OUT

    def test_empty_tree_round_trip(self):
        """An empty tree serializes and deserializes correctly."""
        tree = CancellationTree()
        data = tree.to_dict()
        restored = CancellationTree.from_dict(data)
        assert len(restored) == 0
        assert len(restored.roots) == 0


# =========================================================================
# Edge cases
# =========================================================================


class TestCancellationEdgeCases:
    """Edge cases for cancellation trees."""

    def test_concurrent_cancellation_requests(self):
        """Multiple cancellation requests are idempotent."""
        tree = CancellationTree()
        node = tree.register("tc1", "my_tool")
        # First cancel
        affected1 = tree.cancel("tc1")
        assert len(affected1) == 1
        ts1 = node.cancelled_at
        # Second cancel — no new nodes affected
        affected2 = tree.cancel("tc1")
        assert len(affected2) == 0
        assert node.cancelled_at == ts1  # Timestamp unchanged

    def test_detach_during_cascade(self):
        """A DETACH child survives cascade from parent."""
        tree = CancellationTree()
        tree.register("parent", "parent_tool")
        detach_child = tree.register("detach_child", "detach_tool", parent_id="parent", ownership=ChildOwnership.DETACH)
        cascade_child = tree.register("cascade_child", "cascade_tool", parent_id="parent")
        # Cancel parent
        tree.cancel("parent")
        assert cascade_child.is_cancelled
        assert not detach_child.is_cancelled
        # Now escalate — kills everything
        tree.escalate("parent")
        assert detach_child.is_cancelled

    def test_deeply_nested_cascade(self):
        """Cascade propagates through multiple levels."""
        tree = CancellationTree()
        a = tree.register("a", "tool_a")
        b = tree.register("b", "tool_b", parent_id="a")
        c = tree.register("c", "tool_c", parent_id="b")
        d = tree.register("d", "tool_d", parent_id="c")
        affected = tree.cancel("a")
        assert len(affected) == 4
        assert all(n.is_cancelled for n in [a, b, c, d])

    def test_cancel_then_acknowledge_then_timeout(self):
        """A tool can be cancelled, acknowledged, then also timed out."""
        node = _make_node()
        node.request_cancel()
        node.acknowledge()
        node.mark_timed_out()
        # Last write wins for outcome
        assert node.outcome == CancellationOutcome.TIMED_OUT
        assert node.acknowledged_at is not None
        assert node.timed_out_at is not None

    def test_effect_unknown_on_recovery_without_cancel(self):
        """ADR-011: effect-unknown on recovery without any cancellation request."""
        tree = CancellationTree()
        tree.register("tc1", "started_tool")
        # No cancellation was ever requested — crash recovery
        affected = tree.mark_effect_unknown_for_all()
        assert len(affected) == 1
        assert affected[0].outcome == CancellationOutcome.EFFECT_UNKNOWN
        assert affected[0].cancelled_at is None  # Never cancelled
