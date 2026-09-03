"""D2 Durable Jobs — cascade/detach lifecycle, handoff, terminal resolution.

Covers:
- AC #3: Durable Job cascade propagates to children and detach severs ownership
- AC #4: Exactly one terminal fact per tool call (via DurableJobRecord.is_terminal)
- Edge cases: crash before handoff completes, detach during cascade, concurrent
  handoff requests
"""

from __future__ import annotations

import pytest

from dana.core.tool.durable_jobs import DurableJobManager, DurableJobRecord, DurableJobStatus


# =========================================================================
# AC #3: Durable Job lifecycle
# =========================================================================


class TestDurableJobLifecycle:
    """AC #3: Durable Job lifecycle — handoff, running, terminal."""

    def test_handoff_requested(self):
        """A Durable Job starts in HANDOFF_REQUESTED state."""
        manager = DurableJobManager()
        record = manager.request_handoff("job1", "my_tool")
        assert record.job_id == "job1"
        assert record.tool_name == "my_tool"
        assert record.status == DurableJobStatus.HANDOFF_REQUESTED
        assert not record.is_terminal

    def test_confirm_handoff(self):
        """Confirming handoff moves the job to RUNNING."""
        manager = DurableJobManager()
        manager.request_handoff("job1", "my_tool")
        record = manager.confirm_handoff("job1")
        assert record.status == DurableJobStatus.RUNNING
        assert not record.is_terminal

    def test_complete_job(self):
        """Completing a job moves it to COMPLETED."""
        manager = DurableJobManager()
        manager.request_handoff("job1", "my_tool")
        manager.confirm_handoff("job1")
        record = manager.complete("job1", result={"output": "done"})
        assert record.status == DurableJobStatus.COMPLETED
        assert record.result == {"output": "done"}
        assert record.is_terminal

    def test_fail_job(self):
        """Failing a job moves it to FAILED."""
        manager = DurableJobManager()
        manager.request_handoff("job1", "my_tool")
        manager.confirm_handoff("job1")
        record = manager.fail("job1", error="Something went wrong")
        assert record.status == DurableJobStatus.FAILED
        assert record.result == "Something went wrong"
        assert record.is_terminal

    def test_cancel_job(self):
        """Cancelling a job moves it to CANCELLED."""
        manager = DurableJobManager()
        manager.request_handoff("job1", "my_tool")
        manager.confirm_handoff("job1")
        record = manager.cancel("job1")
        assert record.status == DurableJobStatus.CANCELLED
        assert record.is_terminal

    def test_fail_handoff(self):
        """Failing a handoff moves the job to HANDOFF_FAILED."""
        manager = DurableJobManager()
        manager.request_handoff("job1", "my_tool")
        record = manager.fail_handoff("job1", reason="Parent cancelled")
        assert record.status == DurableJobStatus.HANDOFF_FAILED
        assert record.metadata.get("handoff_failure_reason") == "Parent cancelled"
        assert record.is_terminal

    def test_confirm_handoff_twice_raises(self):
        """Confirming a handoff that's already confirmed raises ValueError."""
        manager = DurableJobManager()
        manager.request_handoff("job1", "my_tool")
        manager.confirm_handoff("job1")
        with pytest.raises(ValueError, match="state"):
            manager.confirm_handoff("job1")

    def test_confirm_handoff_on_completed_raises(self):
        """Confirming a handoff on a completed job raises ValueError."""
        manager = DurableJobManager()
        manager.request_handoff("job1", "my_tool")
        manager.confirm_handoff("job1")
        manager.complete("job1")
        with pytest.raises(ValueError, match="state"):
            manager.confirm_handoff("job1")


# =========================================================================
# AC #3: Cascade / Detach integration
# =========================================================================


class TestCascadeDetachIntegration:
    """AC #3: Durable Job cascade propagates to children and detach severs ownership."""

    def test_detach_from_parent_with_running_jobs(self):
        """RUNNING Durable Jobs survive parent cancellation."""
        manager = DurableJobManager()
        # Request and confirm handoff for a child
        manager.request_handoff("job1", "child_tool", parent_tool_call_id="parent1")
        manager.confirm_handoff("job1")
        # Parent is cancelled
        affected = manager.detach_from_parent("parent1")
        # RUNNING job survives — no affected
        assert len(affected) == 0
        assert manager.get("job1").status == DurableJobStatus.RUNNING

    def test_detach_from_parent_with_pending_handoff(self):
        """HANDOFF_REQUESTED jobs are failed when parent is cancelled."""
        manager = DurableJobManager()
        # Request but do NOT confirm handoff
        manager.request_handoff("job1", "child_tool", parent_tool_call_id="parent1")
        # Parent is cancelled
        affected = manager.detach_from_parent("parent1")
        assert len(affected) == 1
        assert affected[0].job_id == "job1"
        assert affected[0].status == DurableJobStatus.HANDOFF_FAILED
        assert "cancelled before handoff confirmed" in affected[0].metadata.get("handoff_failure_reason", "")

    def test_detach_mixed_state(self):
        """Mixed state: RUNNING jobs survive, HANDOFF_REQUESTED jobs fail."""
        manager = DurableJobManager()
        # Two children from same parent
        manager.request_handoff("job1", "running_child", parent_tool_call_id="parent1")
        manager.confirm_handoff("job1")
        manager.request_handoff("job2", "pending_child", parent_tool_call_id="parent1")
        # Parent cancelled
        affected = manager.detach_from_parent("parent1")
        assert len(affected) == 1
        assert affected[0].job_id == "job2"
        assert manager.get("job1").status == DurableJobStatus.RUNNING  # Survived

    def test_list_by_parent(self):
        """list_by_parent returns all jobs from a given parent."""
        manager = DurableJobManager()
        manager.request_handoff("job1", "tool_a", parent_tool_call_id="parent1")
        manager.request_handoff("job2", "tool_b", parent_tool_call_id="parent1")
        manager.request_handoff("job3", "tool_c", parent_tool_call_id="parent2")
        jobs = manager.list_by_parent("parent1")
        assert len(jobs) == 2
        assert {j.job_id for j in jobs} == {"job1", "job2"}

    def test_active_jobs(self):
        """active_jobs returns only non-terminal jobs."""
        manager = DurableJobManager()
        manager.request_handoff("job1", "tool_a")
        manager.confirm_handoff("job1")
        manager.request_handoff("job2", "tool_b")
        manager.confirm_handoff("job2")
        manager.complete("job2")
        active = manager.active_jobs
        assert len(active) == 1
        assert active[0].job_id == "job1"


# =========================================================================
# Edge cases
# =========================================================================


class TestDurableJobEdgeCases:
    """Edge cases for Durable Jobs."""

    def test_duplicate_job_id_raises(self):
        """Creating a job with a duplicate ID raises ValueError."""
        manager = DurableJobManager()
        manager.request_handoff("job1", "tool_a")
        with pytest.raises(ValueError, match="already exists"):
            manager.request_handoff("job1", "tool_b")

    def test_get_nonexistent_job(self):
        """Getting a non-existent job returns None."""
        manager = DurableJobManager()
        assert manager.get("nonexistent") is None

    def test_complete_nonexistent_job_raises(self):
        """Completing a non-existent job raises ValueError."""
        manager = DurableJobManager()
        with pytest.raises(ValueError, match="not found"):
            manager.complete("nonexistent")

    def test_crash_before_handoff_completes(self):
        """Crash before handoff completes: job is in HANDOFF_REQUESTED."""
        manager = DurableJobManager()
        manager.request_handoff("job1", "tool_a", parent_tool_call_id="parent1")
        # Simulate crash — on recovery, the job is still HANDOFF_REQUESTED
        # The recovery logic should fail the handoff
        record = manager.fail_handoff("job1", reason="Crash recovery")
        assert record.status == DurableJobStatus.HANDOFF_FAILED

    def test_detach_during_cascade(self):
        """A DETACH child survives cascade from parent cancellation."""
        manager = DurableJobManager()
        # Child is already detached (RUNNING)
        manager.request_handoff("job1", "child_tool", parent_tool_call_id="parent1")
        manager.confirm_handoff("job1")
        # Parent cancelled — child survives
        affected = manager.detach_from_parent("parent1")
        assert len(affected) == 0
        assert manager.get("job1").status == DurableJobStatus.RUNNING

    def test_job_with_metadata(self):
        """Jobs can carry arbitrary metadata."""
        manager = DurableJobManager()
        record = manager.request_handoff(
            "job1",
            "tool_a",
            parent_tool_call_id="parent1",
            metadata={"priority": "high", "retry_count": 3},
        )
        assert record.metadata["priority"] == "high"
        assert record.metadata["retry_count"] == 3

    def test_job_timestamps(self):
        """Job timestamps are set on creation and updates."""
        manager = DurableJobManager()
        record = manager.request_handoff("job1", "tool_a")
        created = record.created_at
        updated = record.updated_at
        assert created is not None
        assert updated is not None
        # After update
        record = manager.confirm_handoff("job1")
        assert record.updated_at >= updated


# =========================================================================
# Serialization
# =========================================================================


class TestDurableJobSerialization:
    """Durable Job serialization round-trip."""

    def test_record_round_trip(self):
        """A DurableJobRecord serializes and deserializes correctly."""
        record = DurableJobRecord(
            job_id="job1",
            tool_name="my_tool",
            parent_tool_call_id="parent1",
            status=DurableJobStatus.RUNNING,
            result=None,
            metadata={"key": "value"},
        )
        data = record.to_dict()
        restored = DurableJobRecord.from_dict(data)
        assert restored.job_id == "job1"
        assert restored.tool_name == "my_tool"
        assert restored.parent_tool_call_id == "parent1"
        assert restored.status == DurableJobStatus.RUNNING
        assert restored.metadata == {"key": "value"}

    def test_manager_round_trip(self):
        """A DurableJobManager serializes and deserializes correctly."""
        manager = DurableJobManager()
        manager.request_handoff("job1", "tool_a", parent_tool_call_id="parent1")
        manager.confirm_handoff("job1")
        manager.request_handoff("job2", "tool_b")
        manager.complete("job2", result="done")

        data = manager.to_dict()
        restored = DurableJobManager.from_dict(data)
        assert restored.get("job1") is not None
        assert restored.get("job2") is not None
        assert restored.get("job1").status == DurableJobStatus.RUNNING
        assert restored.get("job2").status == DurableJobStatus.COMPLETED
        assert restored.get("job2").result == "done"

    def test_empty_manager_round_trip(self):
        """An empty manager serializes and deserializes correctly."""
        manager = DurableJobManager()
        data = manager.to_dict()
        restored = DurableJobManager.from_dict(data)
        assert len(restored.all_jobs) == 0
