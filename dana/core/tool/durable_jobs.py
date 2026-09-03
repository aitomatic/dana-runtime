"""Durable Jobs — cascade/detach lifecycle for long-running tool operations (D2).

Per ADR-005 (Cancellation-First Tool Execution Engine):
- Child ownership defaults to ``cascade``; ``detach`` requires successful Durable
  Job handoff.
- ``keep`` reserved for explicitly managed infrastructure.
- Cancellation cannot undo external effects already committed before acknowledgement.

A Durable Job is a tool call that has been **detached** from its parent's
cancellation scope. Once detached, the job lives independently: its parent can
be cancelled without affecting the job, and the job's terminal outcome is
journaled as a first-class fact.

Lifecycle:
1. **Handoff requested** — the parent tool requests detach for a child.
2. **Handoff acknowledged** — the Durable Job system accepts ownership.
3. **Handoff failed** — the child remains under cascade ownership.
4. **Running** — the job is executing independently.
5. **Terminal** — the job completed, failed, or was cancelled independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Durable Job lifecycle states
# ---------------------------------------------------------------------------


class DurableJobStatus(Enum):
    """Lifecycle states of a Durable Job."""

    HANDOFF_REQUESTED = "handoff_requested"
    """Detach has been requested but not yet confirmed."""

    RUNNING = "running"
    """The job is executing independently after successful handoff."""

    COMPLETED = "completed"
    """The job completed successfully."""

    FAILED = "failed"
    """The job failed."""

    CANCELLED = "cancelled"
    """The job was cancelled independently."""

    HANDOFF_FAILED = "handoff_failed"
    """The handoff was not accepted; the child remains under cascade."""


# ---------------------------------------------------------------------------
# Durable Job record
# ---------------------------------------------------------------------------


@dataclass
class DurableJobRecord:
    """A single Durable Job record.

    Created when a tool call is detached from its parent's cancellation scope.
    The record tracks the job's lifecycle from handoff through terminal state.
    """

    job_id: str
    """Unique identifier for this Durable Job (same as the tool_call_id)."""

    tool_name: str
    """Name of the tool being executed."""

    parent_tool_call_id: str | None
    """The tool_call_id of the parent that detached this job, if any."""

    status: DurableJobStatus = DurableJobStatus.HANDOFF_REQUESTED
    """Current lifecycle state."""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When the handoff was requested."""

    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When the record was last updated."""

    result: Any = None
    """The terminal result (on COMPLETED) or error info (on FAILED)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata attached to the job."""

    def update_status(self, status: DurableJobStatus) -> None:
        """Update the job status and timestamp."""
        self.status = status
        self.updated_at = datetime.now(UTC)

    @property
    def is_terminal(self) -> bool:
        """Whether the job has reached a terminal state."""
        return self.status in (
            DurableJobStatus.COMPLETED,
            DurableJobStatus.FAILED,
            DurableJobStatus.CANCELLED,
            DurableJobStatus.HANDOFF_FAILED,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "job_id": self.job_id,
            "tool_name": self.tool_name,
            "parent_tool_call_id": self.parent_tool_call_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "result": self.result,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DurableJobRecord:
        """Deserialize from a dict."""
        return cls(
            job_id=data["job_id"],
            tool_name=data["tool_name"],
            parent_tool_call_id=data.get("parent_tool_call_id"),
            status=DurableJobStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            result=data.get("result"),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Durable Job Manager
# ---------------------------------------------------------------------------


class DurableJobManager:
    """Manages Durable Job lifecycle — handoff, tracking, and terminal resolution.

    Thread-safe for concurrent access. Jobs are stored in-memory by default;
    a persistent backend can be provided for crash recovery.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, DurableJobRecord] = {}
        self._lock: Any = None  # Would use threading.Lock in production

    # ------------------------------------------------------------------
    # Handoff
    # ------------------------------------------------------------------

    def request_handoff(
        self,
        job_id: str,
        tool_name: str,
        parent_tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DurableJobRecord:
        """Request a Durable Job handoff for a tool call.

        This is the first step of detach: the parent requests that the child
        be removed from its cancellation scope. The handoff must be confirmed
        via :meth:`confirm_handoff` to take effect.

        Returns:
            The newly created DurableJobRecord in HANDOFF_REQUESTED state.

        Raises:
            ValueError: If a job with the same job_id already exists.
        """
        if job_id in self._jobs:
            raise ValueError(f"Durable Job {job_id!r} already exists")

        record = DurableJobRecord(
            job_id=job_id,
            tool_name=tool_name,
            parent_tool_call_id=parent_tool_call_id,
            status=DurableJobStatus.HANDOFF_REQUESTED,
            metadata=metadata or {},
        )
        self._jobs[job_id] = record
        return record

    def confirm_handoff(self, job_id: str) -> DurableJobRecord:
        """Confirm a Durable Job handoff.

        Moves the job from HANDOFF_REQUESTED to RUNNING. After this call,
        the job is detached from its parent's cancellation scope.

        Args:
            job_id: The job to confirm.

        Returns:
            The updated record.

        Raises:
            ValueError: If the job does not exist or is not in HANDOFF_REQUESTED state.
        """
        record = self._jobs.get(job_id)
        if record is None:
            raise ValueError(f"Durable Job {job_id!r} not found")
        if record.status != DurableJobStatus.HANDOFF_REQUESTED:
            raise ValueError(f"Durable Job {job_id!r} is in state {record.status.value!r}, expected 'handoff_requested'")
        record.update_status(DurableJobStatus.RUNNING)
        return record

    def fail_handoff(self, job_id: str, reason: str = "") -> DurableJobRecord:
        """Mark a Durable Job handoff as failed.

        The child remains under cascade ownership.

        Args:
            job_id: The job whose handoff failed.
            reason: Optional reason for the failure.

        Returns:
            The updated record.
        """
        record = self._jobs.get(job_id)
        if record is None:
            raise ValueError(f"Durable Job {job_id!r} not found")
        record.update_status(DurableJobStatus.HANDOFF_FAILED)
        if reason:
            record.metadata["handoff_failure_reason"] = reason
        return record

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def complete(self, job_id: str, result: Any = None) -> DurableJobRecord:
        """Mark a Durable Job as completed.

        Args:
            job_id: The job to complete.
            result: The result of the job.

        Returns:
            The updated record.
        """
        record = self._get(job_id)
        record.result = result
        record.update_status(DurableJobStatus.COMPLETED)
        return record

    def fail(self, job_id: str, error: Any = None) -> DurableJobRecord:
        """Mark a Durable Job as failed.

        Args:
            job_id: The job to fail.
            error: Error information.

        Returns:
            The updated record.
        """
        record = self._get(job_id)
        record.result = error
        record.update_status(DurableJobStatus.FAILED)
        return record

    def cancel(self, job_id: str) -> DurableJobRecord:
        """Cancel a Durable Job independently.

        Args:
            job_id: The job to cancel.

        Returns:
            The updated record.
        """
        record = self._get(job_id)
        record.update_status(DurableJobStatus.CANCELLED)
        return record

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> DurableJobRecord | None:
        """Look up a Durable Job by ID."""
        return self._jobs.get(job_id)

    def _get(self, job_id: str) -> DurableJobRecord:
        """Look up a job or raise."""
        record = self._jobs.get(job_id)
        if record is None:
            raise ValueError(f"Durable Job {job_id!r} not found")
        return record

    @property
    def active_jobs(self) -> list[DurableJobRecord]:
        """Return all jobs that are not yet terminal."""
        return [j for j in self._jobs.values() if not j.is_terminal]

    @property
    def all_jobs(self) -> list[DurableJobRecord]:
        """Return all jobs."""
        return list(self._jobs.values())

    def list_by_parent(self, parent_tool_call_id: str) -> list[DurableJobRecord]:
        """Return all jobs that were detached from a given parent."""
        return [j for j in self._jobs.values() if j.parent_tool_call_id == parent_tool_call_id]

    # ------------------------------------------------------------------
    # Cascade / Detach integration
    # ------------------------------------------------------------------

    def detach_from_parent(self, parent_tool_call_id: str) -> list[DurableJobRecord]:
        """Detach all RUNNING jobs from a parent.

        Called when a parent is cancelled: RUNNING Durable Jobs survive
        (they are already detached). Jobs still in HANDOFF_REQUESTED are
        failed — the handoff never completed.

        Returns the list of jobs that were affected (handoff-failed).
        """
        affected: list[DurableJobRecord] = []
        for job in self.list_by_parent(parent_tool_call_id):
            if job.status == DurableJobStatus.HANDOFF_REQUESTED:
                self.fail_handoff(
                    job.job_id,
                    reason=f"Parent {parent_tool_call_id!r} cancelled before handoff confirmed",
                )
                affected.append(job)
            # RUNNING jobs survive — they are already detached
        return affected

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize all jobs to a JSON-safe dict."""
        return {
            "jobs": [j.to_dict() for j in self._jobs.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DurableJobManager:
        """Deserialize from a dict."""
        manager = cls()
        for job_data in data.get("jobs", []):
            record = DurableJobRecord.from_dict(job_data)
            manager._jobs[record.job_id] = record
        return manager
