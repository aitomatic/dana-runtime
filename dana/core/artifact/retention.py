"""
Artifact retention policy — independent from journal-fact retention.

Per ADR-009: artifact retention is independent from journal-fact retention.
Per ADR-002: journal facts hold artifact references, not large content;
retention is per-session explicit deletion.

This module provides retention policy management that is decoupled from the
Session Journal lifecycle. Artifacts can be retained beyond the session that
created them, or deleted independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

from dana.core.artifact.store import ArtifactStore
from dana.core.session.models import OwnerScope


# ---------------------------------------------------------------------------
# Retention policy types
# ---------------------------------------------------------------------------


class RetentionPolicyType(Enum):
    """Types of retention policies for artifacts."""

    # Keep the artifact indefinitely
    KEEP_INDEFINITE = "keep_indefinite"
    # Keep for a specified duration after creation
    KEEP_FOR_DURATION = "keep_for_duration"
    # Keep until the session is deleted
    KEEP_UNTIL_SESSION_DELETED = "keep_until_session_deleted"
    # Delete immediately (transient artifact)
    DELETE_IMMEDIATELY = "delete_immediately"


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """A retention policy for artifacts.

    Attributes:
        policy_type: The type of retention policy.
        duration_seconds: The retention duration in seconds (only for
            ``KEEP_FOR_DURATION``).
    """

    policy_type: RetentionPolicyType = RetentionPolicyType.KEEP_UNTIL_SESSION_DELETED
    duration_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.policy_type is RetentionPolicyType.KEEP_FOR_DURATION:
            if self.duration_seconds is None or self.duration_seconds < 0:
                raise ValueError(
                    "KEEP_FOR_DURATION requires a non-negative duration_seconds"
                )


# ---------------------------------------------------------------------------
# Artifact retention manager
# ---------------------------------------------------------------------------


@dataclass
class ArtifactRetentionEntry:
    """A tracked artifact with its retention policy.

    Attributes:
        sha256: The SHA-256 hash of the artifact.
        owner_scope: The owner scope that owns the artifact.
        session_id: The session that created the artifact (may be empty for
            cross-session artifacts).
        policy: The retention policy for this artifact.
        created_at: ISO-8601 timestamp of when the artifact was created.
    """

    sha256: str
    owner_scope: OwnerScope
    session_id: str
    policy: RetentionPolicy
    created_at: str  # ISO-8601 timestamp


class ArtifactRetentionManager:
    """Manages artifact retention policies independently from journal facts.

    Usage::

        manager = ArtifactRetentionManager(store)
        await manager.track(sha256, owner_scope, session_id, policy)
        await manager.enforce_retention(owner_scope)
    """

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        # In-memory retention tracking. In production, this would be backed by
        # a database table (see spec §15 — production artifact backend).
        self._entries: dict[str, ArtifactRetentionEntry] = {}

    async def track(
        self,
        sha256: str,
        owner_scope: OwnerScope,
        session_id: str,
        policy: RetentionPolicy | None = None,
    ) -> None:
        """Track an artifact with a retention policy.

        Args:
            sha256: The SHA-256 hash of the artifact.
            owner_scope: The owner scope that owns the artifact.
            session_id: The session that created the artifact.
            policy: The retention policy. Defaults to
                ``KEEP_UNTIL_SESSION_DELETED``.
        """
        if policy is None:
            policy = RetentionPolicy()

        now = datetime.now(UTC).isoformat()
        key = self._entry_key(sha256, owner_scope)
        self._entries[key] = ArtifactRetentionEntry(
            sha256=sha256,
            owner_scope=owner_scope,
            session_id=session_id,
            policy=policy,
            created_at=now,
        )

    async def enforce_retention(self, owner_scope: OwnerScope) -> int:
        """Enforce retention policies, deleting expired artifacts.

        Args:
            owner_scope: The owner scope to enforce policies for.

        Returns:
            The number of artifacts deleted.
        """
        now = datetime.now(UTC)
        deleted = 0
        keys_to_delete: list[str] = []

        for key, entry in list(self._entries.items()):
            if entry.owner_scope != owner_scope:
                continue

            should_delete = self._should_delete(entry, now)
            if should_delete:
                try:
                    await self._store.delete(entry.sha256, entry.owner_scope)
                except Exception:
                    # Log and continue — don't let one failure block the sweep
                    pass
                keys_to_delete.append(key)
                deleted += 1

        for key in keys_to_delete:
            self._entries.pop(key, None)

        return deleted

    async def on_session_deleted(self, session_id: str, owner_scope: OwnerScope) -> int:
        """Handle session deletion: delete artifacts with KEEP_UNTIL_SESSION_DELETED.

        Args:
            session_id: The session that was deleted.
            owner_scope: The owner scope.

        Returns:
            The number of artifacts deleted.
        """
        deleted = 0
        keys_to_delete: list[str] = []

        for key, entry in list(self._entries.items()):
            if (
                entry.owner_scope == owner_scope
                and entry.session_id == session_id
                and entry.policy.policy_type is RetentionPolicyType.KEEP_UNTIL_SESSION_DELETED
            ):
                try:
                    await self._store.delete(entry.sha256, entry.owner_scope)
                except Exception:
                    pass
                keys_to_delete.append(key)
                deleted += 1

        for key in keys_to_delete:
            self._entries.pop(key, None)

        return deleted

    async def get_entry(
        self,
        sha256: str,
        owner_scope: OwnerScope,
    ) -> ArtifactRetentionEntry | None:
        """Get the retention entry for an artifact.

        Args:
            sha256: The SHA-256 hash of the artifact.
            owner_scope: The owner scope.

        Returns:
            The retention entry, or ``None`` if not tracked.
        """
        key = self._entry_key(sha256, owner_scope)
        return self._entries.get(key)

    def _should_delete(self, entry: ArtifactRetentionEntry, now: datetime) -> bool:
        """Check if an artifact should be deleted based on its retention policy."""
        if entry.policy.policy_type is RetentionPolicyType.DELETE_IMMEDIATELY:
            return True
        if entry.policy.policy_type is RetentionPolicyType.KEEP_FOR_DURATION:
            if entry.policy.duration_seconds is not None:
                created = datetime.fromisoformat(entry.created_at)
                expiry = created + timedelta(seconds=entry.policy.duration_seconds)
                return now >= expiry
        return False

    @staticmethod
    def _entry_key(sha256: str, owner_scope: OwnerScope) -> str:
        """Build a unique key for a retention entry."""
        return f"{owner_scope.owner_id}:{owner_scope.workspace}:{sha256}"
