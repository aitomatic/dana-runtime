"""
Unit tests for artifact retention — independent retention policy management.

Covers:
- Retention policy types
- Track artifacts with policies
- Enforce retention (delete expired)
- Session deletion handling
- Missing artifacts during sweep
"""

from __future__ import annotations

import pytest

from dana.core.artifact.retention import (
    ArtifactRetentionManager,
    RetentionPolicy,
    RetentionPolicyType,
)
from dana.core.artifact.store import ArtifactStore
from dana.core.session.models import OwnerScope


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def owner() -> OwnerScope:
    return OwnerScope(owner_id="test-owner", workspace="test-ws")


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(base_path=str(tmp_path / "artifacts"))


@pytest.fixture
def manager(store: ArtifactStore) -> ArtifactRetentionManager:
    return ArtifactRetentionManager(store)


# ===========================================================================
# Retention policy
# ===========================================================================


class TestRetentionPolicy:
    """Retention policy validation."""

    def test_default_policy(self) -> None:
        policy = RetentionPolicy()
        assert policy.policy_type is RetentionPolicyType.KEEP_UNTIL_SESSION_DELETED
        assert policy.duration_seconds is None

    def test_keep_indefinite(self) -> None:
        policy = RetentionPolicy(policy_type=RetentionPolicyType.KEEP_INDEFINITE)
        assert policy.policy_type is RetentionPolicyType.KEEP_INDEFINITE

    def test_keep_for_duration_requires_duration(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            RetentionPolicy(
                policy_type=RetentionPolicyType.KEEP_FOR_DURATION,
                duration_seconds=None,
            )

    def test_keep_for_duration_valid(self) -> None:
        policy = RetentionPolicy(
            policy_type=RetentionPolicyType.KEEP_FOR_DURATION,
            duration_seconds=3600,
        )
        assert policy.duration_seconds == 3600

    def test_delete_immediately(self) -> None:
        policy = RetentionPolicy(policy_type=RetentionPolicyType.DELETE_IMMEDIATELY)
        assert policy.policy_type is RetentionPolicyType.DELETE_IMMEDIATELY


# ===========================================================================
# Track artifacts
# ===========================================================================


class TestTrack:
    """Artifacts can be tracked with retention policies."""

    @pytest.mark.asyncio
    async def test_track_default_policy(self, manager: ArtifactRetentionManager, owner: OwnerScope) -> None:
        await manager.track("hash1", owner, "session-1")
        entry = await manager.get_entry("hash1", owner)
        assert entry is not None
        assert entry.sha256 == "hash1"
        assert entry.session_id == "session-1"
        assert entry.policy.policy_type is RetentionPolicyType.KEEP_UNTIL_SESSION_DELETED

    @pytest.mark.asyncio
    async def test_track_custom_policy(self, manager: ArtifactRetentionManager, owner: OwnerScope) -> None:
        policy = RetentionPolicy(policy_type=RetentionPolicyType.KEEP_INDEFINITE)
        await manager.track("hash2", owner, "session-1", policy)
        entry = await manager.get_entry("hash2", owner)
        assert entry is not None
        assert entry.policy.policy_type is RetentionPolicyType.KEEP_INDEFINITE

    @pytest.mark.asyncio
    async def test_get_entry_nonexistent(self, manager: ArtifactRetentionManager, owner: OwnerScope) -> None:
        entry = await manager.get_entry("nonexistent", owner)
        assert entry is None


# ===========================================================================
# Enforce retention
# ===========================================================================


class TestEnforceRetention:
    """Enforce retention deletes expired artifacts."""

    @pytest.mark.asyncio
    async def test_delete_immediately(self, manager: ArtifactRetentionManager, store: ArtifactStore, owner: OwnerScope) -> None:
        # Store an artifact and track it with DELETE_IMMEDIATELY
        ref = await store.store(b"transient content", "text/plain", owner)
        policy = RetentionPolicy(policy_type=RetentionPolicyType.DELETE_IMMEDIATELY)
        await manager.track(ref.sha256, owner, "session-1", policy)

        deleted = await manager.enforce_retention(owner)
        assert deleted == 1
        assert not await store.exists(ref.sha256, owner)

    @pytest.mark.asyncio
    async def test_keep_indefinite_not_deleted(self, manager: ArtifactRetentionManager, store: ArtifactStore, owner: OwnerScope) -> None:
        ref = await store.store(b"permanent content", "text/plain", owner)
        policy = RetentionPolicy(policy_type=RetentionPolicyType.KEEP_INDEFINITE)
        await manager.track(ref.sha256, owner, "session-1", policy)

        deleted = await manager.enforce_retention(owner)
        assert deleted == 0
        assert await store.exists(ref.sha256, owner)

    @pytest.mark.asyncio
    async def test_scope_isolation(self, manager: ArtifactRetentionManager, store: ArtifactStore, owner: OwnerScope) -> None:
        other = OwnerScope(owner_id="other", workspace="ws")
        ref = await store.store(b"content", "text/plain", owner)
        policy = RetentionPolicy(policy_type=RetentionPolicyType.DELETE_IMMEDIATELY)
        await manager.track(ref.sha256, owner, "session-1", policy)

        # Enforce on different scope — should not delete
        deleted = await manager.enforce_retention(other)
        assert deleted == 0
        assert await store.exists(ref.sha256, owner)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestRetentionEdgeCases:
    """Edge cases: retention expiry race, missing artifact during sweep."""

    @pytest.mark.asyncio
    async def test_missing_artifact_during_sweep_does_not_block(
        self, manager: ArtifactRetentionManager, store: ArtifactStore, owner: OwnerScope
    ) -> None:
        """If an artifact was already deleted externally, the sweep should not crash."""
        # Track an artifact that was never stored
        policy = RetentionPolicy(policy_type=RetentionPolicyType.DELETE_IMMEDIATELY)
        await manager.track("nonexistent_hash", owner, "session-1", policy)

        # Sweep should not raise even though the artifact doesn't exist
        deleted = await manager.enforce_retention(owner)
        assert deleted == 1  # Counts as deleted (entry removed from tracking)

    @pytest.mark.asyncio
    async def test_retention_expiry_race(
        self, manager: ArtifactRetentionManager, store: ArtifactStore, owner: OwnerScope
    ) -> None:
        """Multiple sweeps should be idempotent."""
        ref = await store.store(b"race content", "text/plain", owner)
        policy = RetentionPolicy(policy_type=RetentionPolicyType.DELETE_IMMEDIATELY)
        await manager.track(ref.sha256, owner, "session-1", policy)

        deleted1 = await manager.enforce_retention(owner)
        assert deleted1 == 1
        # Second sweep should find nothing to delete
        deleted2 = await manager.enforce_retention(owner)
        assert deleted2 == 0


# ===========================================================================
# Session deletion
# ===========================================================================


class TestOnSessionDeleted:
    """Session deletion cleans up KEEP_UNTIL_SESSION_DELETED artifacts."""

    @pytest.mark.asyncio
    async def test_session_deleted_cleans_up(self, manager: ArtifactRetentionManager, store: ArtifactStore, owner: OwnerScope) -> None:
        ref = await store.store(b"session content", "text/plain", owner)
        await manager.track(ref.sha256, owner, "session-1")

        deleted = await manager.on_session_deleted("session-1", owner)
        assert deleted == 1
        assert not await store.exists(ref.sha256, owner)

    @pytest.mark.asyncio
    async def test_different_session_not_affected(self, manager: ArtifactRetentionManager, store: ArtifactStore, owner: OwnerScope) -> None:
        ref = await store.store(b"other session content", "text/plain", owner)
        await manager.track(ref.sha256, owner, "session-1")

        deleted = await manager.on_session_deleted("session-2", owner)
        assert deleted == 0
        assert await store.exists(ref.sha256, owner)

    @pytest.mark.asyncio
    async def test_keep_indefinite_not_deleted_on_session_delete(
        self, manager: ArtifactRetentionManager, store: ArtifactStore, owner: OwnerScope
    ) -> None:
        ref = await store.store(b"permanent", "text/plain", owner)
        policy = RetentionPolicy(policy_type=RetentionPolicyType.KEEP_INDEFINITE)
        await manager.track(ref.sha256, owner, "session-1", policy)

        deleted = await manager.on_session_deleted("session-1", owner)
        assert deleted == 0
        assert await store.exists(ref.sha256, owner)
