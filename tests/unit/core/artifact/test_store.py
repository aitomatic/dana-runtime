"""
Unit tests for artifact store — hash-based deduplication with OwnerScope isolation.

Covers:
- Store and load artifacts
- Hash-based dedup (same content → same reference)
- Missing artifacts fail explicitly
- OwnerScope isolation
- Delete and exists operations
"""

from __future__ import annotations

import hashlib

import pytest

from dana.core.artifact.store import ArtifactNotFound, ArtifactStore
from dana.core.session.models import ArtifactRef, OwnerScope


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def owner() -> OwnerScope:
    return OwnerScope(owner_id="test-owner", workspace="test-ws")


@pytest.fixture
def other_owner() -> OwnerScope:
    return OwnerScope(owner_id="other-owner", workspace="other-ws")


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(base_path=str(tmp_path / "artifacts"))


# ===========================================================================
# Store and load
# ===========================================================================


class TestStoreAndLoad:
    """Artifacts can be stored and loaded by hash."""

    @pytest.mark.asyncio
    async def test_store_and_load(self, store: ArtifactStore, owner: OwnerScope) -> None:
        content = b"hello world"
        ref = await store.store(content, "text/plain", owner)
        assert isinstance(ref, ArtifactRef)
        assert ref.sha256 == hashlib.sha256(content).hexdigest()
        assert ref.size == len(content)
        assert ref.media_type == "text/plain"
        assert ref.uri.startswith("artifact://")

        loaded = await store.load(ref.sha256, owner)
        assert loaded == content

    @pytest.mark.asyncio
    async def test_store_empty_content(self, store: ArtifactStore, owner: OwnerScope) -> None:
        ref = await store.store(b"", "text/plain", owner)
        loaded = await store.load(ref.sha256, owner)
        assert loaded == b""


# ===========================================================================
# Hash-based deduplication
# ===========================================================================


class TestDeduplication:
    """Same content produces the same artifact reference."""

    @pytest.mark.asyncio
    async def test_same_content_same_hash(self, store: ArtifactStore, owner: OwnerScope) -> None:
        content = b"deduplicated content"
        ref1 = await store.store(content, "text/plain", owner)
        ref2 = await store.store(content, "text/plain", owner)
        assert ref1.sha256 == ref2.sha256
        assert ref1.uri == ref2.uri
        assert ref1.size == ref2.size

    @pytest.mark.asyncio
    async def test_different_content_different_hash(self, store: ArtifactStore, owner: OwnerScope) -> None:
        ref1 = await store.store(b"content a", "text/plain", owner)
        ref2 = await store.store(b"content b", "text/plain", owner)
        assert ref1.sha256 != ref2.sha256


# ===========================================================================
# Missing artifacts
# ===========================================================================


class TestMissingArtifacts:
    """Missing artifacts fail explicitly."""

    @pytest.mark.asyncio
    async def test_load_nonexistent_raises(self, store: ArtifactStore, owner: OwnerScope) -> None:
        with pytest.raises(ArtifactNotFound, match="not found"):
            await store.load("nonexistenthash0000000000000000000000000000000000000000000000", owner)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_raises(self, store: ArtifactStore, owner: OwnerScope) -> None:
        with pytest.raises(ArtifactNotFound):
            await store.delete("nonexistenthash0000000000000000000000000000000000000000000000", owner)

    @pytest.mark.asyncio
    async def test_exists_returns_false_for_missing(self, store: ArtifactStore, owner: OwnerScope) -> None:
        exists = await store.exists("nonexistenthash0000000000000000000000000000000000000000000000", owner)
        assert not exists

    @pytest.mark.asyncio
    async def test_exists_returns_true_for_stored(self, store: ArtifactStore, owner: OwnerScope) -> None:
        content = b"exists test"
        ref = await store.store(content, "text/plain", owner)
        exists = await store.exists(ref.sha256, owner)
        assert exists


# ===========================================================================
# OwnerScope isolation
# ===========================================================================


class TestOwnerScopeIsolation:
    """Artifacts are isolated by OwnerScope."""

    @pytest.mark.asyncio
    async def test_different_owner_cannot_access(self, store: ArtifactStore, owner: OwnerScope, other_owner: OwnerScope) -> None:
        content = b"secret data"
        ref = await store.store(content, "text/plain", owner)
        with pytest.raises(ArtifactNotFound):
            await store.load(ref.sha256, other_owner)

    @pytest.mark.asyncio
    async def test_same_owner_different_workspace_isolation(self, store: ArtifactStore, owner: OwnerScope) -> None:
        ws1 = OwnerScope(owner_id="test-owner", workspace="ws1")
        ws2 = OwnerScope(owner_id="test-owner", workspace="ws2")
        content = b"workspace data"
        ref = await store.store(content, "text/plain", ws1)
        with pytest.raises(ArtifactNotFound):
            await store.load(ref.sha256, ws2)


# ===========================================================================
# Delete
# ===========================================================================


class TestDelete:
    """Artifacts can be deleted."""

    @pytest.mark.asyncio
    async def test_delete_removes_artifact(self, store: ArtifactStore, owner: OwnerScope) -> None:
        content = b"to be deleted"
        ref = await store.store(content, "text/plain", owner)
        assert await store.exists(ref.sha256, owner)
        await store.delete(ref.sha256, owner)
        assert not await store.exists(ref.sha256, owner)

    @pytest.mark.asyncio
    async def test_delete_then_load_raises(self, store: ArtifactStore, owner: OwnerScope) -> None:
        content = b"to be deleted"
        ref = await store.store(content, "text/plain", owner)
        await store.delete(ref.sha256, owner)
        with pytest.raises(ArtifactNotFound):
            await store.load(ref.sha256, owner)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases: concurrent duplicate uploads, zero-byte content."""

    @pytest.mark.asyncio
    async def test_concurrent_duplicate_uploads(
        self, store: ArtifactStore, owner: OwnerScope
    ) -> None:
        """Simulate concurrent duplicate uploads — both should succeed and return same ref."""
        content = b"concurrent content"
        import asyncio

        ref1, ref2 = await asyncio.gather(
            store.store(content, "text/plain", owner),
            store.store(content, "text/plain", owner),
        )
        assert ref1.sha256 == ref2.sha256
        assert ref1.uri == ref2.uri

    @pytest.mark.asyncio
    async def test_zero_byte_artifact(self, store: ArtifactStore, owner: OwnerScope) -> None:
        ref = await store.store(b"", "text/plain", owner)
        loaded = await store.load(ref.sha256, owner)
        assert loaded == b""
        assert ref.size == 0

    @pytest.mark.asyncio
    async def test_large_content(self, store: ArtifactStore, owner: OwnerScope) -> None:
        content = b"x" * 100_000  # 100KB
        ref = await store.store(content, "application/octet-stream", owner)
        loaded = await store.load(ref.sha256, owner)
        assert loaded == content
        assert ref.size == 100_000


# ===========================================================================
# List artifacts
# ===========================================================================


class TestListArtifacts:
    """Artifacts can be listed by owner scope."""

    @pytest.mark.asyncio
    async def test_list_empty(self, store: ArtifactStore, owner: OwnerScope) -> None:
        records = await store.list_artifacts(owner)
        assert records == []

    @pytest.mark.asyncio
    async def test_list_after_store(self, store: ArtifactStore, owner: OwnerScope) -> None:
        await store.store(b"content1", "text/plain", owner)
        await store.store(b"content2", "text/plain", owner)
        records = await store.list_artifacts(owner)
        assert len(records) == 2
        hashes = {r.sha256 for r in records}
        assert len(hashes) == 2
