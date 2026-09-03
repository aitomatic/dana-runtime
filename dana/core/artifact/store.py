"""
Authorized artifact store — hash-based deduplication with OwnerScope isolation.

Per ADR-009 (Multimodal Content and Artifact References):
- Large payloads are artifact references (hash/URI/media-type/size/access-metadata)
- Hash-based dedup: same content produces the same artifact reference
- Missing artifacts fail explicitly

Per ADR-003 (Dual SQLite PostgreSQL Journal Adapters):
- ``OwnerScope`` required at the artifact-store boundary
- Artifact access is scoped
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os

from dana.core.session.models import ArtifactRef, OwnerScope


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ArtifactError(Exception):
    """Base exception for artifact store failures."""


class ArtifactNotFound(ArtifactError):
    """Raised when an artifact is not found in the store."""

    def __init__(self, sha256: str, owner_scope: OwnerScope) -> None:
        self.sha256 = sha256
        self.owner_scope = owner_scope
        super().__init__(
            f"artifact with hash {sha256!r} not found for owner "
            f"{owner_scope.owner_id!r}/{owner_scope.workspace!r}"
        )


class ArtifactStoreError(ArtifactError):
    """Raised on storage backend failures."""


# ---------------------------------------------------------------------------
# Artifact record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """A stored artifact with metadata."""

    sha256: str
    owner_scope: OwnerScope
    media_type: str
    size: int
    storage_path: str
    created_at: str  # ISO-8601 timestamp


# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------


class ArtifactStore:
    """Authorized, owner-scoped artifact store with hash-based deduplication.

    Stores artifacts on the local filesystem under a configurable base path.
    Each artifact is stored at ``{base_path}/{owner_id}/{workspace}/{sha256[:2]}/{sha256[2:4]}/{sha256}``
    to avoid directory fan-out issues.

    Usage::

        store = ArtifactStore(base_path="/tmp/dana/artifacts")
        ref = await store.store(b"large content", "image/png", owner_scope)
        data = await store.load(ref.sha256, owner_scope)
    """

    def __init__(self, base_path: str | None = None) -> None:
        self._base_path = base_path or os.environ.get(
            "DANA_ARTIFACT_STORE_PATH",
            os.path.expanduser("~/.dana/artifacts"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def store(
        self,
        content: bytes,
        media_type: str,
        owner_scope: OwnerScope,
    ) -> ArtifactRef:
        """Store content and return an artifact reference.

        If content with the same SHA-256 hash already exists for this owner
        scope, returns the existing reference (dedup).

        Args:
            content: The raw bytes to store.
            media_type: The MIME type of the content.
            owner_scope: The owner scope for access isolation.

        Returns:
            An ``ArtifactRef`` with URI, media type, size, and SHA-256 hash.

        Raises:
            ArtifactStoreError: On storage backend failures.
        """
        sha256 = hashlib.sha256(content).hexdigest()
        size = len(content)
        storage_path = self._storage_path(sha256, owner_scope)

        # Check for existing artifact (dedup)
        if os.path.isfile(storage_path):
            existing_size = os.path.getsize(storage_path)
            if existing_size == size:
                return ArtifactRef(
                    uri=f"artifact://{sha256}",
                    media_type=media_type,
                    size=size,
                    sha256=sha256,
                )

        # Store the content
        try:
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            with open(storage_path, "wb") as f:
                f.write(content)
        except OSError as exc:
            raise ArtifactStoreError(f"failed to store artifact: {exc}") from exc

        return ArtifactRef(
            uri=f"artifact://{sha256}",
            media_type=media_type,
            size=size,
            sha256=sha256,
        )

    async def load(self, sha256: str, owner_scope: OwnerScope) -> bytes:
        """Load artifact content by hash.

        Args:
            sha256: The SHA-256 hash of the artifact.
            owner_scope: The owner scope for access isolation.

        Returns:
            The raw bytes of the artifact.

        Raises:
            ArtifactNotFound: If the artifact does not exist.
            ArtifactStoreError: On storage backend failures.
        """
        storage_path = self._storage_path(sha256, owner_scope)
        if not os.path.isfile(storage_path):
            raise ArtifactNotFound(sha256, owner_scope)

        try:
            with open(storage_path, "rb") as f:
                return f.read()
        except OSError as exc:
            raise ArtifactStoreError(f"failed to load artifact: {exc}") from exc

    async def delete(self, sha256: str, owner_scope: OwnerScope) -> None:
        """Delete an artifact by hash.

        Args:
            sha256: The SHA-256 hash of the artifact.
            owner_scope: The owner scope for access isolation.

        Raises:
            ArtifactNotFound: If the artifact does not exist.
        """
        storage_path = self._storage_path(sha256, owner_scope)
        if not os.path.isfile(storage_path):
            raise ArtifactNotFound(sha256, owner_scope)

        try:
            os.remove(storage_path)
        except OSError as exc:
            raise ArtifactStoreError(f"failed to delete artifact: {exc}") from exc

    async def exists(self, sha256: str, owner_scope: OwnerScope) -> bool:
        """Check if an artifact exists in the store.

        Args:
            sha256: The SHA-256 hash of the artifact.
            owner_scope: The owner scope for access isolation.

        Returns:
            ``True`` if the artifact exists, ``False`` otherwise.
        """
        return os.path.isfile(self._storage_path(sha256, owner_scope))

    async def list_artifacts(self, owner_scope: OwnerScope) -> list[ArtifactRecord]:
        """List all artifacts for a given owner scope.

        Args:
            owner_scope: The owner scope to list artifacts for.

        Returns:
            A list of ``ArtifactRecord`` instances.
        """
        scope_dir = self._scope_dir(owner_scope)
        if not os.path.isdir(scope_dir):
            return []

        records: list[ArtifactRecord] = []
        for root, _dirs, files in os.walk(scope_dir):
            for filename in files:
                if len(filename) == 64:  # SHA-256 hex digest
                    filepath = os.path.join(root, filename)
                    try:
                        stat = os.stat(filepath)
                        records.append(
                            ArtifactRecord(
                                sha256=filename,
                                owner_scope=owner_scope,
                                media_type="application/octet-stream",
                                size=stat.st_size,
                                storage_path=filepath,
                                created_at="",
                            )
                        )
                    except OSError:
                        continue
        return records

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _storage_path(self, sha256: str, owner_scope: OwnerScope) -> str:
        """Compute the on-disk storage path for an artifact.

        Uses a two-level directory prefix to avoid filesystem fan-out:
        ``{base}/{owner_id}/{workspace}/{sha256[:2]}/{sha256[2:4]}/{sha256}``
        """
        return os.path.join(
            self._base_path,
            owner_scope.owner_id,
            owner_scope.workspace,
            sha256[:2],
            sha256[2:4],
            sha256,
        )

    def _scope_dir(self, owner_scope: OwnerScope) -> str:
        """Compute the scope-level directory for listing."""
        return os.path.join(
            self._base_path,
            owner_scope.owner_id,
            owner_scope.workspace,
        )
