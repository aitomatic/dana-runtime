"""
Session Journal models — the durable data layer for Dana agent sessions.

The Session Journal is the sole durable authority for session history. Each
:class:`JournalFact` is an immutable, typed, and ordered statement about session
activity. Provider Replay State (e.g. OpenAI ``encrypted_content``, reasoning
items) required for continuity is never placed in the regular ``payload``; it is
envelope-encrypted and carried only in ``protected_payload``.

This module is limited to the D1 (text-only conversation) fact set plus the
LEGACY_TIMELINE_MIGRATED import marker. Tool, permission, model, and MCP fact
types belong to later phases.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
from typing import Union


# Recursive JSON-safe value alias. A payload is a mapping from str keys to
# values drawn only from this type.
JSONValue = Union[None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]]

# Best-effort denylist of secret-bearing substrings. Provider replay material
# and credentials must travel ONLY in the encrypted protected_payload; this list
# is a defense-in-depth check, NOT a hard guarantee. The PRIMARY guarantee is
# that provider material is carried in protected_payload. Matching is
# case-insensitive and ignores underscores/hyphens, so variants like "apikey",
# "api_keys", "API_KEY", and "my-api-key" are all caught.
_FORBIDDEN_PAYLOAD_SUBSTRINGS = frozenset(
    {
        "encryptedcontent",
        "apikey",
        "accesskey",
        "secret",
        "password",
        "passphrase",
        "token",
        "privatekey",
        "bearer",
        "credential",
        "authorization",
    }
)


def _is_forbidden_key(key: str) -> bool:
    normalized = key.lower().replace("_", "").replace("-", "")
    return any(term in normalized for term in _FORBIDDEN_PAYLOAD_SUBSTRINGS)


class PayloadSanitizationError(ValueError):
    """Raised when a payload contains non-JSON-safe values or forbidden secret-bearing keys."""


class FactType(Enum):
    """Typed statements about session activity (D1 text-only conversation set + D2 tool lifecycle)."""

    # D1: Text-only conversation
    SESSION_CREATED = "session_created"
    SESSION_LOADED = "session_loaded"
    SESSION_RESUMED = "session_resumed"
    TURN_STARTED = "turn_started"
    USER_CONTENT_FINAL = "user_content_final"
    ASSISTANT_CONTENT_CHUNK = "assistant_content_chunk"
    ASSISTANT_CONTENT_FINAL = "assistant_content_final"
    TURN_COMPLETED = "turn_completed"
    TURN_INTERRUPTED = "turn_interrupted"
    TURN_ERROR = "turn_error"
    TURN_CANCELLED = "turn_cancelled"
    LEGACY_TIMELINE_MIGRATED = "legacy_timeline_migrated"

    # D4: Model change fact (ADR-002, ADR-007)
    MODEL_CHANGED = "model_changed"

    # D2: Tool lifecycle facts (ADR-002, ADR-005)
    # Non-terminal facts
    TOOL_REQUESTED = "tool_requested"
    TOOL_AUTHORIZED_OR_DENIED = "tool_authorized_or_denied"
    TOOL_STARTED = "tool_started"
    TOOL_PROGRESS = "tool_progress"
    TOOL_CANCELLATION_REQUESTED = "tool_cancellation_requested"
    # Terminal facts — exactly one per tool call
    TOOL_RESULT = "tool_result"
    TOOL_FAILURE = "tool_failure"
    TOOL_ACKNOWLEDGED = "tool_acknowledged"
    TOOL_TIMED_OUT = "tool_timed_out"
    TOOL_EFFECT_UNKNOWN = "tool_effect_unknown"


def validate_payload(payload: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
    """Validate that a payload contains only JSON-safe values and no secret-bearing keys.

    Returns the payload unchanged on success. Raises :class:`PayloadSanitizationError`
    if any value is not JSON-safe (including NaN/Infinity), any dict key is not a
    string, or any key matches the secret-bearing denylist (case-insensitive,
    underscore/hyphen-insensitive substring match).
    """
    _validate_json_safe(payload, "payload")
    return payload


def _validate_json_safe(value: object, path: str) -> None:
    # bool is a subclass of int; the combined isinstance covers both correctly.
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise PayloadSanitizationError(f"{path}: float NaN/Infinity is not JSON-safe")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _validate_json_safe(item, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise PayloadSanitizationError(f"{path}: dict key {k!r} must be a string")
            if _is_forbidden_key(k):
                raise PayloadSanitizationError(
                    f"{path}.{k}: secret-bearing key {k!r} is forbidden in payload; use protected_payload for protected material"
                )
            _validate_json_safe(v, f"{path}.{k}")
        return
    raise PayloadSanitizationError(f"{path}: value of type {type(value).__name__} is not JSON-safe")


@dataclass(frozen=True, slots=True)
class OwnerScope:
    """The immutable tenant or principal scope that owns a Session Journal."""

    owner_id: str
    workspace: str

    def __post_init__(self) -> None:
        if not self.owner_id:
            raise ValueError("OwnerScope.owner_id must be a non-empty string")
        if not self.workspace:
            raise ValueError("OwnerScope.workspace must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Reference to a large payload retained outside the Session Journal."""

    uri: str
    media_type: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("ArtifactRef.uri must be a non-empty string")
        if not self.media_type:
            raise ValueError("ArtifactRef.media_type must be a non-empty string")
        if self.size < 0:
            raise ValueError("ArtifactRef.size must be non-negative")
        if not self.sha256:
            raise ValueError("ArtifactRef.sha256 must be a non-empty string")


@dataclass(frozen=True, slots=True)
class JournalFact:
    """The durable, stored form of a Journal Fact after persistence assigns identity."""

    fact_id: str
    owner_scope: OwnerScope
    session_id: str
    sequence: int
    fact_type: FactType
    timestamp: datetime
    correlation_id: str
    causation_id: str | None
    schema_version: int
    payload: Mapping[str, JSONValue]
    protected_payload: bytes | None = None
    artifact_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.fact_id:
            raise ValueError("JournalFact.fact_id must be a non-empty string")
        if not isinstance(self.owner_scope, OwnerScope):
            raise ValueError("JournalFact.owner_scope must be an OwnerScope")
        if not self.session_id:
            raise ValueError("JournalFact.session_id must be a non-empty string")
        if self.sequence < 1:
            raise ValueError("JournalFact.sequence must be >= 1")
        if not self.correlation_id:
            raise ValueError("JournalFact.correlation_id must be a non-empty string")
        if self.schema_version < 1:
            raise ValueError("JournalFact.schema_version must be >= 1")
        validate_payload(self.payload)


@dataclass(frozen=True, slots=True)
class NewJournalFact:
    """The input form of a Journal Fact, before persistence assigns identity/sequence."""

    fact_type: FactType
    correlation_id: str
    causation_id: str | None
    payload: Mapping[str, JSONValue]
    protected_payload: bytes | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.correlation_id:
            raise ValueError("NewJournalFact.correlation_id must be a non-empty string")
        if self.schema_version < 1:
            raise ValueError("NewJournalFact.schema_version must be >= 1")
        validate_payload(self.payload)
