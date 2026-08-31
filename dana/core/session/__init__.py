"""Session Journal package — durable, owner-scoped facts for Dana agent sessions.

Flat re-exports (D8): hosts can build a hello-world session from this package
alone — ``AgentSession``, ``TextBlock``, ``FactType``, ``JournalFact`` — plus
``SessionRecord`` for direct journal access. Deep import paths
(``dana.core.session.agent_session``, ``...models``, ``...journal.models``)
remain the source of truth and keep working.
"""

from __future__ import annotations

from dana.core.session.agent_session import AgentSession, SessionBusy, TextBlock, TurnTerminal
from dana.core.session.journal.models import SessionRecord
from dana.core.session.models import (
    ArtifactRef,
    FactType,
    JournalFact,
    JSONValue,
    NewJournalFact,
    OwnerScope,
    PayloadSanitizationError,
    validate_payload,
)
from dana.core.session.protected_state import (
    EnvProtectedStateKeyProvider,
    ProtectedStateCodec,
    ProtectedStateKeyProvider,
    ProtectedStateKeyUnavailable,
)


__all__ = [
    "AgentSession",
    "ArtifactRef",
    "EnvProtectedStateKeyProvider",
    "FactType",
    "JournalFact",
    "JSONValue",
    "NewJournalFact",
    "OwnerScope",
    "PayloadSanitizationError",
    "ProtectedStateCodec",
    "ProtectedStateKeyProvider",
    "ProtectedStateKeyUnavailable",
    "SessionBusy",
    "SessionRecord",
    "TextBlock",
    "TurnTerminal",
    "validate_payload",
]
