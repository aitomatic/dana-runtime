"""
Conversation projection — the model-facing view of a Session Journal.

Projects ordered :class:`JournalFact` values into model-facing conversation
context (:class:`ConversationView`): the ordered message history, the decrypted
provider replay state, and an interruption observation when an interrupted turn
is found.

Key rule: partial assistant output from an Interrupted Turn is NOT treated as a
completed response. Only an ``ASSISTANT_CONTENT_FINAL`` fact whose turn is
closed by a matching ``TURN_COMPLETED`` terminal fact becomes an assistant
message. Interrupted turns surface as an observation string instead, and that
observation reflects ONLY the most recent terminated turn — a later
completed/errored/cancelled turn clears it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dana.common.llm.types import LLMMessage
from dana.core.session.models import FactType, JournalFact
from dana.core.session.protected_state import ProtectedStateCodec


# Observation injected into the model context when an interrupted turn is found.
# Unfinished tool outcomes are unknown, so the model must not assume they ran.
_INTERRUPTED_OBSERVATION = "The previous turn was interrupted; do not assume unfinished effects completed."


@dataclass(frozen=True, slots=True)
class ConversationView:
    """Model-facing projection of a Session Journal.

    Attributes:
        messages: Ordered model-facing messages (user always included; assistant
            only from committed turns closed by ``TURN_COMPLETED``).
        replay_state: Decrypted provider replay state from the most recent fact
            carrying a ``protected_payload``. ``None`` when no codec is supplied
            or no protected payload is present.
        interruption_observation: Set when the most recent terminated turn was
            interrupted, so the model knows not to assume unfinished effects
            completed. Cleared by a later completed/errored/cancelled turn.
        last_sequence: Highest fact sequence projected (0 for empty input).
    """

    messages: tuple[LLMMessage, ...]
    replay_state: bytes | None
    interruption_observation: str | None
    last_sequence: int


class ConversationProjector:
    """Pure projector turning ordered Journal Facts into a ConversationView.

    Deterministic given the same ordered facts. The optional
    :class:`ProtectedStateCodec` decrypts the most recent protected payload; if
    decryption fails (e.g. wrong key / tampered blob) the underlying crypto
    error propagates rather than being swallowed.
    """

    def __init__(self, protected_state_codec: ProtectedStateCodec | None = None) -> None:
        self._codec = protected_state_codec

    def project(self, facts: Sequence[JournalFact]) -> ConversationView:
        """Project ordered facts into model-facing conversation context.

        - User messages come from ``USER_CONTENT_FINAL`` (always included).
        - Assistant messages come only from ``ASSISTANT_CONTENT_FINAL`` facts
          whose turn is closed by a matching ``TURN_COMPLETED``.
        - Interrupted turns set ``interruption_observation`` instead of adding
          the partial text as a completed message.
        - ``interruption_observation`` reflects ONLY the most recent terminated
          turn: a later completed/errored/cancelled turn clears it, so a stale
          historical interruption is never wrongly injected on session resume.
        - ``replay_state`` is the decrypted bytes of the most recent
          ``protected_payload``.
        """
        messages: list[LLMMessage] = []
        pending_final: dict[str, str] = {}
        interruption_observation: str | None = None
        last_replay_ciphertext: bytes | None = None
        last_sequence = 0

        for fact in facts:
            if fact.sequence > last_sequence:
                last_sequence = fact.sequence
            if fact.fact_type is FactType.USER_CONTENT_FINAL:
                messages.append(LLMMessage(role="user", content=str(fact.payload["text"])))
            elif fact.fact_type is FactType.ASSISTANT_CONTENT_FINAL:
                pending_final[fact.correlation_id] = str(fact.payload["text"])
            elif fact.fact_type is FactType.TURN_COMPLETED:
                text = pending_final.pop(fact.correlation_id, None)
                if text is not None:
                    messages.append(LLMMessage(role="assistant", content=text))
                interruption_observation = None
            elif fact.fact_type is FactType.TURN_INTERRUPTED:
                interruption_observation = _INTERRUPTED_OBSERVATION
            elif fact.fact_type in (FactType.TURN_ERROR, FactType.TURN_CANCELLED):
                interruption_observation = None
            if fact.protected_payload is not None:
                last_replay_ciphertext = fact.protected_payload

        replay_state: bytes | None = None
        if last_replay_ciphertext is not None and self._codec is not None:
            replay_state = self._codec.decrypt(last_replay_ciphertext)

        return ConversationView(
            messages=tuple(messages),
            replay_state=replay_state,
            interruption_observation=interruption_observation,
            last_sequence=last_sequence,
        )
