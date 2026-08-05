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

D4 adds model-change tracking: ``MODEL_CHANGED`` facts are projected into
``model_changes``, and protected replay state is included only when its
provider matches the current ``provider_key`` (compatibility gating).

D6 adds multimodal content projection: ``USER_CONTENT_FINAL`` facts may carry
``content_blocks`` in their payload (a list of normalized content block dicts).
When present, the user message is projected as an ``LLMMessage`` with
``content: list[ContentBlock]`` instead of a plain string. Assistant messages
remain text-only in D6.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dana.common.llm.types import ContentBlock, LLMMessage
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
            carrying a ``protected_payload`` whose provider matches the current
            ``provider_key``. ``None`` when no codec is supplied, no compatible
            protected payload is present, or the provider key is incompatible.
        interruption_observation: Set when the most recent terminated turn was
            interrupted, so the model knows not to assume unfinished effects
            completed. Cleared by a later completed/errored/cancelled turn.
        last_sequence: Highest fact sequence projected (0 for empty input).
        model_changes: Ordered list of model-change events projected from
            ``MODEL_CHANGED`` facts, each with ``provider``, ``model``,
            ``timestamp``, and ``sequence`` keys.
        current_provider: The provider from the most recent ``MODEL_CHANGED``
            fact, or ``None`` if no model change has occurred.
        current_model: The model from the most recent ``MODEL_CHANGED`` fact,
            or ``None`` if no model change has occurred.
    """

    messages: tuple[LLMMessage, ...]
    replay_state: bytes | None
    interruption_observation: str | None
    last_sequence: int
    model_changes: tuple[dict, ...] = ()
    current_provider: str | None = None
    current_model: str | None = None


class ConversationProjector:
    """Pure projector turning ordered Journal Facts into a ConversationView.

    Deterministic given the same ordered facts. The optional
    :class:`ProtectedStateCodec` decrypts the most recent protected payload; if
    decryption fails (e.g. wrong key / tampered blob) the underlying crypto
    error propagates rather than being swallowed.

    D4: When ``provider_key`` is provided, protected replay state is included
    only when its fact's provider matches the current provider (compatibility
    gating per ADR-007/ADR-010). ``MODEL_CHANGED`` facts are tracked to
    determine the current provider for each fact.
    """

    def __init__(self, protected_state_codec: ProtectedStateCodec | None = None) -> None:
        self._codec = protected_state_codec

    def project(self, facts: Sequence[JournalFact], provider_key: str | None = None) -> ConversationView:
        """Project ordered facts into model-facing conversation context.

        Args:
            facts: Ordered journal facts to project.
            provider_key: The current provider key for compatibility gating.
                When set, only protected payloads from facts whose provider
                matches this key are included in ``replay_state``.

        - User messages come from ``USER_CONTENT_FINAL`` (always included).
          D6: when the payload carries ``content_blocks`` (a list of normalized
          content block dicts), the user message is projected as
          ``content: list[ContentBlock]`` instead of a plain string.
        - Assistant messages come only from ``ASSISTANT_CONTENT_FINAL`` facts
          whose turn is closed by a matching ``TURN_COMPLETED``.
        - Interrupted turns set ``interruption_observation`` instead of adding
          the partial text as a completed message.
        - ``interruption_observation`` reflects ONLY the most recent terminated
          turn: a later completed/errored/cancelled turn clears it, so a stale
          historical interruption is never wrongly injected on session resume.
        - ``replay_state`` is the decrypted bytes of the most recent compatible
          ``protected_payload`` (compatibility gated by ``provider_key``).
        - ``model_changes`` collects every ``MODEL_CHANGED`` fact in order.
        """
        messages: list[LLMMessage] = []
        pending_final: dict[str, str] = {}
        interruption_observation: str | None = None
        last_replay_ciphertext: bytes | None = None
        last_sequence = 0
        model_changes: list[dict] = []
        current_provider: str | None = None
        current_model: str | None = None

        for fact in facts:
            if fact.sequence > last_sequence:
                last_sequence = fact.sequence

            if fact.fact_type is FactType.USER_CONTENT_FINAL:
                content_blocks = fact.payload.get("content_blocks")
                if content_blocks is not None and isinstance(content_blocks, list) and len(content_blocks) > 0:
                    # Multimodal: project as list[ContentBlock]
                    projected_blocks: list[ContentBlock] = []
                    for cb in content_blocks:
                        if isinstance(cb, dict):
                            projected_blocks.append(cb)  # type: ignore[arg-type]
                    if projected_blocks:
                        messages.append(LLMMessage(role="user", content=projected_blocks))
                    else:
                        messages.append(LLMMessage(role="user", content=str(fact.payload.get("text", ""))))
                else:
                    messages.append(LLMMessage(role="user", content=str(fact.payload.get("text", ""))))
            elif fact.fact_type is FactType.ASSISTANT_CONTENT_FINAL:
                pending_final[fact.correlation_id] = str(fact.payload.get("text", ""))
            elif fact.fact_type is FactType.TURN_COMPLETED:
                text = pending_final.pop(fact.correlation_id, None)
                if text is not None:
                    messages.append(LLMMessage(role="assistant", content=text))
                interruption_observation = None
            elif fact.fact_type is FactType.TURN_INTERRUPTED:
                interruption_observation = _INTERRUPTED_OBSERVATION
            elif fact.fact_type in (FactType.TURN_ERROR, FactType.TURN_CANCELLED):
                interruption_observation = None
            elif fact.fact_type is FactType.MODEL_CHANGED:
                provider = str(fact.payload.get("provider", ""))
                model = str(fact.payload.get("model", ""))
                current_provider = provider
                current_model = model
                model_changes.append(
                    {
                        "provider": provider,
                        "model": model,
                        "timestamp": fact.timestamp.isoformat(),
                        "sequence": fact.sequence,
                    }
                )

            # Track protected payload only when compatible with provider_key.
            if fact.protected_payload is not None:
                if provider_key is None or current_provider == provider_key:
                    last_replay_ciphertext = fact.protected_payload

        replay_state: bytes | None = None
        if last_replay_ciphertext is not None and self._codec is not None:
            replay_state = self._codec.decrypt(last_replay_ciphertext)

        return ConversationView(
            messages=tuple(messages),
            replay_state=replay_state,
            interruption_observation=interruption_observation,
            last_sequence=last_sequence,
            model_changes=tuple(model_changes),
            current_provider=current_provider,
            current_model=current_model,
        )
