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

The fold itself lives in :class:`ConversationProjectionState` so a session can
fold only the facts appended since its last cursor instead of re-reading and
re-projecting the whole journal. :meth:`ConversationProjector.project` is the
fold-all convenience wrapper over the same state, so there is exactly ONE fold
implementation. The state carries every piece of cross-fact bookkeeping
(pending final text, pending native envelopes, tool requests awaiting their
terminal fact, turn start indices, interruption observation, replay ciphertext,
model changes), so a batch boundary is never observable in the projected view.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any, cast

from dana.common.llm.types import ContentBlock, LLMMessage
from dana.core.session.models import FactType, JournalFact, JSONValue
from dana.core.session.protected_state import ProtectedStateCodec
from dana.core.timeline.native_message import NativeMessage, NativeMessageRole, NativeToolCall


# Observation injected into the model context when an interrupted turn is found.
# Unfinished tool outcomes are unknown, so the model must not assume they ran.
_INTERRUPTED_OBSERVATION = "The previous turn was interrupted; do not assume unfinished effects completed."

# Schema tag for the serialized fold state. Bump on any backwards-incompatible
# change to the serialized shape; readers MUST reject unknown tags (and fall back
# to a full re-fold) rather than guess.
CONVERSATION_STATE_SCHEMA = "dana.conversation-projection/1"

# Turn-scoped facts that open a turn (it has not reached a terminal yet), and the
# terminal facts that close one. Kept in the projection (not imported from
# ``legacy_timeline_migration``) so this module has no dependency on the
# migration layer.
_TURN_OPEN_FACT_TYPES = frozenset(
    {
        FactType.TURN_STARTED,
        FactType.USER_CONTENT_FINAL,
        FactType.ASSISTANT_CONTENT_CHUNK,
        FactType.ASSISTANT_CONTENT_FINAL,
        FactType.TOOL_REQUESTED,
        FactType.TOOL_AUTHORIZED_OR_DENIED,
        FactType.TOOL_STARTED,
        FactType.TOOL_PROGRESS,
        FactType.TOOL_CANCELLATION_REQUESTED,
        FactType.TOOL_RESULT,
        FactType.TOOL_FAILURE,
        FactType.TOOL_ACKNOWLEDGED,
        FactType.TOOL_TIMED_OUT,
        FactType.TOOL_EFFECT_UNKNOWN,
    }
)
_TURN_TERMINAL_FACT_TYPES = frozenset(
    {
        FactType.TURN_COMPLETED,
        FactType.TURN_INTERRUPTED,
        FactType.TURN_ERROR,
        FactType.TURN_CANCELLED,
    }
)


class ConversationProviderMismatch(ValueError):
    """Raised when a fold state is viewed under a provider key it was not folded for.

    Provider gating is applied while folding (a gated-out native envelope falls
    back to the pending final text), so the projection cannot be re-gated after
    the fact. A provider switch MUST rebuild the state instead.
    """


class ConversationPrunedStateError(RuntimeError):
    """Raised when a full :meth:`ConversationProjectionState.view` is asked of a pruned state.

    :meth:`ConversationProjectionState.discard_closed_history` intentionally
    forgets the messages the caller now retains in its own working context. A
    "full view" would silently be a tail, so the state refuses instead of
    returning a plausible-looking partial conversation.
    """


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


def _replay_metadata(message: LLMMessage) -> dict[str, Any]:
    """Provider replay carriers of an assistant message, in NativeMessage metadata form."""
    if not message.reasoning_items:
        return {}
    return {
        "reasoning_items": list(message.reasoning_items),
        "fingerprint": message.reasoning_fingerprint,
        "response_id": message.response_id,
    }


def _message_to_dict(message: LLMMessage) -> dict[str, JSONValue]:
    """Serialize an :class:`LLMMessage` through the canonical ``NativeMessage`` codec.

    Reuses the same (role, content, tool_calls, tool_call_id, replay metadata)
    round trip that ``_populate_timeline`` consumes, so multimodal list content
    and reasoning carriers survive verbatim. The NativeMessage ``timestamp`` is
    dropped: the fold never derives one, and stamping ``now()`` would make the
    serialized state non-deterministic.
    """
    native = NativeMessage(
        role=cast("NativeMessageRole", message.role),
        content=cast("str | list[dict[str, Any]]", message.content),
        tool_calls=[NativeToolCall.from_dict(call) for call in message.tool_calls] if message.tool_calls else None,
        tool_call_id=message.tool_call_id,
        metadata=_replay_metadata(message),
    )
    encoded = native.to_dict()
    encoded.pop("timestamp", None)
    return cast("dict[str, JSONValue]", encoded)


def _message_from_dict(data: Mapping[str, JSONValue]) -> LLMMessage:
    """Inverse of :func:`_message_to_dict`."""
    return NativeMessage.from_dict(dict(data)).to_llm_message()


def _require_message(item: JSONValue, where: str) -> LLMMessage:
    """Decode one serialized message, normalizing any failure to ``ValueError``.

    ``from_dict`` promises ``ValueError`` for a malformed payload so callers can
    fall back to a full re-fold on a single except clause.
    """
    if not isinstance(item, dict):
        raise ValueError(f"conversation projection state {where} entries must be objects")
    try:
        return _message_from_dict(item)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"conversation projection state {where} entry is malformed: {exc}") from exc


def _require_list(data: Mapping[str, JSONValue], key: str) -> list[JSONValue]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"conversation projection state field {key!r} must be a list")
    return value


def _require_mapping(data: Mapping[str, JSONValue], key: str) -> dict[str, JSONValue]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"conversation projection state field {key!r} must be an object")
    return value


def _require_int(data: Mapping[str, JSONValue], key: str) -> int:
    value = data.get(key, 0)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"conversation projection state field {key!r} must be an integer")
    return value


def _optional_str(data: Mapping[str, JSONValue], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"conversation projection state field {key!r} must be a string or null")
    return value


class ConversationProjectionState:
    """Incremental fold state for the Conversation projection.

    One canonical fold: :meth:`apply` consumes the next batch of ordered facts
    and updates the state in place; :meth:`view` materializes the model-facing
    :class:`ConversationView` from it. ``projector.project(facts)`` is
    ``new_state(); apply(facts); view()``.

    Batch boundaries are invisible: everything the fold carries between facts
    (pending final text, pending native envelopes, unresolved tool requests,
    turn start indices, interruption observation, replay ciphertext, model
    changes) is state, so ``apply([])`` is a no-op and splitting a fact stream
    into any number of batches yields the same view as folding it in one call.

    The state is O(history) by construction (it holds the projected messages and
    the per-correlation bookkeeping), exactly like the historical one-shot
    projection. Bounding the working set stays compaction's job; this state is
    not a substitute for it. ``view()`` shares the message objects with the
    state, so callers MUST treat the returned tuple as read-only.

    Serialization (:meth:`to_dict` / :meth:`from_dict`) is a deterministic JSON
    -safe mapping tagged :data:`CONVERSATION_STATE_SCHEMA`. It contains the
    pending state, the projected messages and the *ciphertext* of the replay
    payload - never decrypted provider replay. Assistant messages do carry their
    ``reasoning_items`` (needed to reproduce the model context), which can embed
    provider ``encrypted_content``: the caller MUST protect the serialized state
    with :class:`ProtectedStateCodec` before persisting it, exactly as it does
    for the journal's ``protected_payload``.
    """

    __slots__ = (
        "_codec",
        "_provider_key",
        "_messages",
        "_message_turns",
        "_turn_message_counts",
        "_pending_final",
        "_turn_starts",
        "_pending_native",
        "_tool_requests",
        "_open_turns",
        "_discarded_prefix",
        "_rewrite_revision",
        "_interruption_observation",
        "_last_replay_ciphertext",
        "_model_changes",
        "_current_provider",
        "_current_model",
        "_last_sequence",
    )

    def __init__(self, protected_state_codec: ProtectedStateCodec | None = None, provider_key: str | None = None) -> None:
        self._codec = protected_state_codec
        self._provider_key = provider_key
        self._messages: list[LLMMessage] = []
        self._message_turns: list[str] = []
        # Live message count per correlation id. Invariant: counts[c] equals the
        # number of entries in _message_turns equal to c. Maintained in O(1) so
        # the native-turn replacement can prove a suffix scan saw every message
        # of the turn without scanning the whole owner vector.
        self._turn_message_counts: dict[str, int] = {}
        self._pending_final: dict[str, str] = {}
        self._turn_starts: dict[str, int] = {}
        self._pending_native: dict[str, list[LLMMessage]] = {}
        self._tool_requests: dict[tuple[str, str], Mapping[str, JSONValue]] = {}
        # Turns folded but not yet terminated (TURN_STARTED / USER_CONTENT_FINAL
        # seen, no TURN_COMPLETED/INTERRUPTED/ERROR/CANCELLED yet). Distinct from
        # _turn_starts, which maps every *open* turn (and, transiently, the
        # terminal turn being folded in this same batch - ``_retire_unfinished_turn``
        # drops it as soon as that turn reaches its terminal) to where its
        # messages begin.
        self._open_turns: set[str] = set()
        # Number of leading messages dropped by discard_closed_history. Kept so
        # index-based canonical positions stay meaningful across a prune: the
        # canonical message at index i is messages_after(i - discarded_prefix)[0].
        self._discarded_prefix = 0
        # Ephemeral restructure counter (see the rewrite_revision property); never
        # serialized, always restarts at 0.
        self._rewrite_revision = 0
        self._interruption_observation: str | None = None
        self._last_replay_ciphertext: bytes | None = None
        self._model_changes: list[dict[str, JSONValue]] = []
        self._current_provider: str | None = None
        self._current_model: str | None = None
        self._last_sequence = 0

    # ------------------------------------------------------------------
    # Folding
    # ------------------------------------------------------------------

    @property
    def last_sequence(self) -> int:
        """Highest fact sequence folded so far (the resume cursor)."""
        return self._last_sequence

    @property
    def provider_key(self) -> str | None:
        """Provider key this state was folded under (gating identity)."""
        return self._provider_key

    @property
    def message_count(self) -> int:
        """Canonical number of projected messages (O(1)); the prefix boundary.

        Counts messages discarded by :meth:`discard_closed_history` too, so it is
        always the canonical history size, not the retained-window size.
        """
        return self._discarded_prefix + len(self._messages)

    @property
    def retained_message_count(self) -> int:
        """Messages still held in memory (canonical count minus the pruned prefix)."""
        return len(self._messages)

    @property
    def discarded_prefix(self) -> int:
        """Number of leading canonical messages dropped by :meth:`discard_closed_history`."""
        return self._discarded_prefix

    @property
    def rewrite_revision(self) -> int:
        """Monotonic counter of restructures of messages that were already projected.

        Bumped by :meth:`discard_closed_history` and by any :meth:`apply` batch
        that removes or replaces messages that already existed *before that
        batch*. A pure append batch never bumps it - including a batch that
        projects a turn's user/tool messages and then replaces them with that
        turn's native envelope at ``TURN_COMPLETED``, because those messages were
        created inside the same batch and the caller applies the whole batch as
        tail anyway.

        A consumer that mirrors this state into its own working context can use
        it as an O(delta) invalidation signal: an unchanged revision means every
        message below the previous boundary is still exactly the canonical
        prefix, so only ``messages_after(previous_count)`` needs applying; a
        changed revision means the mirrored context must be rebuilt. Call
        :meth:`apply` once per batch for that property to hold.

        Ephemeral by design: it is not serialized, and a state restored with
        :meth:`from_dict` starts at 0 (a restored state has no mirrored context
        to invalidate).
        """
        return self._rewrite_revision

    @property
    def interruption_observation(self) -> str | None:
        """Current interruption observation without materializing the full view."""
        return self._interruption_observation

    @property
    def current_provider(self) -> str | None:
        """Provider of the most recent ``MODEL_CHANGED`` fact, without a full view."""
        return self._current_provider

    @property
    def current_model(self) -> str | None:
        """Model of the most recent ``MODEL_CHANGED`` fact, without a full view."""
        return self._current_model

    @property
    def open_correlation_ids(self) -> tuple[str, ...]:
        """Correlation ids of turns that have not reached a terminal fact yet.

        Includes turns whose only fact so far is ``TURN_STARTED`` or
        ``USER_CONTENT_FINAL`` - a user-only in-flight turn is still open, and a
        later native envelope for it can replace messages at or below the
        current tail. A prefix boundary captured while this is non-empty is not
        final.
        """
        return tuple(sorted(self._open_turns))

    def messages_after(self, index: int) -> tuple[LLMMessage, ...]:
        """Copy the canonical messages from ``index`` onwards (bounded suffix).

        ``index`` is a canonical position - the same one :attr:`message_count`
        reports - so it stays valid after :meth:`discard_closed_history`. Indexes
        inside the pruned prefix return the retained tail only: the discarded
        messages no longer exist and the caller is expected to hold them in its
        own working context.

        The returned tuple shares the message objects with the state; treat it
        as read-only.
        """
        start = index - self._discarded_prefix
        if start < 0:
            start = 0
        return tuple(self._messages[start:])

    def apply(self, delta: Sequence[JournalFact]) -> int:
        """Fold the next ordered batch of facts; returns the new cursor.

        Fold facts as one batch when the caller mirrors this state into a
        working context: :attr:`rewrite_revision` only stays unchanged for
        batches that append after the pre-batch prefix (see its docstring).
        """
        pre_batch_count = len(self._messages)
        for fact in delta:
            self._apply_fact(fact, pre_batch_count)
        return self._last_sequence

    def discard_closed_history(self) -> int:
        """Drop the retained messages and their per-turn bookkeeping.

        Optional bounded-state mode for a session that keeps its model-facing
        working context elsewhere (e.g. a retained, compacted native timeline)
        and persists a checkpoint carrying *that* context plus this cursor. After
        the call the state holds no messages of its own: the next :meth:`apply`
        folds only new facts, so :meth:`to_dict` stays bounded instead of
        re-serializing the whole canonical history on every save.

        The drop is a derived-view retention only - the Session Journal is
        untouched and the cursor, provider key, replay ciphertext, model changes
        and interruption observation are all kept, so folding the tail still
        yields exactly the canonical tail.

        Raises :class:`ValueError` when any turn is still open
        (:attr:`open_correlation_ids`): an un-terminated turn can still rewrite
        messages retroactively, so its messages must not be forgotten.

        :meth:`message_count` keeps counting the dropped messages (via
        :attr:`discarded_prefix`), so a caller that recorded a boundary before
        the prune can still use that same index with :meth:`messages_after`.
        """
        if self._open_turns:
            raise ValueError(f"cannot discard history while turns are open: {sorted(self._open_turns)!r}")
        discarded = len(self._messages)
        if discarded:
            self._rewrite_revision += 1
        self._messages = []
        self._message_turns = []
        self._turn_message_counts = {}
        self._turn_starts = {}
        self._discarded_prefix += discarded
        # Only the most recent model change still matters for provider gating;
        # keeping the full event log would grow without bound while contributing
        # nothing to the canonical tail fold.
        if self._model_changes:
            self._model_changes = self._model_changes[-1:]
        return discarded

    def _apply_fact(self, fact: JournalFact, pre_batch_count: int) -> None:
        if fact.sequence > self._last_sequence:
            self._last_sequence = fact.sequence

        fact_type = fact.fact_type
        # Open-turn bookkeeping: a turn is open from its first non-terminal fact
        # until its terminal fact. A user-only in-flight turn is open too.
        if fact_type in _TURN_OPEN_FACT_TYPES:
            self._open_turns.add(fact.correlation_id)
        elif fact_type in _TURN_TERMINAL_FACT_TYPES:
            self._open_turns.discard(fact.correlation_id)

        if fact.fact_type is FactType.USER_CONTENT_FINAL:
            self._turn_starts[fact.correlation_id] = len(self._messages)
            content_blocks = fact.payload.get("content_blocks")
            if content_blocks is not None and isinstance(content_blocks, list) and len(content_blocks) > 0:
                # Multimodal: project as list[ContentBlock]
                projected_blocks: list[ContentBlock] = []
                for cb in content_blocks:
                    if isinstance(cb, dict):
                        projected_blocks.append(cb)  # type: ignore[arg-type]
                if projected_blocks:
                    self._messages.append(LLMMessage(role="user", content=projected_blocks))
                else:
                    self._messages.append(LLMMessage(role="user", content=str(fact.payload.get("text", ""))))
            else:
                self._messages.append(LLMMessage(role="user", content=str(fact.payload.get("text", ""))))
        elif fact.fact_type is FactType.ASSISTANT_CONTENT_FINAL:
            self._pending_final[fact.correlation_id] = str(fact.payload.get("text", ""))
        elif fact.fact_type is FactType.TOOL_REQUESTED:
            call_id = str(fact.payload.get("tool_call_id", ""))
            self._tool_requests[(fact.correlation_id, call_id)] = fact.payload
        elif fact.fact_type in (
            FactType.TOOL_RESULT,
            FactType.TOOL_FAILURE,
            FactType.TOOL_ACKNOWLEDGED,
            FactType.TOOL_TIMED_OUT,
            FactType.TOOL_EFFECT_UNKNOWN,
        ):
            call_id = str(fact.payload.get("tool_call_id", ""))
            request = self._tool_requests.pop((fact.correlation_id, call_id), None)
            if request is not None:
                self._messages.append(
                    LLMMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            {
                                "id": call_id,
                                "name": request["tool_name"],
                                "arguments": request.get("arguments", {}),
                            }
                        ],
                    )
                )
                result = fact.payload.get("result")
                if result is None:
                    result = {"status": fact.fact_type.value, "error": fact.payload.get("error")}
                self._messages.append(LLMMessage(role="tool", content=json.dumps(result, ensure_ascii=False), tool_call_id=call_id))
        elif fact.fact_type is FactType.TURN_COMPLETED:
            native = self._pending_native.pop(fact.correlation_id, None)
            start = self._turn_starts.get(fact.correlation_id)
            text = self._pending_final.pop(fact.correlation_id, None)
            if native is not None and start is not None:
                self._commit_native_turn(fact.correlation_id, native, start, pre_batch_count)
            elif text is not None:
                self._messages.append(LLMMessage(role="assistant", content=text))
            self._retire_unfinished_turn(fact.correlation_id)
            self._interruption_observation = None
        elif fact.fact_type is FactType.TURN_INTERRUPTED:
            self._retire_unfinished_turn(fact.correlation_id)
            self._interruption_observation = _INTERRUPTED_OBSERVATION
        elif fact.fact_type in (FactType.TURN_ERROR, FactType.TURN_CANCELLED):
            self._retire_unfinished_turn(fact.correlation_id)
            self._interruption_observation = None
        elif fact.fact_type is FactType.MODEL_CHANGED:
            provider = str(fact.payload.get("provider", ""))
            model = str(fact.payload.get("model", ""))
            self._current_provider = provider
            self._current_model = model
            if self._provider_key is not None and provider != self._provider_key:
                self._last_replay_ciphertext = None
            self._model_changes.append(
                {
                    "provider": provider,
                    "model": model,
                    "timestamp": fact.timestamp.isoformat(),
                    "sequence": fact.sequence,
                }
            )

        # Track protected payload only when compatible with provider_key.
        if fact.protected_payload is not None:
            origin_provider = fact.payload.get("provider", self._current_provider)
            if self._provider_key is None or origin_provider == self._provider_key:
                self._last_replay_ciphertext = fact.protected_payload
                if self._codec is not None:
                    plaintext = self._codec.decrypt(fact.protected_payload)
                    try:
                        envelope = json.loads(plaintext)
                    except (ValueError, UnicodeDecodeError):
                        envelope = None
                    if isinstance(envelope, dict) and envelope.get("schema") == "dana.native-turn/1":
                        native = [NativeMessage.from_dict(item).to_llm_message() for item in envelope["messages"]]
                        if not native or native[0].role != "user":
                            raise ValueError("Protected native turn must begin with its user message")
                        self._pending_native[fact.correlation_id] = native

        # Messages appended above belong to the fact's turn; keep the parallel
        # owner vector and the per-turn counts in step.
        added = len(self._messages) - len(self._message_turns)
        if added > 0:
            self._message_turns.extend([fact.correlation_id] * added)
            self._turn_message_counts[fact.correlation_id] = self._turn_message_counts.get(fact.correlation_id, 0) + added

    def _retire_unfinished_turn(self, correlation_id: str) -> None:
        """Drop per-turn bookkeeping for a turn that reached a terminal fact.

        S6: a terminated (or superseded) turn leaves no unfinished state behind.
        The pending final text, the protected native envelope, the turn start
        index and every unresolved tool request of *that* turn are dropped, so a
        long-lived session does not accumulate one dead entry per finished turn.
        Legitimate interruption visibility is unaffected: it lives in
        ``_interruption_observation`` (set by ``TURN_INTERRUPTED``, cleared by any
        later terminal), not in the per-turn pending maps.

        Only the terminal turn's own entries are removed: ``pending_final`` and
        ``pending_native`` are keyed by correlation id, and an unresolved tool
        request is keyed by ``(correlation_id, call_id)``, so a concurrent open
        turn's bookkeeping is never touched. A tool request belonging to a
        *superseded* turn (its request was folded but its terminal never arrived
        while a later turn's messages were appended) is retired here too.

        ``_message_turns`` / ``_turn_message_counts`` are deliberately left intact
        for the terminal turn: they describe messages that are still part of the
        canonical history, and ``_commit_native_turn`` relies on the per-turn
        count to detect a shifted block. They are bounded by
        :meth:`discard_closed_history`, like the rest of the message bookkeeping.
        """
        self._pending_final.pop(correlation_id, None)
        self._pending_native.pop(correlation_id, None)
        self._turn_starts.pop(correlation_id, None)
        for key in [key for key in self._tool_requests if key[0] == correlation_id]:
            del self._tool_requests[key]

    def _commit_native_turn(self, correlation_id: str, native: list[LLMMessage], start: int, pre_batch_count: int) -> None:
        """Replace a completing turn's projected messages with its native envelope.

        Fold rule preserved verbatim: the turn's messages are removed wherever
        they sit and the native messages are re-inserted at the position of the
        turn's FIRST message (or at the end when the turn projected none), so an
        interleaved later user message is neither erased nor reordered.

        ``start`` - the index recorded when the turn's user fact was folded  -
        bounds the search, because a turn's messages are appended at or after
        it. If a previous native replacement shifted this turn's block earlier
        (detectable as a count shortfall: the turn owns more live messages than
        the suffix scan found), the search falls back to the full owner vector
        to keep the result identical to a from-scratch fold.
        """
        found = [index for index in range(start, len(self._message_turns)) if self._message_turns[index] == correlation_id]
        if len(found) != self._turn_message_counts.get(correlation_id, 0):
            # Anomalous journal (e.g. two user facts sharing a correlation id):
            # a message of this turn sits before ``start``. Fall back to the
            # original whole-vector semantics.
            found = [index for index, owner in enumerate(self._message_turns) if owner == correlation_id]
        insertion = found[0] if found else len(self._messages)
        # Removing messages that existed before this batch restructures a prefix
        # a consumer may already have mirrored; replacing messages created inside
        # this same batch does not (the caller applies the whole batch as tail).
        if any(index < pre_batch_count for index in found):
            self._rewrite_revision += 1
        # found[0] is the smallest removed index, so every retained position
        # below it is unchanged and the insertion index stays valid.
        for index in reversed(found):
            del self._messages[index]
            del self._message_turns[index]
        self._messages[insertion:insertion] = native
        self._message_turns[insertion:insertion] = [correlation_id] * len(native)
        self._turn_message_counts[correlation_id] = len(native)

    # ------------------------------------------------------------------
    # View
    # ------------------------------------------------------------------

    def view(self, provider_key: str | None = None, *, allow_partial: bool = False) -> ConversationView:
        """Materialize the model-facing view; ``provider_key`` must match the fold.

        ``provider_key=None`` means "no re-gating requested" and never overrides
        the key the state was folded under.

        Raises :class:`ConversationPrunedStateError` if :meth:`discard_closed_history`
        has dropped messages: the state no longer holds the full canonical
        history, so a "full view" would silently be a tail. Pass
        ``allow_partial=True`` to explicitly accept the retained window, or use
        :meth:`messages_after` / :meth:`message_count` for bounded access.
        """
        if provider_key is not None and provider_key != self._provider_key:
            raise ConversationProviderMismatch(
                f"conversation projection state was folded for provider_key={self._provider_key!r}; "
                f"cannot view it as {provider_key!r} - rebuild the state instead"
            )
        if self._discarded_prefix and not allow_partial:
            raise ConversationPrunedStateError(
                f"conversation projection state dropped {self._discarded_prefix} closed messages; "
                "the full view is no longer available (pass allow_partial=True for the retained window)"
            )
        replay_state: bytes | None = None
        if self._last_replay_ciphertext is not None and self._codec is not None:
            replay_state = self._codec.decrypt(self._last_replay_ciphertext)
        return ConversationView(
            messages=tuple(self._messages),
            replay_state=replay_state,
            interruption_observation=self._interruption_observation,
            last_sequence=self._last_sequence,
            model_changes=tuple(self._model_changes),
            current_provider=self._current_provider,
            current_model=self._current_model,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the fold state as a deterministic JSON-safe mapping.

        Contains the ciphertext of the replay payload (base64), never the
        decrypted bytes. Assistant messages keep their ``reasoning_items`` (they
        are required to reproduce the model context), so the whole payload MUST
        be envelope-encrypted by the caller before it is persisted.
        """
        return {
            "schema": CONVERSATION_STATE_SCHEMA,
            "provider_key": self._provider_key,
            "last_sequence": self._last_sequence,
            "messages": [_message_to_dict(message) for message in self._messages],
            "message_turns": list(self._message_turns),
            "discarded_prefix": self._discarded_prefix,
            "turn_starts": dict(self._turn_starts),
            "pending_final": dict(self._pending_final),
            "pending_native": {
                correlation_id: [_message_to_dict(m) for m in messages] for correlation_id, messages in self._pending_native.items()
            },
            "tool_requests": [
                [correlation_id, call_id, dict(payload)] for (correlation_id, call_id), payload in self._tool_requests.items()
            ],
            "open_turns": cast("list[JSONValue]", sorted(self._open_turns)),
            "interruption_observation": self._interruption_observation,
            "last_replay_ciphertext": b64encode(self._last_replay_ciphertext).decode("ascii")
            if self._last_replay_ciphertext is not None
            else None,
            "model_changes": [dict(entry) for entry in self._model_changes],
            "current_provider": self._current_provider,
            "current_model": self._current_model,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        protected_state_codec: ProtectedStateCodec | None = None,
    ) -> ConversationProjectionState:
        """Rebuild a fold state from :meth:`to_dict` output.

        Raises :class:`ValueError` on an unknown/missing schema tag or a
        malformed payload so the caller can fall back to a full re-fold.
        ``protected_state_codec`` is needed only to decrypt ``replay_state`` and
        any later native envelope; without it the state is still usable, exactly
        as on the one-shot path.
        """
        schema = data.get("schema")
        if schema != CONVERSATION_STATE_SCHEMA:
            raise ValueError(f"unsupported conversation projection state schema: {schema!r}")

        provider_key = _optional_str(data, "provider_key")
        state = cls(protected_state_codec=protected_state_codec, provider_key=provider_key)
        state._last_sequence = _require_int(data, "last_sequence")

        state._messages = [_require_message(item, "messages") for item in _require_list(data, "messages")]
        state._discarded_prefix = _require_int(data, "discarded_prefix")
        if state._discarded_prefix < 0:
            raise ValueError("conversation projection state discarded_prefix must be >= 0")
        raw_turns = _require_list(data, "message_turns")
        if len(raw_turns) != len(state._messages) or any(not isinstance(turn, str) for turn in raw_turns):
            raise ValueError("conversation projection state message_turns must be strings matching messages")
        state._message_turns = cast("list[str]", list(raw_turns))
        state._turn_message_counts = dict(Counter(state._message_turns))

        raw_starts = _require_mapping(data, "turn_starts")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in raw_starts.values()):
            raise ValueError("conversation projection state turn_starts values must be integers")
        state._turn_starts = cast("dict[str, int]", dict(raw_starts))

        raw_pending = _require_mapping(data, "pending_final")
        if any(not isinstance(value, str) for value in raw_pending.values()):
            raise ValueError("conversation projection state pending_final values must be strings")
        state._pending_final = cast("dict[str, str]", dict(raw_pending))

        raw_native = _require_mapping(data, "pending_native")
        pending_native: dict[str, list[LLMMessage]] = {}
        for correlation_id, messages in raw_native.items():
            if not isinstance(messages, list):
                raise ValueError("conversation projection state pending_native values must be lists of messages")
            pending_native[correlation_id] = [_require_message(item, "pending_native") for item in messages]
        state._pending_native = pending_native

        raw_requests = _require_list(data, "tool_requests")
        tool_requests: dict[tuple[str, str], Mapping[str, JSONValue]] = {}
        for entry in raw_requests:
            if (
                not isinstance(entry, list)
                or len(entry) != 3
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], str)
                or not isinstance(entry[2], dict)
            ):
                raise ValueError("conversation projection state tool_requests entries must be [correlation_id, tool_call_id, payload]")
            tool_requests[(entry[0], entry[1])] = entry[2]
        state._tool_requests = tool_requests

        raw_open = _require_list(data, "open_turns")
        if any(not isinstance(turn, str) for turn in raw_open):
            raise ValueError("conversation projection state open_turns must be strings")
        state._open_turns = set(cast("list[str]", list(raw_open)))

        state._interruption_observation = _optional_str(data, "interruption_observation")
        ciphertext_b64 = _optional_str(data, "last_replay_ciphertext")
        state._last_replay_ciphertext = b64decode(ciphertext_b64) if ciphertext_b64 is not None else None
        model_changes: list[dict[str, JSONValue]] = []
        for entry in _require_list(data, "model_changes"):
            if not isinstance(entry, dict):
                raise ValueError("conversation projection state model_changes entries must be objects")
            model_changes.append(dict(entry))
        state._model_changes = model_changes
        state._current_provider = _optional_str(data, "current_provider")
        state._current_model = _optional_str(data, "current_model")
        return state


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

    Incremental use: :meth:`new_state` returns the fold state for a session that
    wants to consume only newly appended facts; :meth:`project` folds a whole
    fact stream through the very same state.
    """

    def __init__(self, protected_state_codec: ProtectedStateCodec | None = None) -> None:
        self._codec = protected_state_codec

    def new_state(self, provider_key: str | None = None) -> ConversationProjectionState:
        """Create an empty fold state for ``provider_key`` (the compatibility gate)."""
        return ConversationProjectionState(protected_state_codec=self._codec, provider_key=provider_key)

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
        state = self.new_state(provider_key)
        state.apply(facts)
        return state.view()
