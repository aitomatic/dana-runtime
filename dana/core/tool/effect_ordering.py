"""Safe ordering of conflicting tool effects (P1 — "keep conflicting operations ordered").

Why this exists
---------------
The async batch path runs independent tool calls concurrently. Two calls from the
same model turn can then interleave: an ``Edit`` read-modify-write racing a ``Write``
to the same file, or two calls whose side effects the model intended to be
sequential. The model emits calls in order, so the harness MUST preserve that order
for calls that actually conflict while still overlapping calls that cannot interfere.

Classification source of truth
------------------------------
Effect metadata is the existing :mod:`dana.core.policy.effects` taxonomy, read from
the session Tool Catalog entry for each call (``dana.core.tool.catalog``). Call
*arguments* supply the concrete target (e.g. ``file_path``) when the catalog metadata
does not name one — the native catalog classifies by kind only. Unknown metadata is
sensitive (ADR-006) and is therefore serialized, never silently parallelized.

Scheduling rule (deterministic, order-preserving)
------------------------------------------------
Process calls in model order and pack them into *waves*:

* A call joins the current wave only if it conflicts with **no** call already in it.
* Otherwise it opens a new wave; earlier waves complete before the new call starts.
  The relative order of every conflicting pair is therefore exactly the model's order.
* Non-conflicting calls in the same wave run concurrently, bounded by admission.

Conflict rule:

1. Two read-only calls never conflict.
2. A call with unknown/sensitive metadata conflicts with everything (fail cautious).
3. Otherwise the calls conflict iff both name concrete targets and those targets
   intersect. Two mutations of *different* files are independent; two mutations of
   the *same* file must be ordered.
4. A call with no derivable target (e.g. ``bash__execute``, ``Task``) is not proven
   disjoint, but it is also not proven conflicting: such calls overlap under the
   admission bound, which is their documented, bounded concurrency model. This is a
   deliberate decision, not a silent assumption — the alternative (serializing all
   opaque execution) would forbid the bounded parallel sub-agent fan-out the
   harness is required to support.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dana.core.policy.effects import EffectKind, EffectMetadata


#: Effect kinds that mutate durable state or the outside world.
MUTATING_KINDS: frozenset[EffectKind] = frozenset(
    {
        EffectKind.WRITE,
        EffectKind.CREATE,
        EffectKind.MODIFY,
        EffectKind.DELETE,
        EffectKind.EXECUTE,
        EffectKind.NETWORK,
        EffectKind.IDENTITY,
        EffectKind.PERSISTENCE,
    }
)

#: Kinds that carry no side effect.
READ_ONLY_KINDS: frozenset[EffectKind] = frozenset({EffectKind.READ})

#: Argument names that name the concrete target of a call. Small, explicit, and
#: shared by the file and search resources — not a general argument-scanning policy.
TARGET_ARGUMENT_NAMES: frozenset[str] = frozenset({"file_path", "path", "target", "file"})


@dataclass(frozen=True)
class EffectProfile:
    """Effect profile of one tool call.

    ``sensitive`` covers ``EffectMetadata.is_sensitive``, an ``UNKNOWN`` kind, and a
    *missing* catalog entry — all mean "blast radius unknown", so the call is
    ordered against everything rather than parallelized.
    """

    name: str
    kinds: frozenset[EffectKind]
    targets: frozenset[str]
    sensitive: bool

    @property
    def is_read_only(self) -> bool:
        """True only when every declared effect is READ and none is sensitive."""
        if self.sensitive:
            return False
        return self.kinds <= READ_ONLY_KINDS  # empty kind set counts as read-only

    @property
    def mutates(self) -> bool:
        """True when the call may mutate or act on the outside world."""
        return bool(self.kinds & MUTATING_KINDS) or self.sensitive

    def conflicts_with(self, other: EffectProfile) -> bool:
        """Whether these two calls must be ordered relative to each other."""
        if self.is_read_only and other.is_read_only:
            return False
        if self.sensitive or other.sensitive:
            return True
        if not (self.mutates or other.mutates):
            return False
        # Both sides declare only known, non-sensitive effects. They conflict only
        # when their concrete targets are known and intersect; an unknown target
        # cannot prove disjointness, and rule 4 keeps that case overlapping.
        return bool(self.targets and other.targets and (self.targets & other.targets))


def _targets_from_arguments(arguments: dict[str, Any] | None) -> frozenset[str]:
    """Concrete targets named by a call's arguments (e.g. ``file_path``)."""
    if not arguments:
        return frozenset()
    out: set[str] = set()
    for key in TARGET_ARGUMENT_NAMES:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            out.add(value)
    return frozenset(out)


def profile_for_call(name: str, arguments: dict[str, Any] | None, metadata: EffectMetadata | None) -> EffectProfile:
    """Build an :class:`EffectProfile` for one call.

    Args:
        name: The tool name as it appears in the tool_call dict.
        arguments: The call arguments (source of the concrete target).
        metadata: Catalog effect metadata for the tool, or ``None`` when the tool is
            not classified (unknown -> sensitive -> ordered).
    """
    if metadata is None:
        return EffectProfile(name=name, kinds=frozenset({EffectKind.UNKNOWN}), targets=frozenset(), sensitive=True)

    kinds = frozenset(effect.kind for effect in metadata.effects)
    targets = {effect.target for effect in metadata.effects if effect.target}
    targets |= _targets_from_arguments(arguments)
    sensitive = metadata.is_sensitive or EffectKind.UNKNOWN in kinds
    return EffectProfile(name=name, kinds=kinds, targets=frozenset(targets), sensitive=sensitive)


def profiles_for_calls(
    tool_calls: list[dict[str, Any]],
    catalog: Any | None,
    metadata_resolver: Callable[[str], EffectMetadata] | None = None,
) -> list[EffectProfile]:
    """Derive an :class:`EffectProfile` per tool call.

    Args:
        tool_calls: The model's tool calls, in model order.
        catalog: The session Tool Catalog (used for effect metadata lookup).
        metadata_resolver: Optional fallback resolver, consulted when the catalog
            has no entry for a tool (e.g. the native-tool classification table).
            Without it an unclassified tool is sensitive and gets serialized.
    """
    profiles: list[EffectProfile] = []
    for call in tool_calls:
        name = call.get("function", "")
        metadata: EffectMetadata | None = None
        if catalog is not None:
            entry = catalog.get(name)
            if entry is not None:
                metadata = getattr(entry, "effects", None) or EffectMetadata.unknown()
        if metadata is None and metadata_resolver is not None:
            metadata = metadata_resolver(name)
        profiles.append(profile_for_call(name, call.get("arguments"), metadata))
    return profiles


def plan_waves(profiles: list[EffectProfile]) -> list[list[int]]:
    """Partition call indices into ordered waves of mutually non-conflicting calls.

    Pure function (no I/O, no scheduling): the batch executor uses it to decide which
    calls may overlap. Index order is preserved — a call never moves into an earlier
    wave than a call it conflicts with, so the relative order of any conflicting pair
    is exactly the model's order.
    """
    waves: list[list[int]] = []
    wave_for_call: list[int] = []
    for index, profile in enumerate(profiles):
        wave_index = 0
        for previous in range(index):
            if profile.conflicts_with(profiles[previous]):
                wave_index = max(wave_index, wave_for_call[previous] + 1)
        if wave_index == len(waves):
            waves.append([])
        waves[wave_index].append(index)
        wave_for_call.append(wave_index)
    return waves


@dataclass
class WavePlan:
    """A planned batch: ordered waves plus the profiles they were derived from."""

    waves: list[list[int]] = field(default_factory=list)
    profiles: list[EffectProfile] = field(default_factory=list)

    @property
    def serialized(self) -> bool:
        """True when the plan degenerates to one call per wave."""
        return all(len(wave) == 1 for wave in self.waves)


def plan_batch(
    tool_calls: list[dict[str, Any]],
    catalog: Any | None,
    metadata_resolver: Callable[[str], EffectMetadata] | None = None,
) -> WavePlan:
    """Plan an ordered, conflict-safe batch for the given calls."""
    profiles = profiles_for_calls(tool_calls, catalog, metadata_resolver)
    return WavePlan(waves=plan_waves(profiles), profiles=profiles)
