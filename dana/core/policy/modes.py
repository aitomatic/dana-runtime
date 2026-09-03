"""Permission modes for the Dana agent runtime.

Per ADR-006: modes are ``default`` / ``acceptEdits`` / ``bypassPermissions``
with hard policy in every mode. Hard deny always wins regardless of mode.

Per ADR-013: ``session/set_mode`` changes mode outside an active turn.

Decision precedence (ADR-006):
    hard deny → durable reject grant → durable allow grant →
    permission mode → interactive prompt → fail-closed

Permission modes control whether the runtime prompts the user for
interactive approval when no grant matches an operation:

- ``default``: Prompt for every operation that is not hard-denied and
  has no matching grant. This is the safest mode.

- ``acceptEdits``: Automatically allow read and write/modify operations
  that are not hard-denied. Operations with DELETE, EXECUTE, NETWORK,
  IDENTITY, PERSISTENCE, or UNKNOWN effects still require a matching
  grant or interactive prompt.

- ``bypassPermissions``: Automatically allow all operations that are not
  hard-denied. No interactive prompts are shown. Use with extreme caution.
"""

from __future__ import annotations

from enum import Enum

from dana.core.policy.effects import EffectKind


class PermissionMode(Enum):
    """Permission mode controlling interactive prompt behavior.

    Modes are ordered from most restrictive to least restrictive.
    Hard deny always wins in every mode.
    """

    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS_PERMISSIONS = "bypassPermissions"

    def allows_without_prompt(
        self,
        effect_kinds: frozenset[EffectKind],
    ) -> bool:
        """Check whether this mode allows an operation without a prompt.

        Args:
            effect_kinds: The set of ``EffectKind`` values for the operation.

        Returns:
            True if the mode auto-allows the operation without a prompt;
            False if the operation still needs a grant or interactive prompt.
        """
        if self is PermissionMode.BYPASS_PERMISSIONS:
            # Bypass mode auto-allows everything that isn't hard-denied.
            return True

        if self is PermissionMode.ACCEPT_EDITS:
            # AcceptEdits auto-allows READ, WRITE, CREATE, MODIFY.
            # Everything else (DELETE, EXECUTE, NETWORK, IDENTITY,
            # PERSISTENCE, UNKNOWN) still needs a grant or prompt.
            auto_allowed = frozenset(
                {
                    EffectKind.READ,
                    EffectKind.WRITE,
                    EffectKind.CREATE,
                    EffectKind.MODIFY,
                }
            )
            return effect_kinds.issubset(auto_allowed)

        # DEFAULT mode: no auto-allow; always needs a grant or prompt.
        return False
