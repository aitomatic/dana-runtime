"""
Protected-state envelope encryption for Provider Replay State.

Provider Replay State (e.g. OpenAI ``encrypted_content``, reasoning items) is
protected model-provider material required to continue a conversation faithfully.
It must never appear in a Journal Fact's regular ``payload``; instead it is
envelope-encrypted via :class:`ProtectedStateCodec` and carried as
``protected_payload`` bytes.

The encryption key is sourced from an explicit provider (by default the
``DANA_SESSION_STATE_KEY`` environment variable), never hard-coded.
``DANA_SESSION_STATE_KEY`` should be a high-entropy random secret (32+ bytes
recommended); HKDF-SHA256 derives the AES key from it but does not substitute
for key entropy.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


# AES-GCM nonce length in bytes (96 bits is the standard/recommended size).
_NONCE_LEN = 12

# HKDF info string binds the derived key to this purpose, and the derived
# AES-256 key length.
_KDF_INFO = b"dana-session-protected-state-v1"
_KEY_LEN = 32


class ProtectedStateKeyUnavailable(RuntimeError):
    """Raised when the protected-state encryption key is missing or unusable."""


@runtime_checkable
class ProtectedStateKeyProvider(Protocol):
    """Provides the raw key material used to envelope-encrypt provider replay state."""

    def key(self) -> bytes:
        """Return the raw key bytes. Raise if unavailable."""
        ...


class EnvProtectedStateKeyProvider:
    """Protected-state key provider backed by the DANA_SESSION_STATE_KEY env var."""

    _ENV_VAR = "DANA_SESSION_STATE_KEY"

    def key(self) -> bytes:
        value = os.environ.get(self._ENV_VAR)
        if not value:
            raise ProtectedStateKeyUnavailable(f"{self._ENV_VAR} is required")
        return value.encode("ascii")


class ProtectedStateCodec:
    """Envelope-encrypt provider replay state with AES-256-GCM.

    A 32-byte key is derived from the provider's raw key bytes via HKDF-SHA256
    (info=b"dana-session-protected-state-v1", salt=None). :meth:`encrypt` returns
    ``nonce || ciphertext``; :meth:`decrypt` reverses it. Authenticated encryption
    (AES-GCM) means tampering is detected on decryption.

    Both methods accept an optional ``aad`` (Associated Authenticated Data)
    argument. When provided, AES-GCM contextually binds the blob to it: a blob
    encrypted with one AAD value will not decrypt with a different AAD. This lets
    callers bind a blob to (owner_id, session_id, sequence) so it cannot be
    relocated across facts. When ``aad`` is None, behavior is unchanged.
    """

    def __init__(self, key_provider: ProtectedStateKeyProvider) -> None:
        self._key_provider = key_provider

    def _aesgcm(self) -> AESGCM:
        derived = HKDF(
            algorithm=hashes.SHA256(),
            length=_KEY_LEN,
            salt=None,
            info=_KDF_INFO,
        ).derive(self._key_provider.key())
        return AESGCM(derived)

    def encrypt(self, plaintext: bytes, aad: bytes | None = None) -> bytes:
        """Envelope-encrypt plaintext; returns ``nonce || ciphertext``.

        If ``aad`` is provided, it is bound as authenticated associated data.
        """
        aesgcm = self._aesgcm()
        nonce = os.urandom(_NONCE_LEN)
        ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
        return nonce + ciphertext

    def decrypt(self, ciphertext: bytes, aad: bytes | None = None) -> bytes:
        """Decrypt a ``nonce || ciphertext`` blob produced by :meth:`encrypt`.

        ``aad`` must equal the value passed to :meth:`encrypt` (or both None).
        """
        if len(ciphertext) < _NONCE_LEN:
            raise ValueError(f"ciphertext too short to contain a {_NONCE_LEN}-byte nonce")
        aesgcm = self._aesgcm()
        nonce = ciphertext[:_NONCE_LEN]
        body = ciphertext[_NONCE_LEN:]
        return aesgcm.decrypt(nonce, body, aad)
