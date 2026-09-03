"""
Unit tests for session journal models and the protected-state codec.

Covers four contract categories from the Task 1 design:
  1. frozen-fact immutability (JournalFact / NewJournalFact / OwnerScope)
  2. required scope / correlation / schema validation
  3. payload sanitization (JSON-safety, no NaN/Inf, no secret-bearing keys)
  4. secret-redaction failures (protected state never in payload) + codec round-trip
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from dana.core.session.models import (
    ArtifactRef,
    FactType,
    JournalFact,
    NewJournalFact,
    OwnerScope,
    PayloadSanitizationError,
    validate_payload,
)
from dana.core.session.protected_state import (
    EnvProtectedStateKeyProvider,
    ProtectedStateCodec,
    ProtectedStateKeyUnavailable,
)


_ENV_VAR = "DANA_SESSION_STATE_KEY"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _owner() -> OwnerScope:
    return OwnerScope(owner_id="owner-1", workspace="ws-1")


def _fact(**overrides: object) -> JournalFact:
    base: dict[str, object] = dict(
        fact_id="fact-1",
        owner_scope=_owner(),
        session_id="sess-1",
        sequence=1,
        fact_type=FactType.SESSION_CREATED,
        timestamp=datetime(2026, 7, 16, 12, 0, 0),
        correlation_id="corr-1",
        causation_id=None,
        schema_version=1,
        payload={"role": "system", "title": "hello"},
    )
    base.update(overrides)
    return JournalFact(**base)  # type: ignore[arg-type]


# ===========================================================================
# 1. frozen-fact immutability
# ===========================================================================


class TestFrozenFacts:
    """Journal facts and owner scope are immutable value objects."""

    def test_journal_fact_is_frozen(self) -> None:
        fact = _fact()
        with pytest.raises(FrozenInstanceError):
            fact.sequence = 5  # type: ignore[misc]

    def test_journal_fact_fact_id_is_frozen(self) -> None:
        fact = _fact()
        with pytest.raises(FrozenInstanceError):
            fact.fact_id = "other"  # type: ignore[misc]

    def test_new_journal_fact_is_frozen(self) -> None:
        new_fact = NewJournalFact(
            fact_type=FactType.USER_CONTENT_FINAL,
            correlation_id="corr-1",
            causation_id=None,
            payload={"text": "hi"},
        )
        with pytest.raises(FrozenInstanceError):
            new_fact.schema_version = 2  # type: ignore[misc]

    def test_owner_scope_is_frozen(self) -> None:
        scope = _owner()
        with pytest.raises(FrozenInstanceError):
            scope.owner_id = "other"  # type: ignore[misc]

    def test_artifact_ref_is_frozen(self) -> None:
        ref = ArtifactRef(uri="file://x", media_type="text/plain", size=10, sha256="abc")
        with pytest.raises(FrozenInstanceError):
            ref.size = 99  # type: ignore[misc]


# ===========================================================================
# 2. required scope / correlation / schema validation
# ===========================================================================


class TestRequiredFields:
    """Every storage boundary requires an Owner Scope and identity fields."""

    def test_owner_scope_requires_owner_id(self) -> None:
        with pytest.raises(ValueError, match="owner_id"):
            OwnerScope(owner_id="", workspace="ws-1")

    def test_owner_scope_requires_workspace(self) -> None:
        with pytest.raises(ValueError, match="workspace"):
            OwnerScope(owner_id="owner-1", workspace="")

    def test_owner_scope_accepts_valid(self) -> None:
        scope = OwnerScope(owner_id="owner-1", workspace="ws-1")
        assert scope.owner_id == "owner-1"
        assert scope.workspace == "ws-1"

    def test_new_journal_fact_requires_fact_type(self) -> None:
        with pytest.raises(TypeError):
            NewJournalFact(  # type: ignore[call-arg]
                correlation_id="corr-1", causation_id=None, payload={}
            )

    def test_new_journal_fact_requires_correlation_id(self) -> None:
        with pytest.raises(ValueError, match="correlation_id"):
            NewJournalFact(
                fact_type=FactType.TURN_STARTED,
                correlation_id="",
                causation_id=None,
                payload={},
            )

    def test_new_journal_fact_schema_version_defaults_to_one(self) -> None:
        new_fact = NewJournalFact(
            fact_type=FactType.TURN_STARTED,
            correlation_id="corr-1",
            causation_id=None,
            payload={},
        )
        assert new_fact.schema_version == 1

    def test_new_journal_fact_rejects_schema_version_below_one(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            NewJournalFact(
                fact_type=FactType.TURN_STARTED,
                correlation_id="corr-1",
                causation_id=None,
                payload={},
                schema_version=0,
            )

    def test_journal_fact_requires_owner_scope_fields(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            _fact(session_id="")

    def test_journal_fact_requires_correlation_id(self) -> None:
        with pytest.raises(ValueError, match="correlation_id"):
            _fact(correlation_id="")

    def test_journal_fact_requires_fact_id(self) -> None:
        with pytest.raises(ValueError, match="fact_id"):
            _fact(fact_id="")

    @pytest.mark.parametrize("bad_sequence", [0, -1])
    def test_journal_fact_rejects_non_positive_sequence(self, bad_sequence: int) -> None:
        with pytest.raises(ValueError, match="sequence"):
            _fact(sequence=bad_sequence)

    def test_journal_fact_allows_causation_id_none(self) -> None:
        fact = _fact(causation_id=None)
        assert fact.causation_id is None

    def test_journal_fact_allows_causation_id_set(self) -> None:
        fact = _fact(causation_id="prev-fact-1")
        assert fact.causation_id == "prev-fact-1"

    def test_journal_fact_artifact_refs_default_to_empty_tuple(self) -> None:
        fact = _fact()
        assert fact.artifact_refs == ()


# ===========================================================================
# 3. payload sanitization
# ===========================================================================


class TestPayloadSanitization:
    """Payloads must contain only JSON-safe values."""

    def test_valid_payload_passes(self) -> None:
        payload = {"a": 1, "b": "s", "c": None, "d": True, "e": [1, {"x": 2.5}]}
        result = validate_payload(payload)
        assert result is payload

    def test_datetime_in_payload_rejected(self) -> None:
        with pytest.raises(PayloadSanitizationError):
            validate_payload({"when": datetime.now()})

    def test_set_in_payload_rejected(self) -> None:
        with pytest.raises(PayloadSanitizationError):
            validate_payload({"items": {1, 2, 3}})

    def test_custom_object_in_payload_rejected(self) -> None:
        class Custom:
            pass

        with pytest.raises(PayloadSanitizationError):
            validate_payload({"obj": Custom()})

    def test_nan_in_payload_rejected(self) -> None:
        with pytest.raises(PayloadSanitizationError):
            validate_payload({"score": float("nan")})

    def test_inf_in_payload_rejected(self) -> None:
        with pytest.raises(PayloadSanitizationError):
            validate_payload({"score": float("inf")})

    def test_negative_inf_in_payload_rejected(self) -> None:
        with pytest.raises(PayloadSanitizationError):
            validate_payload({"score": float("-inf")})

    def test_nan_nested_in_list_rejected(self) -> None:
        with pytest.raises(PayloadSanitizationError):
            validate_payload({"scores": [1.0, float("nan")]})

    def test_non_string_dict_key_rejected(self) -> None:
        with pytest.raises(PayloadSanitizationError):
            validate_payload({1: "x"})  # type: ignore[dict-item]

    def test_fact_construction_validates_payload(self) -> None:
        with pytest.raises(PayloadSanitizationError):
            _fact(payload={"when": datetime.now()})


# ===========================================================================
# 4. secret-redaction failures + protected-state codec
# ===========================================================================


class TestSecretRedaction:
    """Provider replay state never appears in the regular payload."""

    @pytest.mark.parametrize(
        "secret_key",
        [
            "encrypted_content",
            "api_key",
            "access_key",
            "secret",
            "secret_key",
            "password",
            "passphrase",
            "token",
            "access_token",
            "refresh_token",
            "private_key",
            "bearer",
            "credential",
            "credentials",
            "authorization",
        ],
    )
    def test_secret_key_in_payload_rejected(self, secret_key: str) -> None:
        with pytest.raises(PayloadSanitizationError, match="forbidden"):
            validate_payload({secret_key: "value"})

    @pytest.mark.parametrize(
        "secret_key",
        [
            "apikey",  # no underscore
            "api_keys",  # plural
            "my_api_key",  # prefixed
            "API_KEY",  # uppercase
            "my-api-key",  # hyphenated
            "privateKey",  # camelCase
            "bearerToken",  # camelCase compound
            "x-authorization",  # header-style
        ],
    )
    def test_secret_key_substring_variants_rejected(self, secret_key: str) -> None:
        with pytest.raises(PayloadSanitizationError, match="forbidden"):
            validate_payload({secret_key: "value"})

    def test_fact_with_secret_in_payload_rejected(self) -> None:
        with pytest.raises(PayloadSanitizationError):
            _fact(payload={"api_key": "sk-leaked"})

    def test_protected_payload_is_bytes_not_in_payload(self) -> None:
        """protected_payload holds encrypted bytes; the secret never leaks into payload."""
        fact = _fact(protected_payload=b"\x00\x01\x02secret", payload={"role": "assistant"})
        assert fact.protected_payload == b"\x00\x01\x02secret"
        assert "api_key" not in fact.payload
        assert "encrypted_content" not in fact.payload


class TestProtectedStateCodec:
    """Envelope-encrypt provider replay state with AES-GCM and round-trip it."""

    def _codec(self, monkeypatch: pytest.MonkeyPatch, key: str = "test-session-state-key") -> ProtectedStateCodec:
        monkeypatch.setenv(_ENV_VAR, key)
        return ProtectedStateCodec(EnvProtectedStateKeyProvider())

    @staticmethod
    def _codec_with_key(key: bytes) -> ProtectedStateCodec:
        class _FixedProvider:
            def key(self) -> bytes:
                return key

        return ProtectedStateCodec(_FixedProvider())

    def test_encrypt_decrypt_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        codec = self._codec(monkeypatch)
        plaintext = b'{"encrypted_content":{"reasoning":"hidden","api_key":"sk-x"}}'
        ciphertext = codec.encrypt(plaintext)
        assert ciphertext != plaintext
        assert codec.decrypt(ciphertext) == plaintext

    def test_ciphertext_is_not_plaintext(self, monkeypatch: pytest.MonkeyPatch) -> None:
        codec = self._codec(monkeypatch)
        plaintext = b"provider-replay-state"
        ciphertext = codec.encrypt(plaintext)
        assert plaintext not in ciphertext

    def test_two_encryptions_differ_due_to_nonce(self, monkeypatch: pytest.MonkeyPatch) -> None:
        codec = self._codec(monkeypatch)
        plaintext = b"same input"
        a = codec.encrypt(plaintext)
        b = codec.encrypt(plaintext)
        assert a != b
        assert codec.decrypt(a) == codec.decrypt(b) == plaintext

    def test_decrypt_tampered_ciphertext_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cryptography.exceptions import InvalidTag

        codec = self._codec(monkeypatch)
        ciphertext = bytearray(codec.encrypt(b"payload"))
        ciphertext[-1] ^= 0xFF
        with pytest.raises(InvalidTag):
            codec.decrypt(bytes(ciphertext))

    def test_decrypt_short_ciphertext_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        codec = self._codec(monkeypatch)
        with pytest.raises(ValueError):
            codec.decrypt(b"short")

    def test_decrypt_with_wrong_key_fails(self) -> None:
        from cryptography.exceptions import InvalidTag

        codec_a = self._codec_with_key(b"key-a-material")
        codec_b = self._codec_with_key(b"key-b-material")
        ciphertext = codec_a.encrypt(b"provider-replay-state")
        with pytest.raises(InvalidTag):
            codec_b.decrypt(ciphertext)

    def test_aad_round_trip_succeeds(self) -> None:
        codec = self._codec_with_key(b"aad-key-material")
        plaintext = b"protected"
        aad = b"owner-1|sess-1|42"
        ciphertext = codec.encrypt(plaintext, aad=aad)
        assert codec.decrypt(ciphertext, aad=aad) == plaintext

    def test_aad_mismatch_fails(self) -> None:
        from cryptography.exceptions import InvalidTag

        codec = self._codec_with_key(b"aad-key-material")
        ciphertext = codec.encrypt(b"protected", aad=b"owner-1|sess-1|42")
        with pytest.raises(InvalidTag):
            codec.decrypt(ciphertext, aad=b"owner-2|sess-1|42")

    def test_aad_provided_on_decrypt_of_none_aad_blob_fails(self) -> None:
        from cryptography.exceptions import InvalidTag

        codec = self._codec_with_key(b"aad-key-material")
        ciphertext = codec.encrypt(b"protected")  # no AAD at encrypt time
        with pytest.raises(InvalidTag):
            codec.decrypt(ciphertext, aad=b"owner-1|sess-1|42")


class TestEnvProtectedStateKeyProvider:
    """The env-backed key provider is the required source of envelope key material."""

    def test_raises_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_ENV_VAR, raising=False)
        provider = EnvProtectedStateKeyProvider()
        with pytest.raises(ProtectedStateKeyUnavailable):
            provider.key()

    def test_raises_when_env_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_ENV_VAR, "")
        provider = EnvProtectedStateKeyProvider()
        with pytest.raises(ProtectedStateKeyUnavailable):
            provider.key()

    def test_returns_key_bytes_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_ENV_VAR, "my-key-material")
        provider = EnvProtectedStateKeyProvider()
        assert provider.key() == b"my-key-material"


# ===========================================================================
# FactType enum sanity (D1 set only)
# ===========================================================================


class TestFactType:
    def test_d1_fact_types_present(self) -> None:
        expected = {
            "SESSION_CREATED",
            "SESSION_LOADED",
            "SESSION_RESUMED",
            "TURN_STARTED",
            "USER_CONTENT_FINAL",
            "ASSISTANT_CONTENT_CHUNK",
            "ASSISTANT_CONTENT_FINAL",
            "TURN_COMPLETED",
            "TURN_INTERRUPTED",
            "TURN_ERROR",
            "TURN_CANCELLED",
            "LEGACY_TIMELINE_MIGRATED",
        }
        assert expected.issubset({member.name for member in FactType})

    def test_fact_type_values_are_strings(self) -> None:
        for member in FactType:
            assert isinstance(member.value, str)
