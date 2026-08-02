"""Live tests against the real protectai/llm-guard package.

Skipped automatically when llm-guard isn't installed (the default torch-free
test run). Run after `uv sync`:  pytest tests/unit/core/guard/ -m live

These pin behavior that the stubbed unit tests structurally cannot verify —
notably how the real scanners treat ENCODED output (base64/hex), and that the
class-name → snake_case vocabulary translation works against real results.
"""

from __future__ import annotations

import base64
import codecs

import pytest


pytest.importorskip("llm_guard")  # noqa: E402

from dana.core.guard.config import GuardConfig  # noqa: E402
from dana.core.guard.llm_guard_service import LLMGuardService  # noqa: E402
from dana.core.guard.result import GuardDecision  # noqa: E402


pytestmark = [pytest.mark.live, pytest.mark.slow]


def _svc(**cfg) -> LLMGuardService:
    # llm_getter=None → LLM scrub is a no-op; we isolate the rule-scanner behavior.
    return LLMGuardService(GuardConfig(enabled=True, **cfg), lambda: None)


# ----------------------------------------------------- output: encoded PII
def test_output_base64_pii_is_flagged_and_redacted():
    """Agent base64-encodes PII then responds. The real Sensitive scanner flags
    high-entropy/CRYPTO blobs (without decoding), so the output is sanitized —
    the raw blob must not survive."""
    svc = _svc(output_scanners=["sensitive"])
    pii = "Your email is john.doe@company.com and SSN 123-45-6789."
    blob = base64.b64encode(pii.encode()).decode()
    out = svc.scan_output("", f"Here is the data: {blob}")

    assert out.decision is GuardDecision.SANITIZED
    assert blob not in out.text  # raw encoded payload stripped
    assert out.triggered == ["sensitive"]  # snake_case (vocabulary fix, real results)


def test_output_plaintext_benign_passes():
    svc = _svc(output_scanners=["sensitive"])
    out = svc.scan_output("", "The quick brown fox jumps over the lazy dog.")
    assert out.decision is GuardDecision.ALLOW
    assert out.triggered == []


# ----------------------------------------------------- input: encoded secret
def test_input_base64_secret_is_detected():
    """detect-secrets HighEntropyString catches base64/hex blobs on input."""
    svc = _svc(input_scanners=["secrets"], block_on=[])  # don't block, just observe
    blob = "c2VjcmV0X2tleV9hYmMxMjNkZWY0NTZnaGk3ODlqa2xtbm9wcXJzdHV2d3h5eg=="
    out = svc.scan_input(f"my token is {blob}")
    assert out.decision is GuardDecision.SANITIZED
    assert out.triggered == ["secrets"]


# ----------------------------------------------------- KNOWN RESIDUAL GAP
@pytest.mark.xfail(reason="rot13 is low-entropy/reversible — CRYPTO recognizer doesn't trip; needs always-on LLM scrub", strict=False)
def test_output_rot13_pii_is_a_known_gap():
    """Documents the boundary: low-entropy reversible encodings (rot13, leetspeak,
    spaced text) are NOT caught by the rule scanners, so the gated LLM scrub never
    fires. If this ever starts PASSING, the threat surface improved — revisit docs."""
    svc = _svc(output_scanners=["sensitive"])
    pii = "Your email is john.doe@company.com and SSN 123-45-6789."
    out = svc.scan_output("", "decoded form: " + codecs.encode(pii, "rot13"))
    assert out.decision is GuardDecision.SANITIZED  # expected to FAIL today (xfail)
