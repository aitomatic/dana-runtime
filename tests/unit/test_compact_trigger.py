"""Unit tests for compact_trigger resolver."""

from __future__ import annotations

import pytest

from dana.core.timeline import compact_trigger as ct


@pytest.fixture(autouse=True)
def _reset_cache():
    ct._reset_cache_for_tests()
    yield
    ct._reset_cache_for_tests()


def test_env_unset_returns_default(monkeypatch):
    monkeypatch.delenv("DANA_COMPACT_TRIGGER_TOKENS", raising=False)
    assert ct.resolve_trigger_tokens() == ct.DEFAULT_TRIGGER
    assert ct.DEFAULT_TRIGGER == 150_000


def test_env_valid_value(monkeypatch):
    monkeypatch.setenv("DANA_COMPACT_TRIGGER_TOKENS", "100000")
    assert ct.resolve_trigger_tokens() == 100_000


def test_env_below_min_clamps_to_default(monkeypatch):
    monkeypatch.setenv("DANA_COMPACT_TRIGGER_TOKENS", "5000")
    assert ct.resolve_trigger_tokens() == ct.DEFAULT_TRIGGER


def test_env_above_max_clamps_to_default(monkeypatch):
    monkeypatch.setenv("DANA_COMPACT_TRIGGER_TOKENS", "5000000")
    assert ct.resolve_trigger_tokens() == ct.DEFAULT_TRIGGER


def test_env_non_numeric_falls_back(monkeypatch):
    monkeypatch.setenv("DANA_COMPACT_TRIGGER_TOKENS", "abc")
    assert ct.resolve_trigger_tokens() == ct.DEFAULT_TRIGGER


def test_env_negative_falls_back(monkeypatch):
    monkeypatch.setenv("DANA_COMPACT_TRIGGER_TOKENS", "-1")
    assert ct.resolve_trigger_tokens() == ct.DEFAULT_TRIGGER


def test_env_zero_falls_back(monkeypatch):
    monkeypatch.setenv("DANA_COMPACT_TRIGGER_TOKENS", "0")
    assert ct.resolve_trigger_tokens() == ct.DEFAULT_TRIGGER


def test_env_empty_string_falls_back(monkeypatch):
    monkeypatch.setenv("DANA_COMPACT_TRIGGER_TOKENS", "")
    assert ct.resolve_trigger_tokens() == ct.DEFAULT_TRIGGER


def test_min_trigger_boundary(monkeypatch):
    monkeypatch.setenv("DANA_COMPACT_TRIGGER_TOKENS", str(ct.MIN_TRIGGER))
    assert ct.resolve_trigger_tokens() == ct.MIN_TRIGGER


def test_max_trigger_boundary(monkeypatch):
    monkeypatch.setenv("DANA_COMPACT_TRIGGER_TOKENS", str(ct.MAX_TRIGGER))
    assert ct.resolve_trigger_tokens() == ct.MAX_TRIGGER


def test_resolution_cached(monkeypatch):
    """Second call should not re-read env."""
    monkeypatch.setenv("DANA_COMPACT_TRIGGER_TOKENS", "100000")
    first = ct.resolve_trigger_tokens()
    # Change env; cached value should be stable.
    monkeypatch.setenv("DANA_COMPACT_TRIGGER_TOKENS", "200000")
    second = ct.resolve_trigger_tokens()
    assert first == second == 100_000
