"""Routing tests for Responses API selection across OpenAI / Azure providers.

Covers:
- Model-prefix routing (gpt-5*, o3*, o4*) on OpenAI-compatible base.
- Azure api-version gate (Responses API requires >= 2025-03-01).
- Explicit `use_responses_api` config flag overrides both.
"""

from unittest.mock import MagicMock, patch

import pytest

from dana.common.llm.providers.openai_compatible_base import OpenAICompatibleProvider


def _make_azure(api_version: str, model: str = "gpt-5.2", use_responses_api=None):
    config_mock = MagicMock()
    config_mock.get_provider_api_key.return_value = "fake-key"
    config_mock.get_provider_base_url.return_value = "https://test.openai.azure.com"
    config_mock.get_provider_api_version.return_value = api_version
    config_mock.get_provider_config.return_value = {"use_responses_api": use_responses_api} if use_responses_api is not None else {}
    with patch("dana.common.llm.providers.azure.config_manager", config_mock):
        from dana.common.llm.providers.azure import AzureProvider

        return AzureProvider(api_key="fake-key", base_url="https://test.openai.azure.com", model=model)


class TestAzureApiVersionGate:
    """Azure-specific: Responses API requires api-version >= 2025-03-01-preview."""

    @pytest.mark.parametrize(
        "version",
        ["2025-03-01-preview", "2025-04-01-preview", "2025-12-01-preview", "2025-03-01", "2026-01-01-preview"],
    )
    def test_supported_versions(self, version):
        p = _make_azure(api_version=version)
        assert p._responses_api_supported() is True

    @pytest.mark.parametrize(
        "version",
        ["2024-02-15-preview", "2024-12-01-preview", "2025-02-28-preview", "2023-05-15"],
    )
    def test_unsupported_versions(self, version):
        p = _make_azure(api_version=version)
        assert p._responses_api_supported() is False

    @pytest.mark.parametrize("version", ["", "abc", "20250301"])
    def test_malformed_versions_default_to_unsupported(self, version):
        p = _make_azure(api_version=version)
        assert p._responses_api_supported() is False


class TestAzureRoutingDecision:
    """End-to-end routing: model prefix + version + config flag."""

    def test_gpt5_with_supported_version_uses_responses(self):
        p = _make_azure(api_version="2025-04-01-preview", model="gpt-5.2")
        assert p._should_use_responses_api() is True

    def test_gpt5_with_old_version_falls_back_to_chat(self):
        # Was the production-breaking case: would 400 on /openai/responses.
        p = _make_azure(api_version="2024-12-01-preview", model="gpt-5.2")
        assert p._should_use_responses_api() is False

    def test_non_reasoning_model_never_uses_responses(self):
        p = _make_azure(api_version="2025-04-01-preview", model="gpt-4o")
        assert p._should_use_responses_api() is False

    def test_explicit_true_overrides_version_gate(self):
        # Caller takes responsibility — useful for forcing the path under test or
        # against a custom-configured Azure resource.
        p = _make_azure(api_version="2024-12-01-preview", model="gpt-5.2", use_responses_api=True)
        assert p._should_use_responses_api() is True

    def test_explicit_false_overrides_prefix_match(self):
        p = _make_azure(api_version="2025-04-01-preview", model="gpt-5.2", use_responses_api=False)
        assert p._should_use_responses_api() is False

    @pytest.mark.parametrize("model", ["gpt-5", "gpt-5.2", "gpt-5-turbo", "o3-mini", "o4-preview"])
    def test_reasoning_model_prefixes_match(self, model):
        p = _make_azure(api_version="2025-04-01-preview", model=model)
        assert p._should_use_responses_api() is True


class TestEnvVarOverride:
    """Env var forces/disables Responses API, taking precedence over config flag."""

    def test_provider_env_forces_on_over_version_gate(self, monkeypatch):
        monkeypatch.setenv("AZURE_USE_RESPONSES_API", "true")
        p = _make_azure(api_version="2024-12-01-preview", model="gpt-4o")
        assert p._should_use_responses_api() is True

    def test_provider_env_forces_off_over_prefix_match(self, monkeypatch):
        monkeypatch.setenv("AZURE_USE_RESPONSES_API", "0")
        p = _make_azure(api_version="2025-04-01-preview", model="gpt-5.2")
        assert p._should_use_responses_api() is False

    def test_provider_env_overrides_config_flag(self, monkeypatch):
        monkeypatch.setenv("AZURE_USE_RESPONSES_API", "off")
        p = _make_azure(api_version="2025-04-01-preview", model="gpt-5.2", use_responses_api=True)
        assert p._should_use_responses_api() is False

    @pytest.mark.parametrize("raw", ["1", "true", "on", "yes", "TRUE", " Yes "])
    def test_truthy_values(self, monkeypatch, raw):
        monkeypatch.setenv("AZURE_USE_RESPONSES_API", raw)
        p = _make_azure(api_version="2024-12-01-preview", model="gpt-4o")
        assert p._should_use_responses_api() is True

    @pytest.mark.parametrize("raw", ["0", "false", "off", "no", "FALSE"])
    def test_falsy_values(self, monkeypatch, raw):
        monkeypatch.setenv("AZURE_USE_RESPONSES_API", raw)
        p = _make_azure(api_version="2025-04-01-preview", model="gpt-5.2")
        assert p._should_use_responses_api() is False

    def test_invalid_value_ignored_falls_back_to_auto_detect(self, monkeypatch):
        monkeypatch.setenv("AZURE_USE_RESPONSES_API", "maybe")
        p = _make_azure(api_version="2025-04-01-preview", model="gpt-5.2")
        assert p._should_use_responses_api() is True  # prefix match still applies

    def test_generic_env_applies_when_provider_var_unset(self, monkeypatch):
        monkeypatch.delenv("AZURE_USE_RESPONSES_API", raising=False)
        monkeypatch.setenv("LLM_USE_RESPONSES_API", "true")
        p = _make_azure(api_version="2024-12-01-preview", model="gpt-4o")
        assert p._should_use_responses_api() is True

    def test_provider_var_wins_over_generic(self, monkeypatch):
        monkeypatch.setenv("AZURE_USE_RESPONSES_API", "false")
        monkeypatch.setenv("LLM_USE_RESPONSES_API", "true")
        p = _make_azure(api_version="2025-04-01-preview", model="gpt-5.2")
        assert p._should_use_responses_api() is False


class TestOpenAIBaseDefaultSupported:
    """Non-Azure OpenAI-compatible providers support Responses API unconditionally."""

    def test_default_responses_api_supported_is_true(self):
        # The base method returns True; no version constraint.
        class _Stub(OpenAICompatibleProvider):
            client = None
            model = "gpt-5"

        assert _Stub()._responses_api_supported() is True
