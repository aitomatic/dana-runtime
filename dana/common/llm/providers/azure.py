"""Azure Provider Implementation"""

from openai import AsyncAzureOpenAI
import structlog

from ...config import config_manager
from .openai_compatible_base import OpenAICompatibleProvider, make_logging_http_client


logger = structlog.get_logger()


class AzureProvider(OpenAICompatibleProvider):
    """Azure OpenAI provider."""

    # Azure exposes the Responses API only on api-version >= this date.
    # Older versions return HTTP 400 BadRequest for /openai/responses.
    _RESPONSES_API_MIN_DATE = "2025-03-01"

    # Env var operators use to dial reasoning effort for Azure deployments.
    # Valid values: "minimal" | "low" | "medium" | "high".
    _REASONING_EFFORT_ENV_VAR = "AZURE_THINKING_EFFORT"

    @property
    def name(self) -> str:
        return "azure"

    def _endpoint_url(self) -> str:
        return getattr(self, "azure_endpoint", "") or ""

    def _responses_api_supported(self) -> bool:
        # api-version format is "YYYY-MM-DD" or "YYYY-MM-DD-preview"; first 10 chars
        # are the ISO date which sorts correctly lexicographically.
        version = getattr(self, "api_version", None)
        if not version or len(version) < 10:
            return False
        return version[:10] >= self._RESPONSES_API_MIN_DATE

    def __init__(
        self, api_key: str | None = None, model: str = "gpt-35-turbo", base_url: str | None = None, api_version: str | None = None
    ):
        self.model = model
        self.deployment_name = model

        if api_key:
            self.api_key = api_key
        else:
            self.api_key = config_manager.get_provider_api_key("azure")

        if not self.api_key:
            config = config_manager.get_provider_config("azure")
            api_key_env = config.get("api_key_env") if config else "AZURE_OPENAI_API_KEY"
            raise ValueError(f"Azure OpenAI API key not found. Set {api_key_env} environment variable.")

        if base_url:
            azure_endpoint = base_url
        else:
            azure_endpoint = config_manager.get_provider_base_url("azure")

        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint URL not found. Set AZURE_OPENAI_API_URL environment variable.")

        azure_endpoint = azure_endpoint.rstrip("/")
        self.azure_endpoint = azure_endpoint

        if api_version:
            self.api_version = api_version
        else:
            self.api_version = config_manager.get_provider_api_version("azure") or "2024-02-15-preview"

        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=azure_endpoint,
            api_version=self.api_version,
            http_client=make_logging_http_client(self.DEFAULT_TIMEOUT_SECONDS),
        )

        # Check for use_responses_api config flag
        provider_config = config_manager.get_provider_config("azure")
        self._use_responses_api = provider_config.get("use_responses_api") if provider_config else None

        # Embedding support — resolve model from config or default
        self.embedding_model = (provider_config.get("default_embedding_model") if provider_config else None) or "text-embedding-3-small"
