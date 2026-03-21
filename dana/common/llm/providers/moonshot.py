"""Moonshot (Kimi) Provider — OpenAI-compatible with video support.

Moonshot uses an OpenAI-compatible API but supports video input via
base64-encoded ``video_url`` content blocks. Audio is NOT supported.
"""

from openai import AsyncOpenAI
import structlog

from ...config import config_manager
from ..types import read_media_as_base64, unsupported_placeholder
from .openai_compatible_base import OpenAICompatibleProvider


logger = structlog.get_logger()

MOONSHOT_PROVIDER_NAME = "moonshot"


class MoonshotProvider(OpenAICompatibleProvider):
    """Moonshot (Kimi) API provider with video support."""

    def __init__(self, api_key: str | None = None, model: str = "kimi-k2.5", base_url: str | None = None):
        self.model = model

        if api_key:
            self.api_key = api_key
        else:
            self.api_key = config_manager.get_provider_api_key(MOONSHOT_PROVIDER_NAME)

        if not self.api_key:
            config = config_manager.get_provider_config(MOONSHOT_PROVIDER_NAME)
            api_key_env = config.get("api_key_env") if config else "MOONSHOT_API_KEY"
            raise ValueError(f"Moonshot API key not found. Set {api_key_env} environment variable.")

        if base_url:
            self.base_url = base_url
        else:
            self.base_url = (
                config_manager.get_provider_base_url(MOONSHOT_PROVIDER_NAME) or "https://api.moonshot.ai/v1"
            )

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        # Moonshot never uses Responses API
        self._use_responses_api = False

    def convert_multimodal_content(self, blocks: list[dict]) -> list[dict]:
        """Convert canonical path-based blocks to Moonshot wire format.

        Supports images (OpenAI image_url) and videos (video_url with base64).
        Audio is NOT supported.
        """
        result: list[dict] = []
        for block in blocks:
            btype = block.get("type", "text")
            if btype == "text":
                result.append(block)
            elif btype == "image":
                if getattr(self, "supports_vision", True):
                    b64 = read_media_as_base64(block)
                    media_type = block["media_type"]
                    url = f"data:{media_type};base64,{b64}"
                    result.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
                else:
                    result.append(unsupported_placeholder(block))
            elif btype == "video":
                if getattr(self, "supports_video", True):
                    b64 = read_media_as_base64(block)
                    ext = block["media_type"].split("/", 1)[1] if "/" in block["media_type"] else "mp4"
                    url = f"data:video/{ext};base64,{b64}"
                    result.append({"type": "video_url", "video_url": {"url": url}})
                else:
                    result.append(unsupported_placeholder(block))
            elif btype == "audio":
                # Moonshot does not support audio
                result.append(unsupported_placeholder(block))
            elif btype == "document":
                result.append(unsupported_placeholder(block))
            else:
                result.append(block)
        return result
