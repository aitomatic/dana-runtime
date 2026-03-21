"""Live multimodal test — sends real image/audio/video to each LLM provider.

Usage:
    uv run python test_multi_modal.py                  # run all providers
    uv run python test_multi_modal.py openai            # run one provider
    uv run python test_multi_modal.py gemini --audio    # one provider, one modality

Sample files expected in: data/sample_multimodal/
"""

import asyncio
import os
from pathlib import Path
import sys


# Add dana_agent to path
sys.path.insert(0, str(Path(__file__).parent))

from dana.common.llm.types import LLMMessage, SystemLLMMessage


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path.cwd() / "data" / "sample_multimodal"

SAMPLE_FILES = {
    "image": DATA_DIR / "file_example_PNG_2100kB.png",
    "audio": DATA_DIR / "file_example_MP3_1MG.mp3",
    "video": DATA_DIR / "file_example_MOV_1920_2_2MB.mov",
}

MEDIA_TYPES = {
    "image": "image/png",
    "audio": "audio/mp3",
    "video": "video/quicktime",
}

# Which modalities each provider supports (for the test prompt)
PROVIDER_CAPABILITIES = {
    "anthropic": {"image": True, "audio": False, "video": False},
    "openai": {"image": True, "audio": True, "video": False},
    "azure": {"image": True, "audio": True, "video": False},
    "gemini": {"image": True, "audio": True, "video": True},
    "gemini_openai": {"image": True, "audio": True, "video": True},  # Gemini via OpenAI-compat endpoint
    "anthropic_like": {"image": True, "audio": False, "video": False},
    "moonshot": {"image": True, "audio": False, "video": True},
}

# Model to use per provider (pick smaller/cheaper models)
# Note: OpenAI audio requires gpt-4o-audio-preview; gpt-4.1 doesn't support input_audio
# Note: gemini_openai = Gemini via OpenAI-compatible endpoint (GEMINI_BASE_URL)
PROVIDER_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4.1-mini",
    "azure": os.getenv("AZURE_MODEL", "gpt-5.4"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
    "gemini_openai": os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
    "anthropic_like": None,  # uses default
    "moonshot": "kimi-k2.5",
}

# Override model per modality when needed (e.g., audio needs a different model)
PROVIDER_AUDIO_MODELS = {
    "openai": "gpt-4o-audio-preview",
    "azure": "gpt-4o-audio-preview",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_canonical_block(modality: str) -> list[dict]:
    """Build canonical path-based multimodal content blocks for a given modality."""
    file_path = SAMPLE_FILES[modality]
    if not file_path.exists():
        raise FileNotFoundError(f"Sample file not found: {file_path}")

    media_type = MEDIA_TYPES[modality]
    block_type = {"image": "image", "audio": "audio", "video": "video"}.get(modality)
    if not block_type:
        raise ValueError(f"Unknown modality: {modality}")

    return [
        {"type": "text", "text": f"This is a {modality} file. Describe what you see/hear in 1-2 sentences."},
        {"type": block_type, "media_type": media_type, "path": str(file_path)},
    ]


def create_provider(provider_name: str):
    """Create a provider instance."""
    from dana.common.llm.providers.factory import create_provider as _create

    model = PROVIDER_MODELS.get(provider_name)

    if provider_name == "gemini_openai":
        # Gemini via OpenAI-compatible endpoint
        from openai import AsyncOpenAI

        from dana.common.llm.providers.openai_compatible_base import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
        provider.model = model
        provider._use_responses_api = False
        provider.client = AsyncOpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        )
    else:
        provider = _create(provider_name, model=model)

    # Set capability flags (normally done by LLMResource)
    caps = PROVIDER_CAPABILITIES.get(provider_name, {})
    provider.supports_vision = caps.get("image", False)
    provider.supports_audio = caps.get("audio", False)
    provider.supports_video = caps.get("video", False)

    return provider


async def test_provider_modality(provider_name: str, modality: str) -> str:
    """Test one provider with one modality. Returns result string."""
    caps = PROVIDER_CAPABILITIES.get(provider_name, {})

    if not caps.get(modality, False):
        return f"  [{modality:6}] SKIPPED — not supported by {provider_name}"

    try:
        # Use modality-specific model override if needed
        if modality == "audio" and provider_name in PROVIDER_AUDIO_MODELS:
            from dana.common.llm.providers.factory import create_provider as _create

            audio_model = PROVIDER_AUDIO_MODELS[provider_name]
            provider = _create(provider_name, model=audio_model)
            caps = PROVIDER_CAPABILITIES.get(provider_name, {})
            provider.supports_vision = caps.get("image", False)
            provider.supports_audio = caps.get("audio", False)
            provider.supports_video = caps.get("video", False)
        else:
            provider = create_provider(provider_name)

        content_blocks = build_canonical_block(modality)

        messages = [
            SystemLLMMessage(content="You are a helpful assistant. Describe media content briefly."),
            LLMMessage(role="user", content=content_blocks),
        ]

        response = await provider.chat(messages, max_tokens=200)
        # Truncate long responses for display
        text = response.content[:150].replace("\n", " ")
        return f"  [{modality:6}] OK — {text}..."

    except Exception as e:
        error_msg = str(e)[:200].replace("\n", " ")
        return f"  [{modality:6}] ERROR — {error_msg}"


async def test_provider(provider_name: str, modalities: list[str] | None = None):
    """Test all modalities for one provider."""
    if modalities is None:
        modalities = ["image", "audio", "video"]

    print(f"\n{'=' * 60}")
    print(f"Provider: {provider_name.upper()}")
    model = PROVIDER_MODELS.get(provider_name, "default")
    print(f"Model: {model}")
    print(f"{'=' * 60}")

    for modality in modalities:
        result = await test_provider_modality(provider_name, modality)
        print(result)


async def main():
    # Parse args
    args = sys.argv[1:]
    providers = list(PROVIDER_CAPABILITIES.keys())
    modalities = None  # all

    selected_providers = []
    for arg in args:
        if arg.startswith("--"):
            mod = arg.lstrip("-")
            if mod in ("image", "audio", "video"):
                modalities = [mod]
        elif arg in providers:
            selected_providers.append(arg)

    if not selected_providers:
        selected_providers = providers

    # Verify sample files exist
    print("Sample files:")
    for mod, path in SAMPLE_FILES.items():
        exists = "OK" if path.exists() else "MISSING"
        size = f"{path.stat().st_size / 1024:.0f}KB" if path.exists() else "N/A"
        print(f"  {mod:6}: {path.name} ({size}) [{exists}]")

    # Run tests
    for provider_name in selected_providers:
        await test_provider(provider_name, modalities)

    print(f"\n{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
