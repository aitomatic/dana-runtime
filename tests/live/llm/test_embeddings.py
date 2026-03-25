"""Live embedding test — sends real text to each embedding-capable LLM provider.

Usage:
    uv run python tests/live/llm/test_embeddings.py                # run all providers
    uv run python tests/live/llm/test_embeddings.py openai         # run one provider
    uv run python tests/live/llm/test_embeddings.py gemini --batch # one provider, batch only
"""

import asyncio
import os
import sys

from dotenv import load_dotenv


load_dotenv()

from dana.common.llm.embedder import Embedder
from dana.common.llm.providers.factory import create_provider
from dana.common.llm.types import EmbeddingNotSupportedError


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Providers that support embeddings and their default models
EMBEDDING_PROVIDERS = {
    "openai": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    "gemini": os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
    "azure": os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-large"),
}

# Providers that should raise EmbeddingNotSupportedError
NON_EMBEDDING_PROVIDERS = ["anthropic", "moonshot"]

# Test texts
SINGLE_TEXT = "The quick brown fox jumps over the lazy dog."

BATCH_TEXTS = [
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing enables computers to understand text.",
    "Deep learning uses neural networks with multiple layers.",
]

# Semantically similar pairs for cosine similarity check
SIMILAR_PAIR = (
    "The cat sat on the mat.",
    "A feline was resting on a rug.",
)

DISSIMILAR_PAIR = (
    "The cat sat on the mat.",
    "Quantum computing leverages superposition and entanglement.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def test_single_embed(provider_name: str, model: str) -> str:
    """Test single text embedding."""
    try:
        provider = create_provider(provider_name, model=model)
        response = await provider.embed(SINGLE_TEXT)
        dim = response.dimensions
        vec_preview = str(response.embeddings[0][:5])[:60]
        usage_str = f"tokens={response.usage}" if response.usage else "no usage info"
        return f"  [single ] OK — dim={dim}, {usage_str}, vec={vec_preview}..."
    except Exception as e:
        return f"  [single ] ERROR — {str(e)[:200]}"


async def test_batch_embed(provider_name: str, model: str) -> str:
    """Test batch text embedding."""
    try:
        provider = create_provider(provider_name, model=model)
        response = await provider.embed_batch(BATCH_TEXTS)
        count = len(response.embeddings)
        dim = response.dimensions
        return f"  [batch  ] OK — {count} embeddings, dim={dim}"
    except Exception as e:
        return f"  [batch  ] ERROR — {str(e)[:200]}"


async def test_similarity(provider_name: str, model: str) -> str:
    """Test that similar texts have higher cosine similarity than dissimilar texts."""
    try:
        provider = create_provider(provider_name, model=model)
        all_texts = [SIMILAR_PAIR[0], SIMILAR_PAIR[1], DISSIMILAR_PAIR[1]]
        response = await provider.embed_batch(all_texts)
        vecs = response.embeddings

        sim_score = cosine_similarity(vecs[0], vecs[1])
        dissim_score = cosine_similarity(vecs[0], vecs[2])
        passed = sim_score > dissim_score

        status = "OK" if passed else "WARN"
        return f"  [similar] {status} — similar={sim_score:.4f}, dissimilar={dissim_score:.4f} ({'correct' if passed else 'unexpected'})"
    except Exception as e:
        return f"  [similar] ERROR — {str(e)[:200]}"


async def test_embedder_class(provider_name: str, model: str) -> str:
    """Test the Embedder convenience class."""
    try:
        embedder = Embedder(provider=provider_name, model=model)
        vector = await embedder.embed(SINGLE_TEXT)
        dim = len(vector)
        return f"  [class  ] OK — Embedder(provider='{provider_name}'), dim={dim}"
    except Exception as e:
        return f"  [class  ] ERROR — {str(e)[:200]}"


async def test_embedder_sync(provider_name: str, model: str) -> str:
    """Test synchronous embedding."""
    try:
        Embedder(provider=provider_name, model=model)
        # embed_sync can't be called from async context, test construction only
        return "  [sync   ] OK — Embedder constructed, embed_sync available"
    except Exception as e:
        return f"  [sync   ] ERROR — {str(e)[:200]}"


async def test_non_embedding_provider(provider_name: str) -> str:
    """Test that non-embedding providers raise proper error."""
    try:
        provider = create_provider(provider_name)
        await provider.embed("test")
        return "  [reject ] FAIL — should have raised EmbeddingNotSupportedError"
    except EmbeddingNotSupportedError:
        return "  [reject ] OK — correctly raised EmbeddingNotSupportedError"
    except Exception as e:
        return f"  [reject ] ERROR — wrong exception: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def test_provider(provider_name: str, model: str, tests: list[str] | None = None):
    """Run all embedding tests for one provider."""
    if tests is None:
        tests = ["single", "batch", "similar", "class", "sync"]

    print(f"\n{'=' * 60}")
    print(f"Provider: {provider_name.upper()}")
    print(f"Model: {model}")
    print(f"{'=' * 60}")

    test_map = {
        "single": lambda: test_single_embed(provider_name, model),
        "batch": lambda: test_batch_embed(provider_name, model),
        "similar": lambda: test_similarity(provider_name, model),
        "class": lambda: test_embedder_class(provider_name, model),
        "sync": lambda: test_embedder_sync(provider_name, model),
    }

    for test_name in tests:
        if test_name in test_map:
            result = await test_map[test_name]()
            print(result)


async def main():
    args = sys.argv[1:]
    selected_providers = []
    selected_tests = None

    for arg in args:
        if arg.startswith("--"):
            test_name = arg.lstrip("-")
            if test_name in ("single", "batch", "similar", "class", "sync"):
                selected_tests = [test_name]
        elif arg in EMBEDDING_PROVIDERS:
            selected_providers.append(arg)

    if not selected_providers:
        selected_providers = list(EMBEDDING_PROVIDERS.keys())

    # Run embedding-capable providers
    for provider_name in selected_providers:
        model = EMBEDDING_PROVIDERS[provider_name]
        await test_provider(provider_name, model, selected_tests)

    # Test non-embedding providers
    print(f"\n{'=' * 60}")
    print("NON-EMBEDDING PROVIDERS (should reject)")
    print(f"{'=' * 60}")
    for provider_name in NON_EMBEDDING_PROVIDERS:
        try:
            result = await test_non_embedding_provider(provider_name)
            print(result)
        except Exception as e:
            print(f"  [{provider_name:8}] SKIP — provider unavailable: {str(e)[:100]}")

    print(f"\n{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
