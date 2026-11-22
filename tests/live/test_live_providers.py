"""Gated live provider checks — SKIPPED unless explicitly opted in. Never run in CI.

Run manually only after the account owner confirmed model access:

    LLM_ROUTING_LIVE_TESTS=1 LLM_ROUTING_ALLOW_LIVE=1 GROQ_API_KEY=... \\
        uv run pytest -m live tests/live -q

Only a fixed, synthetic, non-customer prompt is ever authorized for sending.
These tests verify wire compatibility and account access, not quality,
latency SLOs or cost.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from llm_routing_eval.adapters.embeddings import RemoteEmbeddingOptions, managed_nvidia_embedding
from llm_routing_eval.adapters.providers import (
    GroqProviderOptions,
    NvidiaBuildProviderOptions,
    managed_httpx_groq_provider,
    managed_httpx_nvidia_provider,
)
from llm_routing_eval.core import CallStatus, MetricOrigin, ModelRequest

pytestmark = pytest.mark.live

SYNTHETIC_PROMPT = "Reply with the single word: OK"
SYNTHETIC_EMBED_TEXTS = ("synthetic test sentence about weather", "synthetic test sentence")


def _require(*names: str) -> None:
    if os.getenv("LLM_ROUTING_LIVE_TESTS") != "1" or os.getenv("LLM_ROUTING_ALLOW_LIVE") != "1":
        pytest.skip("Live tests require LLM_ROUTING_LIVE_TESTS=1 and LLM_ROUTING_ALLOW_LIVE=1")
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        pytest.skip("Missing live-test environment: " + ",".join(missing))


class FixedSyntheticPromptOnly:
    """Authorizes exactly one hard-coded synthetic prompt; nothing else."""

    def authorize(self, request: ModelRequest) -> bool:
        return request.prompt == SYNTHETIC_PROMPT


class FixedSyntheticEmbeddingOnly:
    def authorize_embedding(self, *, model: str, texts: tuple[str, ...]) -> bool:
        _ = model
        return texts == SYNTHETIC_EMBED_TEXTS


def _request(provider: str, model: str) -> ModelRequest:
    return ModelRequest(
        run_id="live-smoke",
        case_id="synthetic",
        trace_id="live-smoke",
        attempt_id="live-smoke-1",
        provider=provider,
        model=model,
        prompt=SYNTHETIC_PROMPT,
        timeout_seconds=30.0,
        temperature=0.0,
        max_output_tokens=256,
    )


def test_groq_live_chat_completion() -> None:
    _require("GROQ_API_KEY")
    model = os.getenv("GROQ_LIVE_TEST_MODEL", "openai/gpt-oss-20b")

    async def run() -> None:
        async with managed_httpx_groq_provider(
            GroqProviderOptions(allowed_models=frozenset({model}), live_enabled=True),
            outbound_authorization=FixedSyntheticPromptOnly(),
        ) as provider:
            response = await provider.generate(_request("groq", model))
        assert response.status is CallStatus.SUCCESS, response.error
        assert response.metric_origin is MetricOrigin.OBSERVED
        assert response.usage is not None and response.usage.input_tokens
        assert os.environ["GROQ_API_KEY"] not in repr(response)

    asyncio.run(run())


def test_nvidia_build_live_chat_completion() -> None:
    # No default model: NVIDIA Build availability must be confirmed by the operator.
    _require("NVIDIA_API_KEY", "NVIDIA_LIVE_TEST_MODEL")
    model = os.environ["NVIDIA_LIVE_TEST_MODEL"]

    async def run() -> None:
        async with managed_httpx_nvidia_provider(
            NvidiaBuildProviderOptions(allowed_models=frozenset({model}), live_enabled=True),
            outbound_authorization=FixedSyntheticPromptOnly(),
        ) as provider:
            response = await provider.generate(_request("nvidia_build", model))
        assert response.status is CallStatus.SUCCESS, response.error
        assert os.environ["NVIDIA_API_KEY"] not in repr(response)

    asyncio.run(run())


def test_nvidia_build_live_embeddings() -> None:
    _require("NVIDIA_API_KEY", "NVIDIA_LIVE_EMBEDDING_MODEL")
    model = os.environ["NVIDIA_LIVE_EMBEDDING_MODEL"]

    async def run() -> None:
        async with managed_nvidia_embedding(
            RemoteEmbeddingOptions(model=model, live_enabled=True),
            outbound_authorization=FixedSyntheticEmbeddingOnly(),
        ) as embedder:
            vectors = await embedder.embed(SYNTHETIC_EMBED_TEXTS)
        assert len(vectors) == 2 and len(vectors[0]) == len(vectors[1]) > 0

    asyncio.run(run())
