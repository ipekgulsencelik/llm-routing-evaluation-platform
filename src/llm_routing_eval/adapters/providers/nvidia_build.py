"""NVIDIA Build (integrate.api.nvidia.com) compatibility provider — secondary phase.

Same fail-closed gates as Groq: explicit per-instance ``live_enabled``, the
``LLM_ROUTING_ALLOW_LIVE=1`` process flag, a reviewed OutboundAuthorization
(deny-all by default), allowlisted model IDs and ``NVIDIA_API_KEY``.

Model availability and account access on NVIDIA Build are **unverified**; this
adapter is contract/offline tested only and never reports models as confirmed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import httpx

from llm_routing_eval.core import LLMProviderPort

from .async_transport import AsyncClientGroqTransport
from .authorization import OutboundAuthorization
from .httpx_runtime import HttpxPoolSettings, HttpxStreamingClient
from .observability import ObservedProvider, ProviderMetricsSink
from .openai_compatible import (
    ChatBudget,
    OpenAICompatibleChatProvider,
    ProviderProfile,
    validate_budget,
)
from .response_mapping import GroqResponseMapper
from .transport import NVIDIA_BUILD_CHAT_ENDPOINT, GroqTransport

NVIDIA_BUILD_PROFILE = ProviderProfile(
    provider_id="nvidia_build",
    api_key_env="NVIDIA_API_KEY",
    endpoint=NVIDIA_BUILD_CHAT_ENDPOINT,
    code_prefix="NVIDIA",
    # NVIDIA Build's OpenAI-compatible endpoint documents ``max_tokens``.
    completion_tokens_field="max_tokens",
)


@dataclass(frozen=True, slots=True)
class NvidiaBuildProviderOptions:
    """Secret-free, validated resource and security budget for NVIDIA Build."""

    allowed_models: frozenset[str]
    live_enabled: bool = False
    api_key_env: Literal["NVIDIA_API_KEY"] = "NVIDIA_API_KEY"
    max_prompt_chars: int = 50_000
    max_completion_tokens: int = 4096
    max_in_flight: int = 2

    def __post_init__(self) -> None:
        validate_budget(self.budget(), "NVIDIA")
        if self.api_key_env != "NVIDIA_API_KEY":
            raise ValueError("NVIDIA_KEY_ENV_INVALID")

    def budget(self) -> ChatBudget:
        return ChatBudget(
            allowed_models=self.allowed_models,
            live_enabled=self.live_enabled,
            max_prompt_chars=self.max_prompt_chars,
            max_completion_tokens=self.max_completion_tokens,
            max_in_flight=self.max_in_flight,
        )


class NvidiaBuildChatProvider(OpenAICompatibleChatProvider):
    """Secondary-phase provider behind the same LLMProviderPort as Groq."""

    def __init__(
        self,
        options: NvidiaBuildProviderOptions,
        *,
        transport: GroqTransport | None = None,
        outbound_authorization: OutboundAuthorization | None = None,
        mapper: GroqResponseMapper | None = None,
    ) -> None:
        super().__init__(
            NVIDIA_BUILD_PROFILE,
            options.budget(),
            transport=transport,
            outbound_authorization=outbound_authorization,
            mapper=mapper,
        )


@asynccontextmanager
async def managed_httpx_nvidia_provider(
    options: NvidiaBuildProviderOptions,
    *,
    outbound_authorization: OutboundAuthorization | None = None,
    pool_settings: HttpxPoolSettings | None = None,
    mock_transport: httpx.AsyncBaseTransport | None = None,
    metrics_sink: ProviderMetricsSink | None = None,
) -> AsyncIterator[LLMProviderPort]:
    """One shared, TLS-verified HTTPX pool for NVIDIA Build; closed on exit."""
    settings = pool_settings or HttpxPoolSettings()
    max_connections = min(settings.max_connections, options.max_in_flight)
    async with httpx.AsyncClient(
        verify=True,
        trust_env=False,
        follow_redirects=False,
        timeout=httpx.Timeout(
            10.0,
            connect=settings.connect_timeout_seconds,
            pool=settings.pool_timeout_seconds,
        ),
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=min(settings.max_keepalive_connections, max_connections),
        ),
        transport=mock_transport,
    ) as client:
        provider = NvidiaBuildChatProvider(
            options,
            transport=AsyncClientGroqTransport(
                HttpxStreamingClient(client, settings), endpoint=NVIDIA_BUILD_CHAT_ENDPOINT
            ),
            outbound_authorization=outbound_authorization,
        )
        yield ObservedProvider(provider, metrics_sink) if metrics_sink is not None else provider
