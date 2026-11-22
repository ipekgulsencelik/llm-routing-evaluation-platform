"""Ports-and-adapters boundary for LLM providers; Core never imports this module."""

from .async_transport import AsyncClientGroqTransport, AsyncStreamingClient, AsyncStreamingResponse
from .authorization import DenyAllOutbound, OutboundAuthorization
from .groq import GroqChatProvider, GroqProviderOptions
from .httpx_runtime import HttpxPoolSettings, managed_httpx_groq_provider
from .nvidia_build import (
    NvidiaBuildChatProvider,
    NvidiaBuildProviderOptions,
    managed_httpx_nvidia_provider,
)
from .observability import (
    InMemoryProviderMetrics,
    ObservedProvider,
    ProviderErrorBudget,
    ProviderMetricsSink,
    ProviderMetricsSnapshot,
    ProviderTelemetryEvent,
)
from .openai_compatible import OpenAICompatibleChatProvider, ProviderProfile
from .response_mapping import GroqResponseMapper
from .transport import GroqTransport, HttpResult, UrllibGroqTransport

__all__ = [
    "AsyncClientGroqTransport",
    "AsyncStreamingClient",
    "AsyncStreamingResponse",
    "DenyAllOutbound",
    "GroqChatProvider",
    "GroqProviderOptions",
    "GroqResponseMapper",
    "GroqTransport",
    "HttpResult",
    "HttpxPoolSettings",
    "InMemoryProviderMetrics",
    "NvidiaBuildChatProvider",
    "NvidiaBuildProviderOptions",
    "OpenAICompatibleChatProvider",
    "ObservedProvider",
    "ProviderErrorBudget",
    "ProviderMetricsSink",
    "ProviderMetricsSnapshot",
    "ProviderProfile",
    "ProviderTelemetryEvent",
    "OutboundAuthorization",
    "UrllibGroqTransport",
    "managed_httpx_groq_provider",
    "managed_httpx_nvidia_provider",
]
