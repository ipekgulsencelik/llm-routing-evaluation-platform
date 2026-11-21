"""Managed, injectable native-async HTTPX client for Groq.

Composition and lifecycle only: no evaluation, retry or privacy decisions.
The provider must still pass all existing fail-closed live authorization gates.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

import httpx

from .async_transport import AsyncClientGroqTransport, AsyncStreamingClient
from .authorization import OutboundAuthorization
from .groq import GroqChatProvider, GroqProviderOptions


@dataclass(frozen=True, slots=True)
class HttpxPoolSettings:
    """Bounded, secret-free pool settings; no environment-controlled proxies."""

    max_connections: int = 4
    max_keepalive_connections: int = 2
    connect_timeout_seconds: float = 3.0
    pool_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_connections <= 32:
            raise ValueError("HTTPX_POOL_CONNECTIONS_INVALID")
        if not 0 <= self.max_keepalive_connections <= self.max_connections:
            raise ValueError("HTTPX_KEEPALIVE_CONNECTIONS_INVALID")
        if not 0 < self.connect_timeout_seconds <= 30:
            raise ValueError("HTTPX_CONNECT_TIMEOUT_INVALID")
        if not 0 < self.pool_timeout_seconds <= 30:
            raise ValueError("HTTPX_POOL_TIMEOUT_INVALID")


@asynccontextmanager
async def managed_httpx_groq_provider(
    options: GroqProviderOptions,
    *,
    outbound_authorization: OutboundAuthorization | None = None,
    pool_settings: HttpxPoolSettings | None = None,
    mock_transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[GroqChatProvider]:
    """Own one shared HTTPX client/pool across requests and close on shutdown.

    Inject mock_transport only in tests. By default, all outgoing operations
    are denied until application and privacy owners supply a reviewed
    outbound_authorization and explicitly enable the existing provider gates.

    No client instances are created per individual model invocation. The
    transport keeps cancellation cooperative through HTTPX's async stream API.
    """
    settings = pool_settings or HttpxPoolSettings()
    # The pool is never looser than Groq's per-instance concurrency budget.
    max_connections = min(settings.max_connections, options.max_in_flight)
    max_keepalive = min(settings.max_keepalive_connections, max_connections)
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
            max_keepalive_connections=max_keepalive,
        ),
        transport=mock_transport,
    ) as client:
        provider = GroqChatProvider(
            options,
            transport=AsyncClientGroqTransport(cast(AsyncStreamingClient, client)),
            outbound_authorization=outbound_authorization,
        )
        yield provider
