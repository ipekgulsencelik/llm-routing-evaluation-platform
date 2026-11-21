"""Managed, injectable native-async HTTPX client for Groq.

Composition and lifecycle only: no evaluation, retry or privacy decisions.
The provider must still pass all existing fail-closed live authorization gates.
"""

from __future__ import annotations

import ssl
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from .async_transport import AsyncClientGroqTransport, AsyncStreamingResponse
from .authorization import OutboundAuthorization
from .groq import GroqChatProvider, GroqProviderOptions
from .observability import ObservedProvider, ProviderMetricsSink
from .transport import GroqNetworkError, GroqTLSVerificationError

from llm_routing_eval.core import LLMProviderPort


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


def _is_tls_verification_failure(exc: BaseException) -> bool:
    """Inspect bounded exception-chain *types*; never inspect/log remote text."""
    current: BaseException | None = exc
    seen: set[int] = set()
    for _ in range(10):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if isinstance(current, (ssl.SSLCertVerificationError, ssl.CertificateError)):
            return True
        current = current.__cause__ or current.__context__
    return False


class HttpxStreamingClient:
    """Bridge HTTPX to the minimal protocol without leaking SDK types into Core."""

    def __init__(self, client: httpx.AsyncClient, settings: HttpxPoolSettings) -> None:
        self._client = client
        self._settings = settings

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        *,
        json: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
        follow_redirects: bool,
    ) -> AsyncIterator[AsyncStreamingResponse]:
        # Preserve per-phase connect and pool budgets instead of allowing the
        # transport's per-request timeout to override the entire pool policy.
        request_timeout = httpx.Timeout(
            timeout,
            connect=min(self._settings.connect_timeout_seconds, timeout),
            pool=min(self._settings.pool_timeout_seconds, timeout),
        )
        try:
            async with self._client.stream(
                method,
                url,
                json=json,
                headers=headers,
                timeout=request_timeout,
                follow_redirects=follow_redirects,
            ) as response:
                yield response
        except httpx.TimeoutException:
            # Normalize SDK timeouts to the existing provider contract; never
            # include URL, request headers, exception strings or provider text.
            raise TimeoutError("GROQ_HTTP_TIMEOUT") from None
        except httpx.TransportError as exc:
            if _is_tls_verification_failure(exc):
                # Certificate, CA or hostname mismatch is not transient:
                # retrying would waste budget or weaken the security boundary.
                raise GroqTLSVerificationError("GROQ_TLS_VERIFICATION_FAILED") from None
            raise GroqNetworkError("GROQ_NETWORK_ERROR") from None


@asynccontextmanager
async def managed_httpx_groq_provider(
    options: GroqProviderOptions,
    *,
    outbound_authorization: OutboundAuthorization | None = None,
    pool_settings: HttpxPoolSettings | None = None,
    mock_transport: httpx.AsyncBaseTransport | None = None,
    metrics_sink: ProviderMetricsSink | None = None,
) -> AsyncIterator[LLMProviderPort]:
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
            transport=AsyncClientGroqTransport(HttpxStreamingClient(client, settings)),
            outbound_authorization=outbound_authorization,
        )
        yield ObservedProvider(provider, metrics_sink) if metrics_sink is not None else provider
