"""Cancellation-cooperative Groq transport over an injected async streaming client.

The adapter is library-neutral and adds zero runtime dependencies. Callers must
supply and lifecycle-manage a reviewed native async HTTP client (e.g. HTTPX).
It is intentionally NOT the default until a pinned, audited client is approved.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from .transport import (
    _ENDPOINT,
    _MAX_RESPONSE_BYTES,
    HttpResult,
    _checked_endpoint,
    _decode_success_body,
    _retry_after_seconds,
)

_CHUNK_SIZE = 65_536


class AsyncStreamingResponse(Protocol):
    """The minimal streaming response contract required from an HTTP client."""

    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    def aiter_bytes(self, chunk_size: int = ...) -> AsyncIterator[bytes]: ...


class AsyncStreamingClient(Protocol):
    """Subset of HTTPX AsyncClient's streaming API; no vendor in the Core."""

    def stream(
        self,
        method: str,
        url: str,
        *,
        json: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
        follow_redirects: bool,
    ) -> AbstractAsyncContextManager[AsyncStreamingResponse]: ...


class AsyncClientGroqTransport:
    """Transport with structured cancellation and bounded response streaming.

    The injected client's context manager is responsible for closing streams on
    normal return, oversize rejection, exception, timeout, and cancellation.
    The transport does not own or close the injected pool/client.
    """

    def __init__(self, client: AsyncStreamingClient, *, endpoint: str = _ENDPOINT) -> None:
        self._client = client
        self._endpoint = _checked_endpoint(endpoint)

    async def post_chat(
        self,
        body: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> HttpResult:
        headers = {
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        }
        # POST only to a fixed HTTPS endpoint, never follow redirects, and
        # only inspect numeric Retry-After on HTTP 429. Never read error bodies.
        async with self._client.stream(
            "POST",
            self._endpoint,
            json=body,
            headers=headers,
            timeout=timeout_seconds,
            follow_redirects=False,
        ) as response:
            status = response.status_code
            if status != 200:
                hint = _retry_after_seconds(response.headers) if status == 429 else None
                return HttpResult(status_code=status, retry_after_seconds=hint)

            raw = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=_CHUNK_SIZE):
                if len(raw) + len(chunk) > _MAX_RESPONSE_BYTES:
                    return HttpResult(status_code=200)
                raw.extend(chunk)
            return _decode_success_body(bytes(raw))
