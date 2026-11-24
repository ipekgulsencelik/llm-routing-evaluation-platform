"""Fail-closed semantic embedding adapter (NVIDIA Build, OpenAI-compatible API).

Implements Core ``EmbeddingPort`` so the kNN router can use semantic vectors
instead of the offline lexical ``HashingEmbedding``. The adapter:

* is deny-by-default: ``live_enabled``, ``LLM_ROUTING_ALLOW_LIVE=1``, a reviewed
  ``EmbeddingOutboundAuthorization`` and ``NVIDIA_API_KEY`` are all required;
* posts only to a fixed HTTPS endpoint, without redirects or proxy env vars;
* never reads non-200 bodies and never puts texts, keys or remote messages in
  exceptions (only safe uppercase codes);
* bounds batch size, text length, response size and vector dimensions.

Texts given to ``embed`` must already have passed the orchestrator's privacy
gate (kNN routers only see sanitized cases and development-only examples).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

NVIDIA_BUILD_EMBEDDINGS_ENDPOINT = "https://integrate.api.nvidia.com/v1/embeddings"
LIVE_ENV_FLAG = "LLM_ROUTING_ALLOW_LIVE"
_MAX_RESPONSE_BYTES = 8_388_608
_MAX_DIMENSIONS = 8192
_SAFE_MODEL_ID = re.compile(r"^[A-Za-z0-9_./-]{1,100}$")


class EmbeddingUnavailableError(RuntimeError):
    """Redacted failure; ``str(exc)`` is always a safe uppercase code."""


class EmbeddingOutboundAuthorization(Protocol):
    """Reviewed capability check for sending already-sanitized texts outbound."""

    def authorize_embedding(self, *, model: str, texts: tuple[str, ...]) -> bool: ...


class DenyAllEmbeddingOutbound:
    """Safe default until privacy owners supply a reviewed authorizer."""

    def authorize_embedding(self, *, model: str, texts: tuple[str, ...]) -> bool:
        _ = model, texts
        return False


@dataclass(frozen=True, slots=True)
class RemoteEmbeddingOptions:
    """Secret-free embedding budget. Model availability is unverified by default."""

    model: str
    live_enabled: bool = False
    api_key_env: Literal["NVIDIA_API_KEY"] = "NVIDIA_API_KEY"
    # Asymmetric retrieval models need an input type; kNN compares like with
    # like, so the same type is used for development examples and queries.
    input_type: Literal["query", "passage"] | None = "query"
    max_batch_size: int = 64
    max_text_chars: int = 8_000
    timeout_seconds: float = 10.0
    expected_dimensions: int | None = None

    def __post_init__(self) -> None:
        if _SAFE_MODEL_ID.fullmatch(self.model) is None:
            raise ValueError("EMBEDDING_MODEL_INVALID")
        if self.api_key_env != "NVIDIA_API_KEY":
            raise ValueError("EMBEDDING_KEY_ENV_INVALID")
        if not 1 <= self.max_batch_size <= 256:
            raise ValueError("EMBEDDING_BATCH_BUDGET_INVALID")
        if not 1 <= self.max_text_chars <= 50_000:
            raise ValueError("EMBEDDING_TEXT_BUDGET_INVALID")
        if not 0 < self.timeout_seconds <= 60 or not math.isfinite(self.timeout_seconds):
            raise ValueError("EMBEDDING_TIMEOUT_INVALID")
        if self.expected_dimensions is not None and not (
            1 <= self.expected_dimensions <= _MAX_DIMENSIONS
        ):
            raise ValueError("EMBEDDING_DIMENSIONS_INVALID")


def _parse_vectors(
    payload: object, count: int, expected: int | None
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(payload, dict):
        raise EmbeddingUnavailableError("EMBEDDING_INVALID_RESPONSE")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != count:
        raise EmbeddingUnavailableError("EMBEDDING_COUNT_MISMATCH")
    ordered: list[tuple[float, ...] | None] = [None] * count
    for position, item in enumerate(data):
        if not isinstance(item, dict):
            raise EmbeddingUnavailableError("EMBEDDING_INVALID_RESPONSE")
        index = item.get("index", position)
        raw = item.get("embedding")
        if type(index) is not int or not 0 <= index < count or ordered[index] is not None:
            raise EmbeddingUnavailableError("EMBEDDING_INDEX_INVALID")
        if not isinstance(raw, list) or not 1 <= len(raw) <= _MAX_DIMENSIONS:
            raise EmbeddingUnavailableError("EMBEDDING_DIMENSIONS_INVALID")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in raw):
            raise EmbeddingUnavailableError("EMBEDDING_NON_FINITE")
        ordered[index] = tuple(float(v) for v in raw)
    vectors = tuple(v for v in ordered if v is not None)
    dimension = expected if expected is not None else len(vectors[0])
    if any(len(v) != dimension for v in vectors):
        raise EmbeddingUnavailableError("EMBEDDING_DIMENSIONS_INVALID")
    if any(not any(value != 0 for value in v) for v in vectors):
        raise EmbeddingUnavailableError("EMBEDDING_ZERO_VECTOR")
    return vectors


class RemoteSemanticEmbedding:
    """EmbeddingPort over an injected, externally managed ``httpx.AsyncClient``."""

    def __init__(
        self,
        options: RemoteEmbeddingOptions,
        client: httpx.AsyncClient,
        *,
        outbound_authorization: EmbeddingOutboundAuthorization | None = None,
        endpoint: str = NVIDIA_BUILD_EMBEDDINGS_ENDPOINT,
    ) -> None:
        if endpoint != NVIDIA_BUILD_EMBEDDINGS_ENDPOINT:
            raise ValueError("EMBEDDING_ENDPOINT_NOT_ALLOWLISTED")
        self._options = options
        self._client = client
        self._endpoint = endpoint
        self._outbound = (
            outbound_authorization
            if outbound_authorization is not None
            else DenyAllEmbeddingOutbound()
        )

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        options = self._options
        if not texts or any(not text.strip() for text in texts):
            raise EmbeddingUnavailableError("EMBEDDING_EMPTY_INPUT")
        if any(len(text) > options.max_text_chars for text in texts):
            raise EmbeddingUnavailableError("EMBEDDING_TEXT_BUDGET_EXCEEDED")
        if not options.live_enabled or os.getenv(LIVE_ENV_FLAG) != "1":
            raise EmbeddingUnavailableError("EMBEDDING_LIVE_DISABLED")
        try:
            permitted = self._outbound.authorize_embedding(model=options.model, texts=texts)
        except Exception:
            permitted = False
        if permitted is not True:
            raise EmbeddingUnavailableError("EMBEDDING_PRIVACY_NOT_AUTHORIZED")
        api_key = os.getenv(options.api_key_env)
        if not api_key or not api_key.strip():
            raise EmbeddingUnavailableError("EMBEDDING_API_KEY_MISSING")
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), options.max_batch_size):
            batch = texts[start : start + options.max_batch_size]
            vectors.extend(await self._post(batch, api_key))
        return tuple(vectors)

    async def _post(self, batch: Sequence[str], api_key: str) -> tuple[tuple[float, ...], ...]:
        body: dict[str, object] = {
            "model": self._options.model,
            "input": list(batch),
            "encoding_format": "float",
        }
        if self._options.input_type is not None:
            body["input_type"] = self._options.input_type
        headers = {
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        }
        try:
            async with self._client.stream(
                "POST",
                self._endpoint,
                json=body,
                headers=headers,
                timeout=self._options.timeout_seconds,
                follow_redirects=False,
            ) as response:
                if response.status_code == 429:
                    raise EmbeddingUnavailableError("EMBEDDING_RATE_LIMIT")
                if response.status_code != 200:
                    raise EmbeddingUnavailableError("EMBEDDING_HTTP_ERROR")
                # Identity encoding is requested; never decompress (bomb risk).
                encoding = response.headers.get("Content-Encoding", "").strip().lower()
                if encoding not in ("", "identity"):
                    raise EmbeddingUnavailableError("EMBEDDING_UNEXPECTED_ENCODING")
                raw = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65_536):
                    if len(raw) + len(chunk) > _MAX_RESPONSE_BYTES:
                        raise EmbeddingUnavailableError("EMBEDDING_RESPONSE_TOO_LARGE")
                    raw.extend(chunk)
        except EmbeddingUnavailableError:
            raise
        except httpx.TimeoutException:
            raise EmbeddingUnavailableError("EMBEDDING_TIMEOUT") from None
        except Exception:
            # Never surface URL, headers, remote text or credentials.
            raise EmbeddingUnavailableError("EMBEDDING_TRANSPORT_ERROR") from None
        try:
            payload = json.loads(bytes(raw))
        except (ValueError, UnicodeError, RecursionError):
            raise EmbeddingUnavailableError("EMBEDDING_INVALID_RESPONSE") from None
        return _parse_vectors(payload, len(batch), self._options.expected_dimensions)


@asynccontextmanager
async def managed_nvidia_embedding(
    options: RemoteEmbeddingOptions,
    *,
    outbound_authorization: EmbeddingOutboundAuthorization | None = None,
    mock_transport: httpx.AsyncBaseTransport | None = None,
    max_connections: int = 2,
) -> AsyncIterator[RemoteSemanticEmbedding]:
    """Own one TLS-verified HTTPX pool; inject ``mock_transport`` only in tests."""
    if not 1 <= max_connections <= 16:
        raise ValueError("EMBEDDING_POOL_INVALID")
    async with httpx.AsyncClient(
        verify=True,
        trust_env=False,
        follow_redirects=False,
        limits=httpx.Limits(
            max_connections=max_connections, max_keepalive_connections=max_connections
        ),
        transport=mock_transport,
    ) as client:
        yield RemoteSemanticEmbedding(
            options, client, outbound_authorization=outbound_authorization
        )
