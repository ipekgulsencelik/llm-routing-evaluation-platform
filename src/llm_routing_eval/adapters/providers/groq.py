"""Groq Chat Completions adapter behind LLMProviderPort.

No network requests happen unless both explicit constructor and environment
opt-in are set and an API key is present. No log output or exception text.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Protocol
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from llm_routing_eval.core import (
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_MAX_RESPONSE_BYTES = 1_048_576


@dataclass(frozen=True, slots=True)
class GroqProviderOptions:
    """Config is explicit, secret-free, and immutable."""

    allowed_models: frozenset[str]
    live_enabled: bool = False
    api_key_env: Literal["GROQ_API_KEY"] = "GROQ_API_KEY"

    def __post_init__(self) -> None:
        if not self.allowed_models or any(not model.strip() for model in self.allowed_models):
            raise ValueError("GROQ_ALLOWED_MODELS_REQUIRED")
        if self.api_key_env != "GROQ_API_KEY":
            raise ValueError("GROQ_KEY_ENV_INVALID")


@dataclass(frozen=True, slots=True)
class HttpResult:
    status_code: int
    body: Mapping[str, object] | None = None


class GroqTransport(Protocol):
    async def post_chat(
        self, body: Mapping[str, object], *, api_key: str, timeout_seconds: float
    ) -> HttpResult: ...


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        # A token and sanitized banking prompt must never follow a redirect.
        return None


class UrllibGroqTransport:
    """Bounded HTTPS transport; injected in tests so they never use a network."""

    async def post_chat(
        self, body: Mapping[str, object], *, api_key: str, timeout_seconds: float
    ) -> HttpResult:
        return await asyncio.to_thread(self._post, body, api_key, timeout_seconds)

    @staticmethod
    def _post(body: Mapping[str, object], api_key: str, timeout_seconds: float) -> HttpResult:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(
            _ENDPOINT,
            data=payload,
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            opener = build_opener(_RejectRedirects())
            with opener.open(request, timeout=timeout_seconds) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    return HttpResult(status_code=502)
                decoded = json.loads(raw)
                if not isinstance(decoded, dict):
                    return HttpResult(status_code=502)
                return HttpResult(status_code=response.status, body=decoded)
        except HTTPError as exc:
            # Deliberately do not read or persist provider error payloads.
            return HttpResult(status_code=exc.code)


def _failed(
    request: ModelRequest,
    code: str,
    *,
    category: ErrorCategory = ErrorCategory.PROVIDER,
    status: CallStatus = CallStatus.FAILED,
    retryable: bool = False,
    origin: MetricOrigin = MetricOrigin.SIMULATED,
    latency_ms: float | None = None,
) -> ModelResponse:
    return ModelResponse(
        provider=request.provider,
        model=request.model,
        attempt_id=request.attempt_id,
        status=status,
        metric_origin=origin,
        latency_ms=latency_ms,
        error=ErrorInfo(category=category, code=code, retryable=retryable),
    )


def _token(value: object) -> int | None:
    # The provider may omit token usage; never convert absent values into zero.
    return value if type(value) is int and value >= 0 else None


def _usage(payload: Mapping[str, object]) -> TokenUsage | None:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return None
    prompt = _token(raw.get("prompt_tokens"))
    completion = _token(raw.get("completion_tokens"))
    if prompt is None and completion is None:
        return None
    return TokenUsage(input_tokens=prompt, output_tokens=completion)


def _content(payload: Mapping[str, object]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) and content.strip() else None


class GroqChatProvider:
    """No retries or reporting here: orchestration owns those policies."""

    def __init__(
        self,
        options: GroqProviderOptions,
        *,
        transport: GroqTransport | None = None,
    ) -> None:
        self._options = options
        self._transport = transport or UrllibGroqTransport()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if request.provider != "groq":
            return _failed(request, "GROQ_PROVIDER_MISMATCH")
        if request.model not in self._options.allowed_models:
            return _failed(request, "GROQ_MODEL_NOT_ALLOWLISTED")
        if request.generation_parameters:
            return _failed(request, "GROQ_UNSUPPORTED_PARAMETERS")
        if not self._options.live_enabled or os.getenv("LLM_ROUTING_ALLOW_LIVE") != "1":
            return _failed(request, "GROQ_LIVE_DISABLED")
        api_key = os.getenv(self._options.api_key_env)
        if not api_key or not api_key.strip():
            return _failed(request, "GROQ_API_KEY_MISSING")

        body: dict[str, object] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            body["max_completion_tokens"] = request.max_output_tokens

        started = perf_counter()
        try:
            result = await self._transport.post_chat(
                body, api_key=api_key, timeout_seconds=request.timeout_seconds
            )
        except TimeoutError:
            return _failed(
                request,
                "GROQ_TIMEOUT",
                category=ErrorCategory.PROVIDER_TIMEOUT,
                status=CallStatus.TIMEOUT,
                retryable=True,
                origin=MetricOrigin.OBSERVED,
                latency_ms=(perf_counter() - started) * 1000,
            )
        except Exception:
            # Never propagate request headers, provider response bodies or SDK errors.
            return _failed(
                request,
                "GROQ_TRANSPORT_ERROR",
                retryable=True,
                origin=MetricOrigin.OBSERVED,
                latency_ms=(perf_counter() - started) * 1000,
            )

        elapsed = (perf_counter() - started) * 1000
        status_code = result.status_code
        if status_code == 429:
            return _failed(
                request,
                "GROQ_RATE_LIMIT",
                category=ErrorCategory.RATE_LIMIT,
                status=CallStatus.RATE_LIMITED,
                retryable=True,
                origin=MetricOrigin.OBSERVED,
                latency_ms=elapsed,
            )
        if status_code in (408, 504):
            return _failed(
                request,
                "GROQ_TIMEOUT",
                category=ErrorCategory.PROVIDER_TIMEOUT,
                status=CallStatus.TIMEOUT,
                retryable=True,
                origin=MetricOrigin.OBSERVED,
                latency_ms=elapsed,
            )
        if status_code >= 500:
            return _failed(
                request,
                "GROQ_SERVER_ERROR",
                retryable=True,
                origin=MetricOrigin.OBSERVED,
                latency_ms=elapsed,
            )
        if status_code != 200:
            return _failed(
                request,
                "GROQ_HTTP_ERROR",
                origin=MetricOrigin.OBSERVED,
                latency_ms=elapsed,
            )
        payload = result.body
        if payload is None or (answer := _content(payload)) is None:
            return _failed(
                request,
                "GROQ_INVALID_RESPONSE",
                origin=MetricOrigin.OBSERVED,
                latency_ms=elapsed,
            )
        return ModelResponse(
            provider=request.provider,
            model=request.model,
            attempt_id=request.attempt_id,
            status=CallStatus.SUCCESS,
            metric_origin=MetricOrigin.OBSERVED,
            content=answer,
            usage=_usage(payload),
            latency_ms=elapsed,
        )
