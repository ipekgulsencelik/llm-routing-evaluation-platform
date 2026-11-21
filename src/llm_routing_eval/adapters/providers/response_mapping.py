"""Pure Groq wire-to-Core mapping; no IO, retries, logs or secrets."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from llm_routing_eval.core import (
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)

from .transport import HttpResult

_SAFE_PROVIDER_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def failed(
    request: ModelRequest,
    code: str,
    *,
    category: ErrorCategory = ErrorCategory.PROVIDER,
    status: CallStatus = CallStatus.FAILED,
    retryable: bool = False,
    origin: MetricOrigin = MetricOrigin.SIMULATED,
    latency_ms: float | None = None,
    usage: TokenUsage | None = None,
    retry_after_seconds: float | None = None,
) -> ModelResponse:
    return ModelResponse(
        provider=request.provider,
        model=request.model,
        attempt_id=request.attempt_id,
        status=status,
        metric_origin=origin,
        latency_ms=latency_ms,
        usage=usage,
        error=ErrorInfo(
            category=category,
            code=code,
            retryable=retryable,
            retry_after_seconds=retry_after_seconds,
        ),
    )


def _token(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _details(raw: Mapping[str, object], name: str, key: str) -> int | None:
    details = raw.get(name)
    return _token(details.get(key)) if isinstance(details, dict) else None


def _usage(payload: Mapping[str, object]) -> TokenUsage | None:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return None
    prompt = _token(raw.get("prompt_tokens"))
    completion = _token(raw.get("completion_tokens"))
    cached = _details(raw, "prompt_tokens_details", "cached_tokens")
    reasoning = _details(raw, "completion_tokens_details", "reasoning_tokens")
    if cached is None:
        cached = _token(raw.get("cached_tokens"))
    if reasoning is None:
        reasoning = _token(raw.get("reasoning_tokens"))
    if prompt is not None and cached is not None and cached > prompt:
        cached = None  # Reject inconsistent sub-count, not all observed usage.
    if all(item is None for item in (prompt, completion, cached, reasoning)):
        return None
    return TokenUsage(
        input_tokens=prompt,
        output_tokens=completion,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )


def _choice(payload: Mapping[str, object]) -> tuple[str | None, str | None]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None, None
    first = choices[0]
    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    finish = first.get("finish_reason")
    return (
        content if isinstance(content, str) and content.strip() else None,
        finish if isinstance(finish, str) else None,
    )


class GroqResponseMapper:
    """A single responsibility: validate and normalize one completed HTTP result."""

    def map(self, request: ModelRequest, result: HttpResult, *, elapsed_ms: float) -> ModelResponse:
        def http_failure(
            code: str,
            *,
            category: ErrorCategory = ErrorCategory.PROVIDER,
            status: CallStatus = CallStatus.FAILED,
            retryable: bool = False,
            usage: TokenUsage | None = None,
            retry_after_seconds: float | None = None,
        ) -> ModelResponse:
            return failed(
                request,
                code,
                category=category,
                status=status,
                retryable=retryable,
                origin=MetricOrigin.OBSERVED,
                latency_ms=elapsed_ms,
                usage=usage,
                retry_after_seconds=retry_after_seconds,
            )

        status = result.status_code
        if status == 429:
            raw_hint = result.retry_after_seconds
            safe_hint = (
                raw_hint
                if raw_hint is not None and math.isfinite(raw_hint) and 0 <= raw_hint <= 86_400
                else None
            )
            return http_failure(
                "GROQ_RATE_LIMIT",
                category=ErrorCategory.RATE_LIMIT,
                status=CallStatus.RATE_LIMITED,
                retryable=True,
                retry_after_seconds=safe_hint,
            )
        if status in (408, 504):
            return http_failure(
                "GROQ_TIMEOUT",
                category=ErrorCategory.PROVIDER_TIMEOUT,
                status=CallStatus.TIMEOUT,
                retryable=True,
            )
        if status == 409:
            # Groq SDKs classify HTTP 409 as transient; attempts remain
            # bounded by the application's existing global call budget.
            return http_failure("GROQ_CONFLICT", retryable=True)
        if status >= 500:
            return http_failure("GROQ_SERVER_ERROR", retryable=True)
        if status != 200:
            return http_failure("GROQ_HTTP_ERROR")
        payload = result.body
        if payload is None:
            return http_failure("GROQ_INVALID_RESPONSE")
        model = payload.get("model")
        if model is not None and model != request.model:
            return http_failure("GROQ_MODEL_MISMATCH")
        answer, finish_reason = _choice(payload)
        usage = _usage(payload)
        if finish_reason == "length":
            return http_failure("GROQ_TRUNCATED", usage=usage)
        if finish_reason not in (None, "stop"):
            return http_failure("GROQ_UNSUPPORTED_FINISH", usage=usage)
        if answer is None:
            return http_failure("GROQ_INVALID_RESPONSE", usage=usage)
        provider_id = payload.get("id")
        safe_id = (
            provider_id
            if isinstance(provider_id, str) and _SAFE_PROVIDER_ID.fullmatch(provider_id)
            else None
        )
        return ModelResponse(
            provider=request.provider,
            model=request.model,
            attempt_id=request.attempt_id,
            provider_request_id=safe_id,
            status=CallStatus.SUCCESS,
            metric_origin=MetricOrigin.OBSERVED,
            content=answer,
            usage=usage,
            latency_ms=elapsed_ms,
        )
