"""Shared fail-closed Chat Completions provider for OpenAI-compatible APIs.

Groq (primary) and NVIDIA Build (secondary compatibility phase) expose the same
wire format. The provider-specific parts are a fixed, reviewed profile: public
provider ID, credential environment variable, HTTPS endpoint, error-code prefix
and the name of the completion-token budget field. Nothing is URL-configurable.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from time import perf_counter

from llm_routing_eval.core import (
    CallStatus,
    ErrorCategory,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
)

from .authorization import DenyAllOutbound, OutboundAuthorization
from .response_mapping import GroqResponseMapper, failed
from .transport import (
    GroqNetworkError,
    GroqTLSVerificationError,
    GroqTransport,
)

SAFE_MODEL_ID = re.compile(r"^[A-Za-z0-9_./-]{1,100}$")
LIVE_ENV_FLAG = "LLM_ROUTING_ALLOW_LIVE"


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    """Static, reviewed description of one OpenAI-compatible provider."""

    provider_id: str
    api_key_env: str
    endpoint: str
    code_prefix: str
    completion_tokens_field: str


@dataclass(frozen=True, slots=True)
class ChatBudget:
    """Validated, secret-free resource budget shared by provider option types."""

    allowed_models: frozenset[str]
    live_enabled: bool
    max_prompt_chars: int
    max_completion_tokens: int
    max_in_flight: int


def validate_budget(budget: ChatBudget, prefix: str) -> None:
    if not budget.allowed_models or any(
        SAFE_MODEL_ID.fullmatch(model) is None for model in budget.allowed_models
    ):
        raise ValueError(prefix + "_ALLOWED_MODELS_REQUIRED")
    if not 1 <= budget.max_prompt_chars <= 200_000:
        raise ValueError(prefix + "_PROMPT_BUDGET_INVALID")
    if not 1 <= budget.max_completion_tokens <= 32_768:
        raise ValueError(prefix + "_COMPLETION_BUDGET_INVALID")
    if not 1 <= budget.max_in_flight <= 32:
        raise ValueError(prefix + "_CONCURRENCY_BUDGET_INVALID")


class OpenAICompatibleChatProvider:
    """Composes privacy authorization, resource limits, transport and response mapper.

    The benchmark orchestrator retains ownership of retry and trace policies.
    """

    def __init__(
        self,
        profile: ProviderProfile,
        budget: ChatBudget,
        *,
        transport: GroqTransport,
        outbound_authorization: OutboundAuthorization | None = None,
        mapper: GroqResponseMapper | None = None,
    ) -> None:
        validate_budget(budget, profile.code_prefix)
        self._profile = profile
        self._budget = budget
        self._transport = transport
        self._outbound = (
            outbound_authorization if outbound_authorization is not None else DenyAllOutbound()
        )
        self._mapper = (
            mapper if mapper is not None else GroqResponseMapper(code_prefix=profile.code_prefix)
        )
        self._semaphore = asyncio.Semaphore(budget.max_in_flight)

    def _code(self, suffix: str) -> str:
        return self._profile.code_prefix + "_" + suffix

    async def generate(self, request: ModelRequest) -> ModelResponse:
        budget = self._budget
        if request.provider != self._profile.provider_id:
            return failed(request, self._code("PROVIDER_MISMATCH"))
        if request.model not in budget.allowed_models:
            return failed(request, self._code("MODEL_NOT_ALLOWLISTED"))
        if request.generation_parameters:
            return failed(request, self._code("UNSUPPORTED_PARAMETERS"))
        if len(request.prompt) > budget.max_prompt_chars:
            return failed(request, self._code("PROMPT_BUDGET_EXCEEDED"))
        if (
            request.max_output_tokens is not None
            and request.max_output_tokens > budget.max_completion_tokens
        ):
            return failed(request, self._code("COMPLETION_BUDGET_EXCEEDED"))
        if not budget.live_enabled or os.getenv(LIVE_ENV_FLAG) != "1":
            return failed(request, self._code("LIVE_DISABLED"))
        try:
            permitted = self._outbound.authorize(request)
        except Exception:
            permitted = False
        if not permitted:
            return failed(
                request, self._code("PRIVACY_NOT_AUTHORIZED"), category=ErrorCategory.PRIVACY
            )
        api_key = os.getenv(self._profile.api_key_env)
        if not api_key or not api_key.strip():
            return failed(request, self._code("API_KEY_MISSING"))

        body: dict[str, object] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            body[self._profile.completion_tokens_field] = request.max_output_tokens

        async with self._semaphore:
            started = perf_counter()

            def observed_failure(
                suffix: str,
                *,
                category: ErrorCategory = ErrorCategory.PROVIDER,
                status: CallStatus = CallStatus.FAILED,
                retryable: bool,
            ) -> ModelResponse:
                return failed(
                    request,
                    self._code(suffix),
                    category=category,
                    status=status,
                    retryable=retryable,
                    origin=MetricOrigin.OBSERVED,
                    latency_ms=(perf_counter() - started) * 1000,
                )

            try:
                result = await self._transport.post_chat(
                    body, api_key=api_key, timeout_seconds=request.timeout_seconds
                )
            except GroqTLSVerificationError:
                return observed_failure("TLS_VERIFICATION_FAILED", retryable=False)
            except GroqNetworkError:
                return observed_failure("NETWORK_ERROR", retryable=True)
            except TimeoutError:
                return observed_failure(
                    "TIMEOUT",
                    category=ErrorCategory.PROVIDER_TIMEOUT,
                    status=CallStatus.TIMEOUT,
                    retryable=True,
                )
            except Exception:
                # Never propagate remote headers, bodies, credentials or exception text.
                return observed_failure("TRANSPORT_ERROR", retryable=True)
            return self._mapper.map(request, result, elapsed_ms=(perf_counter() - started) * 1000)
