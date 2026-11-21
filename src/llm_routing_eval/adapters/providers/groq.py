"""Fail-closed Groq provider composition behind LLMProviderPort.

Single responsibility: validate outbound intent, build a request and delegate
transport and response normalization to independently testable components.
Real live calls require a reviewed OutboundAuthorization implementation.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from llm_routing_eval.core import (
    CallStatus,
    ErrorCategory,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
)

from .authorization import DenyAllOutbound, OutboundAuthorization
from .response_mapping import GroqResponseMapper, failed
from .transport import GroqTransport, UrllibGroqTransport

_SAFE_MODEL_ID = re.compile(r"^[A-Za-z0-9_./-]{1,100}$")


@dataclass(frozen=True, slots=True)
class GroqProviderOptions:
    """Secret-free, validated per-provider resource and security budget."""

    allowed_models: frozenset[str]
    live_enabled: bool = False
    api_key_env: Literal["GROQ_API_KEY"] = "GROQ_API_KEY"
    max_prompt_chars: int = 50_000
    max_completion_tokens: int = 8192
    max_in_flight: int = 4

    def __post_init__(self) -> None:
        if not self.allowed_models or any(
            _SAFE_MODEL_ID.fullmatch(model) is None for model in self.allowed_models
        ):
            raise ValueError("GROQ_ALLOWED_MODELS_REQUIRED")
        if self.api_key_env != "GROQ_API_KEY":
            raise ValueError("GROQ_KEY_ENV_INVALID")
        if not 1 <= self.max_prompt_chars <= 200_000:
            raise ValueError("GROQ_PROMPT_BUDGET_INVALID")
        if not 1 <= self.max_completion_tokens <= 32_768:
            raise ValueError("GROQ_COMPLETION_BUDGET_INVALID")
        if not 1 <= self.max_in_flight <= 32:
            raise ValueError("GROQ_CONCURRENCY_BUDGET_INVALID")


class GroqChatProvider:
    """Composes privacy authorization, resource limits, transport and response mapper.

    The benchmark orchestrator retains ownership of retry and trace policies.
    """

    def __init__(
        self,
        options: GroqProviderOptions,
        *,
        transport: GroqTransport | None = None,
        outbound_authorization: OutboundAuthorization | None = None,
        mapper: GroqResponseMapper | None = None,
    ) -> None:
        self._options = options
        self._transport = transport if transport is not None else UrllibGroqTransport()
        self._outbound = (
            outbound_authorization if outbound_authorization is not None else DenyAllOutbound()
        )
        self._mapper = mapper if mapper is not None else GroqResponseMapper()
        self._semaphore = asyncio.Semaphore(options.max_in_flight)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if request.provider != "groq":
            return failed(request, "GROQ_PROVIDER_MISMATCH")
        if request.model not in self._options.allowed_models:
            return failed(request, "GROQ_MODEL_NOT_ALLOWLISTED")
        if request.generation_parameters:
            return failed(request, "GROQ_UNSUPPORTED_PARAMETERS")
        if len(request.prompt) > self._options.max_prompt_chars:
            return failed(request, "GROQ_PROMPT_BUDGET_EXCEEDED")
        if (
            request.max_output_tokens is not None
            and request.max_output_tokens > self._options.max_completion_tokens
        ):
            return failed(request, "GROQ_COMPLETION_BUDGET_EXCEEDED")
        if not self._options.live_enabled or os.getenv("LLM_ROUTING_ALLOW_LIVE") != "1":
            return failed(request, "GROQ_LIVE_DISABLED")
        try:
            permitted = self._outbound.authorize(request)
        except Exception:
            permitted = False
        if not permitted:
            return failed(request, "GROQ_PRIVACY_NOT_AUTHORIZED", category=ErrorCategory.PRIVACY)
        api_key = os.getenv(self._options.api_key_env)
        if not api_key or not api_key.strip():
            return failed(request, "GROQ_API_KEY_MISSING")

        body: dict[str, object] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            body["max_completion_tokens"] = request.max_output_tokens

        async with self._semaphore:
            started = perf_counter()
            try:
                result = await self._transport.post_chat(
                    body, api_key=api_key, timeout_seconds=request.timeout_seconds
                )
            except TimeoutError:
                return failed(
                    request,
                    "GROQ_TIMEOUT",
                    category=ErrorCategory.PROVIDER_TIMEOUT,
                    status=CallStatus.TIMEOUT,
                    retryable=True,
                    origin=MetricOrigin.OBSERVED,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            except Exception:
                # Never propagate remote headers, bodies, credentials or exception text.
                return failed(
                    request,
                    "GROQ_TRANSPORT_ERROR",
                    retryable=True,
                    origin=MetricOrigin.OBSERVED,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            return self._mapper.map(request, result, elapsed_ms=(perf_counter() - started) * 1000)
