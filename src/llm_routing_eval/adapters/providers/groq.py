"""Fail-closed Groq provider composition behind LLMProviderPort.

Single responsibility: validate outbound intent, build a request and delegate
transport and response normalization to independently testable components.
Real live calls require a reviewed OutboundAuthorization implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .authorization import OutboundAuthorization
from .openai_compatible import (
    ChatBudget,
    OpenAICompatibleChatProvider,
    ProviderProfile,
    validate_budget,
)
from .response_mapping import GroqResponseMapper
from .transport import _ENDPOINT, GroqTransport

GROQ_PROFILE = ProviderProfile(
    provider_id="groq",
    api_key_env="GROQ_API_KEY",
    endpoint=_ENDPOINT,
    code_prefix="GROQ",
    completion_tokens_field="max_completion_tokens",
)


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
        validate_budget(self.budget(), "GROQ")
        if self.api_key_env != "GROQ_API_KEY":
            raise ValueError("GROQ_KEY_ENV_INVALID")

    def budget(self) -> ChatBudget:
        return ChatBudget(
            allowed_models=self.allowed_models,
            live_enabled=self.live_enabled,
            max_prompt_chars=self.max_prompt_chars,
            max_completion_tokens=self.max_completion_tokens,
            max_in_flight=self.max_in_flight,
        )


class GroqChatProvider(OpenAICompatibleChatProvider):
    """Primary provider. Live use: ``managed_httpx_groq_provider`` (cancellable).

    Without an injected transport every request fails closed with
    ``GROQ_TRANSPORT_NOT_CONFIGURED``; ``UrllibGroqTransport`` must be injected
    explicitly and cannot cancel a timed-out request (see review #12 F1).
    """

    def __init__(
        self,
        options: GroqProviderOptions,
        *,
        transport: GroqTransport | None = None,
        outbound_authorization: OutboundAuthorization | None = None,
        mapper: GroqResponseMapper | None = None,
    ) -> None:
        super().__init__(
            GROQ_PROFILE,
            options.budget(),
            transport=transport,
            outbound_authorization=outbound_authorization,
            mapper=mapper,
        )
