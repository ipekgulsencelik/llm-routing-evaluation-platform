"""NVIDIA Build compatibility provider: fail-closed gates and wire mapping (no network)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

import httpx
import pytest

from llm_routing_eval.adapters.providers import (
    AsyncClientGroqTransport,
    GroqChatProvider,
    GroqProviderOptions,
    HttpResult,
    InMemoryProviderMetrics,
    NvidiaBuildChatProvider,
    NvidiaBuildProviderOptions,
    UrllibGroqTransport,
    managed_httpx_nvidia_provider,
)
from llm_routing_eval.core import CallStatus, ErrorCategory, MetricOrigin, ModelRequest

MODEL = "meta/llama-3.1-8b-instruct"
PROMPT = "Explicitly synthetic approved NVIDIA test prompt"
SECRET = "IN_MEMORY_FAKE_NVIDIA_CREDENTIAL"


class SyntheticOnlyApproval:
    def authorize(self, request: ModelRequest) -> bool:
        return request.provider == "nvidia_build" and request.prompt == PROMPT


class RecordingTransport:
    def __init__(self, result: HttpResult | BaseException) -> None:
        self.result = result
        self.bodies: list[Mapping[str, object]] = []
        self.keys: list[str] = []

    async def post_chat(
        self, body: Mapping[str, object], *, api_key: str, timeout_seconds: float
    ) -> HttpResult:
        self.bodies.append(body)
        self.keys.append(api_key)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def request(provider: str = "nvidia_build", **overrides: object) -> ModelRequest:
    values: dict[str, object] = {
        "run_id": "run",
        "case_id": "case",
        "trace_id": "trace",
        "attempt_id": "attempt",
        "provider": provider,
        "model": MODEL,
        "prompt": PROMPT,
        "timeout_seconds": 2.0,
        "max_output_tokens": 64,
    }
    values.update(overrides)
    return ModelRequest.model_validate(values)


def options(**overrides: object) -> NvidiaBuildProviderOptions:
    values: dict[str, object] = {"allowed_models": frozenset({MODEL}), "live_enabled": True}
    values.update(overrides)
    return NvidiaBuildProviderOptions(**values)  # type: ignore[arg-type]


def success_body() -> dict[str, object]:
    return {
        "id": "chatcmpl-synthetic",
        "model": MODEL,
        "choices": [{"message": {"content": "Synthetic answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3},
    }


@pytest.fixture
def live_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("NVIDIA_API_KEY", SECRET)


def test_default_is_deny_all_even_with_key_and_flags(live_env: None) -> None:
    transport = RecordingTransport(HttpResult(200, success_body()))
    provider = NvidiaBuildChatProvider(options(), transport=transport)
    response = asyncio.run(provider.generate(request()))
    assert response.status is CallStatus.FAILED
    assert response.error is not None
    assert response.error.code == "NVIDIA_PRIVACY_NOT_AUTHORIZED"
    assert response.error.category is ErrorCategory.PRIVACY
    assert transport.bodies == []


@pytest.mark.parametrize(
    ("env", "opts", "req", "code"),
    [
        ({}, {}, {}, "NVIDIA_LIVE_DISABLED"),
        ({"LLM_ROUTING_ALLOW_LIVE": "1"}, {"live_enabled": False}, {}, "NVIDIA_LIVE_DISABLED"),
        ({"LLM_ROUTING_ALLOW_LIVE": "1"}, {}, {}, "NVIDIA_API_KEY_MISSING"),
        ({"LLM_ROUTING_ALLOW_LIVE": "1"}, {}, {"provider": "groq"}, "NVIDIA_PROVIDER_MISMATCH"),
        ({"LLM_ROUTING_ALLOW_LIVE": "1"}, {}, {"model": "x/other"}, "NVIDIA_MODEL_NOT_ALLOWLISTED"),
        (
            {"LLM_ROUTING_ALLOW_LIVE": "1"},
            {"max_completion_tokens": 10},
            {},
            "NVIDIA_COMPLETION_BUDGET_EXCEEDED",
        ),
    ],
)
def test_fail_closed_gates_precede_transport(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    opts: dict[str, object],
    req: dict[str, object],
    code: str,
) -> None:
    monkeypatch.delenv("LLM_ROUTING_ALLOW_LIVE", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    transport = RecordingTransport(HttpResult(200, success_body()))
    provider = NvidiaBuildChatProvider(
        options(**opts), transport=transport, outbound_authorization=SyntheticOnlyApproval()
    )
    provider_id = str(req.pop("provider", "nvidia_build"))
    response = asyncio.run(provider.generate(request(provider_id, **req)))
    assert response.error is not None and response.error.code == code
    assert transport.bodies == []


def test_groq_key_is_not_accepted_for_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    provider = NvidiaBuildChatProvider(
        options(),
        transport=RecordingTransport(HttpResult(200, success_body())),
        outbound_authorization=SyntheticOnlyApproval(),
    )
    response = asyncio.run(provider.generate(request()))
    assert response.error is not None and response.error.code == "NVIDIA_API_KEY_MISSING"


def test_success_uses_max_tokens_field_and_maps_usage(live_env: None) -> None:
    transport = RecordingTransport(HttpResult(200, success_body()))
    provider = NvidiaBuildChatProvider(
        options(), transport=transport, outbound_authorization=SyntheticOnlyApproval()
    )
    response = asyncio.run(provider.generate(request(temperature=0.0)))
    assert response.status is CallStatus.SUCCESS
    assert response.provider == "nvidia_build"
    assert response.metric_origin is MetricOrigin.OBSERVED
    assert response.usage is not None and response.usage.input_tokens == 9
    body = transport.bodies[0]
    assert body["max_tokens"] == 64 and "max_completion_tokens" not in body
    assert body["temperature"] == 0.0 and body["stream"] is False
    assert transport.keys == [SECRET]
    assert SECRET not in repr(response)


def test_groq_profile_is_unchanged_by_shared_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", SECRET)

    class GroqApproval:
        def authorize(self, req: ModelRequest) -> bool:
            return req.provider == "groq"

    groq_model = "openai/gpt-oss-20b"
    body = success_body() | {"model": groq_model}
    transport = RecordingTransport(HttpResult(200, body))
    provider = GroqChatProvider(
        GroqProviderOptions(allowed_models=frozenset({groq_model}), live_enabled=True),
        transport=transport,
        outbound_authorization=GroqApproval(),
    )
    response = asyncio.run(provider.generate(request("groq", model=groq_model)))
    assert response.status is CallStatus.SUCCESS
    assert transport.bodies[0]["max_completion_tokens"] == 64
    assert "max_tokens" not in transport.bodies[0]


@pytest.mark.parametrize(
    ("result", "status", "code", "retryable"),
    [
        (
            HttpResult(429, retry_after_seconds=2.0),
            CallStatus.RATE_LIMITED,
            "NVIDIA_RATE_LIMIT",
            True,
        ),
        (HttpResult(503), CallStatus.FAILED, "NVIDIA_SERVER_ERROR", True),
        (HttpResult(401), CallStatus.FAILED, "NVIDIA_HTTP_ERROR", False),
        (HttpResult(200), CallStatus.FAILED, "NVIDIA_INVALID_RESPONSE", False),
        (TimeoutError(), CallStatus.TIMEOUT, "NVIDIA_TIMEOUT", True),
        (RuntimeError("remote secret text"), CallStatus.FAILED, "NVIDIA_TRANSPORT_ERROR", True),
    ],
)
def test_error_mapping_uses_nvidia_codes(
    live_env: None,
    result: HttpResult | BaseException,
    status: CallStatus,
    code: str,
    retryable: bool,
) -> None:
    provider = NvidiaBuildChatProvider(
        options(),
        transport=RecordingTransport(result),
        outbound_authorization=SyntheticOnlyApproval(),
    )
    response = asyncio.run(provider.generate(request()))
    assert response.status is status
    assert response.error is not None
    assert (response.error.code, response.error.retryable) == (code, retryable)
    assert "secret" not in repr(response.error)


def test_truncated_completion_is_not_success(live_env: None) -> None:
    body = success_body()
    body["choices"] = [{"message": {"content": "partial"}, "finish_reason": "length"}]
    provider = NvidiaBuildChatProvider(
        options(),
        transport=RecordingTransport(HttpResult(200, body)),
        outbound_authorization=SyntheticOnlyApproval(),
    )
    response = asyncio.run(provider.generate(request()))
    assert response.error is not None and response.error.code == "NVIDIA_TRUNCATED"


def test_transports_reject_unreviewed_endpoints() -> None:
    with pytest.raises(ValueError, match="CHAT_ENDPOINT_NOT_ALLOWLISTED"):
        UrllibGroqTransport(endpoint="https://example.invalid/v1/chat/completions")
    with pytest.raises(ValueError, match="CHAT_ENDPOINT_NOT_ALLOWLISTED"):
        AsyncClientGroqTransport(object(), endpoint="http://integrate.api.nvidia.com/v1")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"allowed_models": frozenset()}, "NVIDIA_ALLOWED_MODELS_REQUIRED"),
        ({"allowed_models": frozenset({"bad model"})}, "NVIDIA_ALLOWED_MODELS_REQUIRED"),
        ({"api_key_env": "GROQ_API_KEY"}, "NVIDIA_KEY_ENV_INVALID"),
        ({"max_in_flight": 0}, "NVIDIA_CONCURRENCY_BUDGET_INVALID"),
        ({"max_prompt_chars": 0}, "NVIDIA_PROMPT_BUDGET_INVALID"),
    ],
)
def test_options_validate(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        options(**overrides)


def test_managed_httpx_posts_only_to_fixed_nvidia_endpoint(live_env: None) -> None:
    seen: list[httpx.Request] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=success_body())

    metrics = InMemoryProviderMetrics(max_events=20)

    async def run() -> CallStatus:
        async with managed_httpx_nvidia_provider(
            options(),
            outbound_authorization=SyntheticOnlyApproval(),
            mock_transport=httpx.MockTransport(handler),
            metrics_sink=metrics,
        ) as provider:
            return (await provider.generate(request())).status

    assert asyncio.run(run()) is CallStatus.SUCCESS
    assert len(seen) == 1
    sent = seen[0]
    assert str(sent.url) == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert sent.headers["Authorization"] == "Bearer " + SECRET
    assert json.loads(sent.content)["max_tokens"] == 64
    snapshot = metrics.snapshot()
    assert SECRET not in repr(snapshot) and PROMPT not in repr(snapshot)


def test_managed_httpx_redirect_is_not_followed(live_env: None) -> None:
    calls: list[str] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        return httpx.Response(307, headers={"Location": "https://attacker.invalid/"})

    async def run() -> str | None:
        async with managed_httpx_nvidia_provider(
            options(),
            outbound_authorization=SyntheticOnlyApproval(),
            mock_transport=httpx.MockTransport(handler),
        ) as provider:
            response = await provider.generate(request())
            return response.error.code if response.error else None

    assert asyncio.run(run()) == "NVIDIA_HTTP_ERROR"
    assert calls == ["https://integrate.api.nvidia.com/v1/chat/completions"]
