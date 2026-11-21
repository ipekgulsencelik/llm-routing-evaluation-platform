"""Groq adapter contract and security tests. Transport is never real HTTP."""

import asyncio
from collections.abc import Mapping

import pytest

from llm_routing_eval.adapters.providers import GroqChatProvider, GroqProviderOptions, HttpResult
from llm_routing_eval.core import (
    CallStatus,
    ErrorCategory,
    MetricOrigin,
    ModelRequest,
)

MODEL = "openai/gpt-oss-20b"


class StubTransport:
    def __init__(self, result: HttpResult | Exception) -> None:
        self._result = result
        self.calls: list[tuple[Mapping[str, object], str, float]] = []

    async def post_chat(
        self, body: Mapping[str, object], *, api_key: str, timeout_seconds: float
    ) -> HttpResult:
        self.calls.append((body, api_key, timeout_seconds))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def request(
    *,
    provider: str = "groq",
    model: str = MODEL,
    generation_parameters: dict[str, str] | None = None,
) -> ModelRequest:
    return ModelRequest(
        run_id="run",
        case_id="case",
        trace_id="trace",
        attempt_id="attempt",
        provider=provider,
        model=model,
        prompt="Known synthetic test prompt",
        timeout_seconds=3,
        temperature=0,
        max_output_tokens=64,
        generation_parameters=generation_parameters or {},
    )


def options(*, enabled: bool = True) -> GroqProviderOptions:
    return GroqProviderOptions(allowed_models=frozenset({MODEL}), live_enabled=enabled)


def test_fails_closed_before_transport_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_ROUTING_ALLOW_LIVE", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "SECRET_DO_NOT_PRINT")
    stub = StubTransport(HttpResult(200, {}))
    response = asyncio.run(GroqChatProvider(options(), transport=stub).generate(request()))
    assert response.status is CallStatus.FAILED
    assert response.error is not None and response.error.code == "GROQ_LIVE_DISABLED"
    assert stub.calls == []
    assert "SECRET_DO_NOT_PRINT" not in response.model_dump_json()


def test_requires_constructor_opt_in_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    stub = StubTransport(HttpResult(200, {}))
    disabled = asyncio.run(
        GroqChatProvider(options(enabled=False), transport=stub).generate(request())
    )
    missing = asyncio.run(GroqChatProvider(options(), transport=stub).generate(request()))
    assert disabled.error is not None and disabled.error.code == "GROQ_LIVE_DISABLED"
    assert missing.error is not None and missing.error.code == "GROQ_API_KEY_MISSING"
    assert not stub.calls


def test_success_maps_content_usage_and_request_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SECRET_GROQ_TOKEN")
    stub = StubTransport(
        HttpResult(
            200,
            {
                "choices": [{"message": {"content": "Synthetic output"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        )
    )
    result = asyncio.run(GroqChatProvider(options(), transport=stub).generate(request()))
    assert result.status is CallStatus.SUCCESS
    assert result.metric_origin is MetricOrigin.OBSERVED
    assert result.content == "Synthetic output"
    assert result.usage is not None
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    assert stub.calls[0][0] == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Known synthetic test prompt"}],
        "stream": False,
        "temperature": 0.0,
        "max_completion_tokens": 64,
    }
    assert stub.calls[0][1] == "SECRET_GROQ_TOKEN"
    assert stub.calls[0][2] == 3
    assert "SECRET_GROQ_TOKEN" not in result.model_dump_json()
    assert "Synthetic output" not in result.model_dump_json(exclude={"content"})


@pytest.mark.parametrize(
    ("http_status", "expected", "retry", "category"),
    [
        (429, CallStatus.RATE_LIMITED, True, ErrorCategory.RATE_LIMIT),
        (408, CallStatus.TIMEOUT, True, ErrorCategory.PROVIDER_TIMEOUT),
        (503, CallStatus.FAILED, True, ErrorCategory.PROVIDER),
        (401, CallStatus.FAILED, False, ErrorCategory.PROVIDER),
    ],
)
def test_http_errors_are_safe_and_retryable_only_when_expected(
    monkeypatch: pytest.MonkeyPatch,
    http_status: int,
    expected: CallStatus,
    retry: bool,
    category: ErrorCategory,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SECRET")
    stub = StubTransport(HttpResult(http_status, {"error": "PRIVATE_HTTP_BODY"}))
    result = asyncio.run(GroqChatProvider(options(), transport=stub).generate(request()))
    assert result.status is expected
    assert result.error is not None
    assert result.error.category is category
    assert result.error.retryable is retry
    assert "PRIVATE_HTTP_BODY" not in result.model_dump_json()


@pytest.mark.parametrize(
    "body",
    [{}, {"choices": []}, {"choices": [{"message": {"content": "   "}}]}],
)
def test_invalid_response_does_not_fabricate_success(
    monkeypatch: pytest.MonkeyPatch, body: Mapping[str, object]
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SECRET")
    result = asyncio.run(
        GroqChatProvider(options(), transport=StubTransport(HttpResult(200, body))).generate(
            request()
        )
    )
    assert result.status is CallStatus.FAILED
    assert result.error is not None and result.error.code == "GROQ_INVALID_RESPONSE"


def test_rejects_wrong_provider_model_and_unsupported_controls_without_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SECRET")
    stub = StubTransport(HttpResult(200, {}))
    adapter = GroqChatProvider(options(), transport=stub)
    results = [
        asyncio.run(adapter.generate(request(provider="fake"))),
        asyncio.run(adapter.generate(request(model="different"))),
        asyncio.run(adapter.generate(request(generation_parameters={"top_k": "5"}))),
    ]
    assert all(item.status is CallStatus.FAILED for item in results)
    assert stub.calls == []


def test_transport_timeout_and_exception_are_redacted_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SECRET")
    timeout = asyncio.run(
        GroqChatProvider(options(), transport=StubTransport(TimeoutError("TOKEN_PRIVATE"))).generate(
            request()
        )
    )
    failed = asyncio.run(
        GroqChatProvider(
            options(), transport=StubTransport(RuntimeError("PRIVATE_SERVICE_DETAILS"))
        ).generate(request())
    )
    assert timeout.status is CallStatus.TIMEOUT
    assert timeout.error is not None and timeout.error.retryable
    assert failed.status is CallStatus.FAILED
    assert failed.error is not None and failed.error.retryable
    assert "TOKEN_PRIVATE" not in timeout.model_dump_json()
    assert "PRIVATE_SERVICE_DETAILS" not in failed.model_dump_json()
