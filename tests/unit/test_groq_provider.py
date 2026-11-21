"""Groq adapter contract and security tests. Transport is never real HTTP."""

import asyncio
from collections.abc import Mapping

import pytest

from llm_routing_eval.adapters.providers import (
    GroqChatProvider,
    GroqProviderOptions,
    GroqTransport,
    HttpResult,
)
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


class FixtureApproval:
    """Allow only the literal synthetic test prompt."""

    def authorize(self, outgoing: ModelRequest) -> bool:
        return outgoing.prompt == "Known synthetic test prompt"


class ExplodingApproval:
    def authorize(self, outgoing: ModelRequest) -> bool:
        _ = outgoing
        raise RuntimeError("SECRET_FROM_PRIVACY_SYSTEM")


def fixture_provider(config: GroqProviderOptions, *, transport: GroqTransport) -> GroqChatProvider:
    return GroqChatProvider(config, transport=transport, outbound_authorization=FixtureApproval())


def test_fails_closed_before_transport_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_ROUTING_ALLOW_LIVE", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "SECRET_DO_NOT_PRINT")
    stub = StubTransport(HttpResult(200, {}))
    response = asyncio.run(fixture_provider(options(), transport=stub).generate(request()))
    assert response.status is CallStatus.FAILED
    assert response.error is not None and response.error.code == "GROQ_LIVE_DISABLED"
    assert stub.calls == []
    assert "SECRET_DO_NOT_PRINT" not in response.model_dump_json()


def test_requires_constructor_opt_in_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    stub = StubTransport(HttpResult(200, {}))
    disabled = asyncio.run(
        fixture_provider(options(enabled=False), transport=stub).generate(request())
    )
    missing = asyncio.run(fixture_provider(options(), transport=stub).generate(request()))
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
    result = asyncio.run(fixture_provider(options(), transport=stub).generate(request()))
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
    result = asyncio.run(fixture_provider(options(), transport=stub).generate(request()))
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
        fixture_provider(options(), transport=StubTransport(HttpResult(200, body))).generate(
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
    adapter = fixture_provider(options(), transport=stub)
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
        fixture_provider(
            options(), transport=StubTransport(TimeoutError("TOKEN_PRIVATE"))
        ).generate(request())
    )
    failed = asyncio.run(
        fixture_provider(
            options(), transport=StubTransport(RuntimeError("PRIVATE_SERVICE_DETAILS"))
        ).generate(request())
    )
    assert timeout.status is CallStatus.TIMEOUT
    assert timeout.error is not None and timeout.error.retryable
    assert failed.status is CallStatus.FAILED
    assert failed.error is not None and failed.error.retryable
    assert "TOKEN_PRIVATE" not in timeout.model_dump_json()
    assert "PRIVATE_SERVICE_DETAILS" not in failed.model_dump_json()


def test_outbound_authorization_defaults_to_deny_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "PRESENT_BUT_NOT_ENOUGH")
    stub = StubTransport(HttpResult(200, {}))
    provider = GroqChatProvider(options(), transport=stub)
    result = asyncio.run(provider.generate(request()))
    assert result.status is CallStatus.FAILED
    assert result.error is not None
    assert result.error.category is ErrorCategory.PRIVACY
    assert result.error.code == "GROQ_PRIVACY_NOT_AUTHORIZED"
    assert stub.calls == []


def test_outbound_authorization_exception_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SECRET")
    stub = StubTransport(HttpResult(200, {}))
    provider = GroqChatProvider(
        options(), transport=stub, outbound_authorization=ExplodingApproval()
    )
    result = asyncio.run(provider.generate(request()))
    assert result.status is CallStatus.FAILED
    assert result.error is not None and result.error.category is ErrorCategory.PRIVACY
    assert "SECRET_FROM_PRIVACY_SYSTEM" not in result.model_dump_json()
    assert stub.calls == []


@pytest.mark.parametrize(
    ("override", "expected_code"),
    [
        ({"max_prompt_chars": 5}, "GROQ_PROMPT_BUDGET_EXCEEDED"),
        ({"max_completion_tokens": 20}, "GROQ_COMPLETION_BUDGET_EXCEEDED"),
    ],
)
def test_budget_rejections_precede_network(
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, int],
    expected_code: str,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SECRET")
    stub = StubTransport(HttpResult(200, {}))
    config = GroqProviderOptions(
        allowed_models=frozenset({MODEL}),
        live_enabled=True,
        **override,
    )
    result = asyncio.run(fixture_provider(config, transport=stub).generate(request()))
    assert result.error is not None and result.error.code == expected_code
    assert stub.calls == []


@pytest.mark.parametrize(
    ("http_result", "expected"),
    [
        (
            HttpResult(
                200,
                {
                    "model": MODEL,
                    "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 12},
                },
            ),
            "GROQ_TRUNCATED",
        ),
        (
            HttpResult(
                200,
                {
                    "model": "OTHER_MODEL",
                    "choices": [{"message": {"content": "answer"}}],
                },
            ),
            "GROQ_MODEL_MISMATCH",
        ),
        (
            HttpResult(
                200,
                {
                    "model": MODEL,
                    "choices": [
                        {"message": {"content": "function call"}, "finish_reason": "tool_calls"}
                    ],
                },
            ),
            "GROQ_UNSUPPORTED_FINISH",
        ),
    ],
)
def test_invalid_or_incomplete_responses_are_not_scored_successfully(
    monkeypatch: pytest.MonkeyPatch, http_result: HttpResult, expected: str
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SECRET")
    result = asyncio.run(
        fixture_provider(options(), transport=StubTransport(http_result)).generate(request())
    )
    assert result.status is CallStatus.FAILED
    assert result.error is not None and result.error.code == expected
    assert result.content is None
    if expected == "GROQ_TRUNCATED":
        assert result.usage is not None and result.usage.output_tokens == 12


def test_observed_usage_details_and_safe_provider_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SECRET")
    payload = {
        "id": "chatcmpl_safe-42",
        "model": MODEL,
        "choices": [{"message": {"content": "valid"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 7},
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }
    response = asyncio.run(
        fixture_provider(options(), transport=StubTransport(HttpResult(200, payload))).generate(
            request()
        )
    )
    assert response.status is CallStatus.SUCCESS
    assert response.provider_request_id == "chatcmpl_safe-42"
    assert response.usage is not None
    assert response.usage.cached_tokens == 7
    assert response.usage.reasoning_tokens == 4


def test_inconsistent_cached_usage_is_not_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SECRET")
    payload = {
        "id": "BAD PRIVATE ID WITH SPACES",
        "choices": [{"message": {"content": "valid"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "cached_tokens": 200},
    }
    response = asyncio.run(
        fixture_provider(options(), transport=StubTransport(HttpResult(200, payload))).generate(
            request()
        )
    )
    assert response.status is CallStatus.SUCCESS
    assert response.provider_request_id is None
    assert response.usage is not None
    assert response.usage.input_tokens == 2
    assert response.usage.cached_tokens is None


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"allowed_models": frozenset({"bad model"})}, "GROQ_ALLOWED_MODELS_REQUIRED"),
        ({"max_prompt_chars": 0}, "GROQ_PROMPT_BUDGET_INVALID"),
        ({"max_completion_tokens": 0}, "GROQ_COMPLETION_BUDGET_INVALID"),
        ({"max_in_flight": 0}, "GROQ_CONCURRENCY_BUDGET_INVALID"),
    ],
)
def test_invalid_resource_policy_is_rejected_before_execution(
    override: dict[str, object], reason: str
) -> None:
    arguments: dict[str, object] = {"allowed_models": frozenset({MODEL})}
    arguments.update(override)
    with pytest.raises(ValueError, match=reason):
        GroqProviderOptions(**arguments)  # type: ignore[arg-type]


def test_concurrency_limit_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SECRET")

    class WaitingTransport:
        def __init__(self) -> None:
            self.active = 0
            self.peak = 0

        async def post_chat(
            self, body: Mapping[str, object], *, api_key: str, timeout_seconds: float
        ) -> HttpResult:
            _ = body, api_key, timeout_seconds
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.002)
            self.active -= 1
            return HttpResult(
                200, {"choices": [{"message": {"content": "test"}, "finish_reason": "stop"}]}
            )

    async def run() -> int:
        transport = WaitingTransport()
        conf = GroqProviderOptions(
            allowed_models=frozenset({MODEL}), live_enabled=True, max_in_flight=2
        )
        provider = fixture_provider(conf, transport=transport)
        await asyncio.gather(*(provider.generate(request()) for _ in range(10)))
        return transport.peak

    assert asyncio.run(run()) == 2
