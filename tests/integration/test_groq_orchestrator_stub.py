"""Synthetic Groq wire-contract integration: never contacts an external API."""

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest

from llm_routing_eval.adapters.providers import (
    DenyAllOutbound,
    GroqChatProvider,
    GroqProviderOptions,
    HttpResult,
)
from llm_routing_eval.application import BenchmarkOrchestrator, RunOptions
from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    EvaluationStatus,
    ExecutionStatus,
    ModelRequest,
    StrategyId,
    TraceStepKind,
)
from llm_routing_eval.demo.fakes import (
    DEMO_PROMPTS,
    DEMO_RESPONSE,
    AllowlistedDemoPrivacy,
    DemoFixtureEvaluator,
    InMemoryDemoRepository,
)
from llm_routing_eval.routing import build_router, read_routing_settings

ROOT = Path(__file__).resolve().parents[2]


class SyntheticOnlyApproval:
    """A TEST DOUBLE. Must never be wired into a production CLI."""

    def authorize(self, request: ModelRequest) -> bool:
        return request.provider == "groq" and request.prompt in DEMO_PROMPTS


class StubGroqTransport:
    """Deterministic mock wire transport; no urllib, socket or external call."""

    def __init__(self, *, rate_limit_once: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self._rate_limit_once = rate_limit_once

    async def post_chat(
        self, body: Mapping[str, object], *, api_key: str, timeout_seconds: float
    ) -> HttpResult:
        assert api_key == "SYNTHETIC_TEST_CREDENTIAL"
        assert timeout_seconds == 3.0
        messages = body["messages"]
        assert isinstance(messages, list)
        message = messages[0]
        assert isinstance(message, dict)
        prompt = message["content"]
        assert isinstance(prompt, str)
        assert prompt in DEMO_PROMPTS
        model = body["model"]
        assert isinstance(model, str)
        self.calls.append((model, prompt))
        if self._rate_limit_once and len(self.calls) == 1:
            return HttpResult(429, {"error": "UNSAFE_PROVIDER_ERROR_SHOULD_NOT_APPEAR"})
        return HttpResult(
            200,
            {
                "id": "test-safe-id",
                "model": model,
                "choices": [{"message": {"content": DEMO_RESPONSE}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 16, "completion_tokens": 8},
            },
        )


def synthetic_cases() -> tuple[BenchmarkCase, ...]:
    return (
        BenchmarkCase(
            case_id="sandbox-card",
            input_text="Banka kartı ile kredi kartı arasındaki fark nedir?",
            domain="turkish_banking",
            scenario_type="card_general",
            metadata={"synthetic": True},
        ),
        BenchmarkCase(
            case_id="sandbox-account",
            input_text="Banka hesabı açmak için genel süreç nedir?",
            domain="turkish_banking",
            scenario_type="account_general",
            metadata={"synthetic": True},
        ),
    )


async def execute(
    strategy: StrategyId,
    transport: StubGroqTransport,
    *,
    authorized: bool,
    max_retries: int = 0,
):
    policy = read_routing_settings(ROOT / "config/routing.yaml")
    provider = GroqChatProvider(
        GroqProviderOptions(
            allowed_models=frozenset({policy.models.small, policy.models.large}),
            live_enabled=True,
        ),
        transport=transport,
        outbound_authorization=SyntheticOnlyApproval() if authorized else DenyAllOutbound(),
    )

    async def no_sleep(delay: float) -> None:
        _ = delay

    runner = BenchmarkOrchestrator(
        router=await build_router(strategy, policy),
        provider=provider,
        evaluator=DemoFixtureEvaluator(),
        privacy=AllowlistedDemoPrivacy(),
        repository=InMemoryDemoRepository(),
        sleep=no_sleep,
    )
    return await runner.run(
        synthetic_cases(),
        RunOptions(
            strategy=strategy,
            provider="groq",
            allowed_models=(policy.models.small, policy.models.large),
            routing_policy_sha256=policy.fingerprint(),
            timeout_seconds=3.0,
            max_model_calls_per_case=2,
            max_retries=max_retries,
        ),
    )


@pytest.mark.parametrize(
    ("strategy", "expected_models"),
    [
        (StrategyId.ALWAYS_LARGE, ("openai/gpt-oss-120b", "openai/gpt-oss-120b")),
        (StrategyId.RULE_BASED, ("openai/gpt-oss-20b", "openai/gpt-oss-20b")),
        (StrategyId.KNN, ("openai/gpt-oss-20b", "openai/gpt-oss-20b")),
        (
            StrategyId.CASCADE,
            ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"),
        ),
    ],
)
def test_all_strategies_compose_with_groq_stub_without_network(
    monkeypatch: pytest.MonkeyPatch, strategy: StrategyId, expected_models: tuple[str, ...]
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SYNTHETIC_TEST_CREDENTIAL")
    stub = StubGroqTransport()
    result = asyncio.run(execute(strategy, stub, authorized=True))
    calls = [
        step
        for trace in result.traces
        for step in trace.steps
        if step.kind is TraceStepKind.PROVIDER_ATTEMPT
    ]
    assert tuple(call.model for call in calls) == expected_models
    assert tuple(model for model, _ in stub.calls) == expected_models
    assert all(call.call_status is CallStatus.SUCCESS for call in calls)
    assert all(trace.status is ExecutionStatus.COMPLETED for trace in result.traces)
    assert all(ev.status is EvaluationStatus.COMPLETED for ev in result.evaluations)
    assert all(ev.quality_scores["demo_fixture_match"] == 1.0 for ev in result.evaluations)
    assert result.report.provenance["routing_policy_sha256"].startswith("sha256:")
    assert result.report.run_status == "COMPLETED"
    # No prompt or provider content may enter trace/report JSON.
    stored = result.report.model_dump_json() + "".join(t.model_dump_json() for t in result.traces)
    assert "Banka kartı" not in stored
    assert "Banka hesabı" not in stored
    assert DEMO_RESPONSE not in stored
    assert "SYNTHETIC_TEST_CREDENTIAL" not in stored


def test_outbound_authorization_denies_even_with_key_and_live_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SYNTHETIC_TEST_CREDENTIAL")
    transport = StubGroqTransport()
    result = asyncio.run(execute(StrategyId.ALWAYS_LARGE, transport, authorized=False))
    assert transport.calls == []
    assert all(trace.status is ExecutionStatus.FAILED for trace in result.traces)
    assert all(ev.status is EvaluationStatus.NOT_EVALUATED for ev in result.evaluations)
    events = [
        step
        for trace in result.traces
        for step in trace.steps
        if step.kind is TraceStepKind.PROVIDER_ATTEMPT
    ]
    assert all(ev.call_status is CallStatus.FAILED for ev in events)
    assert all(
        ev.error is not None and ev.error.code == "GROQ_PRIVACY_NOT_AUTHORIZED" for ev in events
    )


def test_rate_limit_retries_are_traced_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SYNTHETIC_TEST_CREDENTIAL")
    stub = StubGroqTransport(rate_limit_once=True)
    result = asyncio.run(execute(StrategyId.ALWAYS_LARGE, stub, authorized=True, max_retries=1))
    steps = [
        step
        for trace in result.traces
        for step in trace.steps
        if step.kind is TraceStepKind.PROVIDER_ATTEMPT
    ]
    assert len(stub.calls) == 3
    assert len(steps) == 3
    assert [step.call_status for step in steps] == [
        CallStatus.RATE_LIMITED,
        CallStatus.SUCCESS,
        CallStatus.SUCCESS,
    ]
    assert all(trace.status is ExecutionStatus.COMPLETED for trace in result.traces)
    assert "UNSAFE_PROVIDER_ERROR_SHOULD_NOT_APPEAR" not in result.report.model_dump_json()
