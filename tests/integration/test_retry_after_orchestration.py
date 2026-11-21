"""429 retry policy through Core orchestration with synthetic Groq transport."""

import asyncio
from collections.abc import Mapping

import pytest

from llm_routing_eval.adapters.providers import GroqChatProvider, GroqProviderOptions, HttpResult
from llm_routing_eval.application import BenchmarkOrchestrator, RunOptions
from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    ErrorCategory,
    ModelRequest,
    StrategyId,
    TraceStepKind,
)
from llm_routing_eval.demo.fakes import (
    DEMO_RESPONSE,
    AllowlistedDemoPrivacy,
    DemoFixtureEvaluator,
    InMemoryDemoRepository,
)
from llm_routing_eval.routing import AlwaysLargeRouter, ModelPair

PROMPT = "Banka hesabı açmak için genel süreç nedir?"
LARGE = "openai/gpt-oss-120b"


class SyntheticApproval:
    def authorize(self, outgoing: ModelRequest) -> bool:
        return outgoing.prompt == PROMPT


class RateLimitThenSuccess:
    def __init__(self, hint: float | None) -> None:
        self.calls = 0
        self.hint = hint

    async def post_chat(
        self, body: Mapping[str, object], *, api_key: str, timeout_seconds: float
    ) -> HttpResult:
        _ = api_key, timeout_seconds
        self.calls += 1
        if self.calls == 1:
            return HttpResult(status_code=429, retry_after_seconds=self.hint)
        return HttpResult(
            status_code=200,
            body={
                "model": body["model"],
                "choices": [{"message": {"content": DEMO_RESPONSE}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 16, "completion_tokens": 8},
            },
        )


async def run_case(hint: float | None, max_delay: float) -> tuple[object, list[float], int]:
    stub = RateLimitThenSuccess(hint)
    slept: list[float] = []

    async def record_sleep(delay: float) -> None:
        slept.append(delay)

    provider = GroqChatProvider(
        GroqProviderOptions(
            allowed_models=frozenset({LARGE}),
            live_enabled=True,
        ),
        transport=stub,
        outbound_authorization=SyntheticApproval(),
    )
    runner = BenchmarkOrchestrator(
        router=AlwaysLargeRouter(ModelPair(small="openai/gpt-oss-20b", large=LARGE)),
        provider=provider,
        evaluator=DemoFixtureEvaluator(),
        privacy=AllowlistedDemoPrivacy(),
        repository=InMemoryDemoRepository(),
        sleep=record_sleep,
    )
    result = await runner.run(
        [
            BenchmarkCase(
                case_id="synthetic-example",
                input_text=PROMPT,
                scenario_type="account_general",
                domain="turkish_banking",
                metadata={"synthetic": True},
            )
        ],
        RunOptions(
            provider="groq",
            strategy=StrategyId.ALWAYS_LARGE,
            allowed_models=(LARGE,),
            max_model_calls_per_case=2,
            max_retries=1,
            max_retry_wait_seconds=max_delay,
            timeout_seconds=3,
        ),
    )
    return result, slept, stub.calls


@pytest.mark.parametrize("hint", [0.0, 0.125, 3.75])
def test_429_obeys_accepted_retry_after_seconds(
    monkeypatch: pytest.MonkeyPatch, hint: float
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "TEST_ONLY")
    result, slept, calls = asyncio.run(run_case(hint, max_delay=5.0))
    assert slept == [hint]
    assert calls == 2
    trace = result.traces[0]
    attempts = [step for step in trace.steps if step.kind is TraceStepKind.PROVIDER_ATTEMPT]
    assert [s.call_status for s in attempts] == [CallStatus.RATE_LIMITED, CallStatus.SUCCESS]
    assert attempts[0].error is not None
    assert attempts[0].error.category is ErrorCategory.RATE_LIMIT
    assert result.report.run_status == "COMPLETED"


def test_rate_limit_exceeding_wait_budget_does_not_retry_too_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "TEST_ONLY")
    result, slept, calls = asyncio.run(run_case(40.0, max_delay=2.0))
    assert slept == []
    assert calls == 1
    trace = result.traces[0]
    attempts = [step for step in trace.steps if step.kind is TraceStepKind.PROVIDER_ATTEMPT]
    assert len(attempts) == 1
    assert attempts[0].call_status is CallStatus.RATE_LIMITED
    assert attempts[0].error is not None and attempts[0].error.retryable
    assert result.report.run_status != "COMPLETED"


def test_429_without_valid_hint_falls_back_to_bounded_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "TEST_ONLY")
    result, slept, calls = asyncio.run(run_case(None, max_delay=0.25))
    assert len(slept) == 1 and 0 <= slept[0] <= 0.25
    assert calls == 2
    assert result.report.run_status == "COMPLETED"
