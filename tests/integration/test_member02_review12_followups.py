"""Regression tests for review #12 follow-ups F1–F3 (upstream issues #15, #16, #17).

All offline: HTTPX MockTransport and in-memory fakes; no network, no real keys.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic import ValidationError

from llm_routing_eval.adapters.providers import (
    GroqChatProvider,
    GroqProviderOptions,
    NvidiaBuildChatProvider,
    NvidiaBuildProviderOptions,
    managed_httpx_groq_provider,
)
from llm_routing_eval.application import BenchmarkOrchestrator, RunOptions
from llm_routing_eval.application.comparison import StrategyComparisonRunner
from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    ExecutionStatus,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
    StrategyId,
    TraceStepKind,
)
from llm_routing_eval.demo.fakes import (
    DEMO_PROMPTS,
    AllowlistedDemoPrivacy,
    DemoAlwaysLargeRouter,
    DemoFakeProvider,
    DemoFixtureEvaluator,
    InMemoryDemoRepository,
)
from llm_routing_eval.routing import CascadeRouter, ModelPair, build_router

SMALL = "openai/gpt-oss-20b"
LARGE = "openai/gpt-oss-120b"
SECRET = "IN_MEMORY_FAKE_CREDENTIAL"


def demo_case(scenario: str = "account_general") -> BenchmarkCase:
    return BenchmarkCase(
        case_id="review12-case",
        input_text=sorted(DEMO_PROMPTS)[0],
        domain="turkish_banking",
        scenario_type=scenario,
    )


async def no_sleep(_: float) -> None:
    return None


# --------------------------------------------------------------------- F1 (#17)


class DemoPromptApproval:
    def authorize(self, request: ModelRequest) -> bool:
        return request.prompt in DEMO_PROMPTS


def test_f1_providers_have_no_implicit_thread_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setenv("NVIDIA_API_KEY", SECRET)
    groq = GroqChatProvider(
        GroqProviderOptions(allowed_models=frozenset({SMALL}), live_enabled=True),
        outbound_authorization=DemoPromptApproval(),
    )
    nvidia = NvidiaBuildChatProvider(
        NvidiaBuildProviderOptions(allowed_models=frozenset({SMALL}), live_enabled=True),
        outbound_authorization=DemoPromptApproval(),
    )
    for provider, provider_id, code in (
        (groq, "groq", "GROQ_TRANSPORT_NOT_CONFIGURED"),
        (nvidia, "nvidia_build", "NVIDIA_TRANSPORT_NOT_CONFIGURED"),
    ):
        response = asyncio.run(
            provider.generate(
                ModelRequest(
                    run_id="r",
                    case_id="c",
                    trace_id="t",
                    attempt_id="a",
                    provider=provider_id,
                    model=SMALL,
                    prompt=sorted(DEMO_PROMPTS)[0],
                    timeout_seconds=1.0,
                )
            )
        )
        assert response.status is CallStatus.FAILED
        assert response.error is not None and response.error.code == code
        assert not response.error.retryable


def test_f1_timeout_leaves_no_call_in_flight_and_retries_never_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #17 repro: transport blocks 0.6 s, timeout 0.2 s, max_retries=1."""
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    in_flight = 0
    max_concurrent = 0
    started = 0

    async def slow(_: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_concurrent, started
        started += 1
        in_flight += 1
        max_concurrent = max(max_concurrent, in_flight)
        try:
            await asyncio.sleep(0.6)
            return httpx.Response(500)
        finally:
            in_flight -= 1

    async def run() -> tuple[int, list[CallStatus | None]]:
        async with managed_httpx_groq_provider(
            GroqProviderOptions(allowed_models=frozenset({LARGE}), live_enabled=True),
            outbound_authorization=DemoPromptApproval(),
            mock_transport=httpx.MockTransport(slow),
        ) as provider:
            engine = BenchmarkOrchestrator(
                router=DemoAlwaysLargeRouter(LARGE),
                provider=provider,
                evaluator=DemoFixtureEvaluator(),
                privacy=AllowlistedDemoPrivacy(),
                repository=InMemoryDemoRepository(),
                sleep=no_sleep,
            )
            result = await engine.run(
                [demo_case()],
                RunOptions(
                    provider="groq",
                    allowed_models=(LARGE,),
                    max_retries=1,
                    timeout_seconds=0.2,
                ),
            )
            # Checked immediately when run() returns, before the pool is closed.
            leaked = in_flight
        statuses = [
            step.call_status
            for step in result.traces[0].steps
            if step.kind is TraceStepKind.PROVIDER_ATTEMPT
        ]
        return leaked, statuses

    leaked, statuses = asyncio.run(run())
    assert leaked == 0
    assert max_concurrent == 1
    assert started == 2
    assert statuses == [CallStatus.TIMEOUT, CallStatus.TIMEOUT]


# --------------------------------------------------------------------- F2 (#16)


class LargeModelFails:
    """Small model answers; large model fails non-retryably."""

    def __init__(self) -> None:
        self.models: list[str] = []
        self._ok = DemoFakeProvider()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.models.append(request.model)
        if request.model == LARGE:
            return ModelResponse(
                provider=request.provider,
                model=request.model,
                attempt_id=request.attempt_id,
                status=CallStatus.FAILED,
                metric_origin=MetricOrigin.SIMULATED,
                error=ErrorInfo(category=ErrorCategory.PROVIDER, code="SYNTHETIC_LARGE_DOWN"),
            )
        return await self._ok.generate(request)


def _engine(router: object, provider: object) -> BenchmarkOrchestrator:
    return BenchmarkOrchestrator(
        router=router,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
        evaluator=DemoFixtureEvaluator(),
        privacy=AllowlistedDemoPrivacy(),
        repository=InMemoryDemoRepository(),
        sleep=no_sleep,
    )


def test_f2_failed_escalation_is_visible_and_counted_as_degraded() -> None:
    """Issue #16 repro: escalate account_general, small OK, large fails."""
    models = ModelPair(small=SMALL, large=LARGE)
    provider = LargeModelFails()
    cascade = _engine(
        CascadeRouter(models, escalate_scenarios=frozenset({"account_general"})), provider
    )
    result = asyncio.run(
        cascade.run(
            [demo_case()],
            RunOptions(strategy=StrategyId.CASCADE, allowed_models=(SMALL, LARGE)),
        )
    )
    trace = result.traces[0]
    assert provider.models == [SMALL, LARGE]
    # Decision D-M02-F2: keep the lower-tier answer, but never silently.
    assert trace.status is ExecutionStatus.COMPLETED
    codes = [s.error.code for s in trace.steps if s.kind is TraceStepKind.ERROR and s.error]
    assert codes == ["ESCALATION_FAILED_RETAINED_LOWER_TIER"]
    metrics = result.report.strategy_metrics[StrategyId.CASCADE]
    assert (metrics.successful_cases, metrics.degraded_cases) == (1, 1)

    baseline = _engine(DemoAlwaysLargeRouter(LARGE), LargeModelFails())
    base = asyncio.run(baseline.run([demo_case()], RunOptions(allowed_models=(SMALL, LARGE))))
    base_metrics = base.report.strategy_metrics[StrategyId.ALWAYS_LARGE]
    assert base.traces[0].status is ExecutionStatus.FAILED
    # Strict success is what is comparable with the baseline: 0 vs 0.
    assert metrics.successful_cases - metrics.degraded_cases == base_metrics.successful_cases


def test_f2_accepted_small_answer_is_not_degraded() -> None:
    cascade = _engine(CascadeRouter(ModelPair(small=SMALL, large=LARGE)), LargeModelFails())
    result = asyncio.run(
        cascade.run(
            [demo_case("card_general")],
            RunOptions(strategy=StrategyId.CASCADE, allowed_models=(SMALL, LARGE)),
        )
    )
    metrics = result.report.strategy_metrics[StrategyId.CASCADE]
    assert (metrics.successful_cases, metrics.degraded_cases) == (1, 0)
    assert not any(s.kind is TraceStepKind.ERROR for s in result.traces[0].steps)


def test_f2_comparison_reports_strict_success() -> None:
    from llm_routing_eval.routing import read_routing_settings

    settings = read_routing_settings(__import__("pathlib").Path("config/routing.yaml"))

    async def build(strategy: StrategyId) -> BenchmarkOrchestrator:
        return _engine(await build_router(strategy, settings), LargeModelFails())

    cases = [
        BenchmarkCase(
            case_id=f"c{i}",
            input_text=text,
            domain="turkish_banking",
            scenario_type=scenario,
            metadata={"synthetic": True},
        )
        for i, (text, scenario) in enumerate(
            zip(sorted(DEMO_PROMPTS), ("account_general", "card_general"), strict=True)
        )
    ]
    summary = asyncio.run(
        StrategyComparisonRunner(orchestrator_factory=build).compare(
            cases,
            RunOptions(
                allowed_models=(settings.models.small, settings.models.large),
                dataset_version="review12",
                routing_policy_sha256=settings.fingerprint(),
            ),
        )
    )
    rows = {row.strategy: row for row in summary.strategies}
    cascade = rows[StrategyId.CASCADE]
    # config/routing.yaml escalates card_general; the large model is down.
    assert cascade.degraded_cases == 1
    assert cascade.strict_successful_cases == cascade.successful_cases - 1
    assert rows[StrategyId.ALWAYS_LARGE].degraded_cases == 0


# --------------------------------------------------------------------- F3 (#15)


@pytest.mark.parametrize(("calls", "retries"), [(2, 1), (3, 1), (5, 2), (7, 3)])
def test_f3_cascade_budget_that_starves_escalation_is_rejected(calls: int, retries: int) -> None:
    with pytest.raises(ValidationError, match="CASCADE_CALL_BUDGET_TOO_SMALL_FOR_RETRIES"):
        RunOptions(
            strategy=StrategyId.CASCADE,
            allowed_models=(SMALL, LARGE),
            max_model_calls_per_case=calls,
            max_retries=retries,
        )


@pytest.mark.parametrize(("calls", "retries"), [(2, 0), (4, 1), (6, 2), (8, 3)])
def test_f3_minimum_cascade_budget_is_accepted(calls: int, retries: int) -> None:
    RunOptions(
        strategy=StrategyId.CASCADE,
        allowed_models=(SMALL, LARGE),
        max_model_calls_per_case=calls,
        max_retries=retries,
    )


def test_f3_rule_does_not_affect_single_call_strategies() -> None:
    for strategy in (StrategyId.ALWAYS_LARGE, StrategyId.RULE_BASED, StrategyId.KNN):
        RunOptions(strategy=strategy, max_model_calls_per_case=2, max_retries=1)


def test_f3_issue_repro_now_escalates_after_small_timeouts() -> None:
    """Issue #15 repro with the minimum valid budget: small times out, large is tried."""

    class SmallTimesOut:
        def __init__(self) -> None:
            self.models: list[str] = []

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.models.append(request.model)
            if request.model == SMALL:
                return ModelResponse(
                    provider=request.provider,
                    model=request.model,
                    attempt_id=request.attempt_id,
                    status=CallStatus.TIMEOUT,
                    metric_origin=MetricOrigin.SIMULATED,
                    error=ErrorInfo(
                        category=ErrorCategory.PROVIDER_TIMEOUT, code="TIMEOUT", retryable=True
                    ),
                )
            return await DemoFakeProvider().generate(request)

    provider = SmallTimesOut()
    engine = _engine(CascadeRouter(ModelPair(small=SMALL, large=LARGE)), provider)
    result = asyncio.run(
        engine.run(
            [demo_case()],
            RunOptions(
                strategy=StrategyId.CASCADE,
                allowed_models=(SMALL, LARGE),
                max_model_calls_per_case=4,
                max_retries=1,
            ),
        )
    )
    assert provider.models == [SMALL, SMALL, LARGE]
    assert result.traces[0].status is ExecutionStatus.COMPLETED


def test_f3_comparison_revalidates_cascade_budget_before_running_anything() -> None:
    """Codex review on PR #20: model_copy must not bypass the Cascade guard."""
    from llm_routing_eval.application import BenchmarkInputError
    from llm_routing_eval.routing import read_routing_settings

    settings = read_routing_settings(__import__("pathlib").Path("config/routing.yaml"))
    built: list[StrategyId] = []

    async def build(strategy: StrategyId) -> BenchmarkOrchestrator:
        built.append(strategy)
        return _engine(await build_router(strategy, settings), DemoFakeProvider())

    base = RunOptions(  # valid for Always-Large, invalid for Cascade
        allowed_models=(settings.models.small, settings.models.large),
        max_model_calls_per_case=2,
        max_retries=1,
        routing_policy_sha256=settings.fingerprint(),
    )
    case = BenchmarkCase(
        case_id="c",
        input_text=sorted(DEMO_PROMPTS)[0],
        domain="turkish_banking",
        scenario_type="account_general",
        metadata={"synthetic": True},
    )
    with pytest.raises(BenchmarkInputError, match="CASCADE_CALL_BUDGET_TOO_SMALL_FOR_RETRIES"):
        asyncio.run(StrategyComparisonRunner(orchestrator_factory=build).compare([case], base))
    assert built == []
