"""Cross-strategy comparison correctness, provenance, coverage and privacy tests."""

import asyncio

import pytest

from llm_routing_eval.application import (
    BenchmarkInputError,
    BenchmarkOrchestrator,
    RunOptions,
    StrategyComparisonRunner,
)
from llm_routing_eval.application.comparison import _digest_cases, _row
from llm_routing_eval.core import (
    BenchmarkCase,
    MetricOrigin,
    StrategyId,
    TraceStepKind,
)
from llm_routing_eval.demo.fakes import (
    AllowlistedDemoPrivacy,
    DemoFakeProvider,
    DemoFixtureEvaluator,
    InMemoryDemoRepository,
)
from llm_routing_eval.routing import build_router, read_routing_settings

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = (
    BenchmarkCase(
        case_id="synthetic-a",
        input_text="Banka kartı ile kredi kartı arasındaki fark nedir?",
        domain="turkish_banking",
        scenario_type="card_general",
        metadata={"synthetic": True},
    ),
    BenchmarkCase(
        case_id="synthetic-b",
        input_text="Banka hesabı açmak için genel süreç nedir?",
        domain="turkish_banking",
        scenario_type="account_general",
        metadata={"synthetic": True},
    ),
)


async def make_comparison(cases: tuple[BenchmarkCase, ...] = CASES):
    routing = read_routing_settings(ROOT / "config/routing.yaml")

    async def factory(strategy: StrategyId) -> BenchmarkOrchestrator:
        return BenchmarkOrchestrator(
            router=await build_router(strategy, routing),
            provider=DemoFakeProvider(),
            evaluator=DemoFixtureEvaluator(),
            privacy=AllowlistedDemoPrivacy(),
            repository=InMemoryDemoRepository(),
        )

    options = RunOptions(
        provider="fake",
        allowed_models=(routing.models.small, routing.models.large),
        dataset_version="test-synthetic-v1",
        routing_policy_sha256=routing.fingerprint(),
        max_model_calls_per_case=2,
    )
    return await StrategyComparisonRunner(orchestrator_factory=factory).compare(cases, options)


def test_four_strategies_use_same_workload_and_preserve_call_accounting() -> None:
    report = asyncio.run(make_comparison())
    assert report.dataset_content_sha256.startswith("sha256:")
    assert report.dataset_version == "test-synthetic-v1"
    assert len({row.run_id for row in report.strategies}) == 4
    assert [row.strategy.value for row in report.strategies] == [
        "always_large",
        "rule_based",
        "knn",
        "cascade",
    ]
    assert [row.provider_attempts for row in report.strategies] == [2, 2, 2, 3]
    assert [row.attempts_per_case for row in report.strategies] == [1.0, 1.0, 1.0, 1.5]
    for row in report.strategies:
        assert row.case_count == row.successful_cases == 2
        assert row.failed_cases == row.blocked_cases == 0
        assert sum(row.calls_by_model.values()) == row.provider_attempts
        assert row.successful_attempts == row.provider_attempts
        assert row.token_origin == "SIMULATED"
        assert row.token_usage_samples == row.provider_attempts
        assert row.input_tokens_total == 16 * row.provider_attempts
        assert row.output_tokens_total == 8 * row.provider_attempts
        assert row.input_tokens_per_attempt == 16.0
        assert row.output_tokens_per_attempt == 8.0
        assert row.fixture_score_samples == 2
        assert row.fixture_match_mean == 1.0
        assert row.cost_amount is None
        assert row.local_case_latency_mean_ms is not None


def test_summary_does_not_expose_prompts_answers_or_credentials() -> None:
    result = asyncio.run(make_comparison())
    output = result.model_dump_json()
    for private_value in (
        "Banka kartı ile kredi",
        "Banka hesabı açmak",
        "OFFLINE_FAKE_RESPONSE_NOT_FINANCIAL_ADVICE",
        "GROQ_API_KEY",
    ):
        assert private_value not in output
    assert "not answer accuracy" in "Fixture-match is a fake-provider contract test, not answer accuracy."


def test_content_hash_changes_for_modified_input_even_if_case_ids_are_same() -> None:
    altered = (CASES[0].model_copy(update={"input_text": "different synthetic fixture"}), CASES[1])
    assert _digest_cases(CASES) != _digest_cases(altered)


def test_token_totals_are_unknown_if_one_attempt_has_missing_usage() -> None:
    from llm_routing_eval.application import RunResult

    async def produce() -> RunResult:
        routing = read_routing_settings(ROOT / "config/routing.yaml")
        runner = BenchmarkOrchestrator(
            router=await build_router(StrategyId.ALWAYS_LARGE, routing),
            provider=DemoFakeProvider(),
            evaluator=DemoFixtureEvaluator(),
            privacy=AllowlistedDemoPrivacy(),
            repository=InMemoryDemoRepository(),
        )
        return await runner.run(CASES, RunOptions(strategy=StrategyId.ALWAYS_LARGE))

    run = asyncio.run(produce())
    first = run.traces[0]
    steps = tuple(
        step.model_copy(update={"usage": None})
        if step.kind is TraceStepKind.PROVIDER_ATTEMPT
        else step
        for step in first.steps
    )
    traces = (first.model_copy(update={"steps": steps}), run.traces[1])
    modified = run.model_copy(update={"traces": traces})
    row = _row(modified, StrategyId.ALWAYS_LARGE, tuple(c.case_id for c in CASES))
    assert row.provider_attempts == 2
    assert row.token_usage_samples == 1
    assert row.input_tokens_total is None
    assert row.output_tokens_total is None
    assert row.input_tokens_per_attempt is None
    assert row.output_tokens_per_attempt is None
    assert row.token_origin == "SIMULATED"


def test_comparison_rejects_live_provider_and_unsupported_workloads() -> None:
    async def never_factory(strategy: StrategyId) -> BenchmarkOrchestrator:
        raise AssertionError(f"Factory must not be called for {strategy}")

    runner = StrategyComparisonRunner(orchestrator_factory=never_factory)
    with pytest.raises(BenchmarkInputError, match="COMPARISON_OFFLINE_ONLY"):
        asyncio.run(runner.compare(CASES, RunOptions(provider="groq")))
    with pytest.raises(BenchmarkInputError, match="COMPARISON_INVALID_WORKLOAD"):
        asyncio.run(runner.compare((CASES[0], CASES[0]), RunOptions(provider="fake")))
    with pytest.raises(BenchmarkInputError, match="COMPARISON_REQUIRES_SYNTHETIC_POLICY"):
        asyncio.run(runner.compare(CASES, RunOptions(provider="fake")))
    with pytest.raises(BenchmarkInputError, match="COMPARISON_REQUIRES_SYNTHETIC_POLICY"):
        bad = CASES[0].model_copy(update={"metadata": {"synthetic": False}})
        asyncio.run(
            runner.compare(
                (bad,),
                RunOptions(
                    provider="fake",
                    routing_policy_sha256="sha256:" + "a" * 64,
                ),
            )
        )


def test_benchmark_isolation_does_not_mix_results_between_strategies() -> None:
    report = asyncio.run(make_comparison())
    assert report.strategies[0].calls_by_model == {"openai/gpt-oss-120b": 2}
    assert report.strategies[1].calls_by_model == {"openai/gpt-oss-20b": 2}
    assert report.strategies[2].calls_by_model == {"openai/gpt-oss-20b": 2}
    assert report.strategies[3].calls_by_model == {
        "openai/gpt-oss-120b": 1,
        "openai/gpt-oss-20b": 2,
    }


def test_trace_counts_and_origin_remain_simulated_not_observed() -> None:
    report = asyncio.run(make_comparison())
    assert all(row.token_origin == "SIMULATED" for row in report.strategies)
    assert all(row.provider_attempts >= row.case_count for row in report.strategies)
    assert MetricOrigin.SIMULATED.value == report.strategies[0].token_origin
