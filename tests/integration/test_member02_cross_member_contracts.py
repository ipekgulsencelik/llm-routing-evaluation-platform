"""Cross-member interface acceptance with *real* routing/provider and test-only M03/M04 ports.

This suite deliberately does not pretend that privacy, expert evaluation or
durable reporting adapters have been implemented by other team members.
Every HTTP request is served through HTTPX MockTransport (zero sockets).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from llm_routing_eval.adapters.providers import (
    GroqProviderOptions,
    InMemoryProviderMetrics,
    managed_httpx_groq_provider,
)
from llm_routing_eval.application import BenchmarkOrchestrator, RunOptions
from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    EvaluationResult,
    EvaluationStatus,
    ExecutionStatus,
    GoldenReference,
    ModelRequest,
    ModelResponse,
    RouterPort,
    RoutingContext,
    RoutingDecision,
    SanitizationStatus,
    SanitizedText,
    StrategyId,
    TraceStepKind,
    VerificationStatus,
)
from llm_routing_eval.demo.fakes import InMemoryDemoRepository
from llm_routing_eval.routing import build_router, read_routing_settings

ROOT = Path(__file__).resolve().parents[2]
POLICY = read_routing_settings(ROOT / "config/routing.yaml")
RAW_PROMPT = "RAW_UNSANITIZED_SYNTHETIC_PROMPT_CANARY"
SAFE_PROMPT = "Banka kartı ile kredi kartı arasındaki fark nedir?"
GOLDEN_CANARY = "PRIVATE_HELDOUT_REFERENCE_NOT_FOR_ROUTER_OR_PROVIDER"
METADATA_CANARY = "PRIVATE_RAW_METADATA_NOT_FOR_ROUTER"
ANSWER_CANARY = "SYNTHETIC_RESPONSE_NOT_FOR_REPORT"
NOTE_CANARY = "PRIVATE_EVALUATOR_PROSE_NOT_FOR_REPORT"
KEY_CANARY = "TEST_ONLY_CREDENTIAL_NEVER_LOG"
CASE_ID = "synthetic-contract"
ALL_STRATEGIES = tuple(StrategyId)


def synthetic_case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id=CASE_ID,
        input_text=RAW_PROMPT,
        domain="turkish_banking",
        scenario_type="card_general",
        metadata={"private_field": METADATA_CANARY, "synthetic": True},
    )


def synthetic_reference() -> GoldenReference:
    return GoldenReference(
        case_id=CASE_ID,
        dataset_version="test_contract_v0",
        expected_facts=(GOLDEN_CANARY,),
        rubric_version="unverified_test_fixture_v0",
        verification_status=VerificationStatus.UNREVIEWED,
    )


class ContractPrivacyDouble:
    """Test-only placeholder for M04. Explicit allowlist, not a real PII filter."""

    def __init__(self, *, deny_at: str | None = None) -> None:
        self.deny_at = deny_at
        self.purposes: list[str] = []

    def sanitize(self, text: str, *, purpose: str) -> SanitizedText:
        self.purposes.append(purpose)
        expected = RAW_PROMPT if purpose == "routing" else SAFE_PROMPT
        allow = purpose in ("routing", "provider") and purpose != self.deny_at and text == expected
        return SanitizedText(
            text=SAFE_PROMPT if allow else None,
            policy_version="test_allowlist_only_v0",
            status=SanitizationStatus.ALLOWED if allow else SanitizationStatus.BLOCKED,
            reason_code="TEST_ALLOWLIST" if allow else "TEST_DENIED",
        )


class RouterCanarySpy:
    """Verify M02 routers receive sanitized case + metadata, never golden facts."""

    def __init__(self, router: RouterPort) -> None:
        self.router = router
        self.seen: list[BenchmarkCase] = []

    async def decide(self, case: BenchmarkCase, context: RoutingContext) -> RoutingDecision:
        self.seen.append(case)
        assert case.input_text == SAFE_PROMPT
        assert case.metadata == {}
        assert GOLDEN_CANARY not in case.model_dump_json()
        assert RAW_PROMPT not in case.model_dump_json()
        assert METADATA_CANARY not in case.model_dump_json()
        return await self.router.decide(case, context)


class EvaluatorContractDouble:
    """Only this M03 test double sees held-out references, after provider success."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def evaluate(
        self,
        case: BenchmarkCase,
        response: ModelResponse | None,
        reference: GoldenReference | None,
        *,
        run_id: str,
        trace_id: str,
    ) -> EvaluationResult:
        self.calls += 1
        assert case.input_text == SAFE_PROMPT
        assert case.metadata == {}
        assert response is not None and response.status is CallStatus.SUCCESS
        assert reference is not None
        assert reference.verification_status is VerificationStatus.UNREVIEWED
        assert reference.expected_facts == (GOLDEN_CANARY,)
        if self.fail:
            raise RuntimeError("PRIVATE_EVALUATOR_EXCEPTION_BODY")
        return EvaluationResult(
            run_id=run_id,
            case_id=case.case_id,
            trace_id=trace_id,
            quality_scores={"contract_fixture_only": 0.75},
            critical_errors=(),
            evaluation_method="fixture_contract_only",
            evaluator_version="test_v0",
            status=EvaluationStatus.COMPLETED,
            safe_notes=(NOTE_CANARY,),
            review_status="TEST_UNVERIFIED",
        )


class SyntheticOnlyOutboundApproval:
    """Unsafe for production; permits exactly the test's sanitized fake content."""

    def authorize(self, request: ModelRequest) -> bool:
        return request.provider == "groq" and request.prompt == SAFE_PROMPT


async def execute(
    strategy: StrategyId,
    *,
    deny_at: str | None = None,
    outbound_authorized: bool = True,
    evaluator_failure: bool = False,
):
    """Compose M01 orchestrator + real M02 routers/provider + M03/M04 *test doubles*."""
    privacy = ContractPrivacyDouble(deny_at=deny_at)
    evaluator = EvaluatorContractDouble(fail=evaluator_failure)
    router = RouterCanarySpy(await build_router(strategy, POLICY))
    repository = InMemoryDemoRepository()
    metrics = InMemoryProviderMetrics(max_events=20)
    requests: list[dict[str, object]] = []

    async def fake_groq(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {KEY_CANARY}"
        payload = json.loads(request.content)
        assert isinstance(payload, dict)
        assert payload["messages"] == [{"role": "user", "content": SAFE_PROMPT}]
        assert payload["model"] in (POLICY.models.small, POLICY.models.large)
        assert GOLDEN_CANARY not in request.content.decode()
        assert RAW_PROMPT not in request.content.decode()
        assert METADATA_CANARY not in request.content.decode()
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "test-wire-id",
                "model": payload["model"],
                "choices": [{"message": {"content": ANSWER_CANARY}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    async def no_sleep(delay: float) -> None:
        raise AssertionError(f"No retry should occur in the clean contract path: {delay}")

    async with managed_httpx_groq_provider(
        GroqProviderOptions(
            allowed_models=frozenset({POLICY.models.small, POLICY.models.large}),
            live_enabled=True,
        ),
        outbound_authorization=SyntheticOnlyOutboundApproval() if outbound_authorized else None,
        metrics_sink=metrics,
        mock_transport=httpx.MockTransport(fake_groq),
    ) as provider:
        runner = BenchmarkOrchestrator(
            router=router,
            provider=provider,
            evaluator=evaluator,
            privacy=privacy,
            repository=repository,
            sleep=no_sleep,
        )
        result = await runner.run(
            [synthetic_case()],
            RunOptions(
                strategy=strategy,
                provider="groq",
                allowed_models=(POLICY.models.small, POLICY.models.large),
                routing_policy_sha256=POLICY.fingerprint(),
                dataset_version="synthetic_contract_v0",
                max_model_calls_per_case=2,
                max_retries=0,
                timeout_seconds=2.0,
            ),
            references={CASE_ID: synthetic_reference()},
        )
        return result, privacy, router, evaluator, repository, metrics.snapshot(), requests


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_all_real_routers_compose_with_cross_member_ports_and_redact_golden(
    monkeypatch: pytest.MonkeyPatch, strategy: StrategyId
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", KEY_CANARY)

    result, privacy, router, evaluator, repository, metrics, requests = asyncio.run(
        execute(strategy)
    )
    assert result.report.run_status == "COMPLETED"
    assert result.report.provenance["routing_policy_sha256"] == POLICY.fingerprint()
    assert len(router.seen) >= 1
    assert len(requests) in (1, 2)
    assert privacy.purposes[0] == "routing"
    assert privacy.purposes.count("provider") == len(requests)
    assert evaluator.calls == 1
    assert result.traces[0].status is ExecutionStatus.COMPLETED
    assert result.evaluations[0].status is EvaluationStatus.COMPLETED
    assert result.evaluations[0].quality_scores == {"contract_fixture_only": 0.75}
    assert result.evaluations[0].safe_notes == ()
    assert result.evaluations[0].review_status is None
    assert len(repository.traces) == len(repository.evaluations) == len(repository.reports) == 1
    trace_attempts = [
        step for step in result.traces[0].steps if step.kind is TraceStepKind.PROVIDER_ATTEMPT
    ]
    assert len(trace_attempts) == metrics.observed_attempts == len(requests)
    assert metrics.observed_successes == len(requests)
    assert metrics.pretransport_results == 0

    serialized = "".join(
        item.model_dump_json()
        for item in (result.report, *repository.traces, *repository.evaluations)
    )
    for forbidden in (
        RAW_PROMPT,
        METADATA_CANARY,
        GOLDEN_CANARY,
        ANSWER_CANARY,
        NOTE_CANARY,
        KEY_CANARY,
        "test-wire-id",
    ):
        assert forbidden not in serialized
    assert metrics.scope == "DIAGNOSTIC_ONLY"
    assert metrics.budget_status == "INSUFFICIENT_DATA"


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
@pytest.mark.parametrize("deny_at", ["routing", "provider"])
def test_privacy_block_at_either_gate_stops_provider_and_judge(
    monkeypatch: pytest.MonkeyPatch, strategy: StrategyId, deny_at: str
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", KEY_CANARY)
    result, privacy, router, evaluator, repository, metrics, requests = asyncio.run(
        execute(strategy, deny_at=deny_at)
    )
    assert requests == []
    assert evaluator.calls == 0
    assert result.traces[0].status is ExecutionStatus.BLOCKED
    assert result.evaluations[0].status is EvaluationStatus.NOT_EVALUATED
    assert metrics.observed_attempts == 0
    assert metrics.pretransport_results == 0
    assert len(repository.traces) == len(repository.evaluations) == len(repository.reports) == 1
    if deny_at == "routing":
        assert privacy.purposes == ["routing"]
        assert router.seen == []
    else:
        assert privacy.purposes[:2] == ["routing", "provider"]
        assert len(router.seen) == 1


def test_provider_default_deny_remains_in_force_even_after_privacy_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", KEY_CANARY)
    result, privacy, router, evaluator, _, metrics, requests = asyncio.run(
        execute(StrategyId.ALWAYS_LARGE, outbound_authorized=False)
    )
    assert requests == []
    assert len(router.seen) == 1
    assert privacy.purposes == ["routing", "provider"]
    assert evaluator.calls == 0
    assert result.traces[0].status is ExecutionStatus.FAILED
    assert result.evaluations[0].status is EvaluationStatus.NOT_EVALUATED
    assert metrics.observed_attempts == 0
    assert metrics.pretransport_results == 1


def test_evaluator_failure_fails_safe_without_quality_claims_or_data_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", KEY_CANARY)
    result, _, _, evaluator, repository, metrics, requests = asyncio.run(
        execute(StrategyId.ALWAYS_LARGE, evaluator_failure=True)
    )
    assert len(requests) == metrics.observed_successes == 1
    assert evaluator.calls == 1
    assert result.traces[0].status is ExecutionStatus.PARTIAL
    assert result.evaluations[0].status is EvaluationStatus.FAILED
    assert result.evaluations[0].quality_scores == {}
    assert "PRIVATE_EVALUATOR_EXCEPTION_BODY" not in (
        result.report.model_dump_json()
        + "".join(item.model_dump_json() for item in repository.evaluations)
    )
