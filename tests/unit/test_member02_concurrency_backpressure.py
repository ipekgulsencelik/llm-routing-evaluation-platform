"""Deterministic provider concurrency/backpressure tests; no external networking."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import httpx
import pytest

from llm_routing_eval.adapters.providers import (
    GroqChatProvider,
    GroqProviderOptions,
    GroqTransport,
    HttpResult,
    managed_httpx_groq_provider,
)
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
    AllowlistedDemoPrivacy,
    DemoFixtureEvaluator,
    InMemoryDemoRepository,
)
from llm_routing_eval.routing import AlwaysLargeRouter, ModelPair

_MODEL = "openai/gpt-oss-120b"
_PROMPT = "Banka hesabı açmak için genel süreç nedir?"
_RESPONSE = "OFFLINE_FAKE_RESPONSE_NOT_FINANCIAL_ADVICE"


class SyntheticApproval:
    """Test-only allowlist; never use this approval policy in live composition."""

    def authorize(self, request: ModelRequest) -> bool:
        return request.provider == "groq" and request.prompt == _PROMPT


def _request(index: int) -> ModelRequest:
    return ModelRequest(
        run_id="synthetic-load",
        case_id=f"synthetic-{index}",
        trace_id=f"synthetic-trace-{index}",
        attempt_id=f"synthetic-attempt-{index}",
        provider="groq",
        model=_MODEL,
        prompt=_PROMPT,
        timeout_seconds=2.0,
        max_output_tokens=32,
    )


def _successful_http_result() -> HttpResult:
    return HttpResult(
        status_code=200,
        body={
            "model": _MODEL,
            "choices": [{"message": {"content": _RESPONSE}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        },
    )


def _provider(transport: GroqTransport, *, max_in_flight: int) -> GroqChatProvider:
    return GroqChatProvider(
        GroqProviderOptions(
            allowed_models=frozenset({_MODEL}),
            live_enabled=True,
            max_in_flight=max_in_flight,
        ),
        transport=transport,
        outbound_authorization=SyntheticApproval(),
    )


def test_many_requests_respect_provider_semaphore_without_exceeding_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SYNTHETIC_TEST_ONLY")

    class GatedTransport:
        def __init__(self) -> None:
            self.release = asyncio.Event()
            self.at_capacity = asyncio.Event()
            self.in_flight = 0
            self.peak = 0
            self.calls = 0

        async def post_chat(
            self, body: Mapping[str, object], *, api_key: str, timeout_seconds: float
        ) -> HttpResult:
            _ = body, api_key, timeout_seconds
            self.calls += 1
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            if self.in_flight == 3:
                self.at_capacity.set()
            try:
                await self.release.wait()
                return _successful_http_result()
            finally:
                self.in_flight -= 1

    async def run() -> tuple[list[CallStatus], int, int, int]:
        transport = GatedTransport()
        provider = _provider(transport, max_in_flight=3)
        tasks = [asyncio.create_task(provider.generate(_request(i))) for i in range(18)]
        try:
            await asyncio.wait_for(transport.at_capacity.wait(), timeout=2.0)
            assert transport.in_flight == 3
            transport.release.set()
            responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=3.0)
            return (
                [r.status for r in responses],
                transport.calls,
                transport.peak,
                transport.in_flight,
            )
        finally:
            transport.release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    statuses, calls, peak, remaining = asyncio.run(run())
    assert len(statuses) == calls == 18
    assert all(status is CallStatus.SUCCESS for status in statuses)
    assert peak == 3
    assert remaining == 0


def test_cancelling_queued_requests_does_not_send_or_leak_semaphore_permits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SYNTHETIC_TEST_ONLY")

    class StallingTransport:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def post_chat(
            self, body: Mapping[str, object], *, api_key: str, timeout_seconds: float
        ) -> HttpResult:
            _ = body, api_key, timeout_seconds
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                await self.release.wait()
            return _successful_http_result()

    async def run() -> tuple[int, list[CallStatus]]:
        transport = StallingTransport()
        provider = _provider(transport, max_in_flight=1)
        first = asyncio.create_task(provider.generate(_request(0)))
        await asyncio.wait_for(transport.started.wait(), timeout=1.0)
        queued = [asyncio.create_task(provider.generate(_request(i))) for i in range(1, 13)]
        await asyncio.sleep(0)
        for task in queued:
            task.cancel()
        cancelled = await asyncio.gather(*queued, return_exceptions=True)
        assert all(isinstance(item, asyncio.CancelledError) for item in cancelled)
        assert transport.calls == 1
        transport.release.set()
        first_result = await asyncio.wait_for(first, timeout=1.0)
        followup = await asyncio.wait_for(provider.generate(_request(13)), timeout=1.0)
        return transport.calls, [first_result.status, followup.status]

    calls, statuses = asyncio.run(run())
    assert calls == 2
    assert statuses == [CallStatus.SUCCESS, CallStatus.SUCCESS]


def test_denied_concurrent_requests_never_reach_httpx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SYNTHETIC_TEST_ONLY")
    actual_calls: list[str] = []

    async def never_outbound(request: httpx.Request) -> httpx.Response:
        actual_calls.append(str(request.url))
        raise AssertionError("Default deny must prevent every attempted HTTP call")

    async def run() -> list[tuple[CallStatus, ErrorCategory]]:
        async with managed_httpx_groq_provider(
            GroqProviderOptions(
                allowed_models=frozenset({_MODEL}),
                live_enabled=True,
                max_in_flight=3,
            ),
            mock_transport=httpx.MockTransport(never_outbound),
        ) as provider:
            responses = await asyncio.gather(
                *(provider.generate(_request(i)) for i in range(24))
            )
            return [
                (r.status, r.error.category)
                for r in responses
                if r.error is not None
            ]

    outcomes = asyncio.run(run())
    assert len(outcomes) == 24
    assert outcomes == [(CallStatus.FAILED, ErrorCategory.PRIVACY)] * 24
    assert not actual_calls


@pytest.mark.parametrize("status", [409, 503])
def test_retryable_provider_failures_never_exceed_global_attempt_budget(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SYNTHETIC_TEST_ONLY")
    calls = 0
    unsafe_remote_body = {"error": "SYNTHETIC_REMOTE_ERROR_BODY_SHOULD_NOT_PERSIST"}

    async def failing_endpoint(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json=unsafe_remote_body)

    async def no_sleep(delay: float) -> None:
        _ = delay

    async def run() -> tuple[int, int, str]:
        async with managed_httpx_groq_provider(
            GroqProviderOptions(
                allowed_models=frozenset({_MODEL}),
                live_enabled=True,
                max_in_flight=2,
            ),
            outbound_authorization=SyntheticApproval(),
            mock_transport=httpx.MockTransport(failing_endpoint),
        ) as provider:
            service = BenchmarkOrchestrator(
                router=AlwaysLargeRouter(
                    ModelPair(small="openai/gpt-oss-20b", large=_MODEL)
                ),
                provider=provider,
                evaluator=DemoFixtureEvaluator(),
                privacy=AllowlistedDemoPrivacy(),
                repository=InMemoryDemoRepository(),
                sleep=no_sleep,
            )
            cases = [
                BenchmarkCase(
                    case_id="synthetic-rate-limit",
                    input_text=_PROMPT,
                    domain="turkish_banking",
                    scenario_type="account_general",
                    metadata={"synthetic": True},
                )
            ]
            result = await service.run(
                cases,
                RunOptions(
                    provider="groq",
                    strategy=StrategyId.ALWAYS_LARGE,
                    allowed_models=(_MODEL,),
                    max_model_calls_per_case=2,
                    max_retries=3,
                    timeout_seconds=2.0,
                ),
            )
            attempts = [
                step
                for trace in result.traces
                for step in trace.steps
                if step.kind is TraceStepKind.PROVIDER_ATTEMPT
            ]
            serialized = (
                result.report.model_dump_json()
                + "".join(trace.model_dump_json() for trace in result.traces)
            )
            return len(attempts), calls, serialized

    attempt_count, http_calls, serialized = asyncio.run(run())
    assert attempt_count == http_calls == 2
    assert "SYNTHETIC_REMOTE_ERROR_BODY_SHOULD_NOT_PERSIST" not in serialized
    assert _PROMPT not in serialized


def test_429_retry_after_larger_than_budget_performs_no_early_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SYNTHETIC_TEST_ONLY")
    calls = 0

    async def rate_limited(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429, headers={"Retry-After": "15"}, json={"error": "redacted"}
        )

    async def run() -> int:
        async with managed_httpx_groq_provider(
            GroqProviderOptions(allowed_models=frozenset({_MODEL}), live_enabled=True),
            outbound_authorization=SyntheticApproval(),
            mock_transport=httpx.MockTransport(rate_limited),
        ) as provider:
            async def no_sleep(delay: float) -> None:
                raise AssertionError("Retry-After above budget must not sleep/retry")

            service = BenchmarkOrchestrator(
                router=AlwaysLargeRouter(
                    ModelPair(small="openai/gpt-oss-20b", large=_MODEL)
                ),
                provider=provider,
                evaluator=DemoFixtureEvaluator(),
                privacy=AllowlistedDemoPrivacy(),
                repository=InMemoryDemoRepository(),
                sleep=no_sleep,
            )
            result = await service.run(
                [
                    BenchmarkCase(
                        case_id="synthetic-hint",
                        input_text=_PROMPT,
                        domain="turkish_banking",
                        scenario_type="account_general",
                        metadata={"synthetic": True},
                    )
                ],
                RunOptions(
                    provider="groq",
                    strategy=StrategyId.ALWAYS_LARGE,
                    allowed_models=(_MODEL,),
                    max_model_calls_per_case=3,
                    max_retries=2,
                    max_retry_wait_seconds=1.0,
                    timeout_seconds=2.0,
                ),
            )
            steps = [
                step
                for trace in result.traces
                for step in trace.steps
                if step.kind is TraceStepKind.PROVIDER_ATTEMPT
            ]
            assert len(steps) == 1
            assert steps[0].call_status is CallStatus.RATE_LIMITED
            return calls

    assert asyncio.run(run()) == 1
