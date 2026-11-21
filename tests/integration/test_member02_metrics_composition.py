"""End-to-end in-memory HTTPX + provider observer integration, zero network."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from llm_routing_eval.adapters.providers import (
    GroqProviderOptions,
    InMemoryProviderMetrics,
    ObservedProvider,
    ProviderErrorBudget,
    managed_httpx_groq_provider,
)
from llm_routing_eval.core import CallStatus, ModelRequest

PROMPT_MARKER = "SYNTHETIC_PRIVATE_PROMPT_MUST_NOT_BE_EXPORTED"
KEY_MARKER = "SYNTHETIC_PRIVATE_CREDENTIAL_MUST_NOT_BE_EXPORTED"
MODEL = "openai/gpt-oss-20b"


class TestOnlyAuthorization:
    def authorize(self, request: ModelRequest) -> bool:
        return request.prompt == PROMPT_MARKER and request.model == MODEL


def request(i: int) -> ModelRequest:
    return ModelRequest(
        run_id="test-synthetic-run",
        case_id=f"test-case-{i}",
        trace_id=f"test-trace-{i}",
        attempt_id=f"test-attempt-{i}",
        provider="groq",
        model=MODEL,
        prompt=PROMPT_MARKER,
        timeout_seconds=1.0,
        max_output_tokens=32,
    )


def test_optional_httpx_observer_records_safe_outcomes_and_keeps_privacy_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", KEY_MARKER)
    calls = 0

    async def handler(outgoing: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert str(outgoing.url) == "https://api.groq.com/openai/v1/chat/completions"
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(
            200,
            json={
                "id": "test-id",
                "model": MODEL,
                "choices": [
                    {"message": {"content": "SYNTHETIC_ANSWER_NOT_EXPORTED"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3},
            },
        )

    async def run() -> tuple[list[CallStatus], str, int]:
        collector = InMemoryProviderMetrics(
            budget=ProviderErrorBudget(min_observed_attempts=2, max_failure_fraction=0.25)
        )
        async with managed_httpx_groq_provider(
            GroqProviderOptions(
                allowed_models=frozenset({MODEL}), live_enabled=True
            ),
            outbound_authorization=TestOnlyAuthorization(),
            metrics_sink=collector,
            mock_transport=httpx.MockTransport(handler),
        ) as provider:
            assert isinstance(provider, ObservedProvider)
            results = [await provider.generate(request(i)) for i in range(2)]
            status = collector.snapshot()
            assert status.observed_attempts == 2
            assert status.observed_failures == 1
            assert status.observed_status_counts[CallStatus.RATE_LIMITED] == 1
            assert status.observed_status_counts[CallStatus.SUCCESS] == 1
            assert status.budget_status == "EXHAUSTED"
            assert status.latency_samples == 2
            return [r.status for r in results], status.model_dump_json(), provider.lost_events

    statuses, safe_json, lost = asyncio.run(run())
    assert calls == 2
    assert statuses == [CallStatus.RATE_LIMITED, CallStatus.SUCCESS]
    assert lost == 0
    for secret in (
        PROMPT_MARKER,
        KEY_MARKER,
        "SYNTHETIC_ANSWER_NOT_EXPORTED",
        "test-attempt",
        "test-id",
        MODEL,
    ):
        assert secret not in safe_json


def test_composed_observer_records_privacy_rejection_without_transport_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", KEY_MARKER)

    async def handler(outgoing: httpx.Request) -> httpx.Response:
        raise AssertionError("Denied request must not go to network")

    async def run() -> tuple[int, int]:
        collector = InMemoryProviderMetrics()
        async with managed_httpx_groq_provider(
            GroqProviderOptions(allowed_models=frozenset({MODEL}), live_enabled=True),
            metrics_sink=collector,
            mock_transport=httpx.MockTransport(handler),
        ) as provider:
            result = await provider.generate(request(1))
            assert result.status is CallStatus.FAILED
            snap = collector.snapshot()
            return snap.pretransport_results, snap.observed_attempts

    blocked, observed = asyncio.run(run())
    assert blocked == 1
    assert observed == 0


def test_no_observer_keeps_existing_provider_behavior_and_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", KEY_MARKER)

    async def run() -> CallStatus:
        async with managed_httpx_groq_provider(
            GroqProviderOptions(allowed_models=frozenset({MODEL}), live_enabled=True),
            mock_transport=httpx.MockTransport(
                lambda outgoing: httpx.Response(500)
            ),
        ) as provider:
            assert not isinstance(provider, ObservedProvider)
            return (await provider.generate(request(0))).status

    assert asyncio.run(run()) is CallStatus.FAILED
