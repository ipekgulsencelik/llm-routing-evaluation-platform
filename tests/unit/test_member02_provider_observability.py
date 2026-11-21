"""Strict, privacy-safe provider telemetry and per-attempt diagnostic budgets."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from llm_routing_eval.adapters.providers import (
    GroqChatProvider,
    GroqProviderOptions,
    InMemoryProviderMetrics,
    ObservedProvider,
    ProviderErrorBudget,
    ProviderTelemetryEvent,
)
from llm_routing_eval.core import (
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
)

_FAKE_PROMPT = "ABSOLUTELY_SYNTHETIC_PRIVATE_PROMPT_MARKER"
_SECRET_MARKER = "TEST_ONLY_SECRET_NOT_A_REAL_KEY"


def _request() -> ModelRequest:
    return ModelRequest(
        run_id="synthetic-run",
        case_id="synthetic-case",
        trace_id="synthetic-trace",
        attempt_id="synthetic-attempt",
        provider="groq",
        model="openai/gpt-oss-20b",
        prompt=_FAKE_PROMPT,
        timeout_seconds=2.0,
    )


def _response(
    *,
    status: CallStatus = CallStatus.SUCCESS,
    origin: MetricOrigin = MetricOrigin.OBSERVED,
    latency_ms: float | None = 12.5,
) -> ModelResponse:
    return ModelResponse(
        provider="groq",
        model="openai/gpt-oss-20b",
        attempt_id="synthetic-attempt",
        status=status,
        metric_origin=origin,
        content="SYNTHETIC_RESPONSE_SHOULD_NOT_APPEAR" if status is CallStatus.SUCCESS else None,
        error=(
            None
            if status is CallStatus.SUCCESS
            else ErrorInfo(
                category=(
                    ErrorCategory.RATE_LIMIT
                    if status is CallStatus.RATE_LIMITED
                    else ErrorCategory.PROVIDER
                ),
                code="SYNTHETIC_PROVIDER_DETAIL_NOT_EXPORTED",
                retryable=True,
            )
        ),
        latency_ms=latency_ms,
    )


def test_event_contract_forbids_request_metadata_or_secrets() -> None:
    event = ProviderTelemetryEvent.from_response(_response())
    assert event.kind == "RESULT"
    content = event.model_dump_json()
    for forbidden in (
        _FAKE_PROMPT,
        "SYNTHETIC_RESPONSE_SHOULD_NOT_APPEAR",
        "synthetic-run",
        "synthetic-case",
        "synthetic-attempt",
        "openai/gpt-oss-20b",
        _SECRET_MARKER,
    ):
        assert forbidden not in content
    with pytest.raises(ValidationError):
        ProviderTelemetryEvent.model_validate(
            {
                "kind": "RESULT",
                "status": "SUCCESS",
                "metric_origin": "OBSERVED",
                "prompt": _FAKE_PROMPT,
            }
        )
    with pytest.raises(ValidationError):
        ProviderTelemetryEvent(kind="CANCELLED", status=CallStatus.TIMEOUT)
    with pytest.raises(ValidationError):
        ProviderTelemetryEvent(
            kind="RESULT",
            status=CallStatus.SUCCESS,
            metric_origin=MetricOrigin.OBSERVED,
            latency_ms=float("nan"),
        )


def test_observed_budget_exhausts_at_five_percent_threshold_and_tail_sample_gate() -> None:
    collector = InMemoryProviderMetrics(max_events=50)
    for i in range(20):
        status = CallStatus.SUCCESS if i < 18 else CallStatus.RATE_LIMITED
        collector.record(
            ProviderTelemetryEvent.from_response(_response(status=status, latency_ms=float(i + 1)))
        )
    summary = collector.snapshot()
    assert summary.scope == "DIAGNOSTIC_ONLY"
    assert summary.retained_events == 20
    assert summary.evicted_events == 0
    assert summary.observed_attempts == 20
    assert summary.observed_successes == 18
    assert summary.observed_failures == 2
    assert summary.latency_samples == 20
    assert summary.latency_p50_ms == 10.5
    assert summary.latency_p95_ms == pytest.approx(19.05)
    assert summary.failure_fraction == pytest.approx(0.1)
    assert summary.budget_status == "EXHAUSTED"
    assert summary.observed_status_counts[CallStatus.RATE_LIMITED] == 2
    assert summary.observed_error_categories[ErrorCategory.RATE_LIMIT] == 2

    collector.record(
        ProviderTelemetryEvent.from_response(
            _response(origin=MetricOrigin.SIMULATED, latency_ms=None)
        )
    )
    summary = collector.snapshot()
    assert summary.pretransport_results == 1
    assert summary.observed_attempts == 20


def test_equal_budget_boundary_is_within_budget() -> None:
    collector = InMemoryProviderMetrics()
    for i in range(20):
        collector.record(
            ProviderTelemetryEvent.from_response(
                _response(
                    status=CallStatus.TIMEOUT if i == 19 else CallStatus.SUCCESS,
                )
            )
        )
    state = collector.snapshot()
    assert state.failure_fraction == pytest.approx(0.05)
    assert state.budget_status == "WITHIN_BUDGET"


def test_insufficient_coverage_must_not_claim_budget_or_p95() -> None:
    collector = InMemoryProviderMetrics()
    for _ in range(19):
        collector.record(
            ProviderTelemetryEvent.from_response(
                _response(status=CallStatus.TIMEOUT, latency_ms=0.0)
            )
        )
    status = collector.snapshot()
    assert status.budget_status == "INSUFFICIENT_DATA"
    assert status.failure_fraction is None
    assert status.latency_p50_ms == 0.0
    assert status.latency_p95_ms is None


def test_missing_latency_and_mixed_non_observed_are_excluded_from_percentiles() -> None:
    collector = InMemoryProviderMetrics()
    collector.record(ProviderTelemetryEvent.from_response(_response(latency_ms=None)))
    collector.record(
        ProviderTelemetryEvent.from_response(
            _response(origin=MetricOrigin.SIMULATED, latency_ms=987654.0)
        )
    )
    collector.record(ProviderTelemetryEvent(kind="CANCELLED"))
    collector.record(ProviderTelemetryEvent(kind="EXCEPTION"))
    snapshot = collector.snapshot()
    assert snapshot.retained_events == 4
    assert snapshot.observed_attempts == 1
    assert snapshot.latency_samples == 0
    assert snapshot.latency_p50_ms is None
    assert snapshot.latency_p95_ms is None
    assert snapshot.pretransport_results == 1
    assert snapshot.cancelled_invocations == 1
    assert snapshot.raised_exceptions == 1
    assert snapshot.budget_status == "INSUFFICIENT_DATA"


def test_window_is_bounded_and_history_does_not_inflate_active_budget() -> None:
    collector = InMemoryProviderMetrics(max_events=20)
    for _ in range(10):
        collector.record(ProviderTelemetryEvent.from_response(_response(status=CallStatus.TIMEOUT)))
    for _ in range(20):
        collector.record(ProviderTelemetryEvent.from_response(_response()))
    summary = collector.snapshot()
    assert summary.retained_events == 20
    assert summary.evicted_events == 10
    assert summary.observed_attempts == 20
    assert summary.observed_failures == 0
    assert summary.budget_status == "WITHIN_BUDGET"


@pytest.mark.parametrize("max_events", [0, 19, 100001])
def test_window_rejects_invalid_limits(max_events: int) -> None:
    with pytest.raises(ValueError, match="METRICS_WINDOW_OUT_OF_RANGE"):
        InMemoryProviderMetrics(max_events=max_events)


def test_budget_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValidationError):
        ProviderErrorBudget(max_failure_fraction=1.5)
    with pytest.raises(ValidationError):
        ProviderErrorBudget(min_observed_attempts=0)


def test_observed_provider_preserves_blocked_result_without_external_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", _SECRET_MARKER)
    provider = GroqChatProvider(
        GroqProviderOptions(
            allowed_models=frozenset({"openai/gpt-oss-20b"}),
            live_enabled=True,
        )
    )
    sink = InMemoryProviderMetrics()
    monitored = ObservedProvider(provider, sink)
    result = asyncio.run(monitored.generate(_request()))
    assert result.status is CallStatus.FAILED
    assert result.error is not None
    assert result.error.category is ErrorCategory.PRIVACY
    summary = sink.snapshot()
    assert summary.pretransport_results == 1
    assert summary.observed_attempts == 0
    assert summary.budget_status == "INSUFFICIENT_DATA"
    serialized = summary.model_dump_json()
    assert _FAKE_PROMPT not in serialized
    assert _SECRET_MARKER not in serialized
    assert "GROQ_PRIVACY_NOT_AUTHORIZED" not in serialized


def test_observer_failure_does_not_change_success_response_or_leak_exception() -> None:
    class FakeProvider:
        async def generate(self, request: ModelRequest) -> ModelResponse:
            _ = request
            return _response()

    class BrokenSink:
        def record(self, event: ProviderTelemetryEvent) -> None:
            _ = event
            raise RuntimeError("NEVER_LOG_THIS_SINK_EXCEPTION")

    observer = ObservedProvider(FakeProvider(), BrokenSink())
    result = asyncio.run(observer.generate(_request()))
    assert result.status is CallStatus.SUCCESS
    assert observer.lost_events == 1


def test_cancelled_task_records_only_cancel_event_and_propagates() -> None:
    class BlockingProvider:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def generate(self, request: ModelRequest) -> ModelResponse:
            _ = request
            self.started.set()
            await asyncio.Event().wait()
            return _response()

    async def run() -> tuple[bool, int]:
        provider = BlockingProvider()
        sink = InMemoryProviderMetrics()
        observed = ObservedProvider(provider, sink)
        task = asyncio.create_task(observed.generate(_request()))
        await asyncio.wait_for(provider.started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        summary = sink.snapshot()
        return task.cancelled(), summary.cancelled_invocations

    is_cancelled, count = asyncio.run(run())
    assert is_cancelled
    assert count == 1


def test_unexpected_delegate_exception_is_not_converted_to_success() -> None:
    class ExplodingProvider:
        async def generate(self, request: ModelRequest) -> ModelResponse:
            _ = request
            raise RuntimeError("SENSITIVE_RUNTIME_ERROR")

    collector = InMemoryProviderMetrics()
    wrapped = ObservedProvider(ExplodingProvider(), collector)
    with pytest.raises(RuntimeError):
        asyncio.run(wrapped.generate(_request()))
    snap = collector.snapshot()
    assert snap.raised_exceptions == 1
    assert snap.observed_attempts == 0
    assert "SENSITIVE_RUNTIME_ERROR" not in snap.model_dump_json()
