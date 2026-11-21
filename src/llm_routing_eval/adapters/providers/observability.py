"""Privacy-safe provider diagnostics, isolated from routing and persistence.

No prompts, outputs, identifiers, keys, model IDs, request URLs, exception
messages, or arbitrary labels can enter the telemetry event schema.
This bounded in-memory diagnostic is NOT production-grade monitoring or billing.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from collections.abc import Sequence
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_routing_eval.core import (
    CallStatus,
    ErrorCategory,
    LLMProviderPort,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class ProviderTelemetryEvent(_Frozen):
    """A fixed-cardinality, scrubbed outcome without any request-level metadata."""

    kind: Literal["RESULT", "CANCELLED", "EXCEPTION"]
    status: CallStatus | None = None
    metric_origin: MetricOrigin | None = None
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    error_category: ErrorCategory | None = None

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if self.kind == "RESULT":
            if self.status is None or self.metric_origin is None:
                raise ValueError("TELEMETRY_RESULT_INCOMPLETE")
            if self.status is CallStatus.SUCCESS and self.error_category is not None:
                raise ValueError("TELEMETRY_SUCCESS_HAS_ERROR")
        elif any(
            field is not None
            for field in (self.status, self.metric_origin, self.latency_ms, self.error_category)
        ):
            raise ValueError("TELEMETRY_NON_RESULT_HAS_FIELDS")
        return self

    @classmethod
    def from_response(cls, response: ModelResponse) -> ProviderTelemetryEvent:
        """Explicitly project only enum-valued safe fields and bounded metrics."""
        return cls(
            kind="RESULT",
            status=response.status,
            metric_origin=response.metric_origin,
            latency_ms=response.latency_ms,
            error_category=response.error.category if response.error is not None else None,
        )


class ProviderMetricsSink(Protocol):
    """ISP: recording events requires one synchronous, bounded method."""

    def record(self, event: ProviderTelemetryEvent) -> None: ...


class ObservedProvider:
    """Decorator preserving LLMProviderPort and original cancellation semantics.

    Sinks are not allowed to change provider outcomes. A defective optional sink
    increments an explicit lost_events counter rather than breaking the request.
    """

    def __init__(self, provider: LLMProviderPort, sink: ProviderMetricsSink) -> None:
        self._provider = provider
        self._sink = sink
        self._lost_events = 0

    @property
    def lost_events(self) -> int:
        return self._lost_events

    def _record(self, event: ProviderTelemetryEvent) -> None:
        try:
            self._sink.record(event)
        except Exception:
            # Observability may fail, but it must not alter the provider result.
            # Never re-emit error details; report the counter at the call site.
            self._lost_events += 1

    async def generate(self, request: ModelRequest) -> ModelResponse:
        try:
            response = await self._provider.generate(request)
        except asyncio.CancelledError:
            self._record(ProviderTelemetryEvent(kind="CANCELLED"))
            raise
        except Exception:
            self._record(ProviderTelemetryEvent(kind="EXCEPTION"))
            raise
        self._record(ProviderTelemetryEvent.from_response(response))
        return response


class ProviderErrorBudget(_Frozen):
    """Diagnostic threshold; not an approved production error budget or SLO."""

    max_failure_fraction: float = Field(default=0.05, ge=0, le=1, allow_inf_nan=False)
    min_observed_attempts: int = Field(default=20, ge=1, le=1_000_000)


class ProviderMetricsSnapshot(_Frozen):
    """Bounded-window summary. Provider attempts differ from user-facing cases."""

    scope: Literal["DIAGNOSTIC_ONLY"] = "DIAGNOSTIC_ONLY"
    retained_events: int = Field(ge=0)
    evicted_events: int = Field(ge=0)
    observed_attempts: int = Field(ge=0)
    observed_successes: int = Field(ge=0)
    observed_failures: int = Field(ge=0)
    pretransport_results: int = Field(ge=0)
    cancelled_invocations: int = Field(ge=0)
    raised_exceptions: int = Field(ge=0)
    observed_status_counts: dict[CallStatus, int]
    observed_error_categories: dict[ErrorCategory, int]
    latency_samples: int = Field(ge=0)
    latency_p50_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    latency_p95_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    failure_fraction: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    max_failure_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    budget_status: Literal["INSUFFICIENT_DATA", "WITHIN_BUDGET", "EXHAUSTED"]


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


class InMemoryProviderMetrics:
    """Small, bounded event window; no background exporter or cardinality risk."""

    def __init__(
        self,
        *,
        max_events: int = 1_000,
        budget: ProviderErrorBudget | None = None,
    ) -> None:
        if not 20 <= max_events <= 100_000:
            raise ValueError("METRICS_WINDOW_OUT_OF_RANGE")
        self._events: deque[ProviderTelemetryEvent] = deque(maxlen=max_events)
        self._evicted = 0
        self._budget = budget if budget is not None else ProviderErrorBudget()

    def record(self, event: ProviderTelemetryEvent) -> None:
        if len(self._events) == self._events.maxlen:
            self._evicted += 1
        self._events.append(event)

    def snapshot(self) -> ProviderMetricsSnapshot:
        retained = tuple(self._events)
        # Metrics only assess transport-invoked outcomes; blocked validation and
        # pre-authorization results carry SIMULATED and must not contaminate the
        # observed provider-attempt error denominator.
        observed = [
            event
            for event in retained
            if event.kind == "RESULT" and event.metric_origin is MetricOrigin.OBSERVED
        ]
        statuses = Counter(
            event.status for event in observed if event.status is not None
        )
        errors = Counter(
            event.error_category for event in observed if event.error_category is not None
        )
        samples = [
            event.latency_ms for event in observed if event.latency_ms is not None
        ]
        successful = statuses[CallStatus.SUCCESS]
        total = len(observed)
        failures = total - successful
        enough = total >= self._budget.min_observed_attempts
        fraction = failures / total if enough else None
        state: Literal["INSUFFICIENT_DATA", "WITHIN_BUDGET", "EXHAUSTED"]
        if fraction is None:
            state = "INSUFFICIENT_DATA"
        elif fraction <= self._budget.max_failure_fraction:
            state = "WITHIN_BUDGET"
        else:
            state = "EXHAUSTED"
        return ProviderMetricsSnapshot(
            retained_events=len(retained),
            evicted_events=self._evicted,
            observed_attempts=total,
            observed_successes=successful,
            observed_failures=failures,
            pretransport_results=sum(
                event.kind == "RESULT" and event.metric_origin is not MetricOrigin.OBSERVED
                for event in retained
            ),
            cancelled_invocations=sum(event.kind == "CANCELLED" for event in retained),
            raised_exceptions=sum(event.kind == "EXCEPTION" for event in retained),
            observed_status_counts=dict(statuses),
            observed_error_categories=dict(errors),
            latency_samples=len(samples),
            latency_p50_ms=_quantile(samples, 0.5) if samples else None,
            # Do not imply a stable tail-latency percentile on tiny samples.
            latency_p95_ms=_quantile(samples, 0.95) if len(samples) >= 20 else None,
            failure_fraction=fraction,
            max_failure_fraction=self._budget.max_failure_fraction,
            budget_status=state,
        )
