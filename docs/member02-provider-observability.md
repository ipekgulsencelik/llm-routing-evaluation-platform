# Member 02 — Privacy-safe provider observability and diagnostic error budgets

**Scope:** A bounded, opt-in, in-memory observability adapter for Groq provider outcomes, with safe counters, guarded latency quantiles, and an explicit *diagnostic-only* error-budget assessment. No external exporter, live activation, customer prompts or new dependencies.

## Design and SOLID boundaries

| Component | Responsibility |
| --- | --- |
| `ProviderTelemetryEvent` | Immutable fixed-shape event: outcome kind, status, metric origin, latency and error **category** only |
| `ProviderMetricsSink` | Narrow synchronous recording port, independent of any telemetry vendor or persistence module |
| `ObservedProvider` | Decorates the existing `LLMProviderPort` without changing its signature or request/response values |
| `InMemoryProviderMetrics` | Stores only a bounded FIFO event window (`max_events` 20–100,000, default 1,000) and returns aggregate snapshots |
| `ProviderErrorBudget` | Validates diagnostic failure-fraction and minimum observation count; does **not** approve a production SLO |
| `managed_httpx_groq_provider(metrics_sink=...)` | Opt-in composition; with no sink, the existing provider path is unchanged |

The collected event schema deliberately has **no fields** for prompt, response content, user/case/trace/attempt identifiers, provider request IDs, model identifier, API key, arbitrary error code, remote endpoint, raw headers, timestamps or exception strings. Extra event keys are rejected. There are no dynamic metric labels and no logs or network exporters. Application-level authorization still defaults to `DenyAllOutbound`.

## Counters and denominators

- `observed_attempts`: provider results explicitly tagged `MetricOrigin.OBSERVED`; **per attempted provider call**, not per user case and not per completed routing run. This count may include retries.
- `observed_failures`: all observed outcomes other than `SUCCESS` (including 429, 409, timeout, 5xx and malformed completion). A provider attempt that returned a non-success is not automatically a failed *end-user case*.
- `pretransport_results`: responses not tagged observed; includes input validation, missing opt-in, privacy denial and other fail-closed checks. **Excluded from the provider error-budget denominator.**
- `cancelled_invocations` and `raised_exceptions`: counted separately; when it cannot be proven whether transport was reached, neither is assigned a provider-attempt error rate.
- `observed_status_counts` and `observed_error_categories`: fixed-cardinality enum-based counters only. Neither event nor report includes raw provider error text.
- `latency_samples`: finite nonnegative durations attached to observed results; p50 with at least one such sample; p95 only when **at least 20** latency samples exist. Missing samples are never backfilled as zero.
- `evicted_events`: number of oldest events removed when the memory cap is reached. Calculations describe the **retained window**, not all-time traffic.

## Budget semantics (diagnostic only)

`failure_fraction = observed_failures / observed_attempts` is published **only when** the count reaches `ProviderErrorBudget.min_observed_attempts` (default 20). Before that, the fraction is `null` and `budget_status = INSUFFICIENT_DATA`. Once the threshold is met, a fraction greater than `max_failure_fraction` (default 0.05) produces `EXHAUSTED`; otherwise `WITHIN_BUDGET`. Threshold values are illustrative engineering defaults, **not stakeholder-approved SLOs**.

These are **local diagnostic comparisons**, not independently verified billable performance or production SLA/error-budget evidence. The adapter's `OBSERVED` metric origin only means an HTTP transport produced a result: **HTTPX `MockTransport` tests can also produce `OBSERVED` results**. Therefore, a favorable `WITHIN_BUDGET` score from test traffic does **not** authorize a live rollout or establish reliability on real Groq servers.

## Privacy and failure behavior

- Monitoring is optional; no call to a telemetry service and no exception text emitted.
- Provider cancellation propagates as `asyncio.CancelledError`; a dedicated `CANCELLED` event is attempted.
- An unexpected delegate exception is **re-raised** after attempting to record a minimal `EXCEPTION` event; it is never silently marked successful.
- If an optional sink fails, the provider outcome is preserved, and the `ObservedProvider.lost_events` counter increments. A nonzero counter means snapshot coverage is incomplete. **Missing observations and sink failures must block reliance on telemetry for a real release decision.**
- The provided collector is a single-process, ephemeral diagnostic; it has no durable storage, multi-worker coordination or distributed sampling.

## Usage with the existing controlled HTTPX composition

```python
from llm_routing_eval.adapters.providers import (
    GroqProviderOptions,
    InMemoryProviderMetrics,
    ProviderErrorBudget,
    managed_httpx_groq_provider,
)

collector = InMemoryProviderMetrics(
    max_events=1000,
    budget=ProviderErrorBudget(max_failure_fraction=0.05, min_observed_attempts=20),
)

# NO outbound authorization is granted: this remains deny-by-default.
async with managed_httpx_groq_provider(
    GroqProviderOptions(allowed_models=frozenset({'openai/gpt-oss-20b'})),
    metrics_sink=collector,
) as provider:
    # This example intentionally performs no HTTP requests.
    pass

safe_metrics = collector.snapshot()
print(safe_metrics.model_dump_json())
```

Within synthetic tests, a strictly allowlisted test authorizer and `httpx.MockTransport` can be injected to exercise the complete provider path. This is **not** a substitute for Member 04's approved privacy authorization.

## Validation and next review gates

```powershell
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q --cov=llm_routing_eval
uv run llm-routing-eval groq-preflight
```

Current CI tests verify schema strictness, no sensitive labels, bounded storage, loss tracking, 20-sample p95 gate, insufficient-data behavior, equality/exhaustion boundaries, default-deny calls, sink failures, cancellations and HTTPX composition with in-memory fake responses.

**Outstanding release gates:** Member 01 must approve benchmark metric semantics and run-level attribution; Member 03 must provide an expert-validated golden evaluation set; Member 04 must approve privacy, retention, audit, pricing, operational telemetry/SLOs and any durable exporter. Real Groq account/model access and synthetic-only approved pilot are not yet verified. `groq-preflight` intentionally remains **BLOCKED (exit code 2)**.
