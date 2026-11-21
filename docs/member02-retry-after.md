# Member 02 — Rate-limit retry policy and cancellation contract

**Scope:** Implement a safe, bounded Groq `Retry-After` policy for HTTP 429 responses. No live model calls, new dependencies, or production authorization bypass.

## Responsibilities and SOLID boundaries

| Owner | Responsibility |
| --- | --- |
| `UrllibGroqTransport` | Read only the 429 `Retry-After` response header, validate it as numeric seconds, never expose raw provider headers or error bodies |
| `HttpResult` | Carry a typed optional retry hint from transport to the pure mapper |
| `GroqResponseMapper` | Normalize a valid hint to a rate-limit `ErrorInfo` without coupling Core to Groq |
| Core `ErrorInfo` | Add optional validated `retry_after_seconds` (0–86,400 seconds, finite, rate limit + retryable only) |
| `BenchmarkOrchestrator` | Apply bounded wait policy, global attempt limits and fallback jitter (no provider-specific import) |

The optional `ErrorInfo` field and optional `RunOptions.max_retry_wait_seconds` are backward-compatible Core changes **requiring Member 01 review**. `RunOptions` includes the budget in its configuration fingerprint. The orchestrator's persisted trace continues to strip the raw provider response and excludes the retry header.

## Behavior

- If an HTTP 429 returns a valid nonnegative numeric `Retry-After` value (seconds), that value becomes the minimum delay before retrying.
- If a valid hint is above the run's `max_retry_wait_seconds` (default 10 seconds; allowed 0–60 seconds), **do not retry**. This avoids hammering the provider sooner than it requested and keeps experiments bounded.
- If a valid numeric header exceeds one day, the hint is saturated at 86,400 seconds, which remains larger than the maximum run wait budget and therefore prevents immediate retries.
- If the header is absent or not a valid Groq numeric-seconds value, fall back to the existing bounded exponential-jitter retry delay.
- Any retry still consumes the existing `max_model_calls_per_case` global provider-attempt budget. Hints cannot increase the maximum attempt count.
- No raw provider header, token, exception detail, or model response body is emitted to trace or console by the retry code.

## Cancellation

A `GroqTransport` implemented as a native async coroutine can propagate `CancelledError`; an explicit test verifies cancellation of an in-flight injected transport without swallowing the task cancellation. **The default `UrllibGroqTransport` still uses `asyncio.to_thread`, so cancelling the waiting coroutine does not forcibly cancel the underlying OS thread/socket operation.** This is an outstanding implementation limitation. A separate reviewed switch to a dependency-locked async HTTP client with connection pooling and cancellation tests is required before claiming native network cancellation or production-grade lifecycle control.

## Verification

```powershell
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q --cov=llm_routing_eval
uv run llm-routing-eval groq-preflight
```

Tests verify numeric header parsing and saturation, invalid header rejection, exact waits, no early retry above the wait budget, request-limit accounting, strict Core DTO validation, no error-body read and async-protocol cancellation propagation. The `groq-preflight` result **must remain BLOCKED** until privacy, evaluation and pricing gates are reviewed.

Official sources: [Groq rate limits](https://console.groq.com/docs/rate-limits), [Groq API reference](https://console.groq.com/docs/api-reference).
