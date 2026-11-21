# Member 02 — Groq HTTP Boundary Hardening

> **Historical PR note (2026-11-21):** The 'known open items' below describe the state of this earlier transport PR. Later work added bounded `Retry-After` and optional HTTPX. The default threaded urllib cancellation limitation and live approvals remain unresolved. See the [current handoff audit](member02-final-technical-handoff.md).

**Scope:** Fix incorrect retry classification when the Groq endpoint returns an HTTP 200 response with invalid JSON, an unsupported JSON root, or an oversized body. This is an offline-tested reliability improvement, not a production-live rollout.

## Failure modes and behavior

| HTTP outcome | Transport result | Provider mapping | Retry? |
| --- | --- | --- | --- |
| 200 with valid JSON object | Body retained (bounded to 1 MiB) | Validated by response mapper | Depends on completion status |
| 200 with invalid JSON / UTF-8 or non-object JSON | 200 with no body | `GROQ_INVALID_RESPONSE` | **No** |
| 200 with more than 1 MiB response | 200 with no body | `GROQ_INVALID_RESPONSE` | **No** |
| 202, 204 or other non-200 | Status only; no body read | `GROQ_HTTP_ERROR` | **No** |
| 409 Conflict | Status only | `GROQ_CONFLICT` | **Yes**, subject to Core's global retry budget |
| 429 Too Many Requests | Status only; no error body read | `GROQ_RATE_LIMIT` | **Yes**, subject to Core's global retry budget |
| 408 / 504 | Status only | `GROQ_TIMEOUT` | **Yes**, subject to Core's global retry budget |
| 5xx | Status only; no body read | `GROQ_SERVER_ERROR` | **Yes**, subject to Core's global retry budget |

Transport failures do not contain raw response bodies or error descriptions. HTTP error handles are closed. Redirects remain disabled to protect bearer credentials and outbound text. New tests use a fake opener and make **zero external HTTP requests**, verifying request shape, timeouts, body limits and error normalization.

## Known open items

- The standard-library `asyncio.to_thread` transport is not a genuinely cancellable network operation: canceling the awaiting task does not forcefully kill an in-progress worker thread. A future PR should adopt a vetted native async HTTP client with explicit cancellation and connection pooling, reviewed by Member 04.
- Groq documents a `retry-after` header when a 429 rate limit is hit. The current shared Core error schema does not expose a safe retry-hint field, so orchestrator backoff remains bounded exponential jitter. Preserving/consuming retry hints needs a separately reviewed Core contract amendment; **this PR does not claim to honor Retry-After**.
- Actual provider/model account access, pricing, expert evaluation and production outbound authorization remain unverified and fail closed. Do not run live banking workloads.

References: https://console.groq.com/docs/rate-limits and https://github.com/groq/groq-typescript (SDK retry semantics).

## Validation

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q --cov=llm_routing_eval
uv run llm-routing-eval groq-preflight
```

Preflight `BLOCKED` (exit code 2) is the correct result until cross-member approval gates are implemented.
