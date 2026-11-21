# Member 02 — Concurrent Load and Cancellation Chaos Gates

**Scope:** Deterministic, offline resource-safety tests for the Groq provider and native HTTPX transport. The suite tests concurrency **invariants**, not provider throughput or quality. It creates no external network connections and performs no real banking API calls.

## Why this matters

An otherwise correct sequential router can misbehave when many requests run at once: queued work can continue after cancellation, a bounded provider can open too many HTTP connections, a stream cancellation can leak pool slots, or a retry loop can exceed the global per-case attempt cap. These regressions validate the existing separation of concerns without changing application/Core APIs.

## Automated contract checks

| Test family | Deterministic workload | Required invariant |
| --- | --- | --- |
| Provider backpressure | 18 concurrent synthetic provider requests, `max_in_flight=3` | Exactly three active operations at the barrier, no greater peak, all 18 eventually complete, active counter returns to zero |
| Queued request cancellation | One held provider operation, 12 queued requests | Queued work is canceled without calling transport; a new request succeeds after the held task releases its semaphore |
| Default-deny privacy under concurrency | 24 parallel provider attempts, test HTTPX client | Every request is refused by `DenyAllOutbound`, with **zero** transport calls despite fake live flags and test-only key |
| Global retry budget | Synthetic 409 and 503 failures; retry allowance greater than provider call budget | Attempts never exceed two despite `max_retries=3`; remote error-body content never persists |
| Rate-limit wait ceiling | Synthetic 429 with `Retry-After: 15`, run budget 1 second | One attempt only; no early retry or unnecessary sleep |
| Real local TCP burst | 16 requests on 127.0.0.1, HTTPX pool `max_connections=4` | Never exceed four concurrently held local requests, all 16 return validated synthetic JSON, no sockets remain in test pool |
| Real local TCP cancellation storm | Three blocked streaming responses and one subsequent recovery request | All three cancellations propagate, server sees three disconnects, the fourth request succeeds with the same HTTPX client |

These checks are **functional safety gates**, not stress/soak or latency benchmarks. Test workloads are deliberately small and event-driven rather than delay-driven; no benchmark percentage, error budget or throughput capability is inferred from the passing results.

## SOLID ownership

- The provider owns its per-instance semaphore, request constraints and injected outbound authorization.
- HTTPX owns real TCP connection pooling; `HttpxStreamingClient` adapts client behavior to the narrow `AsyncStreamingClient` port.
- The orchestrator owns retries, retry ceilings and run trace accounting. No HTTPX imports or provider-specific policies are added to Core.
- Injected tests use protocol-compatible doubles for provider concurrency and HTTPX `MockTransport` for controlled 4xx/5xx failures. Socket-backed tests bind **only** `127.0.0.1`.
- No new production features, new dependencies, Core DTO amendments or live-mode opt-ins are introduced in this milestone.

## Run the gates

```powershell
uv sync --locked
uv run pytest -q tests/unit/test_member02_concurrency_backpressure.py
uv run pytest -q tests/integration/test_member02_loopback_chaos_load.py
uv run pytest -q
uv run llm-routing-eval groq-preflight
```

`groq-preflight` should continue returning `BLOCKED` (exit code 2). The normal CI pytest job includes all newly added tests and still checks Ruff, formatting, strict mypy, lock integrity and offline smoke.

## Remaining real-production criteria

- Sustained, parallel *staging* traffic with bounded p95/p99 latency, error budgets, timeouts, retries, memory and open-fd metrics; these numbers require measured workloads and acceptance thresholds.
- Deployment-specific Linux/Windows and container networking, DNS/TLS/proxy settings, external egress controls, signal handling and graceful worker shutdown under pressure.
- Dependency vulnerability and software bill-of-materials (SBOM) policy, log/metric PII filtering, rate-limit coordination across multiple provider instances and end-to-end auditing.
- Member 03 golden expert evaluation; Member 04 production outbound privacy authorization, pricing and data-governance approvals; Groq account/model availability validation before any live pilot.

**Privacy boundary:** Fake test authorizers or synthetic live flags are used **inside tests only**. The normal provider remains deny-by-default and no real personal/banking data is sent.
