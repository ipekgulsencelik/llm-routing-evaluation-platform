# Member 02 — Enterprise Hardening: Privacy, Reliability & Provenance

> **Historical milestone note (2026-11-21):** This file captures the original hardening plan. Follow-up changes implemented bounded `Retry-After`, an opt-in managed HTTPX 0.28.1 transport, local TCP/TLS cancellation tests and privacy-safe metrics. See the [current technical audit and release blockers](member02-final-technical-handoff.md).

**Status:** Proposed architecture and tested implementation, not enterprise certification. No live banking benchmark is exposed by this PR.

## Architecture (SOLID / Ports and Adapters)

| Component | Single responsibility |
| --- | --- |
| `GroqChatProvider` | Compose validated request, outbound approval, transport and response mapper |
| `OutboundAuthorization` | Narrow per-request outbound approval protocol; default `DenyAllOutbound` |
| `GroqTransport` / `UrllibGroqTransport` | HTTPS IO, bounded responses and redirect refusal |
| `GroqResponseMapper` | Pure HTTP status/error/telemetry-to-Core DTO mapping |
| `RoutingSettings.fingerprint()` | Deterministic identity for a versioned routing policy |
| `BenchmarkOrchestrator` | Existing Core-port orchestration, bounded retries, persistence and evaluation |

SRP: mapping, authorization, transport and orchestration are separated. DIP: orchestration imports Core protocols, not concrete providers. ISP: transport and outbound-approval interfaces are narrow. OCP: alternative implementations can be injected. LSP: the unchanged `LLMProviderPort` contract is implemented by the Groq adapter.

## Fail-closed live protections

1. Existing live opt-ins (`GroqProviderOptions.live_enabled=True`, `LLM_ROUTING_ALLOW_LIVE=1`, `GROQ_API_KEY`) are necessary but **not sufficient**. A reviewed `OutboundAuthorization` implementation must explicitly authorize each request. The default denies all live requests, even with credentials.
2. Failed/throwing authorization, non-allowlisted model, unsupported generation settings and prompt or completion-token budget violations reject requests **before** transport.
3. The adapter bounds simultaneous calls with a semaphore. This is an in-process bound, not a distributed quota or rate limiter.
4. The HTTP transport refuses redirects, limits response size, and never reads/provider-logs error bodies. It uses standard-library HTTPS and `asyncio.to_thread`, so cancelling a coroutine does **not** forcibly terminate an active OS-level request.
5. HTTP 429, 408, 409, 5xx errors are normalized to safe categories. Existing orchestration owns bounded retries; server Retry-After, rate budgeting and circuit-breaking remain future work.
6. Truncated (`finish_reason=length`), tool-call, wrong-model and malformed provider responses are not treated as valid scored answers. Observed usage is retained for truncated responses, including cached/reasoning counts where supplied.
7. Provider identifiers are accepted only if they match a conservative safe-character rule. No API keys, raw prompt text, provider error bodies or exception strings are persisted by these adapters.

The HTTP transport also maps malformed/oversized HTTP 200 payloads to a non-retryable invalid response rather than a false server error. See [Groq transport hardening](member02-transport-hardening.md).

**Security limitation:** Implementing this authorization protocol does not make any application privacy adapter automatically compliant or anonymizing. Member 04 must supply a reviewed production authorization implementation. This repository intentionally supplies no permissive production implementation.

## Routing reproducibility

`RoutingSettings.fingerprint()` SHA-256 hashes canonical, versioned policy contents: model pair, rule order, kNN development examples, thresholds and Cascade policy. It stores only a hash (not development-example texts) in report provenance as `routing_policy_sha256`. An optional, backwards-compatible `RunOptions.routing_policy_sha256` field also affects the general configuration fingerprint. This Core extension requires Member 01 review.

## Offline verification

```powershell
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q --cov=llm_routing_eval
uv run llm-routing-eval validate-routing-config --router-config config/routing.yaml
uv run llm-routing-eval run-routing --strategy cascade
```

Tests cover deny-by-default, approval exceptions, budget enforcement, concurrency, HTTP error classification, truncation, usage normalization, safe request IDs and configuration fingerprint stability. No real network or keys required.

## Release gates before claiming enterprise / production-ready

- [ ] Member 01: review the optional provenance schema extension and run-status semantics
- [ ] Member 03: expert-reviewed golden cases and safe evaluation rubrics with held-out isolation
- [ ] Member 04: production privacy/consent adapter, secrets, auditability, retention, secure network posture and durable reporting
- [ ] Member 02: calibrated semantic embeddings, controlled live Groq integration tests, model-access verification, Retry-After and rate/cancellation handling
- [ ] Shared: observed usage and versioned price table, per-attempt/cascade cost accounting, quality-first gates, threat-model review and roll-back plan

Official API references: https://console.groq.com/docs/api-reference and https://console.groq.com/docs/rate-limits.
