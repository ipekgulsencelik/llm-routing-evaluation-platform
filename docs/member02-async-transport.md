# Member 02 — Cancellation-Cooperative Async Streaming Transport

**Status:** Available as a DI-composed transport alternative, not the live default or a completed production HTTP rollout. The evaluation CLI remains offline and Groq authorization remains deny-by-default.

## Design and SOLID boundaries

`AsyncClientGroqTransport` implements the existing `GroqTransport` port. Its only job is to send a bounded Chat Completions request to a **fixed HTTPS URL**, normalize a small safe `HttpResult` and release the streaming response context on all exit paths. It does not implement routing, retries, privacy policy, evaluator logic, billing or client lifecycle ownership.

`AsyncStreamingClient` / `AsyncStreamingResponse` are narrow structural protocols; a separate composition root can supply a reviewed native async HTTP implementation (for example `httpx.AsyncClient`) with connection pooling and cancellation. The adapter and its tests import **no vendor HTTP package**. This deliberately avoids introducing an unlocked dependency into the present offline-only MVP.

Transport dependencies flow inward through the Core-compatible `GroqTransport` interface, while the application orchestrator knows nothing about the HTTP implementation. `GroqResponseMapper` remains a pure mapping adapter, and the provider's privacy authorization remains deny-by-default.

## Protections

- HTTP POST to the hard-coded `https://api.groq.com/openai/v1/chat/completions` endpoint, with `follow_redirects=False` regardless of injected client's default setting.
- Bearer token and payload are only passed to the injected client for the authorized request. They are not included in returned `HttpResult`, logs or errors. Client-owned event hooks/proxies/logging **must** be reviewed separately to ensure they do not record secrets or sanitized bank text.
- For HTTP 429 only, parse a bounded numeric `Retry-After` into the existing retry policy. Do not read any non-200 response body.
- Stream 200 responses in 64 KiB chunks with a 1 MiB accumulated body limit. An oversized or malformed 200 response is returned as `HttpResult(status_code=200)` with no body, so the pure mapper produces a non-retryable invalid completion.
- The async stream context closes on success, premature return, exception and coroutine cancellation. No `except BaseException` or `asyncio.shield` hides `CancelledError`.
- Client session/pool ownership belongs to the composition root; do not create one client per request. Connection-pool sizing, TLS verification, proxies, timeouts, and HTTP client dependency versions require deployment review.

## Lifecycle for a later reviewed integration

1. Select, pin and audit a native async HTTP library (e.g. HTTPX), update `pyproject.toml` and `uv.lock` together, then verify `uv lock --check` on CI.
2. Configure a shared async client with TLS certificate verification, no insecure proxies or credential-logging hooks, bounded connections and redirects disabled.
3. Inject the managed async client into `AsyncClientGroqTransport` and inject that transport into `GroqChatProvider`. Preserve independent `OutboundAuthorization` approval and the existing max-in-flight request guard.
4. Dispose of the client/pool at controlled application shutdown; include connection cancellation/cleanup, timeout, rate-limit and chaos/network-failure tests.
5. Require Member 01 application retry semantics and Member 04 privacy/security approval before enabling any real traffic. Validate model availability and cost policy on synthetic-only pilot cases.

**Important:** The production default is **still `UrllibGroqTransport`**, which runs in `asyncio.to_thread`; canceling a coroutine does not forcibly abort its worker thread. This PR tests cancellation against an *injected async streaming client double*, not a real HTTPX connection, so it does **not** claim end-to-end network cancellation has been proven. The `groq-preflight` command remains BLOCKED.

## Verification

```powershell
uv sync --locked
uv run pytest -q tests/unit/test_async_groq_transport.py
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run llm-routing-eval groq-preflight
```

The tests use in-memory clients to assert cancellation and context disposal, fixed URL, redirect denial, bounded streaming, safe Retry-After handling, non-200 body refusal, provider privacy denial and exception redaction. They do not use a network, key or new dependency.

API reference: [HTTPX async support and response streaming](https://www.python-httpx.org/async/).
