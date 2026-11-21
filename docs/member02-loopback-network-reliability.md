# Member 02 — Local TCP Reliability Verification

**Status:** Automated socket-backed reliability tests, now also extended with local TLS, injected DNS failures and abrupt disconnects in [the TLS/DNS follow-up](member02-tls-dns-network-failures.md). No external requests or Groq credentials. This does not approve live banking benchmarking.

## Why this test suite exists

The earlier `MockTransport` tests validate interface contracts and in-memory cancellation, but cannot establish whether HTTPX actually closes a TCP connection when a request is cancelled or times out. This suite exercises operating-system sockets against an ephemeral **`127.0.0.1`-only** test server. It never resolves or connects to the Groq host.

## Implemented tests

| Test | Verified behavior |
| --- | --- |
| Socket-backed JSON exchange | HTTPX sends two synthetic POSTs, receives complete JSON responses, and the shared client closes after its context exits |
| Stalled response body | A real localhost socket delivers response headers then stalls; HTTPX emits `ReadTimeout`, normalized by the wrapper to safe `GROQ_HTTP_TIMEOUT`, and the server observes connection termination |
| Cancellation mid-response | After receiving one byte, a blocked consumer is cancelled; `CancelledError` propagates, and the server observes that the underlying TCP connection closes |
| Saturated connection pool | A single allowed TCP connection is held open while another request is attempted; HTTPX pool timeout becomes a safe `GROQ_HTTP_TIMEOUT` rather than opening a second socket |

Each test owns its localhost server, bounds waits with `asyncio.wait_for`, and tears down open request handlers. No real user messages, account information, API keys or external provider traffic are used. The test file is `tests/integration/test_httpx_loopback_network.py`.

## Run locally

```powershell
uv sync --locked
uv run pytest -q tests/integration/test_httpx_loopback_network.py
uv run pytest -q
uv run llm-routing-eval groq-preflight
```

`groq-preflight` is expected to report `BLOCKED` and exit with status 2. The localhost tests do not disable or bypass production privacy authorization.

## Remaining enterprise acceptance criteria

- **TLS/DNS failure simulation:** the follow-up adds real loopback TLS handshakes with synthetic test CAs and an injected fake DNS resolver failure; public TLS/DNS infrastructure remains untested.
- **Host network policy:** Docker/eBPF outbound network allowlists, NAT, proxy enforcement, DNS pinning and certificate trust require a security review.
- **Load and chaos:** multiple concurrent connections, connection resets, half-open sockets, longer-duration cancellation storms and resilience measurements require a controlled workload with threshold-based acceptance criteria.
- **Production release gates:** Member 03's expert evaluation, Member 04's reviewed outbound privacy authorization, verified pricing and Groq account/model access are still unresolved. Never use real customer data until approvals are complete.

These tests are part of the existing `pytest` GitHub Actions quality gate, separate from provider availability and billable API latency metrics.
