# LLM Routing Evaluation Platform - V1.1

**Status:** Member 01 core, application and CLI offline demo implemented. Member 02's four offline routing strategies, strict YAML composition and CLI are merged into main. The optional Groq adapter is available but no live benchmark endpoint is exposed. Live providers, production privacy, dataset evaluation and durable reporting remain pending.

An open-source, domain-independent framework to evaluate whether LLM routing strategies improve cost, answer quality and latency on controlled workloads. The first domain pack targets **Turkish Retail Banking & Customer Support**.

## Approved MVP architecture

- Python 3.12+, uv, TOML, YAML, Docker
- Modular monolith, SOLID/OOP and pragmatic Ports & Adapters
- Pydantic v2 contracts, Typer CLI, async dependency injection through Ports
- Strategies: Always-Large, Rule-Based, kNN and Cascade (configured offline RouterPort implementations and CLI)
- Primary live provider: Groq (later, opt-in); NVIDIA Build (post-MVP)

## Member 02 offline strategy evaluation (new)

Configure model pair, rules, development kNN references and Cascade policy in `config/routing.yaml`. This is **offline simulation** (fake provider, synthetic cases, and fixture quality only):

```bash
uv run llm-routing-eval validate-routing-config --router-config config/routing.yaml
uv run llm-routing-eval run-routing --strategy always_large
uv run llm-routing-eval run-routing --strategy rule_based
uv run llm-routing-eval run-routing --strategy knn
uv run llm-routing-eval run-routing --strategy cascade
```

Member 02 also provides a Groq Chat Completions adapter behind the shared provider port. It is **not connected to the CLI**, requires an explicit per-instance and environment opt-in plus `GROQ_API_KEY`, and is covered by network-free tests. The provider additionally requires explicit per-request outbound authorization (default deny), enforces resource budgets, and stores routing policy hashes in benchmark provenance. The production privacy, financial evaluation and cost-accounting gates remain unimplemented, so no live banking benchmark is offered. See [Member 02 enterprise-hardening design](docs/member02-enterprise-hardening.md).

See [Member 02 implementation](docs/member02-implementation.md).

## Member 02 four-strategy comparison (offline)

Run one controlled, same-workload comparison and produce a single safe JSON summary:

```powershell
uv run llm-routing-eval compare-routing --output artifacts/offline_comparison.json
```

The report includes success and model-attempt counts, requests-per-case, model mix, local elapsed times, simulated token totals **and per-attempt averages**, fixture-only score coverage, routing/dataset fingerprints, and explicit unknown cost. Simulated data and two examples are **not** a live financial quality or latency benchmark. See [comparison design and measurement caveats](docs/member02-comparison.md).

## Member 02 Groq integration preflight (network-free)

```powershell
uv run llm-routing-eval groq-preflight
```

The expected result is **BLOCKED (exit code 2)** until separately approved privacy authorization, expert evaluation, verified pricing and actual model-account access are in place. This command **never calls Groq** and does not print secrets. All four strategies have a separate Groq wire-protocol integration test suite using an injected synthetic HTTP stub; it is not a live performance benchmark. See [preflight and cross-member handoff](docs/member02-groq-preflight.md).

## Member 02 bounded Groq rate-limit retries

Groq HTTP 429 numeric `Retry-After` is now parsed without retaining raw headers and honored only within the run's explicitly bounded retry wait; longer hints stop retries rather than retrying early. This adds safe optional retry-hint fields to the provider/Core contract; see [retry policy and remaining cancellation limitations](docs/member02-retry-after.md). Live provider use remains **deny-by-default** and `groq-preflight` remains blocked.

## Member 02 optional native-async Groq transport boundary

An injectable `AsyncClientGroqTransport` supports bounded streaming and coroutine cancellation through an externally managed async HTTP client. It preserves the existing provider privacy gate and does not require an extra dependency to run offline tests. **It is not the live default, does not bundle an HTTPX dependency, and has not been validated against a real network connection.** See [async transport design and rollout requirements](docs/member02-async-transport.md).
## Member 02 managed HTTPX runtime (opt-in, not live-enabled)

The repository now pins `httpx==0.28.1` in `pyproject.toml` and `uv.lock`. A managed `httpx.AsyncClient` can be injected into the Groq provider through `managed_httpx_groq_provider()` with bounded TLS-verified connection pooling, normalized timeouts and asynchronous response disposal. The default provider/CLI authorization is unchanged: **live banking benchmarking remains disabled**. Unit tests use HTTPX mock streams and prove no real network cancellation. See [managed HTTPX integration and remaining production gates](docs/member02-managed-httpx.md).
## Member 02 localhost TCP reliability tests

The [loopback network reliability suite](docs/member02-loopback-network-reliability.md) verifies HTTPX against real **127.0.0.1 sockets**: successful streaming, read timeout, cancellation closing the TCP connection, and connection-pool saturation. It is included in the regular pytest/CI checks. **There are no external Groq calls**, and real TLS/DNS and deployment load/chaos tests remain future security release gates.
## Member 02 TLS / DNS / remote-disconnect checks

A new [network-failure integration suite](docs/member02-tls-dns-network-failures.md) exercises **real local TLS handshakes** using ephemeral test-only CA certificates, verifies that untrusted or hostname-mismatched certificates fail without retries, and tests synthetic DNS resolution failure plus abrupt TCP disconnects. `trustme==1.2.1` is a dev-only locked dependency; no certificates or keys are committed. Live provider access remains denied by default.
## Member 02 concurrency and cancellation chaos tests

The [concurrency/backpressure and local-TCP chaos gates](docs/member02-chaos-backpressure-gates.md) verify provider semaphore limits, default-deny privacy under concurrent load, cancellation of queued requests, retry ceilings, HTTPX connection-pool caps, and cleanup after multiple canceled local socket streams. They are included in the standard offline CI test suite. **These are functional safety checks, not live throughput, financial quality, or Groq performance benchmarks.**
## Member 02 privacy-safe provider observability (opt-in)

An optional [bounded provider metrics adapter](docs/member02-provider-observability.md) now provides fixed-cardinality status/error-category counters, measured-transport latency samples, minimum-coverage p50/p95, and diagnostic-only provider-attempt error-budget checks. It stores **no prompts, outputs, credentials or request identifiers**, and is never automatically attached to a network exporter. Pre-transport privacy denials, missing data and cancellations remain separate from observed provider-attempt denominators. `managed_httpx_groq_provider(metrics_sink=...)` permits explicit injection; the default Groq authorization and live CLI remain blocked.
## Member 02 cross-member contract handoff (offline)

An [offline cross-member interface acceptance suite](docs/member02-cross-member-integration.md) composes Member 01's orchestrator with all four real Member 02 routers, the Groq HTTPX adapter and provider metrics, plus **test-only Member 03 evaluator and Member 04 privacy/repository adapters**. It verifies two privacy gates, golden reference isolation, fail-closed outbound authorization, redacted persistence, evaluator failure and trace/telemetry reconciliation. **The actual Member 03/04 modules are still scaffolds** and live Groq integration is not complete; this suite provides explicit integration acceptance tests for their future real adapters.
## Member 01 offline smoke test

The CLI is deliberately limited to two allowlisted synthetic examples and deterministic test doubles. **It does not judge financial correctness and never contacts a real API.**

```bash
# In the repository root, with PyPI access and Python 3.12+:
uv sync --locked
uv run llm-routing-eval validate-config --config config/settings.yaml
uv run llm-routing-eval run --config config/settings.yaml \
  --dataset examples/offline_demo_cases.json \
  --output artifacts/offline_demo.jsonl
uv run pytest -q
```

The generated JSONL contains a redacted execution trace, fixture-only evaluation data and a comparison report. Cost is `null` (unknown), model telemetry is simulated and p95 is withheld for very small samples.

`--live` and `BENCHMARK_MODE=live` **fail closed** because live support is not implemented. Do not use customer data in this demo.

### Current verification constraints

Current Member 02 observability branch validation (Python 3.12, `uv sync --locked`): `uv lock --check`, `ruff check .`, `ruff format --check .`, strict `mypy` and 230 `pytest` tests pass with 95% statement coverage. The GitHub Actions workflow runs the same offline gates plus a CLI smoke test. No live API calls were made. Docker remains a scaffold and is unverified (Member 04).

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q --cov=llm_routing_eval
```

## Documentation

- [Architecture](docs/architecture.md)
- [Decision log](docs/decisions.md)
- [Member 01 implementation and handoff](docs/member01-implementation.md)
- [Member 02 offline routing implementation](docs/member02-implementation.md)
- [Member 02 enterprise hardening and release gates](docs/member02-enterprise-hardening.md)
- [Member 02 Groq preflight and integration handoff](docs/member02-groq-preflight.md)
- [Core contract specification](docs/contracts.md)
- [Development roadmap](docs/development-roadmap.md)
- [Contributor guidance](docs/contributing.md)

## Ownership

| Owner     | Area                                                                 |
| --------- | -------------------------------------------------------------------- |
| Member 01 | `core/`, `application/`, `cli/`; `demo/` temporary integration fakes |
| Member 02 | `routing/`, `adapters/providers/`, `adapters/embeddings/`            |
| Member 03 | `evaluation/`, `domains/turkish_banking/`, labeled datasets          |
| Member 04 | `privacy/`, `reporting/`, `adapters/persistence/`, Docker/CI         |

Interfaces require review by all affected owners before integration.
