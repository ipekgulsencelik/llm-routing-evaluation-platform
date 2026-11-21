# Member 02 — Offline Routing Implementation

**Scope:** Real offline routing decisions behind the existing Core `RouterPort`.
**Status (2026-11-21):** Four routing strategies, strict YAML/CLI, guarded Groq provider, optional managed HTTPX, bounded retries, safe observability and offline cross-member contract tests are merged. **Live benchmarking remains unauthorized and is not exposed through the CLI.** For verified CI evidence and release blockers, see [final technical audit and handoff](member02-final-technical-handoff.md).

## Architecture and boundaries

- `routing/strategies.py` implements four exchangeable strategies using the shared typed contracts; no direct provider, persistence, CLI, domain-bank, or evaluator imports.
- `routing/embeddings.py` implements a deterministic local SHA-256 lexical feature-hash embedding adapter behind `EmbeddingPort`, without external calls or heavyweight dependencies.
- `ModelPair`, `ScenarioRule`, and `DevelopmentExample` are frozen validated configuration objects. Scenario rules are first-match with configuration-order precedence and a conservative large-model default.
- `KnnRouter.prepare(...)` precomputes an index from explicitly labeled **development-only** examples. Prepared vectors are reused for query routing; stable neighbor ordering and large-model tie/low-similarity fallback are explicit.
- `CascadeRouter` calls the small model first. Successful small responses escalate only for explicitly configured scenario types; failed calls may escalate if policy enables it. No confidence-based decision is claimed, because the current `RoutingFeedback` contract lacks a validated confidence signal.
- The existing `BenchmarkOrchestrator` owns privacy checks, retries, provider invocations, error attribution, trace persistence, and evaluation. Routers only produce `RoutingDecision` objects.

## Run checks

```bash
uv sync --locked
uv run pytest -q tests/unit/test_routing_strategies.py tests/integration/test_real_offline_routers.py
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

The integration test executes all four routers on the **same synthetic, allowlisted offline prompt** with the demo fake provider. The fixture evaluator proves orchestration compatibility, **not financial correctness** or superiority of a strategy.

## Data governance and reproducibility

- No held-out/golden case may be a `DevelopmentExample`; do not load golden references into router processes.
- No routing decision uses evaluator scores or private golden labels.
- All explanations are short allowlisted machine-readable reason codes; user text is never included in them.
- Hashing embeddings are lexical, not semantic, and should **not** be used to make real financial-quality or language-understanding claims. Production-calibrated embeddings require an independently validated adapter.
- Track future benchmark calibration dataset version, index/embedding version, similarity threshold, rule ordering, cascade escalation scenarios, and model identifiers in run provenance before reporting comparative results.
- Do not activate live provider calls until Member 04 privacy controls and Member 03 evaluation datasets/rubrics are integrated and reviewed.

## YAML composition and offline CLI

The versioned and strict `config/routing.yaml` maps model tiers, priority-ordered rules, development-only examples, and Cascade escalation scenarios to the same RouterPort contract.

```bash
uv run llm-routing-eval validate-routing-config --router-config config/routing.yaml
uv run llm-routing-eval run-routing --strategy always_large
uv run llm-routing-eval run-routing --strategy rule_based
uv run llm-routing-eval run-routing --strategy knn
uv run llm-routing-eval run-routing --strategy cascade
```

These commands keep the existing synthetic allowlisted prompts, fake inference provider, and fixture-only evaluator. They do **not** contact Groq. The existing `run` command remains the original offline demo for backwards compatibility.

The model IDs are configured candidate IDs, not access verification. The test kNN embeddings are lexical hash vectors, not semantic model benchmarks. Golden data and held-out labels must not be used for calibration.

## Groq provider adapter

`GroqChatProvider` implements the existing `LLMProviderPort` without changing Core contracts. It talks to the documented Groq Chat Completions endpoint using a bounded standard-library HTTPS transport with redirects disabled. Responses are converted into safe status/error categories, latency, and observed token counts. Provider error payloads, secrets, and raw model answers are not written to traces by the adapter.

The adapter requires model allowlisting, explicit provider opt-in, environment opt-in, a configured API key, and independently reviewed per-request `OutboundAuthorization`. Authorization defaults to deny-all even when a key and environment flags exist. The offline CLI **does not** wire live calls. Member 04 privacy and Member 03 expert evaluation approval remain required.

Transport is injectable. The default provider still uses standard-library `urllib` (its OS worker cannot be forcibly cancelled), while an optional managed `httpx==0.28.1` async runtime supports bounded pooling and local TCP/TLS/cancellation tests. CI uses mocked or loopback-only traffic with no real provider keys. Actual Groq model/account availability, real rate-limit behavior, remote provider usage and billing remain **unverified** until an authorized synthetic-only pilot.

Groq docs: https://console.groq.com/docs/api-reference and https://console.groq.com/docs/models.

## Remaining Member 02 handoff gates (current status)

1. Obtain Member 01/03/04 review of routing/provider contracts, trace/retry accounting, safe telemetry and golden isolation.
2. Replace test-only evaluator/privacy/persistence adapters with actual approved Member 03/04 implementations and rerun the cross-member acceptance suite.
3. Only after privacy, evaluation, account/model access and pricing approval, run a tightly controlled synthetic-only Groq pilot to verify response/usage compatibility and operational behavior.
4. After the first MVP, implement NVIDIA Build as the separate M5 compatibility milestone.

See [the audited handoff checklist and CI evidence](member02-final-technical-handoff.md).
