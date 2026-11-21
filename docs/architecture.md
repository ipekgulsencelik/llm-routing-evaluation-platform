# Architecture

**Status (21 November 2026):** Member 01's Core/application/CLI offline vertical slice and Member 02's four routers, Groq provider, optional managed HTTPX, and safe observability are implemented. Member 03's expert banking evaluator/golden set and Member 04's production privacy, pricing and durable reporting are still missing. **Only offline cross-member contract tests using test doubles have passed; live Groq and full MVP release remain blocked.**

```text
Typer CLI (composition root)
  ├─ load/validate YAML and synthetic fixtures
  └─ constructor-inject Port implementations
     └─ BenchmarkOrchestrator (application)
         ├─ PrivacyPort.sanitize(input) -> fail closed
         ├─ RouterPort.decide(sanitized case, routing context)
         ├─ PrivacyPort.sanitize(provider prompt) -> fail closed
         ├─ LLMProviderPort.generate(request) [bounded attempts]
         ├─ EvaluatorPort.evaluate(safe case, response, golden ref)
         └─ ResultRepositoryPort.save_trace/evaluation/report

core/ (contracts and Protocols) <- application/ <- cli/ and outer adapters
```

**Architecture invariant:** Core and Application depend on contracts, not concrete provider SDKs or other teams' implementations. CLI composes implementations. Always-Large, Rule-Based, kNN and Cascade implement the same `RouterPort` without selecting providers or importing banking scoring logic.

**Member 02 boundaries:** Router modules may import Core and the `routing/` namespace; provider adapters may import Core and other modules in `adapters/providers/`. The strengthened `tests/contract/test_member02_handoff_guards.py` checks source-level absolute/relative imports, namespace aliases, and common dynamic import entry points. This is static regression coverage, **not a sandbox or a proof against indirect/dynamic execution**.

## Current Member 02 implementation (offline or loopback-tested)

- Routing uses strict versioned YAML and development-only lexical hashing for kNN. Its hash embedding is not a production-calibrated semantic model.
- Groq's `LLMProviderPort` adapter enforces model/resource limits and independent `OutboundAuthorization` default deny; the offline CLI does not expose live provider execution.
- Default transport is standard-library `urllib` wrapped with `asyncio.to_thread` and does **not** provide native socket cancellation. The optional managed `httpx==0.28.1` path has bounded connection pooling and mock/localhost TCP/TLS cancellation checks.
- The offline four-strategy comparison uses identical synthetic cases and simulated metrics; fixture scores and local timers are not financial accuracy or remote model latency measurements.
- Cross-member integration tests use the actual Member 01 orchestrator and Member 02 router/provider modules but **test doubles** for Member 03's evaluator and Member 04's privacy/repository ports. Real adapters and independent approvals are outstanding.

See the [Member 02 technical audit](member02-final-technical-handoff.md) for the implementation inventory and review requirements.

## M1 offline smoke demo

`llm-routing-eval run` uses `demo/` test doubles only: an allowlisted synthetic prompt privacy adapter, a deterministic Always-Large router, a fake model, a simple fixture-match evaluator and an in-memory repository. The CLI exports interim safe JSONL. These are not the Member 02–04 production modules.

`--live` is rejected and fake telemetry is explicitly labeled **SIMULATED**. The local end-to-end timer does not represent real model latency. Run input is never embedded in trace/report payloads. Router input metadata is removed. Golden references are sent only to the evaluator.

## Error model

- Invalid dataset/config: CLI exits nonzero and avoids displaying caller data.
- Privacy denial: `BLOCKED` trace, no provider and no remote judge.
- Router mismatch: normalized `ROUTING_FAILED`, without a provider request.
- Provider failure/timeout: one trace event per attempt, bounded same-model retry if enabled; no implicit fallback.
- Evaluation error: `PARTIAL` trace and `FAILED` evaluation, not a fabricated quality score.
- Persistence error: raises `BenchmarkPersistenceError`; no successful report is returned.

**Limitations:** The offline demo exporter does not provide transactional durable storage or crash recovery. Routing fingerprints do not contain secrets. HTTP 429 numeric `Retry-After` is supported in bounded retries for the provider/orchestrator, **but remote Groq behavior is not validated**. Pricing, expert financial evaluation, real privacy controls, distributed rate-limiting, Docker deployment and durable auditing remain unverified/incomplete. A passing offline CI run is not a live release approval.

**Verified evidence:** Audited `main` commit `9e25a2398c4a5fc6d5698703c1367e811bdb9ab5` passed [GitHub Actions run 44930373044](https://github.com/ipekgulsencelik/llm-routing-evaluation-platform/actions/runs/44930373044): **247 tests, 95% repository-wide statement coverage**, Ruff and strict mypy. The deliberately blocked `groq-preflight` (exit code `2`) is the expected safe result.

See [Member 01 implementation](member01-implementation.md) and [decisions.md](decisions.md).
