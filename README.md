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

Latest Member 02 hardening PR verification (Python 3.12, `uv sync --locked`): `uv lock --check`, `ruff check .`, `ruff format --check .`, strict `mypy` and 103 `pytest` tests pass with 93% statement coverage. The GitHub Actions workflow runs the same offline gates plus a CLI smoke test. No live API calls were made. Docker remains a scaffold and is unverified (Member 04).

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
