# Member 02 — Offline Routing Strategy Comparison

**Scope:** Compare Always-Large, Rule-Based, kNN and Cascade over the **same allowlisted synthetic cases**, offline. This is a local engineering validation utility, **not** a live-model quality/cost benchmark.

## Quick start

```powershell
uv sync --locked
uv run llm-routing-eval compare-routing --output artifacts/offline_comparison.json
```

The command writes one machine-readable JSON report and prints a compact console summary. The report is written atomically to avoid incomplete files. The old `run` and `run-routing` commands are unchanged.

## Measurement semantics

| Field | Interpretation |
| --- | --- |
| `provider_attempts` | Number of model call **attempts** (including retries, if enabled) |
| `attempts_per_case` | Attempts / case count; denominator is the identical fixed workload |
| `calls_by_model` | Count of attempts by configured model ID, not a model usage billing statement |
| `successful_cases` | Execution-level success; not financial correctness |
| `fixture_match_mean` | **Fake-provider fixture score** (not expert evaluation); `null` if unavailable |
| `input_tokens_total`, `output_tokens_total` | Sum only if **every attempt** has the corresponding count; else `null` |
| `input_tokens_per_attempt`, `output_tokens_per_attempt` | Mean across all attempted calls **only with complete usage coverage**; else `null` |
| `token_usage_samples` | Number of attempts with a usage record (coverage of input/output fields may differ) |
| `token_origin` | `SIMULATED` vs `OBSERVED` vs `MIXED` vs `UNKNOWN`; offline demo is simulated |
| `local_case_latency_mean_ms` | Local end-to-end application timing (NOT provider API latency) |
| `local_attempt_latency_mean_ms` | Local await time around the fake provider, NOT real model inference |
| `cost_amount` | `null` until independently verified prices and observed billable usage are integrated |

Quality numbers, local latency and simulated tokens are **not comparable to live inference or actual financial performance**. Two cases are insufficient for statistical conclusions; no strategy ranking or production claim is produced.

## Comparability and lineage

- All four strategies receive the same immutable case tuple, input order, locale and version and the same execution options, except `strategy`.
- Each strategy gets an independent orchestrator and isolated fake provider/evaluator/privacy/repository composition; no state leaks between runs.
- The comparison report stores a SHA-256 fingerprint of the full case content (not the prompts themselves) and a fingerprint of the versioned routing rules/model pair/kNN development references.
- One single report has four isolated run IDs and the fixed strategy order; mismatched per-run cases and evaluations are rejected.
- Only synthetic metadata is accepted by the runner; CLI additionally requires the existing two exact approved demo prompts and development examples.
- Output records contain aggregated safe DTO data only. The CLI does not use real Groq or any outbound networking and explicitly refuses `BENCHMARK_MODE=live`.

## Validation

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q --cov=llm_routing_eval
uv run llm-routing-eval compare-routing
```

Tests assert exact attempt counts and model mix, token averages, coverage on missing usage, shared hashes, output redaction, rejection of unallowlisted prompts/live mode, and atomic success-only export.

## Follow-on release criteria

- Member 03: expert-verified golden set with a held-out evaluation path and independent quality gates.
- Member 04: privacy and retention approval for real prompts, verified pricing and cost attribution, durable storage, fault and cancellation review.
- Member 01: approve comparison application service and any shared run-schema changes.
- Member 02: controlled Groq pilot using separately approved outbound authorization; repeat samples, latency distribution, model capability checks, real costs; no mixing of simulated and measured observations.
