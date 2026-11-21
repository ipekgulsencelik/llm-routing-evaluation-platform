# LLM Routing Evaluation Platform — Development Roadmap

**Status (21 November 2026):** Core/offline CLI and Member 02's routing/provider work are implemented; Member 03/04 production adapters, approvals and live benchmark are outstanding. This is a milestone plan with a verified status snapshot, **not proof of formal four-member sign-off**.\
**Delivery model:** Four contributors, modular monolith, CLI-first, offline-first\
**Initial domain:** Turkish Retail Banking & Customer Support\
**Scope:** MVP with Groq as the primary live provider; NVIDIA Build compatibility follows as a separate milestone

## 1. Delivery Principles

- Agree on **shared interface contracts before parallel feature work**. Member 01 coordinates; all four members approve their respective contracts.
- Preserve **SOLID, OOP, and pragmatic Hexagonal Architecture (Ports & Adapters)**. Domain-specific banking logic stays in a domain pack.
- Establish one **offline end-to-end path early**; live API calls are always opt-in.
- Use development/calibration data for router tuning and a **held-out golden evaluation set** for final comparison. Never leak held-out labels or references into router decisions.
- Report **quality, safety, cost, latency, coverage, and decision explanations separately**. No opaque combined score.
- Require objective milestone exit criteria; **no calendar commitments** until work is estimated by the team.

## 2. Current implementation and release status

Successful offline CI is not the same as acceptance of the complete milestone. As of audited `main` commit `9e25a2398c4a5fc6d5698703c1367e811bdb9ab5`: [GitHub Actions run 44930373044](https://github.com/ipekgulsencelik/llm-routing-evaluation-platform/actions/runs/44930373044) passed **247 tests**, **95% repository-wide statement coverage**, Ruff/strict mypy and blocked Groq preflight.

| Milestone | Implementation evidence | Acceptance status |
| --- | --- | --- |
| **M0 — Contracts & Governance** | Core ports, Pydantic schemas, contract and import-direction tests exist | **Partial**: explicit cross-owner contract and security reviews are not recorded |
| **M1 — Offline Vertical Slice** | Member 01 CLI/orchestrator, synthetic fake-provider run and CI smoke implemented | **Offline path verified**; Member 04 Docker build/run still needs acceptance |
| **M2 — Offline Feature Completion** | Member 02's four strategies, Groq/HTTPX adapters, safe telemetry and mock/localhost tests implemented | **Partial**: Member 03 expert evaluation/golden set and Member 04 real privacy/storage/reporting adapters are not present |
| **M3 — Controlled Groq Benchmark** | Provider integration code and fail-closed preflight implemented | **Blocked**: reviewed outbound approval, real model access, banking evaluation and pricing absent |
| **M4 — MVP Release Gate** | Offline tests, technical handoff and documentation exist | **Pending**: team reviews, actual cross-member integration, Docker/security and controlled results |
| **M5 — NVIDIA Build Compatibility** | Disabled configuration placeholder exists | **Not started / post-MVP**; not an initial MVP blocker |

Member 02's [technical handoff](member02-final-technical-handoff.md) contains current code inventory and unresolved release gates. Its cross-member acceptance tests use **test doubles** for Member 03/04, not real approved modules. The measured `OBSERVED` values from mocked HTTP are not actual provider/billing evidence.

## 3. Milestones and Ownership

| Milestone | Primary work | Owner(s) | Exit criteria |
| --- | --- | --- | --- |
| **M0 — Contracts & Governance** | Module boundaries; Pydantic contracts for cases, routing decisions, provider attempts, evaluations, run metadata, traces; YAML config and privacy semantics; dataset split and quality-gate rules | **01 leads; 02–04 review** | Interfaces and example payloads approved by all four; dependency direction and data-leakage boundaries documented |
| **M1 — Offline Vertical Slice** | CLI + orchestrator using fakes; Always-Large baseline; small synthetic development fixtures and deterministic evaluator; uv/Docker/CI bootstrap; privacy-safe JSONL output | **01 orchestration, 02 router, 03 fixtures/evaluator, 04 environment/storage** | One end-to-end offline run succeeds without keys or network; tests, Ruff and mypy pass |
| **M2 — Offline Feature Completion** | Rule-Based, kNN, Cascade; provider call/route traces; financial rubrics and hybrid evaluation interfaces; versioned held-out dataset and expert-review workflow; privacy guards, SQLite, audit and reports | **02 routing, 03 evaluation, 04 privacy/reporting, 01 integration** | All four strategies run on the same offline workload; every attempted call/failure is recorded; held-out split protection and report round-trip tests pass |
| **M3 — Controlled Groq Benchmark** | Verify model access; Groq provider adapter; explicit live-run flag and cost warning; judge calls where approved; quality-first comparison and latency/cost attribution | **02 provider, 03 quality validation, 04 costs/traces, 01 orchestration** | Successful controlled live experiment **if API/model access permits**; complete provenance, error accounting and separate judge costs; no unsupported quality claims |
| **M4 — MVP Release Gate** | Reproducible example, CLI docs, architecture/limitations, privacy review, integration/regression tests, release-ready GitHub documentation | **All; 01 coordinates** | Offline demonstration works from clean environment; quality and safety thresholds are explicit; expert verification status is disclosed; CI green; secrets/PII checks pass |
| **M5 — NVIDIA Build Compatibility** | Provider adapter and compatibility tests using existing contracts; separate provider-level analysis | **02 leads; 01/04 integrate; 03 checks fairness** | Offline adapter tests pass; optional controlled live validation when access is available; provider differences documented |

**Release boundary:** M0–M4 define MVP delivery. M5 is the approved secondary provider phase, **not a blocker** for the first MVP release.

## 4. Integration Dependencies

1. **M0 unlocks parallel work:** Member 01 publishes versioned interfaces; Members 02–04 confirm payload semantics, privacy boundary, and failure handling.
2. **Member 04 enables safe execution:** Docker/uv setup and privacy guard must be integrated before any external model or judge request.
3. **Member 03 protects evaluation integrity:** Development examples may support kNN, Rule-Based and Cascade calibration; held-out gold references must remain isolated and frozen before final comparisons.
4. **Member 02 produces auditable telemetry:** Every provider attempt, escalation, retry, model choice, usage observation, and non-sensitive reason code flows through the common trace contract.
5. **Member 01 coordinates E2E validation:** Orchestrator must compose interchangeable adapters without importing banking-specific rules into Core.

## 5. Shared Quality Gates

- **Functional:** Four strategies run through one router contract; offline baseline and complete comparison report work.
- **Evaluation:** Financial accuracy, completeness, safety, and unsupported-claim measures are versioned; deterministic checks, rubric-based LLM judge, and human review have defined responsibilities. Unverified cases are not presented as expert-verified.
- **Benchmark fairness:** Same held-out cases, common model pool and prompt controls; documented rule/kNN/cascade tuning and measurement boundaries.
- **Measurement:** Report end-to-end latency, router overhead, P50/P95 (with sample counts), token usage, estimated vs. observed vs. unknown charges, and judge costs separately. Failed calls remain in coverage denominators.
- **Security:** `GROQ_API_KEY` and `NVIDIA_API_KEY` come from environment variables; masking happens before external calls and persistence. Masking is not claimed to guarantee anonymization.
- **Reproducibility:** Record dataset/split, model/provider, configuration, prompts, thresholds, pricing snapshot, rubrics, evaluator and run ID; distinguish repeatable setups from non-deterministic live outputs.
- **Engineering:** Python 3.12+, uv, TOML/YAML, Docker, pytest, Ruff, mypy, and offline GitHub Actions checks.

## 6. Collaboration Rules

- **Branch / PR ownership:** Each member owns PRs for their module; shared contract changes require review from every affected owner.
- **Merge order:** M0 contract PRs → M1 offline vertical slice → independent module PRs → integrated benchmark → optional live runs.
- **Definition of Done (per PR):** Working implementation or document, appropriate tests, typed interface compliance, no secret/PII leaks, and updated usage notes.
- **Escalate blockers early:** Expert reviewer availability, provider/model access, unclear quality thresholds, and cost/pricing provenance are release risks—not assumptions to hide.

## 7. Immediate Next Action — Cross-member acceptance

**Do not restart M0 implementation:** Core ports, the offline vertical slice and Member 02 modules already exist. Record the outstanding reviews and continue integration:

1. **Member 01:** review current router/provider contract, trace/attempt accounting and the expanded dependency-direction checks; own the actual approved adapter composition.
2. **Member 03:** deliver an expert-reviewed versioned held-out golden set, banking safety rubric, real `EvaluatorPort` implementation and development/held-out isolation evidence.
3. **Member 04:** deliver approved `PrivacyPort`, per-request outbound authorization, durable `ResultRepositoryPort`, pricing/usage provenance, secrets/PII governance and Docker verification.
4. **Member 02:** validate calibration with development-only cases; after privacy, quality, cost and account approvals, verify candidate Groq models on **synthetic-only**, tightly capped calls.
5. **All members:** replace test doubles in cross-member acceptance with the real reviewed implementations, rerun full offline CI, document approvals and only then consider controlled M3/M4 release gates.

Until this evidence is recorded, keep both providers disabled, pricing unverified and `groq-preflight` returning **BLOCKED (exit code 2)**.

## 8. Related Documents

- [`project-overview.md`](project-overview.md)
- [`member-01-architecture-core.md`](member-01-architecture-core.md)
- [`member-02-routing-integration.md`](member-02-routing-integration.md)
- [`member-03-dataset-evaluation.md`](member-03-dataset-evaluation.md)
- [`member-04-infrastructure-reporting.md`](member-04-infrastructure-reporting.md)
