# Member 02 — Cross-member integration contract acceptance (offline)

**Scope:** An automated, **network-free** interface acceptance harness for Member 01's orchestrator, Member 02's four real routers + Groq/HTTPX provider, and **test-only** adapters that implement Member 03's evaluator and Member 04's privacy/persistence port contracts.

## What is implemented today

The project currently contains `src/llm_routing_eval/evaluation/__init__.py`, `src/llm_routing_eval/privacy/__init__.py` and `src/llm_routing_eval/adapters/persistence/__init__.py` as **module scaffolds**, not production-ready evaluation, privacy or durable storage adapters. The explicit `*Double` classes in `tests/integration/test_member02_cross_member_contracts.py` are **not those teams' implementations** and must never be imported into a live composition root. The test repository is ephemeral in-memory storage.

The test suite uses `managed_httpx_groq_provider()` with `httpx.MockTransport`, an exact-match synthetic-only outbound approval stub, and environment variables scoped to the test. It opens **no network connections**, calls no Groq model, uses no real credentials, and makes no claim about financial correctness. The provider's default-deny approval policy remains unchanged outside these tests.

## Automated contract matrix

| Interface / invariant | Test evidence | Remaining team deliverable |
| --- | --- | --- |
| `RouterPort` / Member 02 | All four routers built from versioned routing config work through the real Member 01 orchestrator and actual Groq provider adapter | Member 01 review of composition/attempt accounting |
| Pre-routing `PrivacyPort` | Synthetic raw input is transformed to a safe exact-match fixture **before** the router; raw metadata is stripped | Member 04 production-approved sanitization policy and PII/consent rules |
| Pre-provider `PrivacyPort` | A second sanitizer gate is invoked before each model attempt; denying either gate yields **zero** HTTP and evaluator calls | Member 04 outbound authorization, policy versioning and audit evidence |
| Golden isolation / `EvaluatorPort` | A synthetic held-out reference is supplied **only** to evaluator stub after success; its private canary is not in router case, HTTP payload, trace or report | Member 03 versioned *expert-reviewed* golden set, blind evaluation rubric and leakage audit |
| `OutboundAuthorization` | Without a separately injected authorizer, Groq remains default-denied even if the test's environment opt-ins and API key placeholders exist | Member 04 reviewed request-scoped authorization; never reuse test-only allowlist |
| `ResultRepositoryPort` | A trace, evaluation and report are saved through the shared interface; raw prompts/metadata, answer content, golden canaries and notes do not persist | Member 04 durable database, retention, encryption, safe audit and transactional integrity |
| Provider telemetry | Observed HTTPX mock attempts correlate with trace attempt count; pretransport denials do not inflate provider error rate | Member 01 run-level aggregation review and Member 04 operational metrics/export policy |
| Evaluator failure | Failure of synthetic evaluator leaves no quality score, yields a partially completed trace and redacts exception text | Member 03 verified scoring/error taxonomy and retry policy, Member 01 review of case semantics |

These tests assert **only contract behavior using deliberately synthetic fixtures**. An evaluator stub may emit `quality_scores={"contract_fixture_only": 0.75}`; that number is **not** an expert-reviewed score and must never be compared to actual banking quality outcomes.

## Acceptance and release handoff

Before calling the cross-member work integrated, all four owners must confirm:

1. **Member 01:** The actual application factory wires approved adapters, keeps budgets and routing metadata isolated, and produces reproducible trace/report provenance.
2. **Member 03:** The concrete `EvaluatorPort` implementation uses the reviewed held-out golden set, verification statuses and financial critical-error gates; no held-out references appear in routing or providers.
3. **Member 04:** The concrete `PrivacyPort`, `OutboundAuthorization` and `ResultRepositoryPort` implementations have independent security review, fail-closed tests and auditable storage/retention governance; pricing and judge costs are correctly attributed.
4. **Member 02:** Groq candidate model access and response schemas are checked with separately approved **synthetic-only** pilot requests, without weakening deny-by-default or exposing customer content.

Actual approved adapter code must **replace the test doubles** in a follow-up integration PR, and the same privacy, golden isolation, trace and failure invariants must be re-run against those concrete adapters. CI being green here **does not** satisfy expert, privacy, cost or provider-availability approvals. `groq-preflight` is expected to remain `BLOCKED` (exit code 2).

## Run

```powershell
uv sync --locked
uv run pytest -q tests/integration/test_member02_cross_member_contracts.py
uv run pytest -q --cov=llm_routing_eval
uv run llm-routing-eval groq-preflight
```

The integration test fixture deliberately uses environment keys such as `TEST_ONLY_CREDENTIAL_NEVER_LOG` as placeholders; they are not real secrets. The tests verify no such tokens are written to reports, traces or the safe provider metrics snapshots.
