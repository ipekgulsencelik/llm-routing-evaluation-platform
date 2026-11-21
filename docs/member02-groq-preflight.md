# Member 02 — Groq Integration Preflight and Contract Handoff

**Scope:** Network-free readiness diagnostics and provider-to-orchestrator contract tests. This is an **integration preparation milestone**, not a live benchmark or security certification.

## Offline preflight command

```powershell
uv sync --locked
uv run llm-routing-eval groq-preflight
```

Expected: `GROQ_PREFLIGHT_STATUS: BLOCKED` and a nonzero exit code (`2`). **A blocked status is correct at this project stage.** The command reads `config/providers.yaml` and `config/routing.yaml`, checks candidate model alignment, and reports only safe machine-readable blocker codes. It makes **zero HTTP requests**, does not output credential values and does not enable live CLI mode.

Even if `GROQ_API_KEY` and `LLM_ROUTING_ALLOW_LIVE=1` are set and a local provider flag is switched on, local toggles do not prove third-party model access, expert evaluation, safety approval, or verified pricing. The preflight intentionally remains BLOCKED until cross-member releases implement real evidence-based approvals, not booleans set by an end user.

## Tested contract: four strategies, one provider port

`tests/integration/test_groq_orchestrator_stub.py` composes all four real `RouterPort` strategies with an injected `GroqChatProvider` and a deterministic in-memory `GroqTransport` test double.

- The injected authorizer permits **only two exact synthetic allowlisted prompts**; the production provider remains **deny-by-default**.
- Fake HTTP 200 responses include a model identifier, completion status and usage fields and exercise the existing `ModelResponse` contract. Fake HTTP 429 exercises tracing and orchestrator-owned retries.
- Each case passes the existing pre-routing and pre-provider privacy guard; no golden references enter routing or transport.
- Tests assert model selection, final status, attempt counts, retry bounds, evaluation correlation and absence of prompts, credentials and model content in persisted traces and reports.
- The tests set placeholder environment values via pytest monkeypatch, without any network credentials or external calls.

**Measurement caveat:** The mocked HTTP transport returns wire-format results that the provider adapter marks `MetricOrigin.OBSERVED`, because the adapter sees a completed transport response. In these tests this is **a transport contract simulation**, not evidence of real measured latency or billable Groq tokens. These mocked observations are never sent to the CLI comparison or presented as live benchmark data.

## Cross-member acceptance contract

| Owner | Required evidence before enabling real traffic |
| --- | --- |
| Member 01 — Application/Core | Confirm stable `LLMProviderPort` semantics, attempt ID correlation, terminal statuses, retry budget and safe trace persistence |
| Member 03 — Evaluation | Produce a versioned, expert-reviewed held-out golden set; rubric and critical-error gate; ensure routing/kNN development set never includes held-out answers |
| Member 04 — Privacy/Reporting | Implement and approve per-request outbound authorization, sanitized input/response policy, consent/retention/audit rules, secret handling, observed price schedule and failure-safe durable persistence |
| Member 02 — Routing/Provider | Verify actual Groq account model availability, response schema, cancellation behavior, rate-limit/retry policy, versioned model pool and provider tokens through controlled synthetic-only tests |

Until the evidence is reviewed, the CLI stays offline; the only production `OutboundAuthorization` default rejects all requests. Do **not** replace it with the permissive test authorizer. A real model-access check, if later approved, must be an explicit, audited opt-in with a capped synthetic workload; no customer data in a pilot.

## Validation

```powershell
uv run pytest -q tests/integration/test_groq_orchestrator_stub.py tests/unit/test_groq_preflight.py
uv run llm-routing-eval groq-preflight
uv run llm-routing-eval compare-routing
```

The preflight is expected to exit `2` until the real release gates are satisfied. The comparison still reports *offline* simulated data only.
