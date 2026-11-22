# Member 02 — Remaining Deliverables: NVIDIA Build, Semantic Embeddings, Calibration, Live Tests

**Date:** 2026-11-22  
**Owner:** Member 02 — Routing & Provider Integration  
**Decision:** The Member 02-owned items that were still open in the [final technical handoff](member02-final-technical-handoff.md) are now **implemented and offline-tested**. **Live use is still deny-by-default and blocked.** It needs the Member 03 and Member 04 approvals listed below.

This change closes the open Member 02 items that need no other team's input. It does not approve customer data transfer, live benchmarking or production readiness.

## 1. What was added

| Area | Artifact | Properties | Status |
| --- | --- | --- | --- |
| Shared OpenAI-compatible provider | [`adapters/providers/openai_compatible.py`](../src/llm_routing_eval/adapters/providers/openai_compatible.py) | One fail-closed implementation of the Groq gate sequence. Each provider has a fixed, reviewed `ProviderProfile`: ID, key env var, HTTPS endpoint, error-code prefix and token-budget field name. Groq behaviour and error codes are unchanged. | **Refactor; all earlier tests pass** |
| NVIDIA Build adapter (M5) | [`adapters/providers/nvidia_build.py`](../src/llm_routing_eval/adapters/providers/nvidia_build.py) | `NvidiaBuildChatProvider` and `managed_httpx_nvidia_provider`. Same gates as Groq: `live_enabled`, `LLM_ROUTING_ALLOW_LIVE=1`, a reviewed `OutboundAuthorization` (deny-all by default), an allowlisted model and `NVIDIA_API_KEY`. Sends `max_tokens`. Codes are `NVIDIA_*`. | **Offline/contract-tested; model availability UNVERIFIED** |
| Endpoint allowlist | [`transport.py`](../src/llm_routing_eval/adapters/providers/transport.py) | Transports accept only the two fixed chat endpoints. Any other URL raises `CHAT_ENDPOINT_NOT_ALLOWLISTED`. | **Tested** |
| Semantic embedding adapter | [`adapters/embeddings/remote.py`](../src/llm_routing_eval/adapters/embeddings/remote.py) | `RemoteSemanticEmbedding` implements Core `EmbeddingPort` on the NVIDIA Build `/v1/embeddings` endpoint. It has its own `EmbeddingOutboundAuthorization`, deny-all by default, and checks live flags and the key. Batch size, text length, response size and dimensions are bounded. Index order, finite values and non-zero vectors are validated. Errors are redacted safe codes. | **Offline-tested; no approved live model** |
| Development-only calibration | [`routing/calibration.py`](../src/llm_routing_eval/routing/calibration.py) | kNN: leave-one-out grid search over `k` × `minimum_similarity`. Cascade: per-scenario escalation from the large-label share or from too few examples. Non-development splits are rejected. Reports contain no example text. | **Implemented; needs Member 03 labeled data** |
| Calibration CLI | `llm-routing-eval calibrate-routing` in [`cli/app.py`](../src/llm_routing_eval/cli/app.py) | Writes a calibrated, strictly validated routing YAML and a text-free evidence JSON. Exits `3` without writing a policy when no setting meets the under-routing ceiling. | **Offline; CLI owner (Member 01) review requested** |
| Gated live tests | [`tests/live/test_live_providers.py`](../tests/live/test_live_providers.py) | `pytest -m live`. Skipped unless `LLM_ROUTING_LIVE_TESTS=1` and `LLM_ROUTING_ALLOW_LIVE=1` are set and the key exists. Only one hard-coded synthetic prompt (or text pair) is ever authorized. | **Written; never run in CI; not yet executed** |
| Import guards | [`tests/contract/test_member02_handoff_guards.py`](../tests/contract/test_member02_handoff_guards.py) | `adapters/embeddings` may import only Core and itself. | **Tested** |

## 2. Calibration method

Input format: [`examples/synthetic_development_calibration.json`](../examples/synthetic_development_calibration.json). Member 03 provides the real file. Each example's `tier` is the **smallest model tier judged sufficient** for it.

```json
{"schema_version": "1.0.0", "dataset_version": "...", "split": "development",
 "examples": [{"example_id": "...", "input_text": "...", "scenario_type": "...",
               "tier": "small|large", "split": "development"}]}
```

- **kNN.** Each example is classified by all *other* examples with the same `knn_vote` function the router uses. Ties and low-similarity neighbourhoods go to the large model. The selection rule:
  1. Only consider settings whose **under-routing rate** (large-labeled examples sent to small, i.e. quality risk) is within `--max-under-routing-rate` (default 0.05).
  2. Among those, pick the highest accuracy.
  3. Break ties by fewer under-routed examples, then a higher threshold (more conservative), then a smaller `k`.

  If no setting meets the ceiling, nothing is applied and the command exits `3`.
- **Cascade.** A scenario is escalated after a successful small-model answer in either of two cases:
  - the large-label share is above `--max-small-risk` (default 0.2);
  - the scenario has fewer than `--min-scenario-examples` development examples (default 3).

  This replaces the hand-picked `escalate_scenarios` list with evidence. It is still **not** confidence-based escalation, because the Core `RoutingFeedback` contract has no validated output-confidence signal.
- **Limits.** A calibration dataset may hold at most 2,000 examples. Each example's neighbour ranking is computed once and reused across the whole grid. The calibrated policy must fit the 128 KiB routing-config limit; otherwise the command exits `2` with `CALIBRATED_POLICY_TOO_LARGE`, writes the text-free report and writes no policy.
- **Output.** The policy YAML loads with `read_routing_settings`, so the existing SHA-256 policy fingerprint covers the calibrated values. The report contains the dataset fingerprint, the base-policy fingerprint, every grid point and per-scenario counts.

```bash
uv run llm-routing-eval calibrate-routing --examples examples/synthetic_development_calibration.json
```

The CLI uses the offline `HashingEmbedding`. To calibrate with semantic vectors, call `calibrate_knn(...)` with `RemoteSemanticEmbedding` after the privacy approval in §4.

## 3. Running the gated live tests (operators only)

Do this only after the account owner has confirmed model access. Each run uses one synthetic prompt and a few tokens.

```bash
LLM_ROUTING_LIVE_TESTS=1 LLM_ROUTING_ALLOW_LIVE=1 GROQ_API_KEY=... uv run pytest -m live tests/live -q
```

Optional environment variables:

- `GROQ_LIVE_TEST_MODEL` (default `openai/gpt-oss-20b`);
- `NVIDIA_API_KEY` together with `NVIDIA_LIVE_TEST_MODEL` and/or `NVIDIA_LIVE_EMBEDDING_MODEL`. There are no NVIDIA defaults on purpose, because availability is unverified.

Record the result (date, model ID, pass/fail) in the handoff sign-off. A pass proves wire compatibility and account access only. It says nothing about quality, latency or cost.

## 4. Still blocked (not Member 02-only)

| Item | Owner | Why Member 02 cannot close it alone |
| --- | --- | --- |
| Real development labels for calibration; held-out golden set | Member 03 | The calibration code exists, but the labeled banking data must come from the data owner |
| Reviewed `OutboundAuthorization` / `EmbeddingOutboundAuthorization` and `PrivacyPort` | Member 04 | Every live path stays deny-all until these exist |
| Executing the live pilot and confirming model IDs and pricing | Member 02 + Member 04 | Needs an approved key, a budget and privacy sign-off |
| Confidence-based Cascade escalation | Member 01 + Member 03 | Needs a Core contract change (a validated confidence/quality signal in `RoutingFeedback`) |

`config/providers.yaml` keeps `groq.enabled: false` and `nvidia_build.enabled: false`, and `candidate_models` for NVIDIA stays empty. The offline-release guard test enforces this.
