"""Development-only calibration of kNN and Cascade routing policy.

Inputs are labeled *development* examples supplied by Member 03 (``tier`` =
smallest model tier judged sufficient). Held-out golden/test data is rejected
at parse time and must never be used here.

Outputs contain only IDs, counts, rates and a dataset fingerprint — never
example text — so calibration reports are safe to persist and review.

* kNN: leave-one-out grid search over ``k`` and ``minimum_similarity``. The
  objective minimizes *under-routing* (a large-labeled example sent to the
  small model, i.e. quality risk) first, subject to a configured ceiling, then
  maximizes accuracy, then prefers conservative settings.
* Cascade: per ``scenario_type`` escalation if the large-label share exceeds a
  risk ceiling **or** the scenario has too few development examples.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llm_routing_eval.core import EmbeddingPort

from .configuration import (
    MAX_ROUTING_CONFIG_BYTES,
    CascadeSettings,
    DevelopmentExampleSettings,
    KnnSettings,
    RoutingSettings,
)
from .strategies import ModelTier, _check_vector, rank_neighbours, vote_ranked

_SAFE_ID = r"^[A-Za-z0-9_./-]{1,100}$"
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 7)
DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(0, 19))


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class CalibrationExample(_Frozen):
    example_id: str = Field(pattern=_SAFE_ID)
    input_text: str = Field(min_length=1, repr=False)
    scenario_type: str = Field(min_length=1, max_length=100)
    tier: ModelTier
    split: Literal["development"] = "development"


class CalibrationDataset(_Frozen):
    """Member 03 hand-off format. Any non-development split is rejected."""

    schema_version: Literal["1.0.0"]
    dataset_version: str = Field(min_length=1, max_length=100)
    split: Literal["development"]
    # Bounded so leave-one-out (O(n^2) similarities) stays practical; the emitted
    # policy must also fit the routing-config reader limit.
    examples: tuple[CalibrationExample, ...] = Field(min_length=2, max_length=2_000)

    @model_validator(mode="after")
    def unique_ids(self) -> CalibrationDataset:
        if len({ex.example_id for ex in self.examples}) != len(self.examples):
            raise ValueError("DUPLICATE_EXAMPLE_ID")
        return self

    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return "sha256:" + hashlib.sha256(("calibration-v1\n" + canonical).encode()).hexdigest()


def read_calibration_dataset(path: Path) -> CalibrationDataset:
    try:
        if not path.is_file() or path.stat().st_size > 16_777_216:
            raise ValueError("CALIBRATION_DATA_MISSING_OR_TOO_LARGE")
        return CalibrationDataset.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        raise ValueError("CALIBRATION_DATA_INVALID") from exc


@dataclass(frozen=True, slots=True)
class KnnGridPoint:
    k: int
    minimum_similarity: float
    correct: int
    under_routed: int  # large-labeled -> small (quality risk)
    over_routed: int  # small-labeled -> large (cost)
    small_selected: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    @property
    def under_routing_rate(self) -> float:
        return self.under_routed / self.total


@dataclass(frozen=True, slots=True)
class KnnCalibration:
    selected: KnnGridPoint | None
    feasible: bool
    max_under_routing_rate: float
    grid: tuple[KnnGridPoint, ...]


@dataclass(frozen=True, slots=True)
class ScenarioEvidence:
    scenario_type: str
    examples: int
    large_labels: int
    escalate: bool
    reason: Literal["LARGE_SHARE_ABOVE_CEILING", "INSUFFICIENT_EVIDENCE", "ACCEPT_SMALL"]


@dataclass(frozen=True, slots=True)
class CascadeCalibration:
    escalate_scenarios: tuple[str, ...]
    max_small_risk: float
    min_scenario_examples: int
    scenarios: tuple[ScenarioEvidence, ...]


def _require_development(examples: Sequence[CalibrationExample]) -> None:
    if any(ex.split != "development" for ex in examples):
        raise ValueError("HELD_OUT_DATA_FORBIDDEN_IN_CALIBRATION")


async def calibrate_knn(
    examples: Sequence[CalibrationExample],
    embedder: EmbeddingPort,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    max_under_routing_rate: float = 0.05,
) -> KnnCalibration:
    """Leave-one-out evaluation; each example is classified by all *other* examples."""
    _require_development(examples)
    if len(examples) < 2:
        raise ValueError("CALIBRATION_REQUIRES_TWO_EXAMPLES")
    if not math.isfinite(max_under_routing_rate) or not 0 <= max_under_routing_rate <= 1:
        raise ValueError("UNDER_ROUTING_CEILING_INVALID")
    valid_k = sorted({k for k in k_values if not isinstance(k, bool) and 1 <= k < len(examples)})
    valid_t = sorted(
        {float(t) for t in thresholds if math.isfinite(t) and -1 <= t <= 1}, reverse=True
    )
    if not valid_k or not valid_t:
        raise ValueError("CALIBRATION_GRID_EMPTY")
    vectors = await embedder.embed(tuple(ex.input_text for ex in examples))
    if len(vectors) != len(examples):
        raise ValueError("Embedding adapter returned incorrect count")
    for vector in vectors:
        _check_vector(vector, len(vectors[0]))
    references = tuple(
        (vector, ex.example_id, ex.tier) for vector, ex in zip(vectors, examples, strict=True)
    )
    # Neighbour rankings depend on neither k nor the threshold: compute each
    # example's leave-one-out ranking once (top max(k) only) and reuse it.
    max_k = valid_k[-1]
    rankings = tuple(
        rank_neighbours(vectors[index], references[:index] + references[index + 1 :])[:max_k]
        for index in range(len(examples))
    )
    grid: list[KnnGridPoint] = []
    for k in valid_k:
        for threshold in valid_t:
            correct = under = over = small = 0
            for ranked, example in zip(rankings, examples, strict=True):
                predicted, _ = vote_ranked(ranked, k=k, minimum_similarity=threshold)
                small += predicted == "small"
                if predicted == example.tier:
                    correct += 1
                elif predicted == "small":
                    under += 1
                else:
                    over += 1
            grid.append(
                KnnGridPoint(k, threshold, correct, under, over, small, total=len(examples))
            )
    feasible = [p for p in grid if p.under_routing_rate <= max_under_routing_rate]
    selected = (
        min(
            feasible,
            key=lambda p: (
                -p.correct,
                p.under_routed,
                -p.minimum_similarity,  # conservative: higher threshold
                p.k,
            ),
        )
        if feasible
        else None
    )
    return KnnCalibration(
        selected=selected,
        feasible=selected is not None,
        max_under_routing_rate=max_under_routing_rate,
        grid=tuple(grid),
    )


def calibrate_cascade(
    examples: Sequence[CalibrationExample],
    *,
    max_small_risk: float = 0.2,
    min_scenario_examples: int = 3,
) -> CascadeCalibration:
    """Escalate a scenario after a small answer if small is too often insufficient."""
    _require_development(examples)
    if not math.isfinite(max_small_risk) or not 0 <= max_small_risk <= 1:
        raise ValueError("SMALL_RISK_CEILING_INVALID")
    if isinstance(min_scenario_examples, bool) or not 1 <= min_scenario_examples <= 1000:
        raise ValueError("MIN_SCENARIO_EXAMPLES_INVALID")
    counts: dict[str, list[int]] = {}
    for ex in examples:
        entry = counts.setdefault(ex.scenario_type, [0, 0])
        entry[0] += 1
        entry[1] += ex.tier == "large"
    evidence: list[ScenarioEvidence] = []
    for scenario in sorted(counts):
        total, large = counts[scenario]
        if total < min_scenario_examples:
            evidence.append(ScenarioEvidence(scenario, total, large, True, "INSUFFICIENT_EVIDENCE"))
        elif large / total > max_small_risk:
            evidence.append(
                ScenarioEvidence(scenario, total, large, True, "LARGE_SHARE_ABOVE_CEILING")
            )
        else:
            evidence.append(ScenarioEvidence(scenario, total, large, False, "ACCEPT_SMALL"))
    return CascadeCalibration(
        escalate_scenarios=tuple(e.scenario_type for e in evidence if e.escalate),
        max_small_risk=max_small_risk,
        min_scenario_examples=min_scenario_examples,
        scenarios=tuple(evidence),
    )


def apply_calibration(
    settings: RoutingSettings,
    dataset: CalibrationDataset,
    knn: KnnCalibration,
    cascade: CascadeCalibration,
) -> RoutingSettings:
    """Return a new policy; an infeasible kNN calibration is never silently applied."""
    if knn.selected is None:
        raise ValueError("KNN_CALIBRATION_INFEASIBLE")
    knn_settings = KnnSettings(
        k=knn.selected.k,
        minimum_similarity=knn.selected.minimum_similarity,
        embedding_dimensions=settings.knn.embedding_dimensions,
        examples=tuple(
            DevelopmentExampleSettings(
                example_id=ex.example_id, input_text=ex.input_text, tier=ex.tier, split=ex.split
            )
            for ex in dataset.examples
        ),
    )
    cascade_settings = CascadeSettings(
        escalate_scenarios=cascade.escalate_scenarios,
        escalate_on_failure=settings.cascade.escalate_on_failure,
    )
    return settings.model_copy(update={"knn": knn_settings, "cascade": cascade_settings})


def calibration_report(
    dataset: CalibrationDataset, knn: KnnCalibration, cascade: CascadeCalibration
) -> dict[str, object]:
    """Text-free, reviewable calibration evidence."""

    def point(p: KnnGridPoint) -> dict[str, object]:
        return {
            "k": p.k,
            "minimum_similarity": p.minimum_similarity,
            "accuracy": round(p.accuracy, 6),
            "under_routing_rate": round(p.under_routing_rate, 6),
            "correct": p.correct,
            "under_routed": p.under_routed,
            "over_routed": p.over_routed,
            "small_selected": p.small_selected,
            "total": p.total,
        }

    return {
        "schema_version": "1.0.0",
        "method": "leave-one-out-development-only",
        "dataset_version": dataset.dataset_version,
        "dataset_fingerprint": dataset.fingerprint(),
        "example_count": len(dataset.examples),
        "knn": {
            "feasible": knn.feasible,
            "max_under_routing_rate": knn.max_under_routing_rate,
            "selected": point(knn.selected) if knn.selected is not None else None,
            "grid": [point(p) for p in knn.grid],
        },
        "cascade": {
            "max_small_risk": cascade.max_small_risk,
            "min_scenario_examples": cascade.min_scenario_examples,
            "escalate_scenarios": list(cascade.escalate_scenarios),
            "scenarios": [
                {
                    "scenario_type": s.scenario_type,
                    "examples": s.examples,
                    "large_labels": s.large_labels,
                    "escalate": s.escalate,
                    "reason": s.reason,
                }
                for s in cascade.scenarios
            ],
        },
        "limitations": [
            "Development-only labels; not a held-out quality result.",
            "Labels encode the smallest sufficient tier as judged by the data owner.",
        ],
    }


def dump_routing_settings(settings: RoutingSettings) -> str:
    """Serialize a policy, refusing output that ``read_routing_settings`` would reject."""
    text = yaml.safe_dump(
        settings.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=100
    )
    if len(text.encode("utf-8")) > MAX_ROUTING_CONFIG_BYTES:
        raise CalibratedPolicyTooLargeError("CALIBRATED_POLICY_TOO_LARGE")
    return text


class CalibratedPolicyTooLargeError(ValueError):
    """The calibrated policy exceeds the routing-config reader size limit."""
