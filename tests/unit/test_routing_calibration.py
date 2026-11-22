"""Development-only kNN/Cascade calibration: correctness, safety and round-trip."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from llm_routing_eval.cli.app import app
from llm_routing_eval.core import BenchmarkCase, RoutingContext, StrategyId
from llm_routing_eval.routing import (
    CalibrationDataset,
    CalibrationExample,
    HashingEmbedding,
    apply_calibration,
    build_router,
    calibrate_cascade,
    calibrate_knn,
    calibration_report,
    read_calibration_dataset,
    read_routing_settings,
)
from llm_routing_eval.routing.calibration import dump_routing_settings

ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = ROOT / "examples" / "synthetic_development_calibration.json"
BASE = ROOT / "config" / "routing.yaml"


class TableEmbedding:
    """Hand-built vectors so leave-one-out outcomes are exactly predictable."""

    def __init__(self, table: dict[str, tuple[float, ...]]) -> None:
        self._table = table

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._table[text] for text in texts)


def ex(i: int, text: str, tier: str, scenario: str = "s") -> CalibrationExample:
    return CalibrationExample.model_validate(
        {"example_id": f"e{i}", "input_text": text, "scenario_type": scenario, "tier": tier}
    )


def test_leave_one_out_selects_zero_under_routing_and_best_accuracy() -> None:
    examples = (
        ex(1, "a1", "small"),
        ex(2, "a2", "small"),
        ex(3, "b1", "large"),
        ex(4, "b2", "large"),
        ex(5, "c", "large"),  # isolated large example: only safe via low-similarity fallback
    )
    table = {
        "a1": (1.0, 0.0, 0.0),
        "a2": (0.95, 0.05, 0.0),
        "b1": (0.0, 1.0, 0.0),
        "b2": (0.05, 0.95, 0.0),
        "c": (0.6, 0.0, 0.8),
    }
    result = asyncio.run(
        calibrate_knn(
            examples,
            TableEmbedding(table),
            k_values=(1,),
            thresholds=(0.0, 0.9),
            max_under_routing_rate=0.0,
        )
    )
    by_threshold = {p.minimum_similarity: p for p in result.grid}
    # threshold 0.0: "c" nearest neighbour is small -> under-routed.
    assert by_threshold[0.0].under_routed == 1
    # threshold 0.9: "c" falls back to large; others still match their cluster.
    assert by_threshold[0.9].under_routed == 0 and by_threshold[0.9].correct == 5
    assert result.feasible and result.selected == by_threshold[0.9]


def test_infeasible_ceiling_is_reported_and_never_applied() -> None:
    examples = (ex(1, "a", "small"), ex(2, "b", "large"))
    table = {"a": (1.0, 0.0), "b": (1.0, 0.01)}
    result = asyncio.run(
        calibrate_knn(
            examples,
            TableEmbedding(table),
            k_values=(1,),
            thresholds=(0.0,),
            max_under_routing_rate=0.0,
        )
    )
    assert not result.feasible and result.selected is None
    dataset = CalibrationDataset(
        schema_version="1.0.0", dataset_version="t", split="development", examples=examples
    )
    with pytest.raises(ValueError, match="KNN_CALIBRATION_INFEASIBLE"):
        apply_calibration(read_routing_settings(BASE), dataset, result, calibrate_cascade(examples))


def test_k_values_exceeding_leave_one_out_pool_are_dropped() -> None:
    examples = (ex(1, "a", "small"), ex(2, "b", "large"))
    table = {"a": (1.0, 0.0), "b": (0.0, 1.0)}
    with pytest.raises(ValueError, match="CALIBRATION_GRID_EMPTY"):
        asyncio.run(calibrate_knn(examples, TableEmbedding(table), k_values=(2, 3)))
    with pytest.raises(ValueError, match="UNDER_ROUTING_CEILING_INVALID"):
        asyncio.run(
            calibrate_knn(examples, TableEmbedding(table), max_under_routing_rate=float("nan"))
        )


def test_cascade_escalates_risky_and_under_evidenced_scenarios() -> None:
    examples = (
        *(ex(i, f"t{i}", "small", "easy") for i in range(5)),
        ex(10, "h1", "small", "mixed"),
        ex(11, "h2", "small", "mixed"),
        ex(12, "h3", "large", "mixed"),
        ex(20, "r1", "small", "rare"),
    )
    result = calibrate_cascade(examples, max_small_risk=0.2, min_scenario_examples=3)
    reasons = {s.scenario_type: s.reason for s in result.scenarios}
    assert reasons == {
        "easy": "ACCEPT_SMALL",
        "mixed": "LARGE_SHARE_ABOVE_CEILING",
        "rare": "INSUFFICIENT_EVIDENCE",
    }
    assert result.escalate_scenarios == ("mixed", "rare")
    with pytest.raises(ValueError):
        calibrate_cascade(examples, max_small_risk=1.5)
    with pytest.raises(ValueError):
        calibrate_cascade(examples, min_scenario_examples=0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(split="test"),
        lambda d: d["examples"][0].update(split="held_out"),
        lambda d: d["examples"][0].update(tier="medium"),
        lambda d: d["examples"][1].update(example_id=d["examples"][0]["example_id"]),
        lambda d: d["examples"][0].update(expected_facts=["golden"]),
        lambda d: d.update(examples=d["examples"][:1]),
    ],
)
def test_held_out_or_malformed_data_is_rejected(tmp_path: Path, mutate: object) -> None:
    data = json.loads(SYNTHETIC.read_text(encoding="utf-8"))
    mutate(data)  # type: ignore[operator]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="CALIBRATION_DATA_INVALID"):
        read_calibration_dataset(path)


def test_validation_error_hides_example_text() -> None:
    with pytest.raises(ValidationError) as info:
        CalibrationExample.model_validate(
            {"example_id": "x", "input_text": "PRIVATE CANARY", "scenario_type": "s", "tier": "x"}
        )
    assert "PRIVATE CANARY" not in str(info.value)


def test_synthetic_calibration_round_trip_and_text_free_report() -> None:
    dataset = read_calibration_dataset(SYNTHETIC)
    base = read_routing_settings(BASE)
    knn = asyncio.run(calibrate_knn(dataset.examples, HashingEmbedding()))
    cascade = calibrate_cascade(dataset.examples)
    assert knn.selected is not None and knn.selected.under_routing_rate <= 0.05
    calibrated = apply_calibration(base, dataset, knn, cascade)
    assert calibrated.fingerprint() != base.fingerprint()
    assert calibrated.knn.k == knn.selected.k
    assert set(calibrated.cascade.escalate_scenarios) == set(cascade.escalate_scenarios)

    report = json.dumps(calibration_report(dataset, knn, cascade), ensure_ascii=False)
    assert all(example.input_text not in report for example in dataset.examples)
    assert dataset.fingerprint() in report

    # The emitted YAML is a valid, strict routing policy usable by the factory.
    reparsed = type(base).model_validate(
        __import__("yaml").safe_load(dump_routing_settings(calibrated))
    )
    assert reparsed.fingerprint() == calibrated.fingerprint()
    router = asyncio.run(build_router(StrategyId.CASCADE, reparsed))
    case = BenchmarkCase(
        case_id="c", input_text="x", domain="banking", scenario_type="loan_complex"
    )
    first = asyncio.run(
        router.decide(case, RoutingContext(current_stage=0, remaining_model_calls=2))
    )
    assert first.reason_code == "CASCADE_SMALL_FIRST"


def test_cli_writes_policy_and_report(tmp_path: Path) -> None:
    config = tmp_path / "routing.yaml"
    report = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "calibrate-routing",
            "--examples",
            str(SYNTHETIC),
            "--router-config",
            str(BASE),
            "--output-config",
            str(config),
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DEVELOPMENT_ONLY_CALIBRATION" in result.output
    assert read_routing_settings(config).knn.examples
    evidence = json.loads(report.read_text(encoding="utf-8"))
    assert evidence["knn"]["feasible"] is True
    assert evidence["base_routing_policy_sha256"].startswith("sha256:")


def test_cli_infeasible_exits_3_with_report_but_no_policy(tmp_path: Path) -> None:
    conflicting = {
        "schema_version": "1.0.0",
        "dataset_version": "synthetic-conflict",
        "split": "development",
        "examples": [
            {"example_id": "a", "input_text": "aynı soru", "scenario_type": "s", "tier": "small"},
            {"example_id": "b", "input_text": "aynı soru", "scenario_type": "s", "tier": "large"},
        ],
    }
    examples = tmp_path / "conflict.json"
    examples.write_text(json.dumps(conflicting, ensure_ascii=False), encoding="utf-8")
    config = tmp_path / "routing.yaml"
    report = tmp_path / "report.json"
    args = ["calibrate-routing", "--router-config", str(BASE)]
    args += ["--output-config", str(config), "--report", str(report)]
    result = CliRunner().invoke(
        app, [*args, "--examples", str(examples), "--max-under-routing-rate", "0"]
    )
    assert result.exit_code == 3
    assert json.loads(report.read_text(encoding="utf-8"))["knn"]["feasible"] is False
    assert not config.exists()


def test_cli_invalid_input_exits_2(tmp_path: Path) -> None:
    config = tmp_path / "routing.yaml"
    args = ["calibrate-routing", "--output-config", str(config)]
    args += ["--report", str(tmp_path / "report.json")]
    missing = CliRunner().invoke(app, [*args, "--examples", str(tmp_path / "missing.json")])
    assert missing.exit_code == 2 and "CALIBRATION_INPUT_INVALID" in missing.output
    bad_ceiling = CliRunner().invoke(
        app, [*args, "--examples", str(SYNTHETIC), "--max-under-routing-rate", "-0.5"]
    )
    assert bad_ceiling.exit_code == 2
    assert not config.exists()
