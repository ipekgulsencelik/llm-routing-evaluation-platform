"""End-to-end CLI compare-routing smoke tests, synthetic data only."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_routing_eval.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
RUNNER = CliRunner()


def test_compare_routing_outputs_safe_atomic_json_for_four_strategies(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    completed = RUNNER.invoke(app, ["compare-routing", "--output", str(output)])
    assert completed.exit_code == 0, completed.output
    assert "OFFLINE_SIMULATION" in completed.output
    assert "always_large" in completed.output
    assert "rule_based" in completed.output
    assert "knn" in completed.output
    assert "cascade" in completed.output
    assert "COST: UNKNOWN" in completed.output
    content = output.read_text(encoding="utf-8")
    report = json.loads(content)
    assert report["execution_mode"] == "OFFLINE_SIMULATION"
    assert len(report["strategies"]) == 4
    assert [row["provider_attempts"] for row in report["strategies"]] == [2, 2, 2, 3]
    assert all(row["cost_amount"] is None for row in report["strategies"])
    assert all(row["token_origin"] == "SIMULATED" for row in report["strategies"])
    assert all(row["fixture_match_mean"] == 1.0 for row in report["strategies"])
    assert report["dataset_content_sha256"].startswith("sha256:")
    assert report["routing_policy_sha256"].startswith("sha256:")
    assert "Banka kartı" not in content
    assert "OFFLINE_FAKE_RESPONSE" not in content
    assert "Banka hesabı" not in content


def test_live_mode_fails_closed_and_does_not_overwrite_previous_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "untouched.json"
    output.write_text('{"earlier":true}', encoding="utf-8")
    monkeypatch.setenv("BENCHMARK_MODE", "live")
    refused = RUNNER.invoke(app, ["compare-routing", "--output", str(output)])
    assert refused.exit_code == 2
    assert "LIVE_MODE_NOT_IMPLEMENTED" in refused.output
    assert output.read_text(encoding="utf-8") == '{"earlier":true}'


def test_custom_synthetic_dataset_is_rejected_unless_allowlisted(tmp_path: Path) -> None:
    raw = json.loads((ROOT / "examples/offline_demo_cases.json").read_text(encoding="utf-8"))
    raw["cases"][0]["input_text"] = "Invented prompt not on the allowlist"
    supplied = tmp_path / "other.json"
    supplied.write_text(json.dumps(raw), encoding="utf-8")
    output = tmp_path / "no.json"
    denied = RUNNER.invoke(
        app,
        [
            "compare-routing",
            "--dataset",
            str(supplied),
            "--output",
            str(output),
        ],
    )
    assert denied.exit_code == 2
    assert "COMPARISON_REQUIRES_ALLOWLISTED_CASES" in denied.output
    assert not output.exists()


def test_untrusted_routing_configuration_cannot_inject_development_examples(
    tmp_path: Path,
) -> None:
    import yaml

    source = yaml.safe_load((ROOT / "config/routing.yaml").read_text(encoding="utf-8"))
    source["knn"]["examples"][0]["input_text"] = "Non-demo private text"
    config = tmp_path / "bad_router.yaml"
    config.write_text(yaml.safe_dump(source), encoding="utf-8")
    out = tmp_path / "no.json"
    refused = RUNNER.invoke(
        app, ["compare-routing", "--router-config", str(config), "--output", str(out)]
    )
    assert refused.exit_code == 2
    assert "ROUTING_DEMO_EXAMPLES_NOT_ALLOWLISTED" in refused.output
    assert not out.exists()
