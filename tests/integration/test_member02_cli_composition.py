"""All four configured Member 02 routers are CLI-accessible without network."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_routing_eval.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
RUNNER = CliRunner()


@pytest.mark.parametrize("strategy", ["always_large", "rule_based", "knn", "cascade"])
def test_configured_offline_routing_command(strategy: str, tmp_path: Path) -> None:
    output = tmp_path / (strategy + ".jsonl")
    result = RUNNER.invoke(
        app,
        [
            "run-routing",
            "--strategy",
            strategy,
            "--config",
            str(ROOT / "config/settings.yaml"),
            "--router-config",
            str(ROOT / "config/routing.yaml"),
            "--dataset",
            str(ROOT / "examples/offline_demo_cases.json"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert f"ROUTER: {strategy}" in result.output
    assert "SUCCESS: 2" in result.output
    assert output.exists()
    rows = [json.loads(row) for row in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert rows[-1]["payload"]["strategy_metrics"][strategy]["case_count"] == 2
    assert "Banka kartı" not in output.read_text(encoding="utf-8")


def test_routing_config_validation_and_live_denial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    valid = RUNNER.invoke(
        app, ["validate-routing-config", "--router-config", str(ROOT / "config/routing.yaml")]
    )
    assert valid.exit_code == 0
    assert "ROUTER_CONFIG_VALID" in valid.output
    monkeypatch.setenv("BENCHMARK_MODE", "live")
    refused = RUNNER.invoke(
        app, ["run-routing", "--strategy", "knn", "--output", str(tmp_path / "no.jsonl")]
    )
    assert refused.exit_code == 2
    assert "LIVE_MODE_NOT_IMPLEMENTED" in refused.output
    assert not (tmp_path / "no.jsonl").exists()


def test_invalid_strategy_denied_without_provider_call(tmp_path: Path) -> None:
    output = tmp_path / "blocked.jsonl"
    result = RUNNER.invoke(
        app, ["run-routing", "--strategy", "unknown", "--output", str(output)]
    )
    assert result.exit_code == 2
    assert "INVALID_ROUTING_STRATEGY" in result.output
    assert not output.exists()
