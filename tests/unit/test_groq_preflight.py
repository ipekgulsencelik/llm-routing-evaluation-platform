"""Groq readiness never treats API-key existence as approval to transmit data."""

import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llm_routing_eval.cli.app import app
from llm_routing_eval.cli.groq_preflight import assess_groq_preflight, read_provider_manifest
from llm_routing_eval.routing import read_routing_settings

ROOT = Path(__file__).resolve().parents[2]
RUNNER = CliRunner()


def test_default_preflight_blocks_live_and_matches_model_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_ROUTING_ALLOW_LIVE", raising=False)
    status = assess_groq_preflight(
        read_provider_manifest(ROOT / "config/providers.yaml"),
        read_routing_settings(ROOT / "config/routing.yaml"),
    )
    assert status.status == "BLOCKED"
    assert status.provider_model_alignment
    assert not status.provider_enabled
    assert not status.credential_configured
    assert not status.environment_opt_in
    assert "GROQ_PROVIDER_DISABLED" in status.blockers
    assert "GROQ_CREDENTIAL_NOT_CONFIGURED" in status.blockers
    assert "OUTBOUND_PRIVACY_AUTHORIZATION_NOT_APPROVED" in status.blockers


def test_even_all_local_toggles_dont_claim_approval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "SECRET_VALUE_NEVER_PRINT")
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    data = yaml.safe_load((ROOT / "config/providers.yaml").read_text(encoding="utf-8"))
    data["providers"]["groq"]["enabled"] = True
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = RUNNER.invoke(app, ["groq-preflight", "--providers-config", str(path)])
    assert result.exit_code == 2
    assert "GROQ_PREFLIGHT_STATUS: BLOCKED" in result.output
    assert "MODEL_CONFIGURATION: MATCH" in result.output
    assert "EXPERT_EVALUATION_NOT_VERIFIED" in result.output
    assert "VERIFIED_PRICING_NOT_AVAILABLE" in result.output
    assert "MODEL_ACCOUNT_ACCESS_NOT_VERIFIED" in result.output
    assert "GROQ_CREDENTIAL_NOT_CONFIGURED" not in result.output
    assert "SECRET_VALUE_NEVER_PRINT" not in result.output
    assert "GROQ_API_KEY" not in result.output


def test_preflight_rejects_provider_model_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    data = yaml.safe_load((ROOT / "config/providers.yaml").read_text(encoding="utf-8"))
    data["providers"]["groq"]["candidate_models"]["small"] = "different-model"
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    output = RUNNER.invoke(app, ["groq-preflight", "--providers-config", str(path)])
    assert output.exit_code == 2
    assert "MODEL_CONFIGURATION: MISMATCH" in output.output
    assert "MODEL_CONFIGURATION_MISMATCH" in output.output


def test_preflight_invalid_config_does_not_leak_inputs(tmp_path: Path) -> None:
    data = tmp_path / "invalid.yaml"
    data.write_text("secret_value: PRIVACY_DATA_SHOULD_NOT_APPEAR", encoding="utf-8")
    with pytest.raises(ValueError, match="PROVIDER_CONFIG_INVALID"):
        read_provider_manifest(data)
    result = RUNNER.invoke(app, ["groq-preflight", "--providers-config", str(data)])
    assert result.exit_code == 2
    assert "GROQ_PREFLIGHT_INVALID" in result.output
    assert "PRIVACY_DATA_SHOULD_NOT_APPEAR" not in result.output


def test_preflight_does_not_modify_environment_or_emit_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "PLACEHOLDER_NOT_REAL")
    snapshot = os.environ["GROQ_API_KEY"]
    result = RUNNER.invoke(app, ["groq-preflight"])
    assert result.exit_code == 2
    assert os.environ["GROQ_API_KEY"] == snapshot
    assert snapshot not in result.output
