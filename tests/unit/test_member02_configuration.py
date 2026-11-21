"""Strict offline routing YAML, factory injection and held-out separation tests."""

import asyncio
from pathlib import Path

import pytest
import yaml

from llm_routing_eval.core import (
    BenchmarkCase,
    RoutingAction,
    RoutingContext,
    StrategyId,
)
from llm_routing_eval.routing import build_router, read_routing_settings

CONFIG = Path(__file__).resolve().parents[2] / "config" / "routing.yaml"


def read_source() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def invalid_file(tmp_path: Path, data: dict[str, object]) -> Path:
    file = tmp_path / "invalid.yaml"
    file.write_text(yaml.safe_dump(data), encoding="utf-8")
    return file


def test_repository_routing_yaml_is_valid_and_versioned() -> None:
    settings = read_routing_settings(CONFIG)
    assert settings.schema_version == "1.0.0"
    assert settings.models.small != settings.models.large
    assert settings.knn.examples
    assert all(ex.split == "development" for ex in settings.knn.examples)
    assert settings.rule_based.default_tier == "large"


@pytest.mark.parametrize(
    ("strategy", "expected_model"),
    [
        (StrategyId.ALWAYS_LARGE, "openai/gpt-oss-120b"),
        (StrategyId.RULE_BASED, "openai/gpt-oss-20b"),
        (StrategyId.KNN, "openai/gpt-oss-20b"),
        (StrategyId.CASCADE, "openai/gpt-oss-20b"),
    ],
)
def test_factory_builds_every_real_router(strategy: StrategyId, expected_model: str) -> None:
    settings = read_routing_settings(CONFIG)
    router = asyncio.run(build_router(strategy, settings))
    chosen = asyncio.run(
        router.decide(
            BenchmarkCase(
                case_id="dev-only",
                input_text="Banka hesabı açmak için genel süreç nedir?",
                domain="turkish_banking",
                scenario_type="account_general",
            ),
            RoutingContext(current_stage=0, remaining_model_calls=2),
        )
    )
    assert chosen.strategy == strategy
    assert chosen.selected_model == expected_model
    assert chosen.action is RoutingAction.INVOKE_MODEL


@pytest.mark.parametrize(
    ("edit", "description"),
    [
        ("held_out", "golden samples"),
        ("duplicate", "duplicate example IDs"),
        ("large_k", "k larger than samples"),
        ("extra", "unknown fields"),
        ("same_model", "identical models"),
        ("unsafe_rule", "unsafe rule codes"),
    ],
)
def test_invalid_router_configuration_is_rejected(
    tmp_path: Path, edit: str, description: str
) -> None:
    _ = description
    data = read_source()
    knn = data["knn"]
    models = data["models"]
    rule_based = data["rule_based"]
    assert isinstance(knn, dict)
    assert isinstance(models, dict)
    assert isinstance(rule_based, dict)
    examples = knn["examples"]
    assert isinstance(examples, list)
    if edit == "held_out":
        examples[0]["split"] = "golden"
    elif edit == "duplicate":
        examples[1]["example_id"] = examples[0]["example_id"]
    elif edit == "large_k":
        knn["k"] = 99
    elif edit == "extra":
        data["api_key"] = "DO_NOT_ACCEPT_SECRETS"
    elif edit == "same_model":
        models["large"] = models["small"]
    elif edit == "unsafe_rule":
        rule_based["rules"][0]["code"] = "PRIVATE DATA"
    with pytest.raises(ValueError, match="ROUTER_CONFIG_INVALID"):
        read_routing_settings(invalid_file(tmp_path, data))


def test_router_config_rejects_oversized_and_non_mapping_documents(tmp_path: Path) -> None:
    large = tmp_path / "large.yaml"
    large.write_text("x" * 150_000, encoding="utf-8")
    with pytest.raises(ValueError, match="ROUTER_CONFIG_INVALID"):
        read_routing_settings(large)
    non_map = tmp_path / "array.yaml"
    non_map.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="ROUTER_CONFIG_INVALID"):
        read_routing_settings(non_map)


def test_routing_policy_fingerprint_changes_for_model_rules_and_dev_samples() -> None:
    base = read_routing_settings(CONFIG)
    fingerprint = base.fingerprint()
    assert fingerprint.startswith("sha256:") and len(fingerprint) == 71
    assert fingerprint == read_routing_settings(CONFIG).fingerprint()
    changed_rule = base.model_copy(
        update={"models": base.models.model_copy(update={"large": "different-model"})}
    )
    assert changed_rule.fingerprint() != fingerprint
    examples = list(base.knn.examples)
    examples[0] = examples[0].model_copy(update={"input_text": "changed development text"})
    changed_development = base.model_copy(
        update={"knn": base.knn.model_copy(update={"examples": tuple(examples)})}
    )
    assert changed_development.fingerprint() != fingerprint
