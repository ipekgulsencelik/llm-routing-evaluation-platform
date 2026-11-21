"""Versioned, fail-closed Member 02 configuration, separate from CLI and providers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .strategies import ModelTier

_SAFE_ID = r"^[A-Za-z0-9_./-]{1,100}$"
_SAFE_REASON = r"^[A-Z][A-Z0-9_]{0,39}$"


class FrozenSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class ModelPairSettings(FrozenSettings):
    small: str = Field(min_length=1, pattern=_SAFE_ID)
    large: str = Field(min_length=1, pattern=_SAFE_ID)

    @model_validator(mode="after")
    def distinct_models(self) -> Self:
        if self.small == self.large:
            raise ValueError("MODEL_IDS_MUST_DIFFER")
        return self


class RuleSettings(FrozenSettings):
    code: str = Field(pattern=_SAFE_REASON)
    scenario_types: tuple[str, ...] = Field(min_length=1)
    tier: ModelTier

    @model_validator(mode="after")
    def valid_scenarios(self) -> Self:
        if any(not name.strip() for name in self.scenario_types):
            raise ValueError("EMPTY_SCENARIO")
        return self


class RuleBasedSettings(FrozenSettings):
    default_tier: ModelTier = "large"
    rules: tuple[RuleSettings, ...] = ()

    @model_validator(mode="after")
    def unique_codes(self) -> Self:
        if len({rule.code for rule in self.rules}) != len(self.rules):
            raise ValueError("DUPLICATE_RULE_CODE")
        return self


class DevelopmentExampleSettings(FrozenSettings):
    example_id: str = Field(min_length=1, pattern=_SAFE_ID)
    input_text: str = Field(min_length=1, repr=False)
    tier: ModelTier
    split: Literal["development"]


class KnnSettings(FrozenSettings):
    k: int = Field(default=1, ge=1, le=100)
    minimum_similarity: float = Field(default=0.35, ge=-1, le=1, allow_inf_nan=False)
    embedding_dimensions: int = Field(default=256, ge=16, le=4096)
    examples: tuple[DevelopmentExampleSettings, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        if self.k > len(self.examples):
            raise ValueError("K_EXCEEDS_DEVELOPMENT_SET")
        if len({ex.example_id for ex in self.examples}) != len(self.examples):
            raise ValueError("DUPLICATE_EXAMPLE_ID")
        return self


class CascadeSettings(FrozenSettings):
    escalate_scenarios: tuple[str, ...] = ()
    escalate_on_failure: bool = True

    @model_validator(mode="after")
    def nonempty_scenarios(self) -> Self:
        if any(not name.strip() for name in self.escalate_scenarios):
            raise ValueError("EMPTY_ESCALATION_SCENARIO")
        return self


class RoutingSettings(FrozenSettings):
    schema_version: Literal["1.0.0"]
    models: ModelPairSettings
    rule_based: RuleBasedSettings
    knn: KnnSettings
    cascade: CascadeSettings

    def fingerprint(self) -> str:
        """Hash the full routing policy without exposing development sample text.

        Model IDs, rule order, development examples, thresholds, and versions
        are intentionally all included so changing routing behavior changes
        the immutable fingerprint.
        """
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        digest = hashlib.sha256(("routing-policy-v1\n" + canonical).encode("utf-8")).hexdigest()
        return "sha256:" + digest


def read_routing_settings(path: Path) -> RoutingSettings:
    """Read a small YAML file with strict keys and no secret-bearing diagnostics."""
    try:
        if not path.is_file() or path.stat().st_size > 131_072:
            raise ValueError("ROUTER_CONFIG_MISSING_OR_TOO_LARGE")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("ROUTER_CONFIG_NOT_MAPPING")
        return RoutingSettings.model_validate(loaded)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ValueError("ROUTER_CONFIG_INVALID") from exc
