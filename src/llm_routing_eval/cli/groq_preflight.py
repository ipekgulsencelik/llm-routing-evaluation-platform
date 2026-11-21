"""Strict, offline-only Groq integration preflight; never reads or prints key values.

This diagnoses configuration, not model-account availability or privacy approval.
All currently missing cross-member release gates must be satisfied independently.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_routing_eval.routing.configuration import ModelPairSettings, RoutingSettings

_SAFE_ENV = "GROQ_API_KEY"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class GroqConfig(_Frozen):
    enabled: bool = False
    api_key_env: Literal["GROQ_API_KEY"]
    candidate_models: ModelPairSettings


class _DisabledProvider(_Frozen):
    enabled: Literal[False]
    api_key_env: str
    candidate_models: dict[str, str] = Field(default_factory=dict)


class _ProviderCollection(_Frozen):
    groq: GroqConfig
    nvidia_build: _DisabledProvider


class ProviderManifest(_Frozen):
    schema_version: Literal["1.0.0"]
    providers: _ProviderCollection


class GroqPreflight(_Frozen):
    """Only machine-readable safe fields; keys and prompts are never represented."""

    status: Literal["BLOCKED"] = "BLOCKED"
    provider_model_alignment: bool
    provider_enabled: bool
    credential_configured: bool
    environment_opt_in: bool
    blockers: tuple[str, ...] = Field(min_length=1)


def read_provider_manifest(path: Path) -> ProviderManifest:
    """Read an immutable, bounded settings file without disclosing parse input."""
    try:
        if not path.is_file() or path.stat().st_size > 131_072:
            raise ValueError("PROVIDER_CONFIG_UNAVAILABLE")
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(content, dict):
            raise ValueError("PROVIDER_CONFIG_NOT_MAPPING")
        return ProviderManifest.model_validate(content)
    except (OSError, UnicodeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        raise ValueError("PROVIDER_CONFIG_INVALID") from exc


def assess_groq_preflight(
    manifest: ProviderManifest,
    router_policy: RoutingSettings,
) -> GroqPreflight:
    """A diagnostic only: there is intentionally no path to READY before reviews."""
    groq = manifest.providers.groq
    aligned = (
        groq.candidate_models.small == router_policy.models.small
        and groq.candidate_models.large == router_policy.models.large
    )
    configured = bool(os.getenv(_SAFE_ENV, "").strip())
    opt_in = os.getenv("LLM_ROUTING_ALLOW_LIVE") == "1"
    blockers: list[str] = []
    if not aligned:
        blockers.append("MODEL_CONFIGURATION_MISMATCH")
    if not groq.enabled:
        blockers.append("GROQ_PROVIDER_DISABLED")
    if not configured:
        blockers.append("GROQ_CREDENTIAL_NOT_CONFIGURED")
    if not opt_in:
        blockers.append("GROQ_LIVE_OPT_IN_DISABLED")

    # Approval *cannot* be inferred from a local toggle or a working API key.
    blockers.extend(
        (
            "OUTBOUND_PRIVACY_AUTHORIZATION_NOT_APPROVED",
            "EXPERT_EVALUATION_NOT_VERIFIED",
            "VERIFIED_PRICING_NOT_AVAILABLE",
            "MODEL_ACCOUNT_ACCESS_NOT_VERIFIED",
        )
    )
    return GroqPreflight(
        provider_model_alignment=aligned,
        provider_enabled=groq.enabled,
        credential_configured=configured,
        environment_opt_in=opt_in,
        blockers=tuple(blockers),
    )
