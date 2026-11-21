"""Typer composition root for the isolated, safe offline M1 smoke workflow."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from llm_routing_eval.application import (
    BenchmarkInputError,
    BenchmarkOrchestrator,
    BenchmarkPersistenceError,
)
from llm_routing_eval.cli.comparison_export import write_comparison
from llm_routing_eval.cli.configuration import demo_options, read_settings
from llm_routing_eval.cli.groq_preflight import assess_groq_preflight, read_provider_manifest
from llm_routing_eval.core import BenchmarkCase
from llm_routing_eval.demo.fakes import (
    DEMO_PROMPTS,
    AllowlistedDemoPrivacy,
    DemoAlwaysLargeRouter,
    DemoFakeProvider,
    DemoFixtureEvaluator,
    InMemoryDemoRepository,
)
from llm_routing_eval.routing import build_router, read_routing_settings

app = typer.Typer(help="LLM routing evaluation: safe offline integration demo.")
DEFAULT_CONFIG = Path("config/settings.yaml")
DEFAULT_DATASET = Path("examples/offline_demo_cases.json")
DEFAULT_OUTPUT = Path("artifacts/offline_demo.jsonl")


def _load_demo_cases(path: Path) -> tuple[str, list[BenchmarkCase]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
            raise BenchmarkInputError("DATASET_INVALID_FORMAT")
        dataset_version = data.get("dataset_version")
        if not isinstance(dataset_version, str) or not dataset_version:
            raise BenchmarkInputError("DATASET_MISSING_VERSION")
        cases = [BenchmarkCase.model_validate(record) for record in data["cases"]]
        # In this interim demo, never accept data other than explicitly marked synthetic.
        if any(case.metadata.get("synthetic") is not True for case in cases):
            raise BenchmarkInputError("DEMO_REQUIRES_SYNTHETIC_CASES")
        return dataset_version, cases
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise BenchmarkInputError("DATASET_INVALID") from exc


def _write_safe_jsonl(path: Path, repository: InMemoryDemoRepository) -> None:
    """Atomic interim demo export; Member 04 owns durable benchmark persistence."""
    temp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            [("trace", item) for item in repository.traces]
            + [("evaluation", item) for item in repository.evaluations]
            + [("report", item) for item in repository.reports]
        )
        # Only schema-safe records: no ModelResponse.content or BenchmarkCase.input_text.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".offline-demo-",
            suffix=".jsonl",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for kind, record in rows:
                handle.write(
                    json.dumps(
                        {"record_type": kind, "payload": record.model_dump(mode="json")},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        os.replace(temp_path, path)
    except OSError as exc:
        raise BenchmarkPersistenceError("OUTPUT_WRITE_FAILED") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.command("validate-config")
def validate_config(
    config: Annotated[Path, typer.Option(help="YAML settings file")] = DEFAULT_CONFIG,
) -> None:
    """Validate offline settings without invoking providers."""
    try:
        demo_options(read_settings(config))
        typer.echo("CONFIG_VALID: offline demo (no network, no API keys)")
    except BenchmarkInputError as exc:
        typer.echo(f"CONFIG_ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from None


@app.command("run")
def run(
    config: Annotated[Path, typer.Option(help="YAML settings file")] = DEFAULT_CONFIG,
    dataset: Annotated[Path, typer.Option(help="Synthetic demo fixtures JSON")] = DEFAULT_DATASET,
    output: Annotated[Path, typer.Option(help="Safe JSONL result destination")] = DEFAULT_OUTPUT,
    live: Annotated[bool, typer.Option(help="Reserved; live mode is NOT implemented")] = False,
) -> None:
    """Run the offline integration test using deterministic fake components."""
    if live:
        typer.echo("LIVE_MODE_NOT_IMPLEMENTED", err=True)
        raise typer.Exit(code=2)
    try:
        options = demo_options(read_settings(config))
        dataset_version, cases = _load_demo_cases(dataset)
        options = options.model_copy(update={"dataset_version": dataset_version})
        repo = InMemoryDemoRepository()
        orchestrator = BenchmarkOrchestrator(
            router=DemoAlwaysLargeRouter(model=options.allowed_models[0]),
            provider=DemoFakeProvider(),
            evaluator=DemoFixtureEvaluator(),
            privacy=AllowlistedDemoPrivacy(),
            repository=repo,
        )
        result = asyncio.run(orchestrator.run(cases, options))
        _write_safe_jsonl(output, repo)
        metric = result.report.strategy_metrics[options.strategy]
        typer.echo(f"RUN_ID: {result.report.run_id}")
        typer.echo(
            f"CASES: {metric.case_count}, SUCCESS: {metric.successful_cases}, "
            f"FAILED: {metric.failed_cases}, BLOCKED: {metric.blocked_cases}"
        )
        typer.echo("MODE: OFFLINE_SIMULATION; financial quality and API cost NOT measured")
        typer.echo(f"REPORT: {output}")
    except (BenchmarkInputError, BenchmarkPersistenceError, ValidationError) as exc:
        # Avoid exposing Pydantic validation inputs, file names or secrets in errors.
        reason = (
            str(exc)
            if isinstance(exc, (BenchmarkInputError, BenchmarkPersistenceError))
            else "INVALID_OPTIONS"
        )
        typer.echo(f"RUN_FAILED: {reason}", err=True)
        raise typer.Exit(code=2) from None


@app.command("validate-routing-config")
def validate_routing_config(
    router_config: Annotated[Path, typer.Option(help="Member 02 YAML router policy")] = Path(
        "config/routing.yaml"
    ),
) -> None:
    """Validate versioned router settings without network access or API keys."""
    try:
        settings = read_routing_settings(router_config)
        if any(example.input_text not in DEMO_PROMPTS for example in settings.knn.examples):
            raise BenchmarkInputError("ROUTING_DEMO_EXAMPLES_NOT_ALLOWLISTED")
        typer.echo("ROUTER_CONFIG_VALID: offline, development-only")
    except (ValueError, BenchmarkInputError):
        typer.echo("ROUTER_CONFIG_INVALID", err=True)
        raise typer.Exit(code=2) from None


@app.command("run-routing")
def run_routing(
    strategy: Annotated[str, typer.Option(help="always_large, rule_based, knn or cascade")] = (
        "always_large"
    ),
    config: Annotated[Path, typer.Option(help="Offline benchmark YAML settings")] = DEFAULT_CONFIG,
    router_config: Annotated[Path, typer.Option(help="Versioned router policy YAML")] = Path(
        "config/routing.yaml"
    ),
    dataset: Annotated[Path, typer.Option(help="Allowlisted synthetic demo fixtures")] = (
        DEFAULT_DATASET
    ),
    output: Annotated[Path, typer.Option(help="Safe JSONL output")] = Path(
        "artifacts/offline_routing.jsonl"
    ),
) -> None:
    """Run any real Member 02 strategy with the offline fake provider only."""
    from llm_routing_eval.application import RunOptions
    from llm_routing_eval.core import StrategyId

    try:
        settings = read_settings(config)
        if settings.execution.mode != "offline" or settings.execution.live_enabled:
            raise BenchmarkInputError("LIVE_MODE_NOT_IMPLEMENTED")
        router_settings = read_routing_settings(router_config)
        if any(example.input_text not in DEMO_PROMPTS for example in router_settings.knn.examples):
            raise BenchmarkInputError("ROUTING_DEMO_EXAMPLES_NOT_ALLOWLISTED")
        try:
            chosen = StrategyId(strategy)
        except ValueError:
            raise BenchmarkInputError("INVALID_ROUTING_STRATEGY") from None
        dataset_version, cases = _load_demo_cases(dataset)
        options = RunOptions(
            strategy=chosen,
            provider="fake",
            allowed_models=(router_settings.models.small, router_settings.models.large),
            dataset_version=dataset_version,
            routing_policy_sha256=router_settings.fingerprint(),
            max_model_calls_per_case=settings.execution.max_model_calls_per_case,
            max_retries=settings.execution.max_retries,
        )
        repository = InMemoryDemoRepository()

        async def execute() -> None:
            router = await build_router(chosen, router_settings)
            orchestrator = BenchmarkOrchestrator(
                router=router,
                provider=DemoFakeProvider(),
                evaluator=DemoFixtureEvaluator(),
                privacy=AllowlistedDemoPrivacy(),
                repository=repository,
            )
            await orchestrator.run(cases, options)

        asyncio.run(execute())
        _write_safe_jsonl(output, repository)
        metric = repository.reports[-1].strategy_metrics[chosen]
        typer.echo(f"ROUTER: {chosen.value}")
        typer.echo(
            f"CASES: {metric.case_count}, SUCCESS: {metric.successful_cases}, "
            f"FAILED: {metric.failed_cases}, BLOCKED: {metric.blocked_cases}"
        )
        typer.echo("MODE: OFFLINE_SIMULATION; financial quality and API cost NOT measured")
        typer.echo(f"REPORT: {output}")
    except (BenchmarkInputError, BenchmarkPersistenceError, ValidationError, ValueError) as exc:
        reason = (
            str(exc)
            if isinstance(exc, (BenchmarkInputError, BenchmarkPersistenceError))
            else "ROUTER_CONFIG_INVALID"
        )
        typer.echo(f"RUN_FAILED: {reason}", err=True)
        raise typer.Exit(code=2) from None


@app.command("compare-routing")
def compare_routing(
    config: Annotated[Path, typer.Option(help="Offline benchmark YAML")] = DEFAULT_CONFIG,
    router_config: Annotated[Path, typer.Option(help="Member 02 routing policy YAML")] = Path(
        "config/routing.yaml"
    ),
    dataset: Annotated[Path, typer.Option(help="Only allowlisted synthetic cases")] = (
        DEFAULT_DATASET
    ),
    output: Annotated[Path, typer.Option(help="Atomic summary JSON destination")] = Path(
        "artifacts/offline_comparison.json"
    ),
) -> None:
    """Run all four strategies on identical synthetic cases; never call live APIs."""
    from llm_routing_eval.application import RunOptions, StrategyComparisonRunner
    from llm_routing_eval.core import StrategyId

    try:
        settings = read_settings(config)
        if settings.execution.mode != "offline" or settings.execution.live_enabled:
            raise BenchmarkInputError("LIVE_MODE_NOT_IMPLEMENTED")
        routing = read_routing_settings(router_config)
        if any(example.input_text not in DEMO_PROMPTS for example in routing.knn.examples):
            raise BenchmarkInputError("ROUTING_DEMO_EXAMPLES_NOT_ALLOWLISTED")
        version, cases = _load_demo_cases(dataset)
        if not cases or any(case.input_text not in DEMO_PROMPTS for case in cases):
            raise BenchmarkInputError("COMPARISON_REQUIRES_ALLOWLISTED_CASES")

        async def build(strategy: StrategyId) -> BenchmarkOrchestrator:
            return BenchmarkOrchestrator(
                router=await build_router(strategy, routing),
                provider=DemoFakeProvider(),
                evaluator=DemoFixtureEvaluator(),
                privacy=AllowlistedDemoPrivacy(),
                repository=InMemoryDemoRepository(),
            )

        options = RunOptions(
            provider="fake",
            allowed_models=(routing.models.small, routing.models.large),
            dataset_version=version,
            routing_policy_sha256=routing.fingerprint(),
            max_model_calls_per_case=settings.execution.max_model_calls_per_case,
            max_retries=settings.execution.max_retries,
        )
        summary = asyncio.run(
            StrategyComparisonRunner(orchestrator_factory=build).compare(cases, options)
        )
        write_comparison(output, summary)
        typer.echo("MODE: OFFLINE_SIMULATION (NOT financial quality or provider latency)")
        typer.echo("STRATEGY        SUCCESS   ATTEMPTS   ATT/CAS   TOKENS/CALL (SIMULATED)")
        for row in summary.strategies:
            total = (
                f"{row.input_tokens_per_attempt:.1f}+{row.output_tokens_per_attempt:.1f}"
                if row.input_tokens_per_attempt is not None
                and row.output_tokens_per_attempt is not None
                else "UNKNOWN"
            )
            typer.echo(
                f"{row.strategy.value:<16}{row.successful_cases:>3}/{row.case_count:<5}"
                f"{row.provider_attempts:>8}{row.attempts_per_case:>10.2f}   {total}"
            )
        typer.echo("COST: UNKNOWN; LATENCY: LOCAL APPLICATION ONLY")
        typer.echo(f"REPORT: {output}")
    except (BenchmarkInputError, BenchmarkPersistenceError, ValueError, ValidationError) as exc:
        reason = (
            str(exc)
            if isinstance(exc, (BenchmarkInputError, BenchmarkPersistenceError))
            else "COMPARISON_CONFIG_INVALID"
        )
        typer.echo(f"COMPARISON_FAILED: {reason}", err=True)
        raise typer.Exit(code=2) from None


@app.command("groq-preflight")
def groq_preflight(
    providers_config: Annotated[Path, typer.Option(help="Versioned provider YAML config")] = Path(
        "config/providers.yaml"
    ),
    router_config: Annotated[Path, typer.Option(help="Versioned router YAML config")] = Path(
        "config/routing.yaml"
    ),
) -> None:
    """Diagnose live integration blockers without external API requests or secrets."""
    try:
        manifest = read_provider_manifest(providers_config)
        policy = read_routing_settings(router_config)
        result = assess_groq_preflight(manifest, policy)
    except ValueError:
        typer.echo("GROQ_PREFLIGHT_INVALID", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(f"GROQ_PREFLIGHT_STATUS: {result.status}")
    typer.echo(
        "MODEL_CONFIGURATION: "
        + ("MATCH" if result.provider_model_alignment else "MISMATCH")
    )
    for reason in result.blockers:
        typer.echo(f"BLOCKER: {reason}")
    typer.echo("NO_NETWORK_REQUESTS; no credential value inspected or printed")
    # Until approval gates exist, this command must never report live readiness.
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
