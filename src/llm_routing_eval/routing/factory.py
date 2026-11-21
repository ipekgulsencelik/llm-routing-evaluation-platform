"""Router composition (dependency injection), independent of CLI and provider adapters."""

from __future__ import annotations

from llm_routing_eval.core import EmbeddingPort, RouterPort, StrategyId

from .configuration import RoutingSettings
from .embeddings import HashingEmbedding
from .strategies import (
    AlwaysLargeRouter,
    CascadeRouter,
    DevelopmentExample,
    KnnRouter,
    ModelPair,
    RuleBasedRouter,
    ScenarioRule,
)


async def build_router(
    strategy: StrategyId,
    settings: RoutingSettings,
    *,
    embedder: EmbeddingPort | None = None,
) -> RouterPort:
    models = ModelPair(small=settings.models.small, large=settings.models.large)
    if strategy is StrategyId.ALWAYS_LARGE:
        return AlwaysLargeRouter(models)
    if strategy is StrategyId.RULE_BASED:
        rules = tuple(
            ScenarioRule(code=r.code, scenario_types=r.scenario_types, tier=r.tier)
            for r in settings.rule_based.rules
        )
        return RuleBasedRouter(models, rules, default_tier=settings.rule_based.default_tier)
    if strategy is StrategyId.KNN:
        examples = tuple(
            DevelopmentExample(
                example_id=e.example_id, input_text=e.input_text, tier=e.tier, split=e.split
            )
            for e in settings.knn.examples
        )
        return await KnnRouter.prepare(
            models,
            embedder or HashingEmbedding(dimensions=settings.knn.embedding_dimensions),
            examples,
            k=settings.knn.k,
            minimum_similarity=settings.knn.minimum_similarity,
        )
    if strategy is StrategyId.CASCADE:
        return CascadeRouter(
            models,
            escalate_scenarios=frozenset(settings.cascade.escalate_scenarios),
            escalate_on_failure=settings.cascade.escalate_on_failure,
        )
    raise ValueError("UNSUPPORTED_ROUTING_STRATEGY")
