"""Routing algorithms behind the shared Core router/embedding contracts."""

from .configuration import RoutingSettings, read_routing_settings
from .embeddings import HashingEmbedding
from .factory import build_router
from .strategies import (
    AlwaysLargeRouter,
    CascadeRouter,
    DevelopmentExample,
    KnnRouter,
    ModelPair,
    RuleBasedRouter,
    ScenarioRule,
)

__all__ = [
    "AlwaysLargeRouter",
    "CascadeRouter",
    "DevelopmentExample",
    "HashingEmbedding",
    "KnnRouter",
    "ModelPair",
    "RuleBasedRouter",
    "RoutingSettings",
    "build_router",
    "read_routing_settings",
    "ScenarioRule",
]
