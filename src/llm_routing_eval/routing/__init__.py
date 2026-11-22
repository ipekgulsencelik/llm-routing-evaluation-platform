"""Routing algorithms behind the shared Core router/embedding contracts."""

from .calibration import (
    CalibrationDataset,
    CalibrationExample,
    apply_calibration,
    calibrate_cascade,
    calibrate_knn,
    calibration_report,
    read_calibration_dataset,
)
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
    knn_vote,
    rank_neighbours,
    vote_ranked,
)

__all__ = [
    "AlwaysLargeRouter",
    "CalibrationDataset",
    "CalibrationExample",
    "CascadeRouter",
    "DevelopmentExample",
    "HashingEmbedding",
    "KnnRouter",
    "ModelPair",
    "RuleBasedRouter",
    "RoutingSettings",
    "apply_calibration",
    "build_router",
    "calibrate_cascade",
    "calibrate_knn",
    "calibration_report",
    "knn_vote",
    "rank_neighbours",
    "vote_ranked",
    "read_calibration_dataset",
    "read_routing_settings",
    "ScenarioRule",
]
