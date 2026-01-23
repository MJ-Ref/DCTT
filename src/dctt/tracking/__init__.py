"""Experiment tracking and reproducibility infrastructure."""

from dctt.tracking.wandb_utils import (
    init_wandb_from_hydra,
    log_metrics,
    log_artifact,
)
from dctt.tracking.artifacts import (
    log_embeddings_artifact,
    log_diagnostic_results_artifact,
    log_repair_results_artifact,
)
from dctt.tracking.reproducibility import (
    RunManifest,
    create_run_manifest,
    set_all_seeds,
    hash_embeddings,
    hash_directory,
)

__all__ = [
    # W&B utils
    "init_wandb_from_hydra",
    "log_metrics",
    "log_artifact",
    # Artifacts
    "log_embeddings_artifact",
    "log_diagnostic_results_artifact",
    "log_repair_results_artifact",
    # Reproducibility
    "RunManifest",
    "create_run_manifest",
    "set_all_seeds",
    "hash_embeddings",
    "hash_directory",
]
