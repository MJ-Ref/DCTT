"""Weights & Biases integration utilities.

This module provides utilities for integrating DCTT with W&B for
experiment tracking, artifact logging, and visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig
    import wandb

logger = logging.getLogger(__name__)


def init_wandb_from_hydra(
    cfg: "DictConfig",
    project: str | None = None,
    entity: str | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
) -> "wandb.Run":
    """Initialize W&B run from Hydra configuration.

    Args:
        cfg: Hydra configuration.
        project: W&B project name (overrides config).
        entity: W&B entity (overrides config).
        name: Run name (overrides config).
        tags: Run tags (overrides config).

    Returns:
        Initialized W&B run.
    """
    import wandb
    from omegaconf import OmegaConf

    # Get W&B settings from config
    wandb_cfg = cfg.get("wandb", {})

    # Resolve parameters
    project = project or wandb_cfg.get("project", "dctt")
    entity = entity or wandb_cfg.get("entity")
    name = name or wandb_cfg.get("run_name")
    tags = tags or wandb_cfg.get("tags", [])

    # Convert OmegaConf to dict for W&B
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config_dict,
        tags=tags,
        settings=wandb.Settings(code_dir="."),
    )

    logger.info(f"Initialized W&B run: {run.name} ({run.id})")

    return run


def log_metrics(
    metrics: dict[str, Any],
    step: int | None = None,
    prefix: str = "",
) -> None:
    """Log metrics to W&B.

    Args:
        metrics: Dictionary of metric name -> value.
        step: Optional step number.
        prefix: Optional prefix for metric names.
    """
    import wandb

    if wandb.run is None:
        logger.warning("W&B not initialized, skipping metric logging")
        return

    # Add prefix if provided
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    wandb.log(metrics, step=step)


def log_artifact(
    artifact_path: str | Path,
    name: str,
    artifact_type: str,
    metadata: dict | None = None,
) -> None:
    """Log an artifact to W&B.

    Args:
        artifact_path: Path to artifact file or directory.
        name: Artifact name.
        artifact_type: Type of artifact (e.g., "embeddings", "results").
        metadata: Optional metadata dictionary.
    """
    import wandb

    if wandb.run is None:
        logger.warning("W&B not initialized, skipping artifact logging")
        return

    artifact = wandb.Artifact(
        name=name,
        type=artifact_type,
        metadata=metadata or {},
    )

    path = Path(artifact_path)
    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))

    wandb.run.log_artifact(artifact)
    logger.info(f"Logged artifact: {name} ({artifact_type})")


def log_table(
    data: list[dict],
    name: str,
    columns: list[str] | None = None,
) -> None:
    """Log a table to W&B.

    Args:
        data: List of row dictionaries.
        name: Table name.
        columns: Column names (inferred from data if not provided).
    """
    import wandb

    if wandb.run is None:
        return

    if not data:
        return

    # Infer columns from first row
    if columns is None:
        columns = list(data[0].keys())

    table = wandb.Table(columns=columns)
    for row in data:
        table.add_data(*[row.get(col) for col in columns])

    wandb.log({name: table})


def log_diagnostic_summary(
    results: list,
    prefix: str = "diagnostics",
) -> None:
    """Log diagnostic results summary to W&B.

    Args:
        results: List of DiagnosticResult objects.
        prefix: Metric prefix.
    """
    import wandb

    if wandb.run is None:
        return

    # Compute summary statistics
    severities = [r.severity for r in results]
    consistencies = [r.consistency for r in results]
    fail_counts = sum(1 for r in results if r.fails_any_stage)

    metrics = {
        f"{prefix}/total_tokens": len(results),
        f"{prefix}/failing_tokens": fail_counts,
        f"{prefix}/fail_rate": fail_counts / len(results) if results else 0,
        f"{prefix}/mean_severity": sum(severities) / len(severities) if severities else 0,
        f"{prefix}/max_severity": max(severities) if severities else 0,
        f"{prefix}/mean_consistency": sum(consistencies) / len(consistencies) if consistencies else 0,
    }

    wandb.log(metrics)


def finish_run() -> None:
    """Finish the current W&B run."""
    import wandb

    if wandb.run is not None:
        wandb.finish()
        logger.info("W&B run finished")
