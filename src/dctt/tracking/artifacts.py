"""W&B artifact management for DCTT.

This module provides specialized functions for logging DCTT-specific
artifacts to Weights & Biases.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import wandb
    from dctt.tracking.reproducibility import RunManifest

logger = logging.getLogger(__name__)


def log_embeddings_artifact(
    run: "wandb.Run",
    embeddings_path: Path,
    manifest: "RunManifest",
) -> "wandb.Artifact":
    """Log embeddings as versioned W&B artifact.

    Args:
        run: Active W&B run.
        embeddings_path: Path to saved embeddings file.
        manifest: Run manifest with metadata.

    Returns:
        Logged artifact.
    """
    import wandb

    artifact = wandb.Artifact(
        name=f"embeddings-{manifest.model_name.replace('/', '-')}",
        type="embeddings",
        metadata={
            "model_name": manifest.model_name,
            "model_revision": manifest.model_revision,
            "embedding_hash": manifest.embedding_matrix_hash,
            "run_id": manifest.run_id,
        },
    )

    artifact.add_file(str(embeddings_path))
    run.log_artifact(artifact)

    logger.info(f"Logged embeddings artifact for {manifest.model_name}")
    return artifact


def log_diagnostic_results_artifact(
    run: "wandb.Run",
    results_path: Path,
    manifest: "RunManifest",
    summary: dict | None = None,
) -> "wandb.Artifact":
    """Log diagnostic results as versioned artifact.

    Args:
        run: Active W&B run.
        results_path: Path to results directory or file.
        manifest: Run manifest.
        summary: Optional summary statistics.

    Returns:
        Logged artifact.
    """
    import wandb

    metadata = {
        "run_id": manifest.run_id,
        "model_name": manifest.model_name,
        "timestamp": manifest.timestamp,
    }
    if summary:
        metadata.update(summary)

    artifact = wandb.Artifact(
        name=f"diagnostics-{manifest.run_id}",
        type="diagnostic_results",
        metadata=metadata,
    )

    path = Path(results_path)
    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))

    run.log_artifact(artifact)

    logger.info(f"Logged diagnostic results artifact: {manifest.run_id}")
    return artifact


def log_repair_results_artifact(
    run: "wandb.Run",
    results_path: Path,
    manifest: "RunManifest",
    n_repaired: int,
    summary: dict | None = None,
) -> "wandb.Artifact":
    """Log repair results as versioned artifact.

    Args:
        run: Active W&B run.
        results_path: Path to repair results.
        manifest: Run manifest.
        n_repaired: Number of tokens repaired.
        summary: Optional summary statistics.

    Returns:
        Logged artifact.
    """
    import wandb

    metadata = {
        "run_id": manifest.run_id,
        "model_name": manifest.model_name,
        "n_repaired": n_repaired,
        "timestamp": manifest.timestamp,
    }
    if summary:
        metadata.update(summary)

    artifact = wandb.Artifact(
        name=f"repairs-{manifest.run_id}",
        type="repair_results",
        metadata=metadata,
    )

    path = Path(results_path)
    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))

    run.log_artifact(artifact)

    logger.info(f"Logged repair results artifact: {n_repaired} tokens")
    return artifact


def save_results_json(
    results: list,
    path: Path,
    include_eigenvalues: bool = False,
) -> None:
    """Save diagnostic results to JSON.

    Args:
        results: List of DiagnosticResult objects.
        path: Output file path.
        include_eigenvalues: Whether to include full eigenvalue spectra.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for result in results:
        if hasattr(result, "to_dict"):
            data.append(result.to_dict(include_eigenvalues=include_eigenvalues))
        else:
            data.append(result)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Saved {len(results)} results to {path}")


def load_results_json(path: Path) -> list[dict]:
    """Load diagnostic results from JSON.

    Args:
        path: Input file path.

    Returns:
        List of result dictionaries.
    """
    with open(path) as f:
        return json.load(f)
