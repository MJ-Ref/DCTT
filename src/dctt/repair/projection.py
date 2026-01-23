"""Projection-based embedding repair (baseline method).

This module implements a simple orthogonalization baseline for
embedding repair. It projects the embedding away from dominant
principal directions to increase dispersion.

This serves as a baseline to compare against the constrained
optimization approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dctt.core.types import RepairResult

if TYPE_CHECKING:
    pass


@dataclass
class ProjectionConfig:
    """Configuration for projection repair."""

    n_components_remove: int = 1  # Number of dominant directions to remove
    scale_factor: float = 0.1  # How much to move in remaining directions
    max_delta: float = 0.3  # Maximum change norm


class ProjectionRepair:
    """Projection-based embedding repair.

    Projects embedding away from dominant PCA directions of the
    local neighborhood to increase geometric dispersion.
    """

    def __init__(self, config: ProjectionConfig | None = None) -> None:
        """Initialize projection repair.

        Args:
            config: Projection configuration.
        """
        self.config = config or ProjectionConfig()

    def repair(
        self,
        embedding: NDArray[np.float64],
        neighbor_embeddings: NDArray[np.float64],
    ) -> RepairResult:
        """Repair embedding via projection.

        Args:
            embedding: Original embedding vector (d,).
            neighbor_embeddings: Neighbor embeddings (k, d).

        Returns:
            RepairResult with repaired embedding.
        """
        original_embedding = embedding.copy()

        # Compute displacement matrix
        displacement = neighbor_embeddings - embedding

        # Center displacements
        displacement_centered = displacement - displacement.mean(axis=0)

        # Compute PCA on displacements
        _, s, vh = np.linalg.svd(displacement_centered, full_matrices=False)

        # Get dominant directions to remove
        n_remove = min(self.config.n_components_remove, len(s))
        dominant_directions = vh[:n_remove]  # (n_remove, d)

        # Project embedding away from dominant directions
        repaired = embedding.copy()
        for direction in dominant_directions:
            projection = np.dot(repaired, direction) * direction
            repaired = repaired - self.config.scale_factor * projection

        # Normalize
        norm = np.linalg.norm(repaired)
        if norm > 0:
            repaired = repaired / norm

        # Enforce max delta
        delta = repaired - original_embedding
        delta_norm = np.linalg.norm(delta)
        if delta_norm > self.config.max_delta:
            delta = delta * (self.config.max_delta / delta_norm)
            repaired = original_embedding + delta
            repaired = repaired / np.linalg.norm(repaired)

        delta_norm = float(np.linalg.norm(repaired - original_embedding))

        return RepairResult(
            token_id=-1,
            original_embedding=original_embedding,
            repaired_embedding=repaired,
            delta_norm=delta_norm,
            geometry_before={},
            geometry_after={},
            semantic_validation={},
            converged=True,
            iterations=1,
            final_loss=0.0,
        )


def orthogonalize_embedding(
    embedding: NDArray[np.float64],
    neighbor_embeddings: NDArray[np.float64],
    n_components_remove: int = 1,
    scale_factor: float = 0.1,
    max_delta: float = 0.3,
) -> NDArray[np.float64]:
    """Orthogonalize embedding away from dominant directions.

    Convenience function for simple projection repair.

    Args:
        embedding: Original embedding vector.
        neighbor_embeddings: Neighbor embeddings.
        n_components_remove: Number of dominant directions to remove.
        scale_factor: How much to move.
        max_delta: Maximum change norm.

    Returns:
        Repaired embedding.
    """
    config = ProjectionConfig(
        n_components_remove=n_components_remove,
        scale_factor=scale_factor,
        max_delta=max_delta,
    )
    repair = ProjectionRepair(config)
    result = repair.repair(embedding, neighbor_embeddings)
    return result.repaired_embedding
