"""Semantic validation for embedding repairs.

This module implements validation to ensure repairs don't cause
unacceptable semantic drift from the original embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dctt.core.types import RepairResult
from dctt.core.exceptions import SemanticDriftError

if TYPE_CHECKING:
    from dctt.neighbors.index import VectorIndex


@dataclass
class ValidationConfig:
    """Configuration for semantic validation."""

    min_similarity: float = 0.7  # Minimum cosine similarity to original
    min_jaccard: float = 0.5  # Minimum neighbor Jaccard overlap
    max_delta: float = 0.3  # Maximum embedding change norm
    raise_on_fail: bool = False  # Raise exception on validation failure


class SemanticValidator:
    """Validates that repairs preserve semantic properties.

    Checks:
    - Cosine similarity to original embedding
    - Jaccard overlap of nearest neighbors
    - Maximum embedding change norm
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """Initialize validator.

        Args:
            config: Validation configuration.
        """
        self.config = config or ValidationConfig()

    def validate(
        self,
        result: RepairResult,
        index: "VectorIndex" | None = None,
        all_embeddings: NDArray[np.float64] | None = None,
        k: int = 50,
    ) -> tuple[bool, dict[str, float]]:
        """Validate a repair result.

        Args:
            result: Repair result to validate.
            index: kNN index (optional, for Jaccard computation).
            all_embeddings: Full embedding matrix (optional).
            k: Number of neighbors for Jaccard.

        Returns:
            Tuple of (is_valid, validation_metrics).
        """
        metrics = {}
        is_valid = True

        # Check similarity to original
        similarity = float(np.dot(
            result.repaired_embedding,
            result.original_embedding
        ))
        metrics["similarity"] = similarity

        if similarity < self.config.min_similarity:
            is_valid = False
            if self.config.raise_on_fail:
                raise SemanticDriftError(
                    result.token_id,
                    "similarity",
                    similarity,
                    self.config.min_similarity,
                )

        # Check delta norm
        metrics["delta_norm"] = result.delta_norm

        if result.delta_norm > self.config.max_delta:
            is_valid = False
            if self.config.raise_on_fail:
                raise SemanticDriftError(
                    result.token_id,
                    "delta_norm",
                    result.delta_norm,
                    self.config.max_delta,
                )

        # Check Jaccard overlap if index provided
        if index is not None and all_embeddings is not None:
            # Get original neighbors
            orig_query = result.original_embedding.reshape(1, -1)
            orig_neighbors, _ = index.query(orig_query, k=k, exclude_self=True)

            # Get repaired neighbors
            rep_query = result.repaired_embedding.reshape(1, -1)
            rep_neighbors, _ = index.query(rep_query, k=k, exclude_self=True)

            # Compute Jaccard
            set_orig = set(orig_neighbors[0].tolist())
            set_rep = set(rep_neighbors[0].tolist())
            intersection = len(set_orig & set_rep)
            union = len(set_orig | set_rep)
            jaccard = intersection / union if union > 0 else 0.0

            metrics["jaccard"] = jaccard

            if jaccard < self.config.min_jaccard:
                is_valid = False
                if self.config.raise_on_fail:
                    raise SemanticDriftError(
                        result.token_id,
                        "jaccard",
                        jaccard,
                        self.config.min_jaccard,
                    )

        return is_valid, metrics

    def validate_batch(
        self,
        results: list[RepairResult],
        index: "VectorIndex" | None = None,
        all_embeddings: NDArray[np.float64] | None = None,
        k: int = 50,
    ) -> tuple[list[bool], list[dict[str, float]]]:
        """Validate multiple repair results.

        Args:
            results: Repair results to validate.
            index: kNN index.
            all_embeddings: Full embedding matrix.
            k: Number of neighbors.

        Returns:
            Tuple of (validity_list, metrics_list).
        """
        validities = []
        all_metrics = []

        for result in results:
            is_valid, metrics = self.validate(
                result, index, all_embeddings, k
            )
            validities.append(is_valid)
            all_metrics.append(metrics)

        return validities, all_metrics


def validate_semantic_preservation(
    result: RepairResult,
    min_similarity: float = 0.7,
    min_jaccard: float = 0.5,
    max_delta: float = 0.3,
) -> bool:
    """Convenience function to validate semantic preservation.

    Args:
        result: Repair result.
        min_similarity: Minimum similarity threshold.
        min_jaccard: Minimum Jaccard threshold.
        max_delta: Maximum delta threshold.

    Returns:
        True if repair passes validation.
    """
    # Check similarity
    similarity = float(np.dot(
        result.repaired_embedding,
        result.original_embedding
    ))
    if similarity < min_similarity:
        return False

    # Check delta
    if result.delta_norm > max_delta:
        return False

    # Check Jaccard from stored validation metrics
    if "neighbor_jaccard" in result.semantic_validation:
        if result.semantic_validation["neighbor_jaccard"] < min_jaccard:
            return False

    return True
