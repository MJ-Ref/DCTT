"""Constrained embedding repair optimizer.

This module implements the two-loop optimization scheme for repairing
token embeddings:

Outer loop: Recompute neighbors based on current embedding
Inner loop: Gradient descent on combined loss with constraints

Constraints:
- L2 normalization (unit norm)
- Maximum delta from original embedding
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dctt.core.types import RepairConfig, RepairResult
from dctt.repair.losses import CombinedRepairLoss, CombinedLossConfig
from dctt.metrics.stage2 import compute_stage2_metrics

if TYPE_CHECKING:
    from dctt.neighbors.index import VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class OptimizerState:
    """Internal state of the optimizer."""

    embedding: NDArray[np.float64]
    loss: float
    iteration: int
    converged: bool
    loss_history: list[float]


class EmbeddingRepairOptimizer:
    """Constrained optimizer for embedding repair.

    Uses projected gradient descent with:
    - L2 normalization constraint
    - Maximum delta constraint
    - Two-loop neighbor recomputation

    Example:
        >>> optimizer = EmbeddingRepairOptimizer(config)
        >>> result = optimizer.repair(
        ...     embedding, neighbors, all_embeddings, index
        ... )
    """

    def __init__(self, config: RepairConfig | None = None) -> None:
        """Initialize optimizer.

        Args:
            config: Repair configuration.
        """
        self.config = config or RepairConfig()

        # Create combined loss
        loss_config = CombinedLossConfig(
            geometry_objective=self.config.geometry_loss,
            lambda_geometry=1.0,
            lambda_anchor=self.config.lambda_anchor,
            lambda_nn_preserve=self.config.lambda_nn_preserve,
            eps=self.config.eps,
        )
        self.loss_fn = CombinedRepairLoss(loss_config)

    def repair(
        self,
        embedding: NDArray[np.float64],
        neighbors: NDArray[np.int64],
        all_embeddings: NDArray[np.float64],
        index: "VectorIndex",
        k: int = 50,
    ) -> RepairResult:
        """Repair a single embedding.

        Args:
            embedding: Original embedding vector (d,).
            neighbors: Initial neighbor indices.
            all_embeddings: Full embedding matrix (for neighbor lookup).
            index: kNN index for neighbor recomputation.
            k: Number of neighbors to use.

        Returns:
            RepairResult with repaired embedding and metrics.
        """
        original_embedding = embedding.copy()
        current_embedding = embedding.copy()

        # Compute initial geometry
        geometry_before = self._compute_geometry_metrics(
            current_embedding, neighbors, all_embeddings
        )

        loss_history = []
        total_iterations = 0
        converged = False

        # Outer loop: recompute neighbors
        for outer_iter in range(self.config.max_outer_iters):
            # Get neighbor embeddings
            neighbor_embeddings = all_embeddings[neighbors]

            # Inner loop: gradient descent
            for inner_step in range(self.config.max_inner_steps):
                # Compute loss and gradient
                loss, components = self.loss_fn.compute(
                    current_embedding, original_embedding, neighbor_embeddings
                )
                gradient = self.loss_fn.gradient(
                    current_embedding, original_embedding, neighbor_embeddings
                )

                grad_norm = np.linalg.norm(gradient)

                # Debug logging for first few iterations
                if total_iterations < 3:
                    logger.debug(f"Iter {total_iterations}: loss={loss:.6f}, grad_norm={grad_norm:.6f}")
                    logger.debug(f"  Components: {components}")

                loss_history.append(loss)
                total_iterations += 1

                # Check convergence
                if len(loss_history) > 10:
                    recent_improvement = loss_history[-10] - loss
                    if recent_improvement < 1e-6:
                        converged = True
                        break

                # Project gradient to tangent space of unit sphere before step
                # This ensures the update stays on the manifold
                grad_tangent = gradient - np.dot(gradient, current_embedding) * current_embedding

                # Gradient step on tangent space
                current_embedding = current_embedding - self.config.learning_rate * grad_tangent

                # Project to constraints
                current_embedding = self._project_constraints(
                    current_embedding, original_embedding
                )

            if converged:
                break

            # Recompute neighbors for next outer iteration
            if outer_iter < self.config.max_outer_iters - 1:
                query_vec = current_embedding.reshape(1, -1)
                new_neighbors, _ = index.query(query_vec, k=k, exclude_self=True)
                neighbors = new_neighbors[0]

        # Compute final geometry
        geometry_after = self._compute_geometry_metrics(
            current_embedding, neighbors, all_embeddings
        )

        # Compute semantic validation metrics
        semantic_validation = self._compute_semantic_validation(
            current_embedding, original_embedding, neighbors, all_embeddings, index, k
        )

        # Compute delta norm
        delta_norm = float(np.linalg.norm(current_embedding - original_embedding))

        return RepairResult(
            token_id=-1,  # Set by caller
            original_embedding=original_embedding,
            repaired_embedding=current_embedding,
            delta_norm=delta_norm,
            geometry_before=geometry_before,
            geometry_after=geometry_after,
            semantic_validation=semantic_validation,
            converged=converged,
            iterations=total_iterations,
            final_loss=loss_history[-1] if loss_history else float("inf"),
        )

    def _project_constraints(
        self,
        embedding: NDArray[np.float64],
        original_embedding: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Project embedding to satisfy constraints.

        Constraints:
        1. Unit L2 norm
        2. Maximum delta from original

        Args:
            embedding: Current embedding.
            original_embedding: Original embedding.

        Returns:
            Projected embedding.
        """
        # First normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Then enforce max delta constraint
        delta = embedding - original_embedding
        delta_norm = np.linalg.norm(delta)

        if delta_norm > self.config.delta_max:
            # Project to ball
            delta = delta * (self.config.delta_max / delta_norm)
            embedding = original_embedding + delta

            # Re-normalize after projection
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    def _compute_geometry_metrics(
        self,
        embedding: NDArray[np.float64],
        neighbors: NDArray[np.int64],
        all_embeddings: NDArray[np.float64],
    ) -> dict[str, float]:
        """Compute geometry metrics for an embedding.

        Args:
            embedding: Embedding vector.
            neighbors: Neighbor indices.
            all_embeddings: Full embedding matrix.

        Returns:
            Dictionary of metric values.
        """
        neighbor_embeddings = all_embeddings[neighbors]
        displacement = neighbor_embeddings - embedding
        displacement_centered = displacement - displacement.mean(axis=0)

        k = displacement.shape[0]
        # Use k×k Gram matrix for efficiency (same non-zero eigenvalues as d×d)
        # This is O(k³) instead of O(d³) when k << d
        gram_matrix = (displacement_centered @ displacement_centered.T) / (k - 1)

        eigenvalues = np.linalg.eigvalsh(gram_matrix)
        eigenvalues = np.maximum(eigenvalues, self.config.eps)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Compute metrics
        dim95 = self._compute_dim95(eigenvalues)
        pr = self._compute_pr(eigenvalues)
        cond = self._compute_cond(eigenvalues)
        logdet = float(np.sum(np.log(eigenvalues + self.config.eps)))
        anisotropy = float(eigenvalues[0] / (eigenvalues.mean() + self.config.eps))

        return {
            "dim95": float(dim95),
            "pr": pr,
            "cond": cond,
            "logdet": logdet,
            "anisotropy": anisotropy,
        }

    def _compute_dim95(self, eigenvalues: NDArray[np.float64]) -> int:
        """Compute effective dimension at 95% variance."""
        total = eigenvalues.sum()
        if total <= 0:
            return 1
        cumsum = np.cumsum(eigenvalues) / total
        indices = np.where(cumsum >= 0.95)[0]
        return int(indices[0] + 1) if len(indices) > 0 else len(eigenvalues)

    def _compute_pr(self, eigenvalues: NDArray[np.float64]) -> float:
        """Compute participation ratio."""
        sum_lambda = eigenvalues.sum()
        sum_lambda_sq = (eigenvalues**2).sum()
        return float((sum_lambda**2) / (sum_lambda_sq + self.config.eps))

    def _compute_cond(self, eigenvalues: NDArray[np.float64]) -> float:
        """Compute condition number."""
        m = min(10, len(eigenvalues) - 1)
        return float((eigenvalues[0] + self.config.eps) / (eigenvalues[m] + self.config.eps))

    def _compute_semantic_validation(
        self,
        repaired_embedding: NDArray[np.float64],
        original_embedding: NDArray[np.float64],
        neighbors: NDArray[np.int64],
        all_embeddings: NDArray[np.float64],
        index: "VectorIndex",
        k: int,
    ) -> dict[str, float]:
        """Compute semantic validation metrics.

        Args:
            repaired_embedding: Repaired embedding.
            original_embedding: Original embedding.
            neighbors: Current neighbors.
            all_embeddings: Full embedding matrix.
            index: kNN index.
            k: Number of neighbors.

        Returns:
            Dictionary of validation metrics.
        """
        # Similarity to original
        similarity = float(np.dot(repaired_embedding, original_embedding))

        # Query new neighbors for repaired embedding
        query_vec = repaired_embedding.reshape(1, -1)
        new_neighbors, _ = index.query(query_vec, k=k, exclude_self=True)
        new_neighbors = new_neighbors[0]

        # Jaccard overlap with original neighbors
        set_old = set(neighbors.tolist())
        set_new = set(new_neighbors.tolist())
        intersection = len(set_old & set_new)
        union = len(set_old | set_new)
        jaccard = intersection / union if union > 0 else 0.0

        return {
            "similarity_to_original": similarity,
            "neighbor_jaccard": jaccard,
        }


def repair_embedding(
    embedding: NDArray[np.float64],
    token_id: int,
    all_embeddings: NDArray[np.float64],
    index: "VectorIndex",
    config: RepairConfig | None = None,
    k: int = 50,
) -> RepairResult:
    """Convenience function to repair a single embedding.

    Args:
        embedding: Original embedding vector.
        token_id: Token ID being repaired.
        all_embeddings: Full embedding matrix.
        index: kNN index.
        config: Repair configuration.
        k: Number of neighbors.

    Returns:
        RepairResult with repaired embedding.
    """
    optimizer = EmbeddingRepairOptimizer(config)

    # Get initial neighbors
    query_vec = embedding.reshape(1, -1)
    neighbors, _ = index.query(query_vec, k=k, exclude_self=True)

    result = optimizer.repair(
        embedding=embedding,
        neighbors=neighbors[0],
        all_embeddings=all_embeddings,
        index=index,
        k=k,
    )
    result.token_id = token_id
    return result


def repair_embeddings_batch(
    embeddings: NDArray[np.float64],
    token_ids: list[int],
    all_embeddings: NDArray[np.float64],
    index: "VectorIndex",
    config: RepairConfig | None = None,
    k: int = 50,
) -> list[RepairResult]:
    """Repair multiple embeddings.

    Args:
        embeddings: Embeddings to repair, shape (n, d).
        token_ids: Token IDs being repaired.
        all_embeddings: Full embedding matrix.
        index: kNN index.
        config: Repair configuration.
        k: Number of neighbors.

    Returns:
        List of RepairResult.
    """
    results = []
    optimizer = EmbeddingRepairOptimizer(config)

    for i, (embedding, token_id) in enumerate(zip(embeddings, token_ids)):
        query_vec = embedding.reshape(1, -1)
        neighbors, _ = index.query(query_vec, k=k, exclude_self=True)

        result = optimizer.repair(
            embedding=embedding,
            neighbors=neighbors[0],
            all_embeddings=all_embeddings,
            index=index,
            k=k,
        )
        result.token_id = token_id
        results.append(result)

        if (i + 1) % 10 == 0:
            logger.info(f"Repaired {i + 1}/{len(token_ids)} embeddings")

    return results
