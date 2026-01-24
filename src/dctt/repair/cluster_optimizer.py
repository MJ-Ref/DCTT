"""Joint cluster repair optimizer.

This module implements joint optimization of token clusters to fix
geometry pathologies. Unlike single-token repair, joint optimization
CAN improve centered displacement covariance because multiple tokens
are moving together.

Key insight: When we compute centered covariance of neighbors, a single
token moving doesn't change the covariance (it just changes the center).
But when multiple neighbors move together, the covariance matrix changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dctt.core.types import RepairConfig
from dctt.repair.cluster import TokenCluster

if TYPE_CHECKING:
    from dctt.neighbors.index import VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class ClusterRepairResult:
    """Result of cluster repair optimization."""

    cluster_id: int
    token_ids: list[int]
    original_embeddings: NDArray[np.float64]
    repaired_embeddings: NDArray[np.float64]

    # Geometry metrics
    geometry_before: dict[str, float]
    geometry_after: dict[str, float]
    geometry_improved: bool

    # Semantic validation
    mean_similarity: float  # Mean similarity to original
    mean_jaccard: float  # Mean neighbor overlap

    # Optimization stats
    converged: bool
    iterations: int
    final_loss: float


class ClusterRepairOptimizer:
    """Joint optimizer for repairing token clusters.

    Optimizes all tokens in a cluster simultaneously using:
    1. Joint geometry loss (centered covariance of cluster + neighbors)
    2. Anchor loss per token (prevent drift)
    3. Inter-token distance preservation (maintain cluster structure)

    The joint optimization allows centered covariance to change because
    multiple reference points (the tokens being optimized) move together.
    """

    def __init__(
        self,
        config: RepairConfig | None = None,
        n_external_neighbors: int = 20,
    ) -> None:
        """Initialize cluster optimizer.

        Args:
            config: Repair configuration.
            n_external_neighbors: External neighbors per cluster token.
        """
        self.config = config or RepairConfig()
        self.n_external_neighbors = n_external_neighbors
        self.eps = self.config.eps

    def repair_cluster(
        self,
        cluster: TokenCluster,
        embeddings: NDArray[np.float64],
        index: "VectorIndex",
        k: int = 50,
    ) -> ClusterRepairResult:
        """Repair a cluster of pathological tokens jointly.

        Args:
            cluster: Cluster to repair.
            embeddings: Full embedding matrix (will not be modified).
            index: kNN index for neighbor queries.
            k: Number of neighbors for geometry computation.

        Returns:
            ClusterRepairResult with repaired embeddings.
        """
        token_ids = cluster.token_ids
        n_tokens = len(token_ids)

        # Extract original embeddings for cluster
        original = embeddings[token_ids].copy()
        current = original.copy()

        # Get external neighbors (tokens not in cluster)
        cluster_set = set(token_ids)
        external_neighbors = self._get_external_neighbors(
            token_ids, embeddings, index, cluster_set
        )

        # Compute initial geometry
        geometry_before = self._compute_cluster_geometry(
            current, external_neighbors, embeddings
        )

        # Optimization loop
        loss_history = []
        converged = False

        for outer_iter in range(self.config.max_outer_iters):
            # Inner optimization steps
            for inner_step in range(self.config.max_inner_steps):
                # Compute joint loss and gradient
                loss, gradients = self._compute_joint_loss_and_gradient(
                    current, original, external_neighbors, embeddings
                )
                loss_history.append(loss)

                # Check convergence
                if len(loss_history) > 10:
                    recent_improvement = loss_history[-10] - loss
                    if recent_improvement < 1e-6:
                        converged = True
                        break

                # Update each token embedding
                for i in range(n_tokens):
                    grad = gradients[i]

                    # Project gradient to tangent space
                    grad_tangent = grad - np.dot(grad, current[i]) * current[i]

                    # Gradient step
                    current[i] = current[i] - self.config.learning_rate * grad_tangent

                    # Project to constraints
                    current[i] = self._project_constraints(current[i], original[i])

            if converged:
                break

            # Recompute external neighbors for next outer iteration
            if outer_iter < self.config.max_outer_iters - 1:
                external_neighbors = self._get_external_neighbors(
                    token_ids, embeddings, index, cluster_set, current_embeddings=current
                )

        # Compute final geometry
        geometry_after = self._compute_cluster_geometry(
            current, external_neighbors, embeddings
        )

        # Compute semantic validation
        similarities = []
        jaccards = []
        for i, tid in enumerate(token_ids):
            sim = float(np.dot(current[i], original[i]))
            similarities.append(sim)

            # Compute neighbor overlap
            query_orig = original[i].reshape(1, -1)
            query_new = current[i].reshape(1, -1)
            orig_neighbors, _ = index.query(query_orig, k=k, exclude_self=True)
            new_neighbors, _ = index.query(query_new, k=k, exclude_self=True)

            orig_set = set(orig_neighbors[0].tolist())
            new_set = set(new_neighbors[0].tolist())
            jaccard = len(orig_set & new_set) / len(orig_set | new_set)
            jaccards.append(jaccard)

        # Check if geometry improved
        cond_improved = geometry_after["cond"] < geometry_before["cond"]
        pr_improved = geometry_after["pr"] > geometry_before["pr"]
        geometry_improved = cond_improved or pr_improved

        return ClusterRepairResult(
            cluster_id=cluster.cluster_id,
            token_ids=token_ids,
            original_embeddings=original,
            repaired_embeddings=current,
            geometry_before=geometry_before,
            geometry_after=geometry_after,
            geometry_improved=geometry_improved,
            mean_similarity=float(np.mean(similarities)),
            mean_jaccard=float(np.mean(jaccards)),
            converged=converged,
            iterations=len(loss_history),
            final_loss=loss_history[-1] if loss_history else float("inf"),
        )

    def _get_external_neighbors(
        self,
        token_ids: list[int],
        embeddings: NDArray[np.float64],
        index: "VectorIndex",
        cluster_set: set[int],
        current_embeddings: NDArray[np.float64] | None = None,
    ) -> NDArray[np.int64]:
        """Get external neighbors for cluster tokens.

        Args:
            token_ids: Cluster token IDs.
            embeddings: Full embedding matrix.
            index: kNN index.
            cluster_set: Set of cluster token IDs (to exclude).
            current_embeddings: Current cluster embeddings (if different from original).

        Returns:
            Array of external neighbor IDs, shape (n_cluster, n_external).
        """
        n_tokens = len(token_ids)
        external = np.zeros((n_tokens, self.n_external_neighbors), dtype=np.int64)

        for i, tid in enumerate(token_ids):
            if current_embeddings is not None:
                query_vec = current_embeddings[i].reshape(1, -1)
            else:
                query_vec = embeddings[tid].reshape(1, -1)

            # Query more neighbors to filter out cluster members
            neighbors, _ = index.query(
                query_vec, k=self.n_external_neighbors + len(cluster_set), exclude_self=True
            )

            # Filter out cluster members
            external_list = [n for n in neighbors[0] if n not in cluster_set]
            n_ext = min(len(external_list), self.n_external_neighbors)
            external[i, :n_ext] = external_list[:n_ext]

        return external

    def _compute_cluster_geometry(
        self,
        cluster_embeddings: NDArray[np.float64],
        external_neighbors: NDArray[np.int64],
        all_embeddings: NDArray[np.float64],
    ) -> dict[str, float]:
        """Compute aggregate geometry metrics for cluster.

        Uses centered covariance computed from:
        - Cluster tokens themselves
        - External neighbors of cluster tokens
        """
        n_cluster = cluster_embeddings.shape[0]

        # Collect all relevant embeddings
        all_points = list(cluster_embeddings)

        # Add external neighbor embeddings
        for i in range(n_cluster):
            for ext_id in external_neighbors[i]:
                if ext_id > 0:  # Valid neighbor
                    all_points.append(all_embeddings[ext_id])

        points = np.array(all_points)

        # Compute centered covariance
        centroid = points.mean(axis=0)
        centered = points - centroid

        # Use k×k Gram matrix for efficiency
        k = len(points)
        gram = (centered @ centered.T) / (k - 1)

        eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = np.maximum(eigenvalues, self.eps)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Compute metrics
        dim95 = self._compute_dim95(eigenvalues)
        pr = self._compute_pr(eigenvalues)
        cond = self._compute_cond(eigenvalues)
        logdet = float(np.sum(np.log(eigenvalues + self.eps)))
        anisotropy = float(eigenvalues[0] / (eigenvalues.mean() + self.eps))

        return {
            "dim95": float(dim95),
            "pr": pr,
            "cond": cond,
            "logdet": logdet,
            "anisotropy": anisotropy,
        }

    def _compute_joint_loss_and_gradient(
        self,
        current: NDArray[np.float64],
        original: NDArray[np.float64],
        external_neighbors: NDArray[np.int64],
        all_embeddings: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        """Compute joint loss and gradients for all cluster tokens.

        Loss = geometry_loss + lambda_anchor * anchor_loss + lambda_structure * structure_loss

        Args:
            current: Current cluster embeddings (n_cluster, d).
            original: Original cluster embeddings (n_cluster, d).
            external_neighbors: External neighbor IDs (n_cluster, n_external).
            all_embeddings: Full embedding matrix.

        Returns:
            Tuple of (loss, gradients) where gradients has shape (n_cluster, d).
        """
        n_cluster, d = current.shape
        gradients = np.zeros_like(current)

        # Geometry loss: based on cluster + external neighbors
        geometry_loss = 0.0
        geometry_grads = np.zeros_like(current)

        # Collect all points for geometry computation
        all_points = list(current)
        for i in range(n_cluster):
            for ext_id in external_neighbors[i]:
                if ext_id > 0:
                    all_points.append(all_embeddings[ext_id])

        points = np.array(all_points)
        n_points = len(points)

        # Compute centered covariance
        centroid = points.mean(axis=0)
        centered = points - centroid

        # Gram matrix
        gram = (centered @ centered.T) / (n_points - 1)
        eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = np.maximum(eigenvalues, self.eps)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Condition number loss
        m = min(10, len(eigenvalues) - 1)
        cond = (eigenvalues[0] + self.eps) / (eigenvalues[m] + self.eps)
        geometry_loss = np.log(cond + 1)

        # Numerical gradient for geometry (for cluster tokens only)
        delta = 1e-5
        for i in range(n_cluster):
            for j in range(d):
                current_plus = current.copy()
                current_plus[i, j] += delta

                # Recompute geometry with perturbed embedding
                all_points_plus = list(current_plus)
                for k in range(n_cluster):
                    for ext_id in external_neighbors[k]:
                        if ext_id > 0:
                            all_points_plus.append(all_embeddings[ext_id])

                points_plus = np.array(all_points_plus)
                centroid_plus = points_plus.mean(axis=0)
                centered_plus = points_plus - centroid_plus
                gram_plus = (centered_plus @ centered_plus.T) / (n_points - 1)
                eigs_plus = np.linalg.eigvalsh(gram_plus)
                eigs_plus = np.maximum(eigs_plus, self.eps)
                eigs_plus = np.sort(eigs_plus)[::-1]

                cond_plus = (eigs_plus[0] + self.eps) / (eigs_plus[m] + self.eps)
                loss_plus = np.log(cond_plus + 1)

                geometry_grads[i, j] = (loss_plus - geometry_loss) / delta

        # Anchor loss: keep close to original
        anchor_loss = 0.0
        anchor_grads = np.zeros_like(current)
        for i in range(n_cluster):
            diff = current[i] - original[i]
            anchor_loss += np.sum(diff ** 2)
            anchor_grads[i] = 2 * diff

        anchor_loss *= self.config.lambda_anchor

        # Structure loss: preserve inter-token distances within cluster
        structure_loss = 0.0
        structure_grads = np.zeros_like(current)

        for i in range(n_cluster):
            for j in range(i + 1, n_cluster):
                orig_dist = np.linalg.norm(original[i] - original[j])
                curr_dist = np.linalg.norm(current[i] - current[j])
                dist_diff = curr_dist - orig_dist

                structure_loss += dist_diff ** 2

                # Gradient
                if curr_dist > self.eps:
                    direction = (current[i] - current[j]) / curr_dist
                    grad = 2 * dist_diff * direction
                    structure_grads[i] += grad * 0.1  # Small weight
                    structure_grads[j] -= grad * 0.1

        # Combine losses and gradients
        total_loss = geometry_loss + anchor_loss + structure_loss
        gradients = (
            geometry_grads +
            self.config.lambda_anchor * anchor_grads +
            structure_grads
        )

        return total_loss, gradients

    def _project_constraints(
        self,
        embedding: NDArray[np.float64],
        original: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Project embedding to satisfy constraints."""
        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Enforce max delta
        delta = embedding - original
        delta_norm = np.linalg.norm(delta)

        if delta_norm > self.config.delta_max:
            delta = delta * (self.config.delta_max / delta_norm)
            embedding = original + delta
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    def _compute_dim95(self, eigenvalues: NDArray[np.float64]) -> int:
        total = eigenvalues.sum()
        if total <= 0:
            return 1
        cumsum = np.cumsum(eigenvalues) / total
        indices = np.where(cumsum >= 0.95)[0]
        return int(indices[0] + 1) if len(indices) > 0 else len(eigenvalues)

    def _compute_pr(self, eigenvalues: NDArray[np.float64]) -> float:
        sum_lambda = eigenvalues.sum()
        sum_lambda_sq = (eigenvalues ** 2).sum()
        return float((sum_lambda ** 2) / (sum_lambda_sq + self.eps))

    def _compute_cond(self, eigenvalues: NDArray[np.float64]) -> float:
        m = min(10, len(eigenvalues) - 1)
        return float((eigenvalues[0] + self.eps) / (eigenvalues[m] + self.eps))


def repair_cluster(
    cluster: TokenCluster,
    embeddings: NDArray[np.float64],
    index: "VectorIndex",
    config: RepairConfig | None = None,
    k: int = 50,
) -> ClusterRepairResult:
    """Convenience function to repair a cluster.

    Args:
        cluster: Cluster to repair.
        embeddings: Full embedding matrix.
        index: kNN index.
        config: Repair configuration.
        k: Number of neighbors.

    Returns:
        ClusterRepairResult.
    """
    optimizer = ClusterRepairOptimizer(config)
    return optimizer.repair_cluster(cluster, embeddings, index, k)
