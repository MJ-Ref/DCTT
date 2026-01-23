"""Loss functions for embedding repair optimization.

This module implements the various loss components used in the
constrained optimization repair process:

1. Geometry losses: Improve local geometric properties
2. Anchor loss: Keep repaired embedding close to original
3. Neighbor preservation loss: Maintain semantic relationships
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


class RepairLoss(ABC):
    """Abstract base class for repair loss functions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Loss function name."""
        pass

    @abstractmethod
    def compute(
        self,
        embedding: NDArray[np.float64],
        **kwargs,
    ) -> float:
        """Compute loss value."""
        pass

    @abstractmethod
    def gradient(
        self,
        embedding: NDArray[np.float64],
        **kwargs,
    ) -> NDArray[np.float64]:
        """Compute gradient with respect to embedding."""
        pass


class GeometryLoss(RepairLoss):
    """Loss functions based on local geometry metrics.

    Supports multiple geometry objectives:
    - "cond": Minimize condition number
    - "logdet": Maximize log-determinant (minimize negative)
    - "pr": Maximize participation ratio (minimize negative)
    - "combined": Weighted combination
    """

    def __init__(
        self,
        objective: str = "cond",
        eps: float = 1e-10,
        m_min: int = 10,
    ) -> None:
        """Initialize geometry loss.

        Args:
            objective: Which geometry objective to optimize.
            eps: Numerical stability constant.
            m_min: Minimum dimension for condition number.
        """
        self.objective = objective
        self.eps = eps
        self.m_min = m_min

    @property
    def name(self) -> str:
        return f"geometry_{self.objective}"

    def compute(
        self,
        embedding: NDArray[np.float64],
        neighbor_embeddings: NDArray[np.float64],
        **kwargs,
    ) -> float:
        """Compute geometry loss.

        Args:
            embedding: Current embedding vector (d,).
            neighbor_embeddings: Neighbor embeddings (k, d).

        Returns:
            Loss value (lower is better).
        """
        # Compute displacement matrix
        displacement = neighbor_embeddings - embedding

        # Center displacements
        displacement_centered = displacement - displacement.mean(axis=0)

        # Compute covariance
        k = displacement.shape[0]
        covariance = (displacement_centered.T @ displacement_centered) / (k - 1)

        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.maximum(eigenvalues, self.eps)
        eigenvalues = np.sort(eigenvalues)[::-1]

        if self.objective == "cond":
            # Minimize condition number
            m = min(self.m_min, len(eigenvalues) - 1)
            cond = (eigenvalues[0] + self.eps) / (eigenvalues[m] + self.eps)
            return np.log(cond + 1)  # Log for numerical stability

        elif self.objective == "logdet":
            # Maximize log-determinant (minimize negative)
            logdet = np.sum(np.log(eigenvalues + self.eps))
            return -logdet

        elif self.objective == "pr":
            # Maximize participation ratio (minimize negative)
            sum_lambda = eigenvalues.sum()
            sum_lambda_sq = (eigenvalues**2).sum()
            pr = (sum_lambda**2) / (sum_lambda_sq + self.eps)
            return -pr

        elif self.objective == "combined":
            # Weighted combination
            m = min(self.m_min, len(eigenvalues) - 1)
            cond = (eigenvalues[0] + self.eps) / (eigenvalues[m] + self.eps)
            logdet = np.sum(np.log(eigenvalues + self.eps))
            return np.log(cond + 1) - 0.1 * logdet

        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def gradient(
        self,
        embedding: NDArray[np.float64],
        neighbor_embeddings: NDArray[np.float64],
        **kwargs,
    ) -> NDArray[np.float64]:
        """Compute gradient via finite differences.

        For simplicity, we use numerical differentiation.
        Analytical gradients could be derived but are complex.

        Args:
            embedding: Current embedding vector (d,).
            neighbor_embeddings: Neighbor embeddings (k, d).

        Returns:
            Gradient vector (d,).
        """
        delta = 1e-5
        d = len(embedding)
        grad = np.zeros(d)

        base_loss = self.compute(embedding, neighbor_embeddings)

        for i in range(d):
            embedding_plus = embedding.copy()
            embedding_plus[i] += delta
            loss_plus = self.compute(embedding_plus, neighbor_embeddings)
            grad[i] = (loss_plus - base_loss) / delta

        return grad


class AnchorLoss(RepairLoss):
    """Loss to keep repaired embedding close to original.

    L_anchor = ||x - x_0||^2

    Prevents excessive drift from the original embedding.
    """

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize anchor loss.

        Args:
            weight: Weight for this loss component.
        """
        self.weight = weight

    @property
    def name(self) -> str:
        return "anchor"

    def compute(
        self,
        embedding: NDArray[np.float64],
        original_embedding: NDArray[np.float64],
        **kwargs,
    ) -> float:
        """Compute anchor loss.

        Args:
            embedding: Current embedding vector.
            original_embedding: Original embedding vector.

        Returns:
            Squared L2 distance.
        """
        diff = embedding - original_embedding
        return float(self.weight * np.sum(diff**2))

    def gradient(
        self,
        embedding: NDArray[np.float64],
        original_embedding: NDArray[np.float64],
        **kwargs,
    ) -> NDArray[np.float64]:
        """Compute gradient of anchor loss.

        Args:
            embedding: Current embedding vector.
            original_embedding: Original embedding vector.

        Returns:
            Gradient vector.
        """
        return 2 * self.weight * (embedding - original_embedding)


class NeighborPreservationLoss(RepairLoss):
    """Loss to preserve similarity to top neighbors.

    L_nn = Σ (x · x_u - x_0 · x_u)^2

    Ensures the repaired embedding maintains similar relationships
    with its most important neighbors.
    """

    def __init__(
        self,
        weight: float = 0.5,
        top_k: int = 10,
    ) -> None:
        """Initialize neighbor preservation loss.

        Args:
            weight: Weight for this loss component.
            top_k: Number of top neighbors to preserve.
        """
        self.weight = weight
        self.top_k = top_k

    @property
    def name(self) -> str:
        return "neighbor_preservation"

    def compute(
        self,
        embedding: NDArray[np.float64],
        original_embedding: NDArray[np.float64],
        neighbor_embeddings: NDArray[np.float64],
        **kwargs,
    ) -> float:
        """Compute neighbor preservation loss.

        Args:
            embedding: Current embedding vector.
            original_embedding: Original embedding vector.
            neighbor_embeddings: Top neighbor embeddings.

        Returns:
            Loss value.
        """
        # Use top_k neighbors
        top_neighbors = neighbor_embeddings[: self.top_k]

        # Current similarities
        current_sims = top_neighbors @ embedding

        # Original similarities
        original_sims = top_neighbors @ original_embedding

        # Sum of squared differences
        diff = current_sims - original_sims
        return float(self.weight * np.sum(diff**2))

    def gradient(
        self,
        embedding: NDArray[np.float64],
        original_embedding: NDArray[np.float64],
        neighbor_embeddings: NDArray[np.float64],
        **kwargs,
    ) -> NDArray[np.float64]:
        """Compute gradient of neighbor preservation loss.

        Args:
            embedding: Current embedding vector.
            original_embedding: Original embedding vector.
            neighbor_embeddings: Top neighbor embeddings.

        Returns:
            Gradient vector.
        """
        top_neighbors = neighbor_embeddings[: self.top_k]

        # Current similarities
        current_sims = top_neighbors @ embedding

        # Original similarities
        original_sims = top_neighbors @ original_embedding

        # Gradient: 2 * weight * Σ (s_i - s_i^0) * x_u_i
        diff = current_sims - original_sims
        grad = 2 * self.weight * (top_neighbors.T @ diff)

        return grad


@dataclass
class CombinedLossConfig:
    """Configuration for combined repair loss."""

    geometry_objective: str = "cond"
    lambda_geometry: float = 1.0
    lambda_anchor: float = 1.0
    lambda_nn_preserve: float = 0.5
    top_k_preserve: int = 10
    eps: float = 1e-10


class CombinedRepairLoss:
    """Combined loss for embedding repair.

    Combines geometry, anchor, and neighbor preservation losses:

    L_total = λ_geom * L_geometry + λ_anchor * L_anchor + λ_nn * L_nn
    """

    def __init__(self, config: CombinedLossConfig | None = None) -> None:
        """Initialize combined loss.

        Args:
            config: Loss configuration.
        """
        self.config = config or CombinedLossConfig()

        self.geometry_loss = GeometryLoss(
            objective=self.config.geometry_objective,
            eps=self.config.eps,
        )
        self.anchor_loss = AnchorLoss(weight=self.config.lambda_anchor)
        self.nn_preserve_loss = NeighborPreservationLoss(
            weight=self.config.lambda_nn_preserve,
            top_k=self.config.top_k_preserve,
        )

    def compute(
        self,
        embedding: NDArray[np.float64],
        original_embedding: NDArray[np.float64],
        neighbor_embeddings: NDArray[np.float64],
    ) -> tuple[float, dict[str, float]]:
        """Compute combined loss.

        Args:
            embedding: Current embedding vector.
            original_embedding: Original embedding vector.
            neighbor_embeddings: Neighbor embeddings.

        Returns:
            Tuple of (total_loss, component_losses).
        """
        components = {}

        # Geometry loss
        geom_loss = self.geometry_loss.compute(embedding, neighbor_embeddings)
        components["geometry"] = geom_loss

        # Anchor loss
        anchor_loss = self.anchor_loss.compute(embedding, original_embedding)
        components["anchor"] = anchor_loss

        # Neighbor preservation loss
        nn_loss = self.nn_preserve_loss.compute(
            embedding, original_embedding, neighbor_embeddings
        )
        components["nn_preserve"] = nn_loss

        # Total loss
        total = (
            self.config.lambda_geometry * geom_loss
            + anchor_loss  # Already weighted
            + nn_loss  # Already weighted
        )
        components["total"] = total

        return total, components

    def gradient(
        self,
        embedding: NDArray[np.float64],
        original_embedding: NDArray[np.float64],
        neighbor_embeddings: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute combined gradient.

        Args:
            embedding: Current embedding vector.
            original_embedding: Original embedding vector.
            neighbor_embeddings: Neighbor embeddings.

        Returns:
            Combined gradient vector.
        """
        # Geometry gradient
        geom_grad = self.geometry_loss.gradient(embedding, neighbor_embeddings)

        # Anchor gradient
        anchor_grad = self.anchor_loss.gradient(embedding, original_embedding)

        # NN preservation gradient
        nn_grad = self.nn_preserve_loss.gradient(
            embedding, original_embedding, neighbor_embeddings
        )

        # Combined gradient
        total_grad = (
            self.config.lambda_geometry * geom_grad
            + anchor_grad
            + nn_grad
        )

        return total_grad
