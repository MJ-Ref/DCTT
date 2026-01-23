"""Stage 2 Spectral Geometry Metrics.

This module implements the core spectral geometry metrics that form the main
contribution of DCTT. All metrics are computed on the displacement matrix
(Option B from spec), which makes the local shape around a token explicit.

Key metrics:
- dim95: Effective dimension at 95% explained variance
- pr: Participation ratio (measures eigenvalue spread)
- cond: Local condition number (numerical stability indicator)
- logdet: Log-determinant of covariance (dispersion measure)
- anisotropy: Ratio of dominant eigenvalue to mean
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dctt.core.types import Stage2Result
from dctt.core.registry import register_metric

if TYPE_CHECKING:
    from collections.abc import Sequence

# Type aliases
FloatArray = NDArray[np.float64]


def compute_displacement_matrix(
    embeddings: FloatArray,
    token_id: int,
    neighbor_ids: NDArray[np.int64],
) -> FloatArray:
    """Compute displacement matrix from token to its neighbors.

    The displacement matrix Δ captures the local geometry around a token:
        Δ[i] = embedding[neighbor_i] - embedding[token]

    This makes "local shape around token" explicit, and editing the token
    embedding affects all rows.

    Args:
        embeddings: Normalized embedding matrix (V, d).
        token_id: Index of the target token.
        neighbor_ids: Indices of k nearest neighbors.

    Returns:
        Displacement matrix of shape (k, d).
    """
    token_embedding = embeddings[token_id]
    neighbor_embeddings = embeddings[neighbor_ids]
    return neighbor_embeddings - token_embedding


def compute_local_covariance(
    displacement_matrix: FloatArray,
    center: bool = True,
) -> FloatArray:
    """Compute local covariance matrix from displacement matrix.

    Args:
        displacement_matrix: Displacement matrix Δ of shape (k, d).
        center: Whether to center the displacements (recommended).

    Returns:
        Covariance matrix of shape (d, d).
    """
    k = displacement_matrix.shape[0]

    if center:
        delta_centered = displacement_matrix - displacement_matrix.mean(axis=0)
    else:
        delta_centered = displacement_matrix

    # Compute covariance: C = (Δc^T @ Δc) / (k - 1)
    covariance = (delta_centered.T @ delta_centered) / (k - 1)

    return covariance


def compute_eigenvalues(
    covariance: FloatArray,
    eps: float = 1e-10,
) -> FloatArray:
    """Compute eigenvalues of covariance matrix.

    Uses symmetric eigenvalue decomposition for numerical stability.
    Eigenvalues are returned in descending order.

    Args:
        covariance: Covariance matrix (d, d).
        eps: Small constant for numerical stability.

    Returns:
        Eigenvalues in descending order, shape (d,).
    """
    # Use eigh for symmetric matrices (more stable than eig)
    eigenvalues = np.linalg.eigvalsh(covariance)

    # Ensure non-negative (numerical issues can cause tiny negatives)
    eigenvalues = np.maximum(eigenvalues, eps)

    # Sort in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]

    return eigenvalues


def compute_effective_dimension(
    eigenvalues: FloatArray,
    variance_threshold: float = 0.95,
) -> int:
    """Compute effective dimension at given variance threshold.

    Finds the minimum m such that the top m eigenvalues explain at least
    `variance_threshold` of the total variance.

    Args:
        eigenvalues: Eigenvalues in descending order.
        variance_threshold: Fraction of variance to explain (default 0.95).

    Returns:
        Effective dimension (minimum 1).
    """
    total_variance = eigenvalues.sum()
    if total_variance <= 0:
        return 1

    cumulative_variance = np.cumsum(eigenvalues) / total_variance

    # Find first index where we exceed threshold
    indices = np.where(cumulative_variance >= variance_threshold)[0]
    if len(indices) > 0:
        return int(indices[0] + 1)  # +1 because 0-indexed
    return len(eigenvalues)


def compute_participation_ratio(
    eigenvalues: FloatArray,
    eps: float = 1e-10,
) -> float:
    """Compute participation ratio.

    PR = (Σλ)² / Σ(λ²)

    The participation ratio measures the "effective number of contributing
    dimensions". It equals:
    - d (embedding dimension) when all eigenvalues are equal
    - 1 when a single eigenvalue dominates

    Args:
        eigenvalues: Eigenvalues (any order).
        eps: Small constant for numerical stability.

    Returns:
        Participation ratio in range [1, d].
    """
    sum_lambda = eigenvalues.sum()
    sum_lambda_sq = (eigenvalues**2).sum()

    if sum_lambda_sq < eps:
        return 1.0

    pr = (sum_lambda**2) / (sum_lambda_sq + eps)
    return float(pr)


def compute_condition_number(
    eigenvalues: FloatArray,
    m_min: int = 10,
    dim95: int | None = None,
    eps: float = 1e-10,
) -> float:
    """Compute local condition number.

    cond = (λ₁ + ε) / (λₘ + ε)

    where m = max(dim95, m_min) to avoid dividing by tiny eigenvalues
    that are essentially numerical noise.

    High condition numbers indicate ill-conditioned local geometry.

    Args:
        eigenvalues: Eigenvalues in descending order.
        m_min: Minimum dimension for denominator.
        dim95: Effective dimension at 95% variance (optional).
        eps: Small constant for numerical stability.

    Returns:
        Condition number (≥ 1).
    """
    if len(eigenvalues) == 0:
        return 1.0

    lambda_max = eigenvalues[0]

    # Determine which eigenvalue to use for denominator
    if dim95 is not None:
        m = max(dim95, m_min)
    else:
        m = m_min

    # Ensure m is within bounds
    m = min(m, len(eigenvalues)) - 1  # Convert to 0-indexed
    m = max(m, 0)

    lambda_m = eigenvalues[m]

    cond = (lambda_max + eps) / (lambda_m + eps)
    return float(cond)


def compute_log_determinant(
    eigenvalues: FloatArray,
    eps: float = 1e-10,
) -> float:
    """Compute log-determinant of covariance matrix.

    logdet(C) = Σ log(λᵢ + ε)

    This is a measure of the local "volume" or dispersion. Low logdet
    indicates collapsed/degenerate geometry.

    Args:
        eigenvalues: Eigenvalues (any order).
        eps: Small constant for numerical stability.

    Returns:
        Log-determinant value.
    """
    log_eigenvalues = np.log(eigenvalues + eps)
    return float(log_eigenvalues.sum())


def compute_anisotropy(
    eigenvalues: FloatArray,
    eps: float = 1e-10,
) -> float:
    """Compute anisotropy measure.

    anisotropy = λ₁ / (mean(λ) + ε)

    High anisotropy indicates the local geometry is dominated by a single
    direction (potentially problematic for gradient-based learning).

    Args:
        eigenvalues: Eigenvalues in descending order.
        eps: Small constant for numerical stability.

    Returns:
        Anisotropy ratio (≥ 1).
    """
    if len(eigenvalues) == 0:
        return 1.0

    lambda_max = eigenvalues[0]
    mean_lambda = eigenvalues.mean()

    return float(lambda_max / (mean_lambda + eps))


def compute_stage2_metrics(
    embeddings: FloatArray,
    token_id: int,
    neighbor_ids: NDArray[np.int64],
    variance_threshold: float = 0.95,
    m_min: int = 10,
    eps: float = 1e-10,
    return_eigenvalues: bool = False,
) -> Stage2Result:
    """Compute all Stage 2 spectral geometry metrics for a token.

    This is the main entry point for Stage 2 analysis. It computes the
    displacement matrix, local covariance, eigenvalues, and all derived
    metrics.

    Args:
        embeddings: Normalized embedding matrix (V, d).
        token_id: Index of the target token.
        neighbor_ids: Indices of k nearest neighbors.
        variance_threshold: Threshold for dim95 computation.
        m_min: Minimum dimension for condition number denominator.
        eps: Numerical stability constant.
        return_eigenvalues: Whether to include full eigenvalue spectrum.

    Returns:
        Stage2Result with all computed metrics.
    """
    # Compute displacement matrix
    displacement = compute_displacement_matrix(embeddings, token_id, neighbor_ids)

    # Compute covariance
    covariance = compute_local_covariance(displacement, center=True)

    # Compute eigenvalues
    eigenvalues = compute_eigenvalues(covariance, eps=eps)

    # Compute all metrics
    dim95 = compute_effective_dimension(eigenvalues, variance_threshold)
    pr = compute_participation_ratio(eigenvalues, eps=eps)
    cond = compute_condition_number(eigenvalues, m_min=m_min, dim95=dim95, eps=eps)
    logdet = compute_log_determinant(eigenvalues, eps=eps)
    anisotropy = compute_anisotropy(eigenvalues, eps=eps)

    return Stage2Result(
        token_id=token_id,
        dim95=dim95,
        pr=pr,
        cond=cond,
        logdet=logdet,
        anisotropy=anisotropy,
        eigenvalues=eigenvalues if return_eigenvalues else None,
        fail=False,  # Threshold check happens in thresholding module
    )


def compute_stage2_metrics_batch(
    embeddings: FloatArray,
    token_ids: Sequence[int],
    neighbor_matrix: NDArray[np.int64],
    variance_threshold: float = 0.95,
    m_min: int = 10,
    eps: float = 1e-10,
    return_eigenvalues: bool = False,
) -> list[Stage2Result]:
    """Compute Stage 2 metrics for multiple tokens.

    Args:
        embeddings: Normalized embedding matrix (V, d).
        token_ids: Sequence of token indices to analyze.
        neighbor_matrix: Neighbor indices, shape (len(token_ids), k).
        variance_threshold: Threshold for dim95 computation.
        m_min: Minimum dimension for condition number.
        eps: Numerical stability constant.
        return_eigenvalues: Whether to include full eigenvalue spectra.

    Returns:
        List of Stage2Result, one per token.
    """
    results = []
    for i, token_id in enumerate(token_ids):
        result = compute_stage2_metrics(
            embeddings=embeddings,
            token_id=token_id,
            neighbor_ids=neighbor_matrix[i],
            variance_threshold=variance_threshold,
            m_min=m_min,
            eps=eps,
            return_eigenvalues=return_eigenvalues,
        )
        results.append(result)
    return results


# Registered metric classes for the registry pattern


@register_metric("dim95")
class EffectiveDimensionMetric:
    """Effective dimension at 95% variance threshold."""

    name = "dim95"
    higher_is_worse = False  # Low dimension is bad

    def __init__(self, variance_threshold: float = 0.95) -> None:
        self.variance_threshold = variance_threshold

    def compute(
        self,
        embeddings: FloatArray,
        token_id: int,
        neighbors: NDArray[np.int64],
    ) -> float:
        displacement = compute_displacement_matrix(embeddings, token_id, neighbors)
        covariance = compute_local_covariance(displacement)
        eigenvalues = compute_eigenvalues(covariance)
        return float(compute_effective_dimension(eigenvalues, self.variance_threshold))


@register_metric("pr")
class ParticipationRatioMetric:
    """Participation ratio metric."""

    name = "pr"
    higher_is_worse = False  # Low PR is bad

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def compute(
        self,
        embeddings: FloatArray,
        token_id: int,
        neighbors: NDArray[np.int64],
    ) -> float:
        displacement = compute_displacement_matrix(embeddings, token_id, neighbors)
        covariance = compute_local_covariance(displacement)
        eigenvalues = compute_eigenvalues(covariance, eps=self.eps)
        return compute_participation_ratio(eigenvalues, eps=self.eps)


@register_metric("cond")
class ConditionNumberMetric:
    """Local condition number metric."""

    name = "cond"
    higher_is_worse = True  # High condition number is bad

    def __init__(self, m_min: int = 10, eps: float = 1e-10) -> None:
        self.m_min = m_min
        self.eps = eps

    def compute(
        self,
        embeddings: FloatArray,
        token_id: int,
        neighbors: NDArray[np.int64],
    ) -> float:
        displacement = compute_displacement_matrix(embeddings, token_id, neighbors)
        covariance = compute_local_covariance(displacement)
        eigenvalues = compute_eigenvalues(covariance, eps=self.eps)
        return compute_condition_number(eigenvalues, m_min=self.m_min, eps=self.eps)


@register_metric("logdet")
class LogDeterminantMetric:
    """Log-determinant of local covariance."""

    name = "logdet"
    higher_is_worse = False  # Low logdet is bad (collapsed geometry)

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def compute(
        self,
        embeddings: FloatArray,
        token_id: int,
        neighbors: NDArray[np.int64],
    ) -> float:
        displacement = compute_displacement_matrix(embeddings, token_id, neighbors)
        covariance = compute_local_covariance(displacement)
        eigenvalues = compute_eigenvalues(covariance, eps=self.eps)
        return compute_log_determinant(eigenvalues, eps=self.eps)


@register_metric("anisotropy")
class AnisotropyMetric:
    """Anisotropy measure (λ₁ / mean(λ))."""

    name = "anisotropy"
    higher_is_worse = True  # High anisotropy is bad

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def compute(
        self,
        embeddings: FloatArray,
        token_id: int,
        neighbors: NDArray[np.int64],
    ) -> float:
        displacement = compute_displacement_matrix(embeddings, token_id, neighbors)
        covariance = compute_local_covariance(displacement)
        eigenvalues = compute_eigenvalues(covariance, eps=self.eps)
        return compute_anisotropy(eigenvalues, eps=self.eps)
