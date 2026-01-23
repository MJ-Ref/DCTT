"""Stage 1 Basic Local Outlier Checks.

This module implements fast, cheap metrics for initial token screening:
- Mean kNN distance (μ_k)
- Median kNN distance (med_k)
- Quantile spread ratio (spread_q)
- Local Outlier Factor (LOF) - optional

These metrics identify obvious outliers and sparse neighborhoods before
the more expensive spectral analysis in Stage 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dctt.core.types import Stage1Result
from dctt.core.registry import register_metric

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_mean_knn_distance(distances: NDArray[np.float64]) -> float:
    """Compute mean distance to k nearest neighbors.

    Args:
        distances: Array of distances to k neighbors.

    Returns:
        Mean distance.
    """
    return float(np.mean(distances))


def compute_median_knn_distance(distances: NDArray[np.float64]) -> float:
    """Compute median distance to k nearest neighbors.

    Args:
        distances: Array of distances to k neighbors.

    Returns:
        Median distance.
    """
    return float(np.median(distances))


def compute_spread_ratio(
    distances: NDArray[np.float64],
    q_low: float = 0.10,
    q_high: float = 0.90,
    eps: float = 1e-10,
) -> float:
    """Compute quantile spread ratio.

    spread_q = q_high(distances) / (q_low(distances) + eps)

    A high spread ratio indicates non-uniform neighbor distribution,
    which may signal geometric pathology.

    Args:
        distances: Array of distances to k neighbors.
        q_low: Lower quantile (default 10%).
        q_high: Upper quantile (default 90%).
        eps: Small constant to prevent division by zero.

    Returns:
        Spread ratio (≥ 1).
    """
    q_low_val = np.quantile(distances, q_low)
    q_high_val = np.quantile(distances, q_high)
    return float(q_high_val / (q_low_val + eps))


def compute_lof(
    embeddings: NDArray[np.float64],
    token_id: int,
    k: int = 20,
    metric: str = "cosine",
) -> float:
    """Compute Local Outlier Factor for a token.

    LOF measures the local density deviation of a point with respect
    to its neighbors. Points with LOF >> 1 are outliers.

    Note: This is more expensive than other Stage 1 metrics and is
    optional. Uses sklearn's LocalOutlierFactor.

    Args:
        embeddings: Normalized embedding matrix (vocab_size, dim).
        token_id: Index of the token to analyze.
        k: Number of neighbors for LOF computation.
        metric: Distance metric ("cosine" or "euclidean").

    Returns:
        LOF score (higher = more anomalous).
    """
    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError:
        raise ImportError("sklearn required for LOF computation")

    # LOF requires fitting on the full dataset
    # We compute it for a single point by using negative_outlier_factor_
    lof = LocalOutlierFactor(
        n_neighbors=min(k, len(embeddings) - 1),
        metric=metric,
        novelty=False,  # For inlier detection
    )

    # Fit on embeddings
    lof.fit(embeddings)

    # Return negative of negative_outlier_factor (so higher = more anomalous)
    return float(-lof.negative_outlier_factor_[token_id])


def compute_stage1_metrics(
    distances: NDArray[np.float64],
    token_id: int,
    q_low: float = 0.10,
    q_high: float = 0.90,
    eps: float = 1e-10,
) -> Stage1Result:
    """Compute all Stage 1 metrics for a token.

    Args:
        distances: Array of distances to k nearest neighbors.
        token_id: Index of the token being analyzed.
        q_low: Lower quantile for spread ratio.
        q_high: Upper quantile for spread ratio.
        eps: Numerical stability constant.

    Returns:
        Stage1Result with all computed metrics.
    """
    mu_k = compute_mean_knn_distance(distances)
    med_k = compute_median_knn_distance(distances)
    spread_q = compute_spread_ratio(distances, q_low, q_high, eps)

    return Stage1Result(
        token_id=token_id,
        mu_k=mu_k,
        med_k=med_k,
        spread_q=spread_q,
        lof=None,  # Computed separately due to expense
        distances=distances,
        fail=False,  # Threshold check happens in thresholding module
    )


def compute_stage1_metrics_batch(
    distance_matrix: NDArray[np.float64],
    token_ids: Sequence[int],
    q_low: float = 0.10,
    q_high: float = 0.90,
    eps: float = 1e-10,
) -> list[Stage1Result]:
    """Compute Stage 1 metrics for multiple tokens.

    Args:
        distance_matrix: Distance matrix of shape (n_tokens, k).
        token_ids: Sequence of token indices.
        q_low: Lower quantile for spread ratio.
        q_high: Upper quantile for spread ratio.
        eps: Numerical stability constant.

    Returns:
        List of Stage1Result, one per token.
    """
    results = []
    for i, token_id in enumerate(token_ids):
        result = compute_stage1_metrics(
            distances=distance_matrix[i],
            token_id=token_id,
            q_low=q_low,
            q_high=q_high,
            eps=eps,
        )
        results.append(result)
    return results


def compute_lof_batch(
    embeddings: NDArray[np.float64],
    token_ids: Sequence[int] | None = None,
    k: int = 20,
    metric: str = "cosine",
) -> NDArray[np.float64]:
    """Compute LOF scores for multiple tokens.

    More efficient than calling compute_lof repeatedly, as it only
    fits the model once.

    Args:
        embeddings: Normalized embedding matrix.
        token_ids: Tokens to compute LOF for (None = all).
        k: Number of neighbors for LOF.
        metric: Distance metric.

    Returns:
        Array of LOF scores.
    """
    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError:
        raise ImportError("sklearn required for LOF computation")

    lof = LocalOutlierFactor(
        n_neighbors=min(k, len(embeddings) - 1),
        metric=metric,
        novelty=False,
    )
    lof.fit(embeddings)

    # Get all LOF scores
    all_scores = -lof.negative_outlier_factor_

    if token_ids is None:
        return all_scores
    return all_scores[list(token_ids)]


# Registered metric classes for the registry pattern


@register_metric("mu_k")
class MeanKNNDistanceMetric:
    """Mean k-NN distance metric."""

    name = "mu_k"
    higher_is_worse = True  # High mean distance suggests outlier

    def compute(
        self,
        embeddings: NDArray[np.float64],
        token_id: int,
        neighbors: NDArray[np.int64],
    ) -> float:
        # Compute distances to neighbors
        token_emb = embeddings[token_id]
        neighbor_embs = embeddings[neighbors]
        # Cosine distance on normalized embeddings = 1 - dot product
        distances = 1.0 - (neighbor_embs @ token_emb)
        return compute_mean_knn_distance(distances)


@register_metric("med_k")
class MedianKNNDistanceMetric:
    """Median k-NN distance metric."""

    name = "med_k"
    higher_is_worse = True

    def compute(
        self,
        embeddings: NDArray[np.float64],
        token_id: int,
        neighbors: NDArray[np.int64],
    ) -> float:
        token_emb = embeddings[token_id]
        neighbor_embs = embeddings[neighbors]
        distances = 1.0 - (neighbor_embs @ token_emb)
        return compute_median_knn_distance(distances)


@register_metric("spread_q")
class SpreadRatioMetric:
    """Quantile spread ratio metric."""

    name = "spread_q"
    higher_is_worse = True  # High spread suggests non-uniform neighborhood

    def __init__(self, q_low: float = 0.10, q_high: float = 0.90) -> None:
        self.q_low = q_low
        self.q_high = q_high

    def compute(
        self,
        embeddings: NDArray[np.float64],
        token_id: int,
        neighbors: NDArray[np.int64],
    ) -> float:
        token_emb = embeddings[token_id]
        neighbor_embs = embeddings[neighbors]
        distances = 1.0 - (neighbor_embs @ token_emb)
        return compute_spread_ratio(distances, self.q_low, self.q_high)
