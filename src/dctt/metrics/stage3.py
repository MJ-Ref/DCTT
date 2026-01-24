"""Stage 3 advanced metrics for DCTT diagnostics.

This module implements optional advanced metrics that provide additional
diagnostic power beyond Stage 2 spectral analysis. These metrics are
computationally expensive and should only be used when justified by ablations.

Includes:
- MLE intrinsic dimension estimation
- Topological Data Analysis (TDA) via persistent homology
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dctt.neighbors.usearch_index import USearchIndex


@dataclass
class Stage3Result:
    """Results from Stage 3 advanced analysis."""

    mle_dimension: float | None = None
    tda_h0_components: int | None = None
    tda_h1_loops: int | None = None
    tda_h1_max_persistence: float | None = None
    tda_h1_total_persistence: float | None = None


class MLEDimensionEstimator:
    """Maximum Likelihood Estimator for intrinsic dimension.

    Based on the method from Levina & Bickel (2004):
    "Maximum Likelihood Estimation of Intrinsic Dimension"
    """

    def __init__(self, k1: int = 10, k2: int = 20) -> None:
        """Initialize estimator.

        Args:
            k1: Minimum number of neighbors.
            k2: Maximum number of neighbors.
        """
        self.k1 = k1
        self.k2 = k2

    def estimate(
        self,
        distances: NDArray[np.floating],
    ) -> float:
        """Estimate intrinsic dimension from neighbor distances.

        Args:
            distances: Sorted distances to k neighbors (excluding self).

        Returns:
            Estimated intrinsic dimension.
        """
        k = len(distances)
        if k < self.k2:
            k2 = k
        else:
            k2 = self.k2

        k1 = min(self.k1, k2 - 1)

        # MLE estimator
        # d_hat = 1 / (1/k * sum_{j=k1}^{k2} log(r_k / r_j))
        r_k = distances[k2 - 1]

        if r_k < 1e-10:
            return 0.0

        log_ratios = []
        for j in range(k1, k2):
            r_j = distances[j]
            if r_j > 1e-10:
                log_ratios.append(np.log(r_k / r_j))

        if not log_ratios:
            return 0.0

        mean_log_ratio = np.mean(log_ratios)
        if mean_log_ratio < 1e-10:
            return float("inf")

        return 1.0 / mean_log_ratio

    def estimate_batch(
        self,
        all_distances: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Estimate dimension for multiple points.

        Args:
            all_distances: (n_points, k) array of distances.

        Returns:
            Array of dimension estimates.
        """
        n_points = all_distances.shape[0]
        dimensions = np.zeros(n_points)

        for i in range(n_points):
            dimensions[i] = self.estimate(all_distances[i])

        return dimensions


class TDAAnalyzer:
    """Topological Data Analysis using persistent homology.

    Computes Vietoris-Rips persistent homology features for
    local neighborhoods.
    """

    def __init__(
        self,
        max_dimension: int = 1,
        max_edge_length: float | None = None,
        n_pca_components: int | None = 50,
    ) -> None:
        """Initialize TDA analyzer.

        Args:
            max_dimension: Maximum homology dimension (0 or 1).
            max_edge_length: Maximum edge length for Rips complex.
            n_pca_components: Number of PCA components for projection.
        """
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.n_pca_components = n_pca_components
        self._ripser_available: bool | None = None

    def _check_ripser(self) -> bool:
        """Check if ripser is available."""
        if self._ripser_available is None:
            try:
                import ripser
                self._ripser_available = True
            except ImportError:
                self._ripser_available = False
        return self._ripser_available

    def _project_to_lower_dim(
        self,
        points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Project points to lower dimension using PCA.

        Args:
            points: (n_points, d) array.

        Returns:
            Projected points.
        """
        if self.n_pca_components is None or points.shape[1] <= self.n_pca_components:
            return points

        # Center the data
        centered = points - points.mean(axis=0)

        # SVD for PCA
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Project to top components
        n_components = min(self.n_pca_components, len(S))
        return centered @ Vt[:n_components].T

    def compute_persistence(
        self,
        points: NDArray[np.floating],
    ) -> dict[int, NDArray[np.floating]]:
        """Compute persistent homology.

        Args:
            points: (n_points, d) array of points.

        Returns:
            Dictionary mapping dimension to persistence diagrams.
        """
        if not self._check_ripser():
            return {}

        import ripser

        # Project to lower dimension if needed
        if self.n_pca_components is not None:
            points = self._project_to_lower_dim(points)

        # Compute persistence
        result = ripser.ripser(
            points,
            maxdim=self.max_dimension,
            thresh=self.max_edge_length,
        )

        return {i: result["dgms"][i] for i in range(self.max_dimension + 1)}

    def extract_features(
        self,
        persistence: dict[int, NDArray[np.floating]],
    ) -> dict[str, float]:
        """Extract features from persistence diagrams.

        Args:
            persistence: Dictionary of persistence diagrams.

        Returns:
            Feature dictionary.
        """
        features = {}

        # H0 features (connected components)
        if 0 in persistence:
            h0 = persistence[0]
            # Filter out infinite persistence (the single connected component)
            finite_h0 = h0[np.isfinite(h0[:, 1])] if len(h0) > 0 else np.array([])
            features["h0_components"] = len(finite_h0) + 1  # +1 for the infinite one

        # H1 features (loops/holes)
        if 1 in persistence:
            h1 = persistence[1]
            if len(h1) > 0:
                # Compute lifetimes (death - birth)
                lifetimes = h1[:, 1] - h1[:, 0]
                finite_lifetimes = lifetimes[np.isfinite(lifetimes)]

                features["h1_loops"] = len(finite_lifetimes)
                features["h1_max_persistence"] = (
                    float(np.max(finite_lifetimes)) if len(finite_lifetimes) > 0 else 0.0
                )
                features["h1_total_persistence"] = float(np.sum(finite_lifetimes))
            else:
                features["h1_loops"] = 0
                features["h1_max_persistence"] = 0.0
                features["h1_total_persistence"] = 0.0

        return features


def compute_stage3_metrics(
    embedding: NDArray[np.floating],
    neighbor_embeddings: NDArray[np.floating],
    neighbor_distances: NDArray[np.floating],
    use_tda: bool = True,
    tda_subsample: int = 100,
) -> Stage3Result:
    """Compute Stage 3 advanced metrics for a token.

    Args:
        embedding: The token embedding (d,).
        neighbor_embeddings: Neighbor embeddings (k, d).
        neighbor_distances: Distances to neighbors (k,).
        use_tda: Whether to compute TDA features.
        tda_subsample: Max neighbors for TDA (expensive).

    Returns:
        Stage3Result with advanced metrics.
    """
    result = Stage3Result()

    # MLE intrinsic dimension
    mle_estimator = MLEDimensionEstimator()
    result.mle_dimension = mle_estimator.estimate(neighbor_distances)

    # TDA features
    if use_tda:
        tda = TDAAnalyzer()
        if tda._check_ripser():
            # Subsample for computational efficiency
            if len(neighbor_embeddings) > tda_subsample:
                indices = np.random.choice(
                    len(neighbor_embeddings),
                    tda_subsample,
                    replace=False,
                )
                points = neighbor_embeddings[indices]
            else:
                points = neighbor_embeddings

            # Include the token itself
            points = np.vstack([embedding, points])

            persistence = tda.compute_persistence(points)
            features = tda.extract_features(persistence)

            result.tda_h0_components = features.get("h0_components")
            result.tda_h1_loops = features.get("h1_loops")
            result.tda_h1_max_persistence = features.get("h1_max_persistence")
            result.tda_h1_total_persistence = features.get("h1_total_persistence")

    return result


def should_run_stage3(
    severity: float,
    severity_threshold: float = 2.0,
    frequency: int = 0,
    frequency_threshold: int = 1000,
) -> bool:
    """Determine if Stage 3 should be run for a token.

    Stage 3 is expensive and should be gated based on:
    - High severity (above threshold)
    - High frequency AND moderate severity
    - Borderline Stage 2 results

    Args:
        severity: Token severity score.
        severity_threshold: Severity threshold for Stage 3.
        frequency: Token frequency.
        frequency_threshold: Frequency threshold for high-freq tokens.

    Returns:
        True if Stage 3 should be run.
    """
    # High severity: always run Stage 3
    if severity >= severity_threshold:
        return True

    # High frequency with moderate severity
    if frequency >= frequency_threshold and severity >= severity_threshold * 0.5:
        return True

    return False
