"""Consistency estimation (cons@k) for diagnostic robustness.

This module implements the consistency metric that measures how
reliably a token is flagged across multiple diagnostic runs with
controlled perturbations (different index seeds, k values, etc.).

High consistency indicates the diagnostic is robust and the token
should be prioritized for repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence
    from dctt.neighbors.index import VectorIndex


@dataclass
class ConsistencyResult:
    """Result of consistency estimation.

    Attributes:
        token_id: Token being analyzed.
        consistency: cons@k score in [0, 1].
        num_runs: Number of diagnostic runs.
        fail_counts: Number of times each stage failed.
        bayesian_mean: Posterior mean (Beta distribution).
        bayesian_ci: 95% credible interval.
    """

    token_id: int
    consistency: float
    num_runs: int
    fail_counts: dict[str, int]
    bayesian_mean: float
    bayesian_ci: tuple[float, float]


class ConsistencyEstimator:
    """Estimates diagnostic consistency across perturbations.

    Consistency is computed by running diagnostics multiple times with:
    - Different random seeds for kNN index construction
    - Different k values for neighborhood size
    - Random projections (optional)

    Example:
        >>> estimator = ConsistencyEstimator(num_runs=5)
        >>> consistency = estimator.estimate(
        ...     embeddings, token_id, build_index_fn, compute_metrics_fn
        ... )
    """

    def __init__(
        self,
        num_runs: int = 5,
        methods: list[str] | None = None,
        k_variants: list[int] | None = None,
        base_seed: int = 42,
    ) -> None:
        """Initialize estimator.

        Args:
            num_runs: Number of diagnostic runs per method.
            methods: Perturbation methods to use.
            k_variants: Different k values to test.
            base_seed: Base random seed for reproducibility.
        """
        self.num_runs = num_runs
        self.methods = methods or ["index_seeds"]
        self.k_variants = k_variants or [25, 50, 75, 100]
        self.base_seed = base_seed

    def estimate(
        self,
        embeddings: NDArray[np.float64],
        token_id: int,
        build_index_fn: Callable[..., "VectorIndex"],
        compute_fail_fn: Callable[[int, NDArray[np.int64]], bool],
        k: int = 50,
    ) -> ConsistencyResult:
        """Estimate consistency for a single token.

        Args:
            embeddings: Normalized embedding matrix.
            token_id: Token to analyze.
            build_index_fn: Function to build kNN index (accepts seed).
            compute_fail_fn: Function that returns True if token fails
                            given (token_id, neighbor_indices).
            k: Default neighborhood size.

        Returns:
            ConsistencyResult with estimated consistency.
        """
        fail_count = 0
        total_runs = 0

        # Method: Different index seeds
        if "index_seeds" in self.methods:
            for i in range(self.num_runs):
                seed = self.base_seed + i
                index = build_index_fn(embeddings, seed=seed)

                # Query neighbors
                query_vec = embeddings[token_id].reshape(1, -1)
                neighbors, _ = index.query(query_vec, k=k, exclude_self=True)

                # Check if fails
                if compute_fail_fn(token_id, neighbors[0]):
                    fail_count += 1
                total_runs += 1

        # Method: Different k values
        if "k_variants" in self.methods:
            # Build single index
            index = build_index_fn(embeddings, seed=self.base_seed)
            max_k = max(self.k_variants)
            query_vec = embeddings[token_id].reshape(1, -1)
            all_neighbors, _ = index.query(query_vec, k=max_k, exclude_self=True)

            for k_var in self.k_variants:
                neighbors = all_neighbors[0, :k_var]
                if compute_fail_fn(token_id, neighbors):
                    fail_count += 1
                total_runs += 1

        # Compute consistency
        consistency = fail_count / total_runs if total_runs > 0 else 0.0

        # Bayesian estimation (Beta posterior)
        alpha = 1 + fail_count  # Prior: Beta(1, 1) = uniform
        beta = 1 + (total_runs - fail_count)
        bayesian_mean = alpha / (alpha + beta)

        # 95% credible interval
        from scipy import stats
        ci_low = stats.beta.ppf(0.025, alpha, beta)
        ci_high = stats.beta.ppf(0.975, alpha, beta)

        return ConsistencyResult(
            token_id=token_id,
            consistency=consistency,
            num_runs=total_runs,
            fail_counts={"total": fail_count},
            bayesian_mean=bayesian_mean,
            bayesian_ci=(ci_low, ci_high),
        )

    def estimate_batch(
        self,
        embeddings: NDArray[np.float64],
        token_ids: Sequence[int],
        build_index_fn: Callable[..., "VectorIndex"],
        compute_fail_fn: Callable[[int, NDArray[np.int64]], bool],
        k: int = 50,
    ) -> list[ConsistencyResult]:
        """Estimate consistency for multiple tokens.

        More efficient than calling estimate() repeatedly as it reuses
        index builds where possible.

        Args:
            embeddings: Normalized embedding matrix.
            token_ids: Tokens to analyze.
            build_index_fn: Function to build kNN index.
            compute_fail_fn: Function that returns True if token fails.
            k: Default neighborhood size.

        Returns:
            List of ConsistencyResult, one per token.
        """
        results = []

        # Pre-build indices for each seed
        indices = {}
        if "index_seeds" in self.methods:
            for i in range(self.num_runs):
                seed = self.base_seed + i
                indices[seed] = build_index_fn(embeddings, seed=seed)

        # Single index for k_variants method
        if "k_variants" in self.methods:
            indices["k_variants"] = build_index_fn(embeddings, seed=self.base_seed)

        for token_id in token_ids:
            fail_count = 0
            total_runs = 0

            # Method: Different index seeds
            if "index_seeds" in self.methods:
                for i in range(self.num_runs):
                    seed = self.base_seed + i
                    index = indices[seed]

                    query_vec = embeddings[token_id].reshape(1, -1)
                    neighbors, _ = index.query(query_vec, k=k, exclude_self=True)

                    if compute_fail_fn(token_id, neighbors[0]):
                        fail_count += 1
                    total_runs += 1

            # Method: Different k values
            if "k_variants" in self.methods:
                index = indices["k_variants"]
                max_k = max(self.k_variants)
                query_vec = embeddings[token_id].reshape(1, -1)
                all_neighbors, _ = index.query(query_vec, k=max_k, exclude_self=True)

                for k_var in self.k_variants:
                    neighbors = all_neighbors[0, :k_var]
                    if compute_fail_fn(token_id, neighbors):
                        fail_count += 1
                    total_runs += 1

            consistency = fail_count / total_runs if total_runs > 0 else 0.0

            # Bayesian estimation
            alpha = 1 + fail_count
            beta = 1 + (total_runs - fail_count)
            bayesian_mean = alpha / (alpha + beta)

            from scipy import stats
            ci_low = stats.beta.ppf(0.025, alpha, beta)
            ci_high = stats.beta.ppf(0.975, alpha, beta)

            results.append(ConsistencyResult(
                token_id=token_id,
                consistency=consistency,
                num_runs=total_runs,
                fail_counts={"total": fail_count},
                bayesian_mean=bayesian_mean,
                bayesian_ci=(ci_low, ci_high),
            ))

        return results


def compute_consistency(
    fail_flags: Sequence[bool],
) -> float:
    """Compute simple consistency from a sequence of fail flags.

    Args:
        fail_flags: Sequence of boolean fail indicators.

    Returns:
        Consistency score (fraction of runs that failed).
    """
    if len(fail_flags) == 0:
        return 0.0
    return sum(fail_flags) / len(fail_flags)
