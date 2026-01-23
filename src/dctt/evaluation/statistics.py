"""Statistical analysis utilities for DCTT experiments.

This module provides statistical tools for analyzing experimental
results, including bootstrap confidence intervals and paired tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass
class ConfidenceInterval:
    """Confidence interval result."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    n_bootstrap: int


def bootstrap_ci(
    data: NDArray[np.float64],
    statistic: Callable[[NDArray], float] = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> ConfidenceInterval:
    """Compute bootstrap confidence interval.

    Args:
        data: Data array.
        statistic: Function to compute statistic (default: mean).
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        ConfidenceInterval with estimate and bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(data)

    # Compute bootstrap distribution
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute percentile CI
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return ConfidenceInterval(
        estimate=statistic(data),
        lower=lower,
        upper=upper,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
    )


def paired_permutation_test(
    before: NDArray[np.float64],
    after: NDArray[np.float64],
    n_permutations: int = 10000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Paired permutation test for difference in means.

    Tests H0: mean(after - before) = 0.

    Args:
        before: Measurements before intervention.
        after: Measurements after intervention.
        n_permutations: Number of permutations.
        seed: Random seed.

    Returns:
        Tuple of (observed_diff, p_value).
    """
    if len(before) != len(after):
        raise ValueError("before and after must have same length")

    rng = np.random.default_rng(seed)
    n = len(before)

    # Compute observed difference
    differences = after - before
    observed_diff = np.mean(differences)

    # Permutation distribution under H0
    count_extreme = 0
    for _ in range(n_permutations):
        # Randomly flip signs
        signs = rng.choice([-1, 1], size=n)
        permuted_diff = np.mean(differences * signs)

        if abs(permuted_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return observed_diff, p_value


def compute_effect_size(
    treatment: NDArray[np.float64],
    control: NDArray[np.float64],
) -> dict[str, float]:
    """Compute effect size measures.

    Args:
        treatment: Treatment group measurements.
        control: Control group measurements.

    Returns:
        Dictionary with effect size measures.
    """
    # Cohen's d
    pooled_std = np.sqrt(
        (np.var(treatment, ddof=1) + np.var(control, ddof=1)) / 2
    )
    cohens_d = (np.mean(treatment) - np.mean(control)) / (pooled_std + 1e-10)

    # Difference in means
    mean_diff = np.mean(treatment) - np.mean(control)

    # Relative improvement
    control_mean = np.mean(control)
    relative_improvement = mean_diff / (abs(control_mean) + 1e-10)

    return {
        "cohens_d": cohens_d,
        "mean_difference": mean_diff,
        "relative_improvement": relative_improvement,
        "treatment_mean": np.mean(treatment),
        "control_mean": control_mean,
        "treatment_std": np.std(treatment, ddof=1),
        "control_std": np.std(control, ddof=1),
    }


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for DCTT experiments."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize analyzer.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed

    def analyze_repair_effect(
        self,
        treatment_before: NDArray[np.float64],
        treatment_after: NDArray[np.float64],
        control_before: NDArray[np.float64],
        control_after: NDArray[np.float64],
    ) -> dict:
        """Analyze treatment effect with difference-in-differences.

        Args:
            treatment_before: Treatment group pre-measurements.
            treatment_after: Treatment group post-measurements.
            control_before: Control group pre-measurements.
            control_after: Control group post-measurements.

        Returns:
            Dictionary with analysis results.
        """
        # Treatment group change
        treatment_change = treatment_after - treatment_before
        treatment_ci = bootstrap_ci(
            treatment_change, seed=self.seed
        )

        # Control group change
        control_change = control_after - control_before
        control_ci = bootstrap_ci(
            control_change, seed=self.seed
        )

        # Difference-in-differences
        did = np.mean(treatment_change) - np.mean(control_change)

        # Permutation test on DiD
        _, p_value = paired_permutation_test(
            np.concatenate([treatment_change, control_change]),
            np.concatenate([
                treatment_change - np.mean(treatment_change) + did,
                control_change,
            ]),
            seed=self.seed,
        )

        return {
            "treatment_change": {
                "mean": treatment_ci.estimate,
                "ci_lower": treatment_ci.lower,
                "ci_upper": treatment_ci.upper,
            },
            "control_change": {
                "mean": control_ci.estimate,
                "ci_lower": control_ci.lower,
                "ci_upper": control_ci.upper,
            },
            "difference_in_differences": did,
            "p_value": p_value,
            "effect_size": compute_effect_size(treatment_change, control_change),
        }
