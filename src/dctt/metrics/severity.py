"""Severity scoring for token geometry pathologies.

This module implements the composite severity score that aggregates
multiple metrics into a single ranking for repair candidate selection.

The severity score uses robust z-scores (based on median and MAD)
computed within frequency-type buckets to ensure fair comparison
across different token categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dctt.core.types import Stage1Result, Stage2Result, DiagnosticResult, Bucket

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class SeverityWeights:
    """Weights for combining metrics into severity score.

    Attributes:
        cond: Weight for condition number (higher is worse).
        pr: Weight for participation ratio (lower is worse, so negated).
        logdet: Weight for log-determinant (lower is worse, so negated).
        spread_q: Weight for spread ratio (higher is worse).
        anisotropy: Weight for anisotropy (higher is worse).
    """

    cond: float = 1.0
    pr: float = 1.0
    logdet: float = 1.0
    spread_q: float = 0.5
    anisotropy: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "cond": self.cond,
            "pr": self.pr,
            "logdet": self.logdet,
            "spread_q": self.spread_q,
            "anisotropy": self.anisotropy,
        }


@dataclass
class BucketStatistics:
    """Statistics for metrics within a bucket.

    Used for computing robust z-scores.
    """

    bucket: Bucket
    count: int = 0
    metric_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def add_metric(
        self,
        metric_name: str,
        median: float,
        mad: float,
    ) -> None:
        """Add statistics for a metric."""
        self.metric_stats[metric_name] = {
            "median": median,
            "mad": mad,
        }


def compute_mad(values: NDArray[np.float64]) -> float:
    """Compute Median Absolute Deviation.

    MAD is a robust measure of dispersion:
        MAD = median(|x_i - median(x)|)

    Args:
        values: Array of values.

    Returns:
        MAD value.
    """
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def compute_robust_zscore(
    value: float,
    median: float,
    mad: float,
    eps: float = 1e-10,
) -> float:
    """Compute robust z-score using median and MAD.

    z = (value - median) / (1.4826 * MAD + eps)

    The 1.4826 factor makes MAD consistent with standard deviation
    for normal distributions.

    Args:
        value: Value to compute z-score for.
        median: Bucket median.
        mad: Bucket MAD.
        eps: Small constant to prevent division by zero.

    Returns:
        Robust z-score.
    """
    # 1.4826 is the consistency constant for MAD
    return (value - median) / (1.4826 * mad + eps)


class SeverityScorer:
    """Computes severity scores for tokens based on geometry metrics.

    The severity score aggregates multiple metrics into a single value
    for ranking repair candidates. It uses robust statistics (median, MAD)
    computed within frequency-type buckets.

    Example:
        >>> scorer = SeverityScorer(weights=SeverityWeights())
        >>> scorer.fit(diagnostic_results)
        >>> severity = scorer.compute_severity(diagnostic_result)
    """

    def __init__(
        self,
        weights: SeverityWeights | None = None,
        eps: float = 1e-10,
    ) -> None:
        """Initialize scorer.

        Args:
            weights: Metric weights for severity combination.
            eps: Numerical stability constant.
        """
        self.weights = weights or SeverityWeights()
        self.eps = eps
        self.bucket_stats: dict[Bucket, BucketStatistics] = {}
        self._is_fitted = False

    def fit(
        self,
        results: Sequence[DiagnosticResult],
    ) -> "SeverityScorer":
        """Fit bucket statistics from diagnostic results.

        Args:
            results: Sequence of diagnostic results.

        Returns:
            Self for method chaining.
        """
        # Group results by bucket
        bucket_values: dict[Bucket, dict[str, list[float]]] = {}

        for result in results:
            bucket = result.bucket
            if bucket is None:
                continue

            if bucket not in bucket_values:
                bucket_values[bucket] = {
                    "cond": [],
                    "pr": [],
                    "logdet": [],
                    "spread_q": [],
                    "anisotropy": [],
                }

            bucket_values[bucket]["cond"].append(result.stage2.cond)
            bucket_values[bucket]["pr"].append(result.stage2.pr)
            bucket_values[bucket]["logdet"].append(result.stage2.logdet)
            bucket_values[bucket]["spread_q"].append(result.stage1.spread_q)
            bucket_values[bucket]["anisotropy"].append(result.stage2.anisotropy)

        # Compute statistics per bucket
        self.bucket_stats = {}
        for bucket, values in bucket_values.items():
            stats = BucketStatistics(bucket=bucket, count=len(values["cond"]))

            for metric_name, metric_values in values.items():
                arr = np.array(metric_values)
                median = float(np.median(arr))
                mad = compute_mad(arr)
                stats.add_metric(metric_name, median, mad)

            self.bucket_stats[bucket] = stats

        self._is_fitted = True
        return self

    def compute_severity(
        self,
        result: DiagnosticResult,
    ) -> float:
        """Compute severity score for a single token.

        Args:
            result: Diagnostic result for the token.

        Returns:
            Severity score (higher = more pathological).
        """
        if not self._is_fitted:
            raise RuntimeError("Scorer not fitted. Call fit() first.")

        bucket = result.bucket
        if bucket is None or bucket not in self.bucket_stats:
            return 0.0

        stats = self.bucket_stats[bucket]
        weights = self.weights

        # Extract metric values
        metrics = {
            "cond": result.stage2.cond,
            "pr": result.stage2.pr,
            "logdet": result.stage2.logdet,
            "spread_q": result.stage1.spread_q,
            "anisotropy": result.stage2.anisotropy,
        }

        # Compute z-scores and combine
        severity = 0.0

        for metric_name, value in metrics.items():
            metric_stats = stats.metric_stats.get(metric_name)
            if metric_stats is None:
                continue

            z = compute_robust_zscore(
                value,
                metric_stats["median"],
                metric_stats["mad"],
                self.eps,
            )

            # Get weight
            weight = getattr(weights, metric_name, 0.0)

            # For metrics where lower is worse (pr, logdet), negate z-score
            if metric_name in ("pr", "logdet"):
                z = -z

            severity += weight * z

        return severity

    def compute_severity_batch(
        self,
        results: Sequence[DiagnosticResult],
    ) -> NDArray[np.float64]:
        """Compute severity scores for multiple tokens.

        Args:
            results: Sequence of diagnostic results.

        Returns:
            Array of severity scores.
        """
        return np.array([self.compute_severity(r) for r in results])


def compute_severity_score(
    stage1: Stage1Result,
    stage2: Stage2Result,
    bucket_stats: BucketStatistics,
    weights: SeverityWeights | None = None,
    eps: float = 1e-10,
) -> float:
    """Compute severity score for a token.

    Standalone function for computing severity when you have
    individual stage results and bucket statistics.

    Args:
        stage1: Stage 1 results.
        stage2: Stage 2 results.
        bucket_stats: Statistics for the token's bucket.
        weights: Metric weights.
        eps: Numerical stability constant.

    Returns:
        Severity score.
    """
    weights = weights or SeverityWeights()

    metrics = {
        "cond": stage2.cond,
        "pr": stage2.pr,
        "logdet": stage2.logdet,
        "spread_q": stage1.spread_q,
        "anisotropy": stage2.anisotropy,
    }

    severity = 0.0
    for metric_name, value in metrics.items():
        metric_stats = bucket_stats.metric_stats.get(metric_name)
        if metric_stats is None:
            continue

        z = compute_robust_zscore(
            value,
            metric_stats["median"],
            metric_stats["mad"],
            eps,
        )

        weight = getattr(weights, metric_name, 0.0)

        if metric_name in ("pr", "logdet"):
            z = -z

        severity += weight * z

    return severity


def compute_priority_score(
    severity: float,
    consistency: float,
    frequency: float,
    alpha: float = 1.0,
) -> float:
    """Compute priority score for repair candidate selection.

    priority = severity * consistency * log(frequency + 1)^alpha

    Combines severity (how bad the geometry is), consistency (how
    reliable the diagnosis is), and frequency importance.

    Args:
        severity: Severity score.
        consistency: Consistency score (cons@k).
        frequency: Token frequency.
        alpha: Exponent for frequency term.

    Returns:
        Priority score.
    """
    freq_importance = np.log(frequency + 1) ** alpha
    return severity * consistency * freq_importance
