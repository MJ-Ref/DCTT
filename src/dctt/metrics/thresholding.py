"""Adaptive thresholding for metric flagging.

This module implements per-bucket adaptive thresholds for determining
when tokens fail diagnostic checks. Thresholds are computed as
quantiles within frequency-type buckets rather than using arbitrary
fixed values.

This approach is more defensible to reviewers as thresholds are
data-driven and account for natural variation across token types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dctt.core.types import (
    Stage1Result,
    Stage2Result,
    DiagnosticResult,
    Bucket,
    FrequencyTier,
    TokenType,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class ThresholdConfig:
    """Configuration for adaptive thresholding.

    Attributes:
        quantile_high: Quantile for "high is bad" metrics (e.g., cond).
        quantile_low: Quantile for "low is bad" metrics (e.g., pr).
        stage1_metrics: Stage 1 metrics to threshold.
        stage2_metrics: Stage 2 metrics to threshold.
    """

    quantile_high: float = 0.99
    quantile_low: float = 0.01
    stage1_metrics: list[str] = field(
        default_factory=lambda: ["mu_k", "spread_q"]
    )
    stage2_metrics: list[str] = field(
        default_factory=lambda: ["cond", "pr", "logdet", "dim95"]
    )


@dataclass
class BucketThresholds:
    """Thresholds for a single bucket.

    Stores threshold values for each metric in the bucket.
    """

    bucket: Bucket
    count: int
    thresholds: dict[str, float] = field(default_factory=dict)

    def get_threshold(self, metric_name: str) -> float | None:
        """Get threshold for a metric."""
        return self.thresholds.get(metric_name)


class AdaptiveThresholder:
    """Computes and applies adaptive per-bucket thresholds.

    Thresholds are computed as quantiles of metric distributions
    within each (frequency_tier, token_type) bucket.

    Example:
        >>> thresholder = AdaptiveThresholder(config)
        >>> thresholder.fit(diagnostic_results)
        >>> fail = thresholder.check_stage2_failure(result)
    """

    # Metrics where high values are bad
    HIGH_IS_BAD = {"cond", "anisotropy", "mu_k", "med_k", "spread_q", "lof"}
    # Metrics where low values are bad
    LOW_IS_BAD = {"pr", "logdet", "dim95"}

    def __init__(self, config: ThresholdConfig | None = None) -> None:
        """Initialize thresholder.

        Args:
            config: Threshold configuration.
        """
        self.config = config or ThresholdConfig()
        self.bucket_thresholds: dict[Bucket, BucketThresholds] = {}
        self._is_fitted = False

    def fit(
        self,
        results: Sequence[DiagnosticResult],
    ) -> "AdaptiveThresholder":
        """Compute thresholds from diagnostic results.

        Args:
            results: Sequence of diagnostic results.

        Returns:
            Self for method chaining.
        """
        # Group metric values by bucket
        bucket_values: dict[Bucket, dict[str, list[float]]] = {}

        for result in results:
            bucket = result.bucket
            if bucket is None:
                continue

            if bucket not in bucket_values:
                bucket_values[bucket] = {}

            # Stage 1 metrics
            for metric in self.config.stage1_metrics:
                if metric not in bucket_values[bucket]:
                    bucket_values[bucket][metric] = []
                value = getattr(result.stage1, metric, None)
                if value is not None:
                    bucket_values[bucket][metric].append(value)

            # Stage 2 metrics
            for metric in self.config.stage2_metrics:
                if metric not in bucket_values[bucket]:
                    bucket_values[bucket][metric] = []
                value = getattr(result.stage2, metric, None)
                if value is not None:
                    bucket_values[bucket][metric].append(value)

        # Compute thresholds per bucket
        self.bucket_thresholds = {}
        for bucket, values in bucket_values.items():
            bt = BucketThresholds(bucket=bucket, count=len(next(iter(values.values()))))

            for metric_name, metric_values in values.items():
                arr = np.array(metric_values)
                if len(arr) == 0:
                    continue

                # Determine which quantile to use based on metric
                if metric_name in self.HIGH_IS_BAD:
                    threshold = float(np.quantile(arr, self.config.quantile_high))
                elif metric_name in self.LOW_IS_BAD:
                    threshold = float(np.quantile(arr, self.config.quantile_low))
                else:
                    # Default to high quantile
                    threshold = float(np.quantile(arr, self.config.quantile_high))

                bt.thresholds[metric_name] = threshold

            self.bucket_thresholds[bucket] = bt

        self._is_fitted = True
        return self

    def check_stage1_failure(self, result: DiagnosticResult) -> bool:
        """Check if token fails Stage 1 thresholds.

        Args:
            result: Diagnostic result.

        Returns:
            True if token fails any Stage 1 threshold.
        """
        if not self._is_fitted:
            raise RuntimeError("Thresholder not fitted. Call fit() first.")

        bucket = result.bucket
        if bucket is None or bucket not in self.bucket_thresholds:
            return False

        bt = self.bucket_thresholds[bucket]

        for metric in self.config.stage1_metrics:
            value = getattr(result.stage1, metric, None)
            threshold = bt.get_threshold(metric)

            if value is None or threshold is None:
                continue

            if metric in self.HIGH_IS_BAD:
                if value > threshold:
                    return True
            elif metric in self.LOW_IS_BAD:
                if value < threshold:
                    return True

        return False

    def check_stage2_failure(self, result: DiagnosticResult) -> bool:
        """Check if token fails Stage 2 thresholds.

        Args:
            result: Diagnostic result.

        Returns:
            True if token fails any Stage 2 threshold.
        """
        if not self._is_fitted:
            raise RuntimeError("Thresholder not fitted. Call fit() first.")

        bucket = result.bucket
        if bucket is None or bucket not in self.bucket_thresholds:
            return False

        bt = self.bucket_thresholds[bucket]

        for metric in self.config.stage2_metrics:
            value = getattr(result.stage2, metric, None)
            threshold = bt.get_threshold(metric)

            if value is None or threshold is None:
                continue

            if metric in self.HIGH_IS_BAD:
                if value > threshold:
                    return True
            elif metric in self.LOW_IS_BAD:
                if value < threshold:
                    return True

        return False

    def check_failure(self, result: DiagnosticResult) -> tuple[bool, bool]:
        """Check if token fails Stage 1 or Stage 2 thresholds.

        Args:
            result: Diagnostic result.

        Returns:
            Tuple of (stage1_fail, stage2_fail).
        """
        return (
            self.check_stage1_failure(result),
            self.check_stage2_failure(result),
        )

    def apply_thresholds(
        self,
        results: Sequence[DiagnosticResult],
    ) -> list[DiagnosticResult]:
        """Apply thresholds to update fail flags in results.

        Args:
            results: Diagnostic results to update.

        Returns:
            Updated results with fail flags set.
        """
        updated = []
        for result in results:
            s1_fail = self.check_stage1_failure(result)
            s2_fail = self.check_stage2_failure(result)

            # Update fail flags (results are mutable dataclasses)
            result.stage1.fail = s1_fail
            result.stage2.fail = s2_fail

            updated.append(result)

        return updated

    def get_threshold_summary(self) -> dict[str, dict[str, float]]:
        """Get summary of all thresholds.

        Returns:
            Dictionary mapping bucket names to metric thresholds.
        """
        summary = {}
        for bucket, bt in self.bucket_thresholds.items():
            bucket_name = f"{bucket[0].name}_{bucket[1].name}"
            summary[bucket_name] = bt.thresholds.copy()
        return summary


def compute_adaptive_thresholds(
    results: Sequence[DiagnosticResult],
    config: ThresholdConfig | None = None,
) -> AdaptiveThresholder:
    """Convenience function to compute adaptive thresholds.

    Args:
        results: Diagnostic results to fit on.
        config: Threshold configuration.

    Returns:
        Fitted AdaptiveThresholder.
    """
    thresholder = AdaptiveThresholder(config)
    return thresholder.fit(results)


def classify_token_type(token_str: str) -> TokenType:
    """Classify a token string into a TokenType.

    Args:
        token_str: String representation of the token.

    Returns:
        TokenType classification.
    """
    if not token_str:
        return TokenType.UNKNOWN

    # Check for special tokens
    if token_str.startswith("<") and token_str.endswith(">"):
        return TokenType.SPECIAL

    # Check for whitespace
    if token_str.strip() == "" or token_str in {" ", "\n", "\t", "\r"}:
        return TokenType.WHITESPACE

    # Check for numeric
    if token_str.replace(".", "").replace("-", "").isdigit():
        return TokenType.NUMERIC

    # Check for code symbols
    code_symbols = set("{}[]()<>:;,.=+-*/%&|^~!@#$\\\"'`")
    if all(c in code_symbols for c in token_str):
        if len(token_str) <= 3:  # Short symbol sequences
            return TokenType.CODE_SYMBOL
        return TokenType.PUNCTUATION

    # Check for punctuation
    if len(token_str) == 1 and not token_str.isalnum():
        return TokenType.PUNCTUATION

    # Check for subword (starts with special subword markers)
    subword_prefixes = ("##", "Ġ", "▁", "_", "Ċ")
    if any(token_str.startswith(p) for p in subword_prefixes):
        return TokenType.SUBWORD

    # Check if it looks like a complete word
    if token_str.isalpha() or (token_str.isalnum() and token_str[0].isalpha()):
        # If it starts with lowercase and no subword prefix, likely subword
        if token_str[0].islower() and len(token_str) < 4:
            return TokenType.SUBWORD
        return TokenType.FULL_WORD

    return TokenType.SUBWORD


def classify_frequency_tier(
    frequency: float,
    all_frequencies: NDArray[np.float64],
) -> FrequencyTier:
    """Classify token frequency into a tier.

    Uses log-frequency quantiles:
    - HIGH: top 20%
    - MID: middle 60%
    - LOW: bottom 20%

    Args:
        frequency: Token frequency.
        all_frequencies: All token frequencies for quantile computation.

    Returns:
        FrequencyTier classification.
    """
    log_freq = np.log(frequency + 1)
    log_all = np.log(all_frequencies + 1)

    q20 = np.quantile(log_all, 0.20)
    q80 = np.quantile(log_all, 0.80)

    if log_freq >= q80:
        return FrequencyTier.HIGH
    elif log_freq <= q20:
        return FrequencyTier.LOW
    else:
        return FrequencyTier.MID


def assign_bucket(
    token_str: str,
    frequency: float,
    all_frequencies: NDArray[np.float64],
) -> Bucket:
    """Assign a token to a (frequency_tier, token_type) bucket.

    Args:
        token_str: Token string representation.
        frequency: Token frequency.
        all_frequencies: All frequencies for tier computation.

    Returns:
        Bucket tuple.
    """
    token_type = classify_token_type(token_str)
    freq_tier = classify_frequency_tier(frequency, all_frequencies)
    return (freq_tier, token_type)
