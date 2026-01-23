"""Repair candidate selection.

This module implements strategies for selecting which tokens to
repair based on severity, consistency, and importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dctt.core.types import DiagnosticResult

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class CandidateSelectionConfig:
    """Configuration for candidate selection."""

    top_n: int = 100  # Number of tokens to repair
    min_consistency: float = 0.6  # Minimum cons@k
    min_severity_percentile: float = 95  # Minimum severity percentile
    use_frequency_weight: bool = True  # Weight by log(freq+1)
    frequency_alpha: float = 1.0  # Exponent for frequency weight


class CandidateSelector:
    """Selects repair candidates based on diagnostic results.

    Selection criteria:
    - High severity score
    - High consistency (reliable diagnosis)
    - Optionally weighted by frequency importance
    """

    def __init__(self, config: CandidateSelectionConfig | None = None) -> None:
        """Initialize selector.

        Args:
            config: Selection configuration.
        """
        self.config = config or CandidateSelectionConfig()

    def select(
        self,
        results: Sequence[DiagnosticResult],
    ) -> list[DiagnosticResult]:
        """Select repair candidates from diagnostic results.

        Args:
            results: Sequence of diagnostic results.

        Returns:
            List of selected candidates, sorted by priority.
        """
        # Filter by minimum consistency
        candidates = [r for r in results if r.consistency >= self.config.min_consistency]

        if not candidates:
            return []

        # Compute severity percentile threshold
        severities = [r.severity for r in candidates]
        severity_threshold = np.percentile(
            severities, self.config.min_severity_percentile
        )

        # Filter by severity
        candidates = [r for r in candidates if r.severity >= severity_threshold]

        if not candidates:
            return []

        # Compute priority scores
        for result in candidates:
            freq = result.token_info.frequency
            if self.config.use_frequency_weight:
                freq_weight = np.log(freq + 1) ** self.config.frequency_alpha
            else:
                freq_weight = 1.0

            result.priority = result.severity * result.consistency * freq_weight

        # Sort by priority (descending) and take top N
        candidates.sort(key=lambda r: r.priority, reverse=True)
        return candidates[: self.config.top_n]

    def select_with_matched_controls(
        self,
        results: Sequence[DiagnosticResult],
    ) -> tuple[list[DiagnosticResult], list[DiagnosticResult]]:
        """Select candidates and matched controls.

        Args:
            results: Sequence of diagnostic results.

        Returns:
            Tuple of (candidates, controls).
        """
        candidates = self.select(results)
        candidate_ids = {r.token_id for r in candidates}

        # Find controls matched on frequency tier and token type
        controls = []
        for candidate in candidates:
            bucket = candidate.bucket
            if bucket is None:
                continue

            # Find tokens in same bucket but not selected
            for result in results:
                if result.token_id in candidate_ids:
                    continue
                if result.bucket == bucket:
                    # Check if severity is not too high (should be "normal")
                    if result.severity < candidate.severity * 0.5:
                        controls.append(result)
                        break

        return candidates, controls


def select_repair_candidates(
    results: Sequence[DiagnosticResult],
    top_n: int = 100,
    min_consistency: float = 0.6,
    min_severity_percentile: float = 95,
) -> list[DiagnosticResult]:
    """Convenience function for candidate selection.

    Args:
        results: Diagnostic results.
        top_n: Number of candidates to select.
        min_consistency: Minimum consistency threshold.
        min_severity_percentile: Minimum severity percentile.

    Returns:
        List of selected candidates.
    """
    config = CandidateSelectionConfig(
        top_n=top_n,
        min_consistency=min_consistency,
        min_severity_percentile=min_severity_percentile,
    )
    selector = CandidateSelector(config)
    return selector.select(results)
